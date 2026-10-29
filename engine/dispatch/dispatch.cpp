// The backend loader. One backend library is opened for the process and its symbols fill the dispatch
// table (generated/lx_dispatch_gen.cpp). The choice is LX_KERNELS (exactly that file), then LX_BACKEND (that
// backend's file), then a probe of the installed libraries. Nothing here calls an lx_ stub: the probe reads
// lx_rt_device_count from each library's own handle.
#include "dispatch/dispatch.h"

#include "dispatch/dispatch_internal.h"

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "platform/platform.h"
#include "platform/posix.h"

#ifndef LX_INSTALL_LIBDIR
#define LX_INSTALL_LIBDIR "lib"
#endif

namespace lx::dispatch {
namespace {

// A backend's file: libdromos_NAME.so on Linux and macOS, dromos_NAME.dll on Windows.
#if defined(_WIN32)
#define LX_BACKEND_LIB(name) name ".dll"
#else
#define LX_BACKEND_LIB(name) "lib" name ".so"
#endif

// The backends LX_BACKEND can name, the file each one is, and how doctor names it.
struct BackendFile {
  const char* name;   // lower case, as LX_BACKEND is compared
  const char* file;
  const char* label;  // as lx_rt_backend names it
  const char* gpus;   // whose GPUs it drives, "" for the CPU
};
const BackendFile kBackendFiles[] = {
    {"cuda", LX_BACKEND_LIB("dromos_kernels"), "CUDA", "NVIDIA"},
    {"hip", LX_BACKEND_LIB("dromos_hip"), "HIP", "AMD"},
    {"metal", "libdromos_metal.dylib", "Metal", "Apple"},
    {"cpu", LX_BACKEND_LIB("dromos_cpu"), "CPU", ""},
};

// The probe's order. The CPU library comes last and qualifies as soon as it loads: it is the fallback.
const char* const kProbeOrder[] = {LX_BACKEND_LIB("dromos_kernels"), LX_BACKEND_LIB("dromos_hip"),
                                   "libdromos_metal.dylib", LX_BACKEND_LIB("dromos_cpu")};
const char* const kCpuFile = LX_BACKEND_LIB("dromos_cpu");

// The names of the backends by the id lx_rt_backend returns.
const char* const kBackendNames[] = {"CUDA", "HIP", "Metal", "CPU"};

std::once_flag g_once;
Info g_info;
void* g_handle = nullptr;  // the chosen library, open for the life of the process

std::mutex g_mutex;  // guards g_last and g_reported
std::string g_last;
std::set<std::string> g_reported;

std::string join(const std::vector<std::string>& items) {
  std::string out;
  for (const std::string& item : items) out += (out.empty() ? "" : ", ") + item;
  return out;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The path with symlinks and "." and ".." resolved, or the path itself when it does not resolve.
std::string canonical(const std::string& path) {
  char* resolved = realpath(path.c_str(), nullptr);
  if (!resolved) return path;
  std::string out = resolved;
  std::free(resolved);
  return out;
}

std::string dir_of(const std::string& path) {
#if defined(_WIN32)
  const std::size_t slash = path.find_last_of("/\\");
#else
  const std::size_t slash = path.rfind('/');
#endif
  if (slash == std::string::npos) return ".";
  return slash == 0 ? "/" : path.substr(0, slash);
}

// The directory of the running executable (engine/platform), or the current directory when it cannot be found.
std::string executable_dir() {
  const std::string exe = lx::platform::executable_path().string();
  if (!exe.empty()) return dir_of(canonical(exe));
  char cwd[4096];
  return getcwd(cwd, sizeof cwd) ? cwd : ".";
}

// Where a backend library is looked for: next to the executable, then in the install library directory,
// which sits beside the executable's own directory.
std::vector<std::string> search_dirs() {
  const std::string exe = executable_dir();
  return {exe, exe + "/../" + LX_INSTALL_LIBDIR};
}

// Opens a library for the process's dynamic loader. Null on failure, with the reason in `why`.
void* open_library(const std::string& path, std::string& why) {
  void* handle = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!handle) {
    const char* text = dlerror();
    why = text ? text : "dlopen failed";
  }
  return handle;
}

// lx_rt_device_count read from the library's own handle, not through the stubs. 0 when it is not exported.
int device_count(void* handle) {
  using Count = int (*)(void);
  void* sym = dlsym(handle, "lx_rt_device_count");
  return sym ? reinterpret_cast<Count>(sym)() : 0;
}

std::string backend_name(void* handle) {
  using Id = int (*)(void);
  void* sym = dlsym(handle, "lx_rt_backend");
  const int id = sym ? reinterpret_cast<Id>(sym)() : -1;
  if (id < 0 || id >= static_cast<int>(std::size(kBackendNames))) return "Unknown";
  return kBackendNames[id];
}

// Makes `handle` the backend: its name and path go into the info, and its symbols fill the table.
void attach(void* handle, const std::string& path) {
  g_handle = handle;
  g_info.backend = backend_name(handle);
  g_info.path = path;
  detail::bind(handle, g_info.missing);
}

// A library the probe passed over, as doctor prints it: "CUDA, dromos_kernels.dll: found no NVIDIA GPU (...)".
// `why` is the loader's text for one that did not load, or "" for one that loaded and sees no device.
std::string skipped_note(const char* file, const std::string& candidate, std::string why) {
  const BackendFile* backend = nullptr;
  for (const BackendFile& b : kBackendFiles)
    if (std::string(file) == b.file) backend = &b;
  const std::string head = std::string(backend ? backend->label : "Unknown") + ", " + file + ": ";
  if (why.empty())
    return head + "found no " + (backend && *backend->gpus ? std::string(backend->gpus) + " " : "") +
           "GPU (there is none, or its driver is missing or too old)";
  // The loader's text may lead with the path it was given, which the row names already.
  if (why.rfind(candidate + ": ", 0) == 0) why.erase(0, candidate.size() + 2);
  return head + "did not load: " + why;
}

// Probes kProbeOrder in each directory. The first library that loads and qualifies is kept (the handle stays
// open); each one that loads and does not qualify is closed, and `skipped` says why. On failure `error` lists
// what was tried.
void probe(const std::vector<std::string>& dirs, void*& handle, std::string& path, std::string& error,
           std::vector<std::string>& skipped) {
  std::vector<std::string> tried;
  std::vector<std::string> notes;
  for (const char* file : kProbeOrder) {
    for (const std::string& dir : dirs) {
      const std::string candidate = dir + "/" + file;
      tried.push_back(candidate);
      if (access(candidate.c_str(), F_OK) != 0) continue;
      std::string why;
      void* lib = open_library(candidate, why);
      if (!lib) {
        notes.push_back(candidate + " did not load: " + why);
        skipped.push_back(skipped_note(file, candidate, why));
        continue;
      }
      if (std::string(file) == kCpuFile || device_count(lib) > 0) {
        handle = lib;
        path = canonical(candidate);
        return;
      }
      dlclose(lib);
      notes.push_back(candidate + " has no device");
      skipped.push_back(skipped_note(file, candidate, ""));
    }
  }
  error = "no backend library could be loaded; looked in " + join(tried);
  if (!notes.empty()) error += " (" + join(notes) + ")";
}

// The three ways in, in order. LX_KERNELS names the file to load, and that file is the answer even when it
// fails. LX_BACKEND names the backend, and its file is the answer even when it fails. Otherwise the probe.
void load() {
  const std::vector<std::string> dirs = search_dirs();
  void* handle = nullptr;
  std::string path, error;
  const char* kernels = std::getenv("LX_KERNELS");
  const char* want = std::getenv("LX_BACKEND");

  if (kernels && *kernels) {
    path = kernels;
    handle = open_library(path, error);
    if (!handle) error = "LX_KERNELS=" + path + ": " + error;
    else path = canonical(path);
  } else if (want && *want) {
    const std::string name = lower(want);
    const char* file = nullptr;
    for (const BackendFile& b : kBackendFiles)
      if (name == b.name) file = b.file;
    if (!file) {
      error = "LX_BACKEND=" + std::string(want) + " is not cuda, hip, metal or cpu";
    } else {
      std::vector<std::string> tried;
      std::string found;
      for (const std::string& dir : dirs) {
        const std::string candidate = dir + "/" + file;
        tried.push_back(candidate);
        if (found.empty() && access(candidate.c_str(), F_OK) == 0) found = candidate;
      }
      if (found.empty()) {
        error = "LX_BACKEND=" + name + ": " + file + " not found; looked in " + join(tried);
      } else {
        handle = open_library(found, error);
        if (!handle) error = "LX_BACKEND=" + name + ": " + error;
        else path = canonical(found);
      }
    }
  } else {
    probe(dirs, handle, path, error, g_info.skipped);
  }

  if (handle) attach(handle, path);
  else g_info.error = error;
}

}  // namespace

namespace detail {

void ensure() { std::call_once(g_once, load); }

}  // namespace detail

const Info& info() {
  detail::ensure();
  return g_info;
}

std::string library_file(const std::string& backend) {
  for (const BackendFile& b : kBackendFiles)
    if (backend == b.name) return b.file;
  return "";
}

std::string last_message() {
  std::lock_guard<std::mutex> lock(g_mutex);
  return g_last;
}

void report_missing(const char* symbol) {
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_reported.insert(symbol).second) return;
  g_last = "backend " + (g_info.backend.empty() ? std::string("none") : g_info.backend) + " lacks " + symbol;
  std::fprintf(stderr, "dromos: %s\n", g_last.c_str());
}

}  // namespace lx::dispatch
