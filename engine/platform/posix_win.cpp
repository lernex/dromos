// posix.h on Windows: the POSIX calls the C runtime lacks, over the Win32 API. Paths arrive as UTF-8 and are opened
// as UTF-16, so a model folder or a user name outside ASCII works whatever the process's code page.
#if defined(_WIN32)

#include "platform/posix.h"

#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

std::wstring widen(const char* s) {
  if (!s || !*s) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s, -1, nullptr, 0);
  if (n <= 0) return std::wstring();
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s, -1, out.data(), n);
  out.resize(static_cast<size_t>(n) - 1);
  return out;
}

std::string narrow(const wchar_t* s) {
  if (!s || !*s) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, s, -1, nullptr, 0, nullptr, nullptr);
  if (n <= 0) return std::string();
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, s, -1, out.data(), n, nullptr, nullptr);
  out.resize(static_cast<size_t>(n) - 1);
  return out;
}

// The system's text for an error code, without its trailing line break.
std::string message(DWORD code) {
  wchar_t* text = nullptr;
  const DWORD n = FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 nullptr, code, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
  std::string out = n ? narrow(text) : "error " + std::to_string(code);
  if (text) LocalFree(text);
  while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' ' || out.back() == '.'))
    out.pop_back();
  return out + " (error " + std::to_string(code) + ")";
}

thread_local std::string t_dl_error;
thread_local bool t_dl_failed = false;

void dl_fail(const std::string& what) {
  t_dl_error = what;
  t_dl_failed = true;
}

// errno for a failed Win32 call, so strerror(errno) says something true after mmap.
void* map_failed() {
  const DWORD code = GetLastError();
  errno = code == ERROR_NOT_ENOUGH_MEMORY || code == ERROR_COMMITMENT_LIMIT || code == ERROR_OUTOFMEMORY ? ENOMEM
          : code == ERROR_ACCESS_DENIED                                                                 ? EACCES
                                                                                                        : EINVAL;
  return MAP_FAILED;
}

// The DLLs the image at `path` imports that cannot be found: not beside it, and not where SearchPathW looks
// (System32, the Windows folder, PATH), which is the order dlopen falls back to. Read from the import table of the
// file mapped as an image resource, so nothing in it runs and none of its imports load. API sets (api-ms-*,
// ext-ms-*) are the loader's own and never missing. Empty when the table cannot be read.
std::vector<std::string> missing_imports(const std::wstring& path) {
  std::vector<std::string> out;
  HMODULE image = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_AS_IMAGE_RESOURCE | LOAD_LIBRARY_AS_DATAFILE);
  if (!image) return out;
  // A resource module's handle carries flags in its low bits; the image starts at the aligned address.
  const auto* base = reinterpret_cast<const unsigned char*>(reinterpret_cast<uintptr_t>(image) & ~uintptr_t{3});
  const auto* dos = reinterpret_cast<const IMAGE_DOS_HEADER*>(base);
  const auto* nt = reinterpret_cast<const IMAGE_NT_HEADERS64*>(base + dos->e_lfanew);
  if (dos->e_magic == IMAGE_DOS_SIGNATURE && nt->Signature == IMAGE_NT_SIGNATURE &&
      nt->OptionalHeader.Magic == IMAGE_NT_OPTIONAL_HDR64_MAGIC) {
    const IMAGE_DATA_DIRECTORY& dir = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    const std::wstring folder = path.substr(0, path.find_last_of(L'\\') + 1);
    for (auto* d = reinterpret_cast<const IMAGE_IMPORT_DESCRIPTOR*>(base + dir.VirtualAddress);
         dir.VirtualAddress && dir.Size && d->Name; ++d) {
      const char* name = reinterpret_cast<const char*>(base + d->Name);
      if (_strnicmp(name, "api-ms-", 7) == 0 || _strnicmp(name, "ext-ms-", 7) == 0) continue;
      const std::wstring wide = widen(name);
      if (GetFileAttributesW((folder + wide).c_str()) != INVALID_FILE_ATTRIBUTES) continue;
      if (SearchPathW(nullptr, wide.c_str(), nullptr, 0, nullptr, nullptr) > 0) continue;
      out.push_back(name);
    }
  }
  FreeLibrary(image);
  return out;
}

// The views mmap made from files, by address: munmap unmaps those and frees everything else.
std::mutex g_views_mu;
std::unordered_map<void*, bool>& views() {
  static auto* m = new std::unordered_map<void*, bool>;
  return *m;
}

}  // namespace

void* dlopen(const char* file, int) {
  if (!file || !*file) {
    dl_fail("dlopen of the main program is not supported on Windows");
    return nullptr;
  }
  std::wstring path = widen(file);
  for (wchar_t& c : path)
    if (c == L'/') c = L'\\';  // LoadLibraryExW wants backslashes in a qualified path
  const bool qualified = path.find(L'\\') != std::wstring::npos;
  HMODULE h = nullptr;
  if (qualified) {
    // The library's own directory first for what it links (cudart beside the CUDA backend), then the system's
    // directories. A dependency found on neither is looked for on PATH, where pip's NVIDIA wheels are put.
    h = LoadLibraryExW(path.c_str(), nullptr, LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h && GetLastError() == ERROR_MOD_NOT_FOUND)
      h = LoadLibraryExW(path.c_str(), nullptr, LOAD_WITH_ALTERED_SEARCH_PATH);
  } else {
    h = LoadLibraryExW(path.c_str(), nullptr, 0);
  }
  if (!h) {
    const DWORD code = GetLastError();
    std::string why = message(code);
    // 126 for a file that is there means one of the DLLs it links is not, and the system's text names neither.
    // Linux's dlerror names the library it could not open; this names the ones the file imports and no folder has.
    if (code == ERROR_MOD_NOT_FOUND && qualified && GetFileAttributesW(path.c_str()) != INVALID_FILE_ATTRIBUTES) {
      std::string names;
      const std::vector<std::string> missing = missing_imports(path);
      for (size_t i = 0; i < missing.size(); ++i)
        names += (i == 0 ? "" : i + 1 == missing.size() ? " and " : ", ") + missing[i];
      why = missing.empty()
                ? "a DLL that one of its DLLs needs could not be found (error 126)"
                : names + (missing.size() == 1 ? ", which it needs, is" : ", which it needs, are") +
                      " not beside it, in System32 or on PATH (error 126)";
    }
    dl_fail(std::string(file) + ": " + why);
    return nullptr;
  }
  return reinterpret_cast<void*>(h);
}

void* dlsym(void* handle, const char* name) {
  FARPROC p = GetProcAddress(reinterpret_cast<HMODULE>(handle), name);
  if (!p) {
    dl_fail(std::string(name) + ": " + message(GetLastError()));
    return nullptr;
  }
  return reinterpret_cast<void*>(p);
}

int dlclose(void* handle) { return FreeLibrary(reinterpret_cast<HMODULE>(handle)) ? 0 : -1; }

char* dlerror(void) {
  if (!t_dl_failed) return nullptr;
  t_dl_failed = false;
  return t_dl_error.data();
}

void* mmap(void* addr, size_t length, int prot, int flags, int fd, int64_t offset) {
  (void)addr;
  if (length == 0) {
    errno = EINVAL;
    return MAP_FAILED;
  }
  if (flags & MAP_ANONYMOUS) {
    // Committed and zeroed, as an anonymous mapping is.
    void* p = VirtualAlloc(nullptr, length, MEM_RESERVE | MEM_COMMIT, (prot & PROT_WRITE) ? PAGE_READWRITE : PAGE_READONLY);
    return p ? p : map_failed();
  }
  if (prot & PROT_WRITE) {  // the engine maps files to read them, never to write them back
    errno = EINVAL;
    return MAP_FAILED;
  }
  const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) {
    errno = EBADF;
    return MAP_FAILED;
  }
  HANDLE mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
  if (!mapping) return map_failed();
  void* view = MapViewOfFile(mapping, FILE_MAP_READ, static_cast<DWORD>(static_cast<uint64_t>(offset) >> 32),
                             static_cast<DWORD>(static_cast<uint64_t>(offset) & 0xffffffffu), length);
  if (!view) {
    void* failed = map_failed();
    CloseHandle(mapping);
    return failed;
  }
  CloseHandle(mapping);  // the view keeps the mapping alive
  std::lock_guard<std::mutex> lock(g_views_mu);
  views()[view] = true;
  return view;
}

int munmap(void* addr, size_t) {
  if (!addr) return -1;
  bool view = false;
  {
    std::lock_guard<std::mutex> lock(g_views_mu);
    view = views().erase(addr) > 0;
  }
  if (view) return UnmapViewOfFile(addr) ? 0 : -1;
  return VirtualFree(addr, 0, MEM_RELEASE) ? 0 : -1;
}

int madvise(void* addr, size_t length, int advice) {
  if (advice == MADV_WILLNEED && addr && length) {
    WIN32_MEMORY_RANGE_ENTRY range{addr, length};
    PrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0);
  }
  return 0;
}

ssize_t pread(int fd, void* buf, size_t count, int64_t offset) {
  const HANDLE file = reinterpret_cast<HANDLE>(_get_osfhandle(fd));
  if (file == INVALID_HANDLE_VALUE) return -1;
  size_t done = 0;
  while (done < count) {
    OVERLAPPED at{};
    const uint64_t pos = static_cast<uint64_t>(offset) + done;
    at.Offset = static_cast<DWORD>(pos & 0xffffffffu);
    at.OffsetHigh = static_cast<DWORD>(pos >> 32);
    const DWORD want = static_cast<DWORD>(count - done < (size_t{1} << 30) ? count - done : (size_t{1} << 30));
    DWORD got = 0;
    if (!ReadFile(file, static_cast<char*>(buf) + done, want, &got, &at)) {
      if (GetLastError() == ERROR_HANDLE_EOF) break;
      return done ? static_cast<ssize_t>(done) : -1;
    }
    if (got == 0) break;
    done += got;
  }
  return static_cast<ssize_t>(done);
}

int lx_open_unbuffered(const char* path) {
  const HANDLE h = CreateFileW(widen(path).c_str(), GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL | FILE_FLAG_NO_BUFFERING, nullptr);
  if (h == INVALID_HANDLE_VALUE) return -1;
  const int fd = _open_osfhandle(reinterpret_cast<intptr_t>(h), _O_RDONLY | _O_BINARY);
  if (fd < 0) CloseHandle(h);
  return fd;
}

int usleep(unsigned int microseconds) {
  Sleep((microseconds + 999) / 1000);
  return 0;
}

int nanosleep(const struct timespec* request, struct timespec* remaining) {
  if (!request) return -1;
  const long long ms = static_cast<long long>(request->tv_sec) * 1000 + (request->tv_nsec + 999999) / 1000000;
  if (ms <= 0) SwitchToThread();
  else Sleep(static_cast<DWORD>(ms));
  if (remaining) *remaining = timespec{};
  return 0;
}

long sysconf(int name) {
  SYSTEM_INFO info;
  GetSystemInfo(&info);
  switch (name) {
    case _SC_PAGESIZE: return static_cast<long>(info.dwPageSize);
    case _SC_NPROCESSORS_ONLN: return static_cast<long>(info.dwNumberOfProcessors);
    default: return -1;
  }
}

char* realpath(const char* path, char* resolved) {
  wchar_t full[32768];
  const DWORD n = GetFullPathNameW(widen(path).c_str(), 32768, full, nullptr);
  if (n == 0 || n >= 32768) return nullptr;
  if (GetFileAttributesW(full) == INVALID_FILE_ATTRIBUTES) return nullptr;
  const std::string out = narrow(full);
  if (resolved) {
    strncpy(resolved, out.c_str(), 259);
    resolved[259] = '\0';
    return resolved;
  }
  return _strdup(out.c_str());
}

#endif
