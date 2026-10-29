#include "platform/platform.h"

#include <cstdlib>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <system_error>

#include "platform/posix.h"

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <winioctl.h>
#else
#include <cerrno>
#include <sys/ioctl.h>
#include <sys/wait.h>
#if defined(__APPLE__)
#include <climits>
#include <mach-o/dyld.h>
#endif
#endif

namespace lx::platform {
namespace fs = std::filesystem;

namespace {

// The engine's folder names. Renaming the engine changes these, install.sh and install.ps1.
constexpr const char* kDotDir = ".dromos";                        // ~/.dromos
constexpr const char* kCacheDir = "dromos";                       // ~/.cache/dromos
[[maybe_unused]] constexpr const wchar_t* kWindowsDir = L"Dromos";  // %LOCALAPPDATA%\[Programs\]Dromos

std::string env(const char* name) {
  const char* v = std::getenv(name);
  return v ? std::string(v) : std::string();
}

#if defined(_WIN32)

std::wstring widen(const std::string& s) {
  if (s.empty()) return std::wstring();
  const int n = MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0);
  std::wstring out(static_cast<size_t>(n), L'\0');
  MultiByteToWideChar(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n);
  return out;
}

std::string narrow(const std::wstring& s) {
  if (s.empty()) return std::string();
  const int n = WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), nullptr, 0, nullptr, nullptr);
  std::string out(static_cast<size_t>(n), '\0');
  WideCharToMultiByte(CP_UTF8, 0, s.data(), static_cast<int>(s.size()), out.data(), n, nullptr, nullptr);
  return out;
}

std::wstring wenv(const wchar_t* name) {
  const DWORD n = GetEnvironmentVariableW(name, nullptr, 0);
  if (n == 0) return std::wstring();
  std::wstring out(n, L'\0');
  const DWORD got = GetEnvironmentVariableW(name, out.data(), n);
  out.resize(got);
  return out;
}

// \\?\C:\x is C:\x and \\?\UNC\server\share is \\server\share: the long forms GetFinalPathNameByHandle returns.
std::wstring strip_long_prefix(std::wstring p) {
  if (p.rfind(L"\\\\?\\UNC\\", 0) == 0) return L"\\\\" + p.substr(8);
  if (p.rfind(L"\\\\?\\", 0) == 0 || p.rfind(L"\\??\\", 0) == 0) return p.substr(4);
  return p;
}

fs::path local_app_data() {
  const std::wstring v = wenv(L"LOCALAPPDATA");
  if (!v.empty()) return fs::path(v);
  return home() / "AppData" / "Local";
}

std::string error_text(DWORD code) {
  return std::system_category().message(static_cast<int>(code)) + " (error " + std::to_string(code) + ")";
}

// REPARSE_DATA_BUFFER (ntifs.h, which user mode does not get) for the two kinds the install reads: a mount point
// (junction) and a symbolic link.
struct ReparseBuffer {
  DWORD tag;
  WORD data_length;
  WORD reserved;
  union {
    struct {
      WORD substitute_offset, substitute_length, print_offset, print_length;
      ULONG flags;
      WCHAR path[1];
    } symlink;
    struct {
      WORD substitute_offset, substitute_length, print_offset, print_length;
      WCHAR path[1];
    } mount;
  };
};

HANDLE open_reparse(const fs::path& p, DWORD access) {
  return CreateFileW(p.c_str(), access, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr, OPEN_EXISTING,
                     FILE_FLAG_OPEN_REPARSE_POINT | FILE_FLAG_BACKUP_SEMANTICS, nullptr);
}

// Points the junction at `link` (an empty directory, or a junction already) at `target`.
void write_junction(const fs::path& link, const fs::path& target) {
  const std::wstring abs = strip_long_prefix(fs::absolute(target).wstring());
  const std::wstring substitute = L"\\??\\" + abs;
  const size_t sub_bytes = substitute.size() * sizeof(wchar_t), print_bytes = abs.size() * sizeof(wchar_t);
  // header (8) + the four offsets (8) + both names, each with its terminating NUL
  const size_t data = 8 + sub_bytes + sizeof(wchar_t) + print_bytes + sizeof(wchar_t);
  std::vector<char> buf(8 + data, 0);
  auto* r = reinterpret_cast<ReparseBuffer*>(buf.data());
  r->tag = IO_REPARSE_TAG_MOUNT_POINT;
  r->data_length = static_cast<WORD>(data);
  r->mount.substitute_offset = 0;
  r->mount.substitute_length = static_cast<WORD>(sub_bytes);
  r->mount.print_offset = static_cast<WORD>(sub_bytes + sizeof(wchar_t));
  r->mount.print_length = static_cast<WORD>(print_bytes);
  std::memcpy(r->mount.path, substitute.c_str(), sub_bytes);
  std::memcpy(reinterpret_cast<char*>(r->mount.path) + sub_bytes + sizeof(wchar_t), abs.c_str(), print_bytes);
  const HANDLE h = open_reparse(link, GENERIC_WRITE);
  if (h == INVALID_HANDLE_VALUE)
    throw std::runtime_error("cannot open " + link.string() + ": " + error_text(GetLastError()));
  DWORD ignored = 0;
  const BOOL ok = DeviceIoControl(h, FSCTL_SET_REPARSE_POINT, buf.data(), static_cast<DWORD>(buf.size()), nullptr, 0,
                                  &ignored, nullptr);
  const DWORD code = GetLastError();
  CloseHandle(h);
  if (!ok) throw std::runtime_error("cannot point " + link.string() + " at " + target.string() + ": " + error_text(code));
}

// Quotes one argument the way the C runtime's command-line parser splits it back (CommandLineToArgvW's rules).
std::wstring quote_arg(const std::wstring& arg) {
  if (!arg.empty() && arg.find_first_of(L" \t\n\v\"") == std::wstring::npos) return arg;
  std::wstring out = L"\"";
  size_t backslashes = 0;
  for (wchar_t c : arg) {
    if (c == L'\\') {
      ++backslashes;
    } else if (c == L'"') {
      out.append(backslashes * 2 + 1, L'\\');
      out.push_back(c);
      backslashes = 0;
    } else {
      out.append(backslashes, L'\\');
      out.push_back(c);
      backslashes = 0;
    }
  }
  out.append(backslashes * 2, L'\\');
  out.push_back(L'"');
  return out;
}

// What console_init changed, so exit can put it back.
UINT g_saved_out_cp = 0, g_saved_in_cp = 0;
DWORD g_saved_out_mode = 0, g_saved_err_mode = 0;
bool g_out_mode_saved = false, g_err_mode_saved = false;

void console_restore() {
  if (g_saved_out_cp) SetConsoleOutputCP(g_saved_out_cp);
  if (g_saved_in_cp) SetConsoleCP(g_saved_in_cp);
  if (g_out_mode_saved) SetConsoleMode(GetStdHandle(STD_OUTPUT_HANDLE), g_saved_out_mode);
  if (g_err_mode_saved) SetConsoleMode(GetStdHandle(STD_ERROR_HANDLE), g_saved_err_mode);
}

HANDLE handle_of(std::FILE* stream) { return reinterpret_cast<HANDLE>(_get_osfhandle(_fileno(stream))); }

#endif

}  // namespace

fs::path executable_path() {
#if defined(_WIN32)
  std::wstring buf(32768, L'\0');
  const DWORD n = GetModuleFileNameW(nullptr, buf.data(), static_cast<DWORD>(buf.size()));
  if (n == 0 || n >= buf.size()) return {};
  buf.resize(n);
  // The final path: an install reached through the current junction resolves to its versions\X.Y.Z folder.
  const HANDLE h = CreateFileW(buf.c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, nullptr,
                               OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr);
  if (h == INVALID_HANDLE_VALUE) return fs::path(buf);
  std::wstring final_path(32768, L'\0');
  const DWORD m = GetFinalPathNameByHandleW(h, final_path.data(), static_cast<DWORD>(final_path.size()),
                                            FILE_NAME_NORMALIZED | VOLUME_NAME_DOS);
  CloseHandle(h);
  if (m == 0 || m >= final_path.size()) return fs::path(buf);
  final_path.resize(m);
  return fs::path(strip_long_prefix(final_path));
#elif defined(__APPLE__)
  char buf[4096];
  uint32_t size = sizeof buf;
  if (_NSGetExecutablePath(buf, &size) != 0) return {};
  char resolved[PATH_MAX];
  if (!realpath(buf, resolved)) return {};
  return resolved;
#else
  char buf[4096];
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return {};
  return std::string(buf, static_cast<size_t>(n));
#endif
}

fs::path home() {
#if defined(_WIN32)
  std::wstring h = wenv(L"USERPROFILE");
  if (h.empty()) h = wenv(L"HOME");
  if (h.empty()) {
    const std::wstring drive = wenv(L"HOMEDRIVE"), path = wenv(L"HOMEPATH");
    if (!drive.empty() && !path.empty()) h = drive + path;
  }
  if (h.empty()) throw std::runtime_error("neither USERPROFILE nor HOME is set");
  return fs::path(h);
#else
  const std::string h = env("HOME");
  if (h.empty()) throw std::runtime_error("HOME is not set");
  return h;
#endif
}

fs::path install_root() {
#if defined(_WIN32)
  return local_app_data() / "Programs" / kWindowsDir;
#else
  return home() / kDotDir;
#endif
}

fs::path cache_root() {
#if defined(_WIN32)
  return local_app_data() / kWindowsDir;
#else
  return home() / ".cache" / kCacheDir;
#endif
}

std::string display(const fs::path& path) {
  const std::string s = path.string();
  auto under = [&s](const std::string& base, const std::string& name) -> std::string {
    if (base.empty() || s.size() < base.size() || s.compare(0, base.size(), base) != 0) return "";
    if (s.size() > base.size() && s[base.size()] != '/' && s[base.size()] != '\\') return "";
    return name + s.substr(base.size());
  };
#if defined(_WIN32)
  if (std::string out = under(local_app_data().string(), "%LOCALAPPDATA%"); !out.empty()) return out;
  try {
    if (std::string out = under(home().string(), "%USERPROFILE%"); !out.empty()) return out;
  } catch (const std::exception&) {
  }
#else
  if (std::string out = under(env("HOME"), "~"); !out.empty()) return out;
#endif
  return s;
}

fs::path expand_home(const std::string& path) {
  if (path == "~") return home();
  if (path.rfind("~/", 0) == 0) return home() / path.substr(2);
#if defined(_WIN32)
  if (path.rfind("~\\", 0) == 0) return home() / path.substr(2);
#endif
  return fs::path(path);
}

int run_process(const std::vector<std::string>& args, const OutputSink& output, bool merge_stderr) {
  if (args.empty()) return -1;
#if defined(_WIN32)
  std::wstring command;
  for (const std::string& a : args) command += (command.empty() ? L"" : L" ") + quote_arg(widen(a));
  SECURITY_ATTRIBUTES inherit{sizeof(SECURITY_ATTRIBUTES), nullptr, TRUE};
  HANDLE read_end = nullptr, write_end = nullptr;
  STARTUPINFOW si{};
  si.cb = sizeof si;
  if (output) {
    if (!CreatePipe(&read_end, &write_end, &inherit, 0)) return -1;
    SetHandleInformation(read_end, HANDLE_FLAG_INHERIT, 0);
    si.dwFlags = STARTF_USESTDHANDLES;
    si.hStdInput = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = write_end;
    si.hStdError = merge_stderr ? write_end : GetStdHandle(STD_ERROR_HANDLE);
  }
  PROCESS_INFORMATION pi{};
  const BOOL started = CreateProcessW(nullptr, command.data(), nullptr, nullptr, output ? TRUE : FALSE, 0, nullptr,
                                     nullptr, &si, &pi);
  if (write_end) CloseHandle(write_end);  // the child holds its own copy; the read below ends when it closes it
  if (!started) {
    if (read_end) CloseHandle(read_end);
    return -1;
  }
  if (output) {
    char buf[4096];
    DWORD n = 0;
    while (ReadFile(read_end, buf, sizeof buf, &n, nullptr) && n > 0) output(buf, n);
    CloseHandle(read_end);
  }
  WaitForSingleObject(pi.hProcess, INFINITE);
  DWORD code = 0;
  const BOOL got = GetExitCodeProcess(pi.hProcess, &code);
  CloseHandle(pi.hThread);
  CloseHandle(pi.hProcess);
  return got ? static_cast<int>(code) : -1;
#else
  int pipefd[2] = {-1, -1};
  if (output && pipe(pipefd) != 0) return -1;
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    if (output) {
      close(pipefd[0]);
      close(pipefd[1]);
    }
    return -1;
  }
  if (pid == 0) {
    if (output) {
      dup2(pipefd[1], 1);
      if (merge_stderr) dup2(pipefd[1], 2);
      close(pipefd[0]);
      close(pipefd[1]);
    }
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (output) {
    close(pipefd[1]);
    char buf[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof buf)) != 0) {
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) break;
      output(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return -1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
#endif
}

void console_init() {
#if defined(_WIN32)
  static bool done = false;
  if (done) return;
  done = true;
  // Only a real console: a pipe, a file or mintty's pty takes the bytes as they are.
  DWORD mode = 0;
  const HANDLE out = GetStdHandle(STD_OUTPUT_HANDLE), err = GetStdHandle(STD_ERROR_HANDLE),
               in = GetStdHandle(STD_INPUT_HANDLE);
  const bool out_console = out != INVALID_HANDLE_VALUE && GetConsoleMode(out, &mode);
  if (out_console) {
    g_saved_out_mode = mode;
    g_out_mode_saved = SetConsoleMode(out, mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
  }
  if (err != INVALID_HANDLE_VALUE && GetConsoleMode(err, &mode)) {
    g_saved_err_mode = mode;
    g_err_mode_saved = SetConsoleMode(err, mode | ENABLE_PROCESSED_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
  }
  if (out_console || g_err_mode_saved) {
    g_saved_out_cp = GetConsoleOutputCP();
    SetConsoleOutputCP(CP_UTF8);
  }
  if (in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &mode)) {
    g_saved_in_cp = GetConsoleCP();
    SetConsoleCP(CP_UTF8);
  }
  if (g_saved_out_cp || g_saved_in_cp || g_out_mode_saved || g_err_mode_saved) std::atexit(console_restore);
#endif
}

bool is_terminal(std::FILE* stream) {
#if defined(_WIN32)
  DWORD mode = 0;
  const HANDLE h = handle_of(stream);
  return h != INVALID_HANDLE_VALUE && GetConsoleMode(h, &mode) != 0;
#else
  return isatty(fileno(stream)) != 0;
#endif
}

int terminal_columns(std::FILE* stream) {
#if defined(_WIN32)
  CONSOLE_SCREEN_BUFFER_INFO info{};
  if (!GetConsoleScreenBufferInfo(handle_of(stream), &info)) return 0;
  return info.srWindow.Right - info.srWindow.Left + 1;
#else
  winsize ws{};
  if (ioctl(fileno(stream), TIOCGWINSZ, &ws) == 0 && ws.ws_col > 0) return ws.ws_col;
  return 0;
#endif
}

int read_line(std::string& line) {
  line.clear();
#if defined(_WIN32)
  const HANDLE in = GetStdHandle(STD_INPUT_HANDLE);
  DWORD mode = 0;
  if (in != INVALID_HANDLE_VALUE && GetConsoleMode(in, &mode)) {
    // A console: read UTF-16 and convert, since the console's own UTF-8 input path drops what is not ASCII.
    std::wstring text;
    for (;;) {
      wchar_t buf[512];
      DWORD n = 0;
      SetLastError(ERROR_SUCCESS);
      if (!ReadConsoleW(in, buf, 512, &n, nullptr)) return GetLastError() == ERROR_OPERATION_ABORTED ? -1 : 0;
      if (n == 0) return GetLastError() == ERROR_OPERATION_ABORTED ? -1 : 0;  // Ctrl-C ends the read with nothing
      text.append(buf, n);
      if (text.find(L'\n') != std::wstring::npos) break;
    }
    if (!text.empty() && text[0] == 0x1a) return 0;  // Ctrl-Z, then Enter: the end of the input
    while (!text.empty() && (text.back() == L'\n' || text.back() == L'\r')) text.pop_back();
    line = narrow(text);
    return 1;
  }
#endif
  if (std::getline(std::cin, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    return 1;
  }
  if (std::cin.eof()) return 0;
  std::cin.clear();
  return -1;
}

void set_link(const fs::path& link, const fs::path& target) {
#if defined(_WIN32)
  // A junction is retargeted in place (one reparse write), so the name never stops existing.
  std::error_code ec;
  const fs::file_status st = fs::symlink_status(link, ec);
  if (ec || !fs::exists(st)) {
    if (!CreateDirectoryW(link.c_str(), nullptr))
      throw std::runtime_error("cannot create " + link.string() + ": " + error_text(GetLastError()));
    try {
      write_junction(link, target);
    } catch (...) {
      RemoveDirectoryW(link.c_str());
      throw;
    }
    return;
  }
  if (link_target(link).empty() && !fs::is_empty(link, ec))
    throw std::runtime_error(link.string() + " is a folder, not a link the installer made");
  write_junction(link, target);
#else
  const fs::path tmp = link.parent_path() / (link.filename().string() + ".tmp" + std::to_string(getpid()));
  std::error_code ec;
  fs::remove(tmp, ec);
  fs::create_symlink(target, tmp);
  fs::rename(tmp, link);
#endif
}

fs::path link_target(const fs::path& link) {
#if defined(_WIN32)
  const DWORD attrs = GetFileAttributesW(link.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES || !(attrs & FILE_ATTRIBUTE_REPARSE_POINT)) return {};
  const HANDLE h = open_reparse(link, 0);
  if (h == INVALID_HANDLE_VALUE) return {};
  std::vector<char> buf(MAXIMUM_REPARSE_DATA_BUFFER_SIZE);
  DWORD got = 0;
  const BOOL ok = DeviceIoControl(h, FSCTL_GET_REPARSE_POINT, nullptr, 0, buf.data(), static_cast<DWORD>(buf.size()),
                                  &got, nullptr);
  CloseHandle(h);
  if (!ok) return {};
  const auto* r = reinterpret_cast<const ReparseBuffer*>(buf.data());
  std::wstring target;
  if (r->tag == IO_REPARSE_TAG_MOUNT_POINT) {
    target.assign(r->mount.path + r->mount.substitute_offset / sizeof(wchar_t), r->mount.substitute_length / sizeof(wchar_t));
  } else if (r->tag == IO_REPARSE_TAG_SYMLINK) {
    target.assign(r->symlink.path + r->symlink.substitute_offset / sizeof(wchar_t),
                  r->symlink.substitute_length / sizeof(wchar_t));
    if (r->symlink.flags & 1 /* SYMLINK_FLAG_RELATIVE */) return (link.parent_path() / target).lexically_normal();
  } else {
    return {};
  }
  return fs::path(strip_long_prefix(target));
#else
  std::error_code ec;
  if (!fs::is_symlink(link, ec)) return {};
  const fs::path target = fs::read_symlink(link, ec);
  if (ec) return {};
  return target.is_absolute() ? target : (link.parent_path() / target).lexically_normal();
#endif
}

void remove_link(const fs::path& link) {
#if defined(_WIN32)
  const DWORD attrs = GetFileAttributesW(link.c_str());
  if (attrs == INVALID_FILE_ATTRIBUTES) return;
  if (!(attrs & FILE_ATTRIBUTE_REPARSE_POINT))
    throw std::runtime_error(link.string() + " is not a link");
  // A junction or a directory symlink is a directory entry: removing it leaves its target alone.
  const BOOL ok = (attrs & FILE_ATTRIBUTE_DIRECTORY) ? RemoveDirectoryW(link.c_str()) : DeleteFileW(link.c_str());
  if (!ok) throw std::runtime_error("cannot remove " + link.string() + ": " + error_text(GetLastError()));
#else
  std::error_code ec;
  if (fs::is_symlink(link, ec)) fs::remove(link);
#endif
}

long process_id() {
#if defined(_WIN32)
  return static_cast<long>(GetCurrentProcessId());
#else
  return static_cast<long>(getpid());
#endif
}

}  // namespace lx::platform
