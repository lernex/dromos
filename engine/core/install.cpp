#include "core/install.h"

#include <unistd.h>

#include <cstdint>
#include <fstream>
#include <string>

#if defined(__APPLE__)
#include <climits>
#include <cstdlib>
#include <mach-o/dyld.h>
#endif

namespace lx::install {

// The executable's path with symlinks resolved, or "".
std::string executable_path() {
  char buf[4096];
#if defined(__APPLE__)
  uint32_t size = sizeof buf;
  if (_NSGetExecutablePath(buf, &size) != 0) return "";
  char resolved[PATH_MAX];
  if (!realpath(buf, resolved)) return "";
  return resolved;
#else
  const ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
  if (n <= 0) return "";
  return std::string(buf, static_cast<size_t>(n));
#endif
}

std::string executable_dir() {
  const std::string exe = executable_path();
  if (exe.empty()) return "";
  const size_t slash = exe.rfind('/');
  if (slash == std::string::npos) return "";
  return slash == 0 ? "/" : exe.substr(0, slash);
}

std::string share_dir() {
  const std::string dir = executable_dir();
  if (dir.empty()) return "";
  return dir + "/../share/lernex";
}

std::string channel() {
  const std::string share = share_dir();
  if (share.empty()) return "";
  std::ifstream in(share + "/channel");
  if (!in) return "";
  std::string name;
  std::getline(in, name);
  while (!name.empty() && (name.back() == '\r' || name.back() == ' ' || name.back() == '\t')) name.pop_back();
  if (name == "tarball" || name == "homebrew" || name == "pip" || name == "docker") return name;
  return "";
}

}  // namespace lx::install
