#include "core/install.h"

#include <fstream>
#include <string>

#include "platform/platform.h"

namespace lx::install {

// The executable's path with symlinks resolved, or "" (engine/platform).
std::string executable_path() { return lx::platform::executable_path().string(); }

std::string executable_dir() {
  const std::string exe = executable_path();
  if (exe.empty()) return "";
#if defined(_WIN32)
  const size_t slash = exe.find_last_of("/\\");
#else
  const size_t slash = exe.rfind('/');
#endif
  if (slash == std::string::npos) return "";
  return slash == 0 ? "/" : exe.substr(0, slash);
}

std::string share_dir() {
  const std::string dir = executable_dir();
  if (dir.empty()) return "";
  return dir + "/../share/dromos";
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
