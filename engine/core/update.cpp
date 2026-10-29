#include "core/update.h"

#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <vector>

#include "core/digest.h"
#include "core/http.h"
#include "core/install.h"
#include "lx_version.h"

namespace lx::update {
namespace fs = std::filesystem;
namespace {

constexpr long long kDay = 24 * 60 * 60;
const char* kDefaultReleases = "https://github.com/lernex/dromos/releases";

std::string home() {
  const char* h = std::getenv("HOME");
  if (!h || !*h) throw std::runtime_error("HOME is not set");
  return h;
}

// The dromos directory of this user: ~/.dromos (versions, current, previous).
fs::path dromos_home() { return fs::path(home()) / ".dromos"; }

// Runs a program and waits for it. With `captured`, its standard output is collected there. Returns its exit status,
// or -1 when it did not run to an exit.
int run_process(const std::vector<std::string>& args, std::string* captured) {
  int pipefd[2] = {-1, -1};
  if (captured && pipe(pipefd) != 0) return -1;
  std::vector<char*> argv;
  argv.reserve(args.size() + 1);
  for (const std::string& a : args) argv.push_back(const_cast<char*>(a.c_str()));
  argv.push_back(nullptr);
  const pid_t pid = fork();
  if (pid < 0) {
    if (captured) {
      close(pipefd[0]);
      close(pipefd[1]);
    }
    return -1;
  }
  if (pid == 0) {
    if (captured) {
      dup2(pipefd[1], 1);
      close(pipefd[0]);
      close(pipefd[1]);
    }
    execvp(argv[0], argv.data());
    _exit(127);
  }
  if (captured) {
    close(pipefd[1]);
    char buf[4096];
    ssize_t n;
    while ((n = read(pipefd[0], buf, sizeof buf)) != 0) {
      if (n < 0 && errno == EINTR) continue;
      if (n < 0) break;
      captured->append(buf, static_cast<size_t>(n));
    }
    close(pipefd[0]);
  }
  int status = 0;
  while (waitpid(pid, &status, 0) < 0) {
    if (errno != EINTR) return -1;
  }
  return WIFEXITED(status) ? WEXITSTATUS(status) : -1;
}

// The body of a GET, as a string. Throws unless the server answers 200.
std::string fetch(const std::string& url) {
  std::string body;
  const http::Response r = http::get_string(url, {}, body);
  if (r.status != 200) throw std::runtime_error("GET " + url + " answered HTTP " + std::to_string(r.status));
  return body;
}

// Downloads url to `file` and returns the SHA-256 of what was written. Throws unless the server answers 200.
std::string download(const std::string& url, const fs::path& file) {
  std::ofstream out(file, std::ios::binary);
  if (!out) throw std::runtime_error("cannot write " + file.string());
  Sha256 hash;
  const http::Response r = http::get(url, {}, [&](const char* data, size_t size, const http::Response&) {
    out.write(data, static_cast<std::streamsize>(size));
    hash.update(data, size);
    return static_cast<bool>(out);
  });
  out.close();
  if (!out) throw std::runtime_error("cannot write " + file.string());
  if (r.status != 200) throw std::runtime_error("download of " + url + " answered HTTP " + std::to_string(r.status));
  return hash.hex_digest();
}

// Points `link` at `target`, replacing what it was, in one step (a rename over the old link).
void set_link(const fs::path& link, const fs::path& target) {
  const fs::path tmp = link.parent_path() / (link.filename().string() + ".tmp" + std::to_string(getpid()));
  std::error_code ec;
  fs::remove(tmp, ec);
  fs::create_symlink(target, tmp);
  fs::rename(tmp, link);
}

// What a symlink points at, or "" when `link` is not one.
fs::path link_target(const fs::path& link) {
  std::error_code ec;
  if (!fs::is_symlink(link, ec)) return "";
  return fs::read_symlink(link, ec);
}

// Installs a verified release tree under ~/.dromos/versions/VERSION and makes it current. The version that was current
// becomes the previous one, and every other installed version is removed.
void install_release(const std::string& releases, const std::string& version) {
  const std::string flavor = LX_FLAVOR_STRING;
  if (flavor == "source") throw std::runtime_error("this build has no release flavor to download");
  const fs::path root = dromos_home();
  const fs::path versions = root / "versions";
  fs::create_directories(versions);

  const fs::path staging = root / ("staging-" + std::to_string(getpid()));
  std::error_code ec;
  fs::remove_all(staging, ec);
  fs::create_directories(staging);
  struct Cleanup {
    fs::path dir;
    ~Cleanup() {
      std::error_code e;
      fs::remove_all(dir, e);
    }
  } cleanup{staging};

  const std::string asset = asset_name(flavor);
  const std::string base = releases + "/download/v" + version + "/";
  std::fprintf(stderr, "downloading dromos %s (%s)\n", version.c_str(), flavor.c_str());
  const std::string expected = checksum_for(fetch(base + "SHA256SUMS"), asset);
  if (expected.empty()) throw std::runtime_error("SHA256SUMS lists no " + asset);
  const fs::path tarball = staging / asset;
  const std::string actual = download(base + asset, tarball);
  if (actual != expected) throw std::runtime_error("checksum mismatch for " + asset + ": nothing was installed");

  const fs::path tree = staging / "tree";
  fs::create_directories(tree);
  if (run_process({"tar", "-xzf", tarball.string(), "-C", tree.string()}, nullptr) != 0)
    throw std::runtime_error("could not unpack " + asset);
  // The unpacked binary has to be the release it claims, before anything points at it.
  std::string printed;
  const std::string expect = "dromos " + version + " (" + flavor + ",";
  if (run_process({(tree / "bin" / "dromos").string(), "--version"}, &printed) != 0 ||
      printed.rfind(expect, 0) != 0)
    throw std::runtime_error("the unpacked dromos does not report " + version + " (" + flavor + ")");

  const fs::path dest = versions / version;
  fs::remove_all(dest, ec);
  fs::rename(tree, dest);

  const fs::path previous = link_target(root / "current");
  set_link(root / "current", dest);
  if (!previous.empty() && previous != dest) set_link(root / "previous", previous);
  const std::string keep_previous = previous.empty() || previous == dest ? "" : previous.filename().string();
  for (const fs::directory_entry& entry : fs::directory_iterator(versions, ec)) {
    const std::string name = entry.path().filename().string();
    if (name == version || name == keep_previous) continue;
    fs::remove_all(entry.path(), ec);
  }

  std::printf("dromos %s is installed and current.", version.c_str());
  if (!keep_previous.empty())
    std::printf(" dromos %s is kept: `dromos update --rollback` returns to it.", keep_previous.c_str());
  std::printf("\n");
}

// Switches ~/.dromos/current back to the version the last update replaced.
int rollback_to_previous() {
  const fs::path root = dromos_home();
  const fs::path previous = link_target(root / "previous");
  const fs::path current = link_target(root / "current");
  if (previous.empty() || !fs::is_directory(previous))
    throw std::runtime_error("there is no previous version to roll back to");
  set_link(root / "current", previous);
  if (!current.empty()) set_link(root / "previous", current);
  std::printf("dromos switched back to %s. The next run uses it.\n", previous.filename().string().c_str());
  return 0;
}

}  // namespace

bool valid(const std::string& version) {
  // X.Y.Z (each part digits), then nothing, or a "-" pre-release or "+" build tail of [0-9A-Za-z.+-].
  size_t i = 0;
  for (int part = 0; part < 3; ++part) {
    const size_t start = i;
    while (i < version.size() && std::isdigit(static_cast<unsigned char>(version[i]))) ++i;
    if (i == start) return false;
    if (part < 2) {
      if (i >= version.size() || version[i] != '.') return false;
      ++i;
    }
  }
  if (i == version.size()) return true;
  if (version[i] != '-' && version[i] != '+') return false;
  for (size_t j = i + 1; j < version.size(); ++j) {
    const char c = version[j];
    if (!(std::isalnum(static_cast<unsigned char>(c)) || c == '.' || c == '-' || c == '+')) return false;
  }
  return i + 1 < version.size();
}

namespace {

struct Parsed {
  bool ok = false;
  long long core[3] = {0, 0, 0};
  std::string pre;  // the "-" tail, without any "+" build metadata, which does not order versions
};

Parsed parse(const std::string& version) {
  Parsed p;
  if (!valid(version)) return p;
  p.ok = true;
  size_t i = 0;
  for (int part = 0; part < 3; ++part) {
    const size_t stop = std::min(version.find_first_not_of("0123456789", i), version.size());
    p.core[part] = std::stoll(version.substr(i, stop - i));
    i = stop + 1;  // past the separator: the "." of the first two parts, the character that begins the tail after the last
  }
  const size_t tail = i - 1;  // valid() guarantees a tail, if any, starts with "-" or "+" here
  if (tail < version.size() && version[tail] == '-') {
    const size_t plus = version.find('+', tail);
    p.pre = version.substr(tail + 1, plus == std::string::npos ? std::string::npos : plus - tail - 1);
  }
  return p;
}

}  // namespace

int compare(const std::string& a, const std::string& b) {
  const Parsed x = parse(a), y = parse(b);
  if (!x.ok || !y.ok) return x.ok == y.ok ? 0 : (x.ok ? 1 : -1);
  for (int part = 0; part < 3; ++part) {
    if (x.core[part] != y.core[part]) return x.core[part] < y.core[part] ? -1 : 1;
  }
  if (x.pre == y.pre) return 0;
  if (x.pre.empty()) return 1;  // a release is newer than its pre-releases
  if (y.pre.empty()) return -1;
  return x.pre < y.pre ? -1 : 1;
}

std::string version_from_url(const std::string& url) {
  const std::string marker = "/tag/v";
  const size_t at = url.rfind(marker);
  if (at == std::string::npos) return "";
  const size_t start = at + marker.size();
  const size_t end = url.find_first_of("?#/", start);
  return url.substr(start, end == std::string::npos ? std::string::npos : end - start);
}

std::string asset_name(const std::string& flavor) { return "dromos-" + flavor + ".tar.gz"; }

std::string checksum_for(const std::string& sums, const std::string& file) {
  std::istringstream in(sums);
  std::string line;
  while (std::getline(in, line)) {
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const size_t space = line.find(' ');
    if (space == std::string::npos) continue;
    std::string name = line.substr(space + 1);
    if (!name.empty() && (name[0] == ' ' || name[0] == '*')) name.erase(0, 1);
    if (name != file) continue;
    std::string hash = line.substr(0, space);
    if (hash.size() != 64) return "";
    for (char& c : hash) {
      if (!std::isxdigit(static_cast<unsigned char>(c))) return "";
      c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    }
    return hash;
  }
  return "";
}

std::string sha256_file(const fs::path& file) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  Sha256 hash;
  std::vector<char> buf(1 << 16);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    hash.update(buf.data(), static_cast<size_t>(in.gcount()));
  }
  return hash.hex_digest();
}

std::string releases_url() {
  if (const char* env = std::getenv("DROMOS_RELEASES"); env && *env) {
    std::string url = env;
    while (!url.empty() && url.back() == '/') url.pop_back();
    return url;
  }
  return kDefaultReleases;
}

std::string latest_version(const std::string& releases, long timeout_seconds) {
  // The redirect ends on the release page, whose URL names the tag. The page itself is not needed: stop reading it.
  const http::Response r = http::get(
      releases + "/latest", {}, [](const char*, size_t, const http::Response&) { return false; }, timeout_seconds);
  const std::string version = version_from_url(r.url);
  if (r.status != 200 || !valid(version) || version.find_first_of("-+") != std::string::npos)
    throw std::runtime_error("the release host names no latest release at " + releases);
  return version;
}

std::string update_command(const std::string& channel) {
  if (channel == "tarball") return "dromos update";
  if (channel == "homebrew") return "brew upgrade dromos";
  if (channel == "pip") return "pip install -U dromos";
  if (channel == "docker") return "docker pull ghcr.io/lernex/dromos";
  return "";
}

fs::path cache_path() {
  const char* h = std::getenv("HOME");
  return fs::path(h && *h ? h : ".") / ".cache" / "dromos" / "update-check";
}

Cache read_cache(const fs::path& file) {
  Cache cache;
  std::ifstream in(file);
  std::string line;
  while (std::getline(in, line)) {
    const size_t eq = line.find('=');
    if (eq == std::string::npos) continue;
    const std::string key = line.substr(0, eq), value = line.substr(eq + 1);
    if (key == "latest") cache.latest = value;
    else if (key == "checked") cache.checked = std::strtoll(value.c_str(), nullptr, 10);
  }
  if (!valid(cache.latest)) cache.latest.clear();
  return cache;
}

void write_cache(const fs::path& file, const Cache& cache) {
  std::error_code ec;
  fs::create_directories(file.parent_path(), ec);
  const fs::path tmp = file.string() + ".tmp" + std::to_string(getpid());
  {
    std::ofstream out(tmp);
    if (!out) return;
    out << "latest=" << cache.latest << "\nchecked=" << cache.checked << "\n";
    if (!out) {
      fs::remove(tmp, ec);
      return;
    }
  }
  fs::rename(tmp, file, ec);
  if (ec) fs::remove(tmp, ec);
}

bool stale(const Cache& cache, long long now) { return now - cache.checked >= kDay; }

std::string notice(const std::string& current, const Cache& cache, const std::string& channel) {
  if (cache.latest.empty() || compare(cache.latest, current) <= 0) return "";
  const std::string how = update_command(channel);
  if (how.empty()) return "";
  return "dromos " + cache.latest + " is out (you have " + current + "). Run `" + how + "` to install it.";
}

int run_update(bool check_only, bool rollback) {
  const std::string channel = lx::install::channel();
  if (channel.empty()) {
    std::printf("this dromos was built from source, so it has no release channel. Update the checkout and build again.\n");
    return 0;
  }
  if (channel != "tarball") {
    std::printf("this dromos was installed with %s. Update it with `%s`.\n", channel.c_str(),
                update_command(channel).c_str());
    return 0;
  }
  if (rollback) return rollback_to_previous();

  const std::string releases = releases_url();
  const std::string latest = latest_version(releases, 30);
  const std::string current = LX_VERSION_STRING;
  if (compare(latest, current) <= 0) {
    std::printf("dromos %s is the latest release.\n", current.c_str());
    return 0;
  }
  if (check_only) {
    std::printf("dromos %s is available (you have %s). Run `dromos update` to install it.\n", latest.c_str(),
                current.c_str());
    return 0;
  }
  install_release(releases, latest);
  return 0;
}

void start_daily_check() {
  try {
    if (const char* off = std::getenv("DROMOS_NO_UPDATE_CHECK"); off && *off && std::string(off) != "0") return;
    const std::string channel = lx::install::channel();
    if (channel.empty()) return;  // a source build has no release to compare with
    const fs::path file = cache_path();
    const Cache cache = read_cache(file);
    const long long now = static_cast<long long>(std::time(nullptr));
    const std::string line = notice(LX_VERSION_STRING, cache, channel);
    if (!line.empty()) std::fprintf(stderr, "%s\n", line.c_str());
    if (!stale(cache, now)) return;
    const std::string releases = releases_url();
    // Detached: the process never waits on it. Whatever happens, the attempt is recorded, so a machine that is
    // offline asks once a day rather than on every start.
    std::thread([releases, file, cache, now] {
      try {
        Cache next{cache.latest, now};
        try {
          next.latest = latest_version(releases, 5);
        } catch (...) {
        }
        write_cache(file, next);
      } catch (...) {
      }
    }).detach();
  } catch (...) {
    // The check is a courtesy. Nothing in it may stop a run.
  }
}

}  // namespace lx::update
