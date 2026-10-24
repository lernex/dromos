#include "core/hub.h"

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <thread>

#include "core/digest.h"
#include "core/http.h"
#include "nlohmann/json.hpp"

namespace lx::hub {
namespace fs = std::filesystem;
using json = nlohmann::json;

namespace {

const char* kHub = "https://huggingface.co/";

std::string trim(std::string s) {
  const size_t first = s.find_first_not_of(" \t\r\n");
  if (first == std::string::npos) return "";
  const size_t last = s.find_last_not_of(" \t\r\n");
  return s.substr(first, last - first + 1);
}

std::string home() {
  const char* h = std::getenv("HOME");
  return h ? h : ".";
}

bool valid_repo(const std::string& repo) {
  const size_t slash = repo.find('/');
  if (slash == std::string::npos || slash == 0 || slash + 1 >= repo.size() || repo.find('/', slash + 1) != std::string::npos)
    return false;
  for (char c : repo) {
    const bool ok = (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '/';
    if (!ok) return false;
  }
  return true;
}

std::string lower(std::string s) {
  for (char& c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  return s;
}

// The manifest records what has verified, so an interrupted pull resumes without re-hashing.
struct Manifest {
  std::string repo, revision;
  bool complete = false;
  std::map<std::string, std::pair<uint64_t, std::string>> verified;  // path -> (size, digest)

  static Manifest load(const fs::path& file) {
    Manifest m;
    std::ifstream in(file);
    if (!in) return m;
    try {
      json j = json::parse(in);
      m.repo = j.value("repo", "");
      m.revision = j.value("revision", "");
      m.complete = j.value("complete", false);
      const json files = j.value("files", json::object());
      for (const auto& [path, entry] : files.items())
        m.verified[path] = {entry.value("size", uint64_t{0}), entry.value("digest", "")};
    } catch (const std::exception&) {
      m = Manifest{};
    }
    return m;
  }

  void save(const fs::path& file) const {
    json j;
    j["repo"] = repo;
    j["revision"] = revision;
    j["complete"] = complete;
    j["files"] = json::object();
    for (const auto& [path, v] : verified) j["files"][path] = {{"size", v.first}, {"digest", v.second}};
    const fs::path tmp = fs::path(file).concat(".tmp");
    {
      std::ofstream out(tmp);
      out << j.dump(2) << "\n";
    }
    std::error_code ec;
    fs::rename(tmp, file, ec);
    if (ec) throw std::runtime_error("cannot write " + file.string() + ": " + ec.message());
  }
};

std::string auth_header(const std::string& token) { return token.empty() ? "" : "Authorization: Bearer " + token; }

[[noreturn]] void refused(const std::string& repo, long status) {
  if (status == 401 || status == 403)
    throw std::runtime_error(repo + " needs access: sign in with `huggingface-cli login`, or set HF_TOKEN");
  if (status == 404) throw std::runtime_error("there is no model named " + repo + " on the hub");
  throw std::runtime_error("the hub answered " + std::to_string(status) + " for " + repo);
}

// Hashes the file's bytes from disk into the verifier.
template <typename Fn>
void each_chunk(const fs::path& file, Fn fn) {
  std::ifstream in(file, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + file.string());
  std::vector<char> buf(1 << 22);
  while (in) {
    in.read(buf.data(), static_cast<std::streamsize>(buf.size()));
    const std::streamsize n = in.gcount();
    if (n > 0) fn(buf.data(), static_cast<size_t>(n));
  }
}

// Digest of a whole file: SHA-256 for LFS files, the git blob id otherwise.
std::string digest_of_file(const fs::path& file, const RemoteFile& f) {
  if (f.lfs) {
    Sha256 s;
    each_chunk(file, [&](const char* p, size_t n) { s.update(p, n); });
    return s.hex_digest();
  }
  GitBlobId g(f.size);
  each_chunk(file, [&](const char* p, size_t n) { g.update(p, n); });
  return g.hex_digest();
}

// The digest the hub gives for a file, to compare with ours.
std::string expected_digest(const RemoteFile& f) { return f.lfs ? f.sha256 : f.blob_id; }

class Verifier {
 public:
  explicit Verifier(const RemoteFile& f) : lfs_(f.lfs), git_(f.size) {}
  void update(const void* p, size_t n) {
    if (lfs_)
      sha_.update(p, n);
    else
      git_.update(p, n);
  }
  std::string hex() { return lfs_ ? sha_.hex_digest() : git_.hex_digest(); }

 private:
  bool lfs_;
  Sha256 sha_;
  GitBlobId git_;
};

}  // namespace

uint64_t RemoteModel::total_bytes() const {
  uint64_t total = 0;
  for (const auto& f : files) total += f.size;
  return total;
}

ModelRef resolve(const std::string& name) {
  const size_t colon = name.find(':');
  const std::string base = lower(name.substr(0, colon));
  const std::string format = colon == std::string::npos ? "bf16" : lower(name.substr(colon + 1));
  for (const Alias& a : aliases()) {
    if (a.name != base) continue;
    if (format != "bf16" && format != "fp8" && format != "nvfp4")
      throw std::invalid_argument("unknown format '" + format + "' for " + base + ": use bf16, fp8 or nvfp4");
    return ModelRef{alias_repo(a, format), a.name, format};
  }
  if (colon == std::string::npos && valid_repo(name)) return ModelRef{name, "", ""};
  throw std::invalid_argument("unknown model '" + name + "': use praxis, logos, or owner/name on the hub");
}

const std::vector<Alias>& aliases() {
  static const std::vector<Alias> list = {
      {"praxis", "Praxis", "Lernex/Metis-1.6-Praxis"},
      {"logos", "Logos", "Lernex/Metis-1.6-Logos"},
  };
  return list;
}

std::string alias_repo(const Alias& alias, const std::string& format) {
  if (format == "fp8") return alias.bf16 + "-FP8";
  if (format == "nvfp4") return alias.bf16 + "-NVFP4";
  return alias.bf16;
}

fs::path cache_root() { return fs::path(home()) / ".cache" / "lernex" / "models"; }

fs::path install_dir(const std::string& repo) {
  const size_t slash = repo.find('/');
  return cache_root() / repo.substr(0, slash) / repo.substr(slash + 1);
}

std::string hf_token() {
  if (const char* env = std::getenv("HF_TOKEN"); env && *env) return trim(env);
  std::ifstream in(fs::path(home()) / ".cache" / "huggingface" / "token");
  if (!in) return "";
  std::stringstream ss;
  ss << in.rdbuf();
  return trim(ss.str());
}

bool installed(const fs::path& dir) {
  const Manifest m = Manifest::load(dir / "lernex.json");
  if (!m.complete) return false;
  for (const auto& [path, v] : m.verified) {
    std::error_code ec;
    if (fs::file_size(dir / path, ec) != v.first || ec) return false;
  }
  return true;
}

bool needed_file(const std::string& path) {
  if (path.find('/') != std::string::npos) return false;
  static const char* exact[] = {"config.json",         "generation_config.json", "tokenizer.json",
                                "tokenizer_config.json", "special_tokens_map.json", "chat_template.jinja",
                                "added_tokens.json",     "model.safetensors.index.json", "LICENSE", "README.md"};
  for (const char* name : exact)
    if (path == name) return true;
  const bool shard = path.rfind("model", 0) == 0 && path.size() > 12 &&
                     path.compare(path.size() - 12, 12, ".safetensors") == 0;
  return shard;
}

RemoteModel describe(const std::string& repo, const std::string& token) {
  if (!http::available()) throw std::runtime_error("libcurl is not installed, so models cannot be downloaded");
  std::string body;
  const http::Response r = http::get_string(std::string(kHub) + "api/models/" + repo + "?blobs=true",
                                            token.empty() ? std::vector<std::string>{} : std::vector<std::string>{auth_header(token)},
                                            body);
  if (r.status != 200) refused(repo, r.status);
  json j;
  try {
    j = json::parse(body);
  } catch (const std::exception&) {
    throw std::runtime_error("the hub sent an unreadable description of " + repo);
  }
  RemoteModel m;
  m.repo = repo;
  m.revision = j.value("sha", "main");
  for (const auto& s : j.value("siblings", json::array())) {
    RemoteFile f;
    f.path = s.value("rfilename", "");
    if (!needed_file(f.path)) continue;
    f.size = s.value("size", uint64_t{0});
    f.blob_id = s.value("blobId", "");
    if (s.contains("lfs") && s["lfs"].is_object()) {
      f.lfs = true;
      f.sha256 = s["lfs"].value("sha256", "");
      if (f.size == 0) f.size = s["lfs"].value("size", uint64_t{0});
    }
    m.files.push_back(f);
  }
  bool weights = false;
  for (const auto& f : m.files) weights = weights || f.path.find(".safetensors") != std::string::npos;
  if (!weights) throw std::runtime_error(repo + " has no safetensors weights to run");
  return m;
}

void pull(const RemoteModel& model, const fs::path& dir, const std::string& token, const ProgressFn& progress) {
  fs::create_directories(dir);
  const fs::path manifest_path = dir / "lernex.json";
  Manifest m = Manifest::load(manifest_path);
  if (m.repo != model.repo || m.revision != model.revision) {
    // another revision (or no manifest): nothing in it is trusted until verified again
    m = Manifest{};
    m.repo = model.repo;
    m.revision = model.revision;
  }
  m.complete = false;

  const uint64_t total = model.total_bytes();
  uint64_t done_before = 0;
  for (const auto& f : model.files) {
    auto it = m.verified.find(f.path);
    if (it != m.verified.end() && it->second.first == f.size) done_before += f.size;
  }
  uint64_t model_done = done_before;

  for (size_t i = 0; i < model.files.size(); ++i) {
    const RemoteFile& f = model.files[i];
    const fs::path final_path = dir / f.path;
    const fs::path part_path = dir / (f.path + ".part");
    auto report = [&](uint64_t done) {
      if (progress) progress(Progress{f.path, done, f.size, i, model.files.size(), model_done + done, total});
    };

    // already verified: nothing to fetch
    auto known = m.verified.find(f.path);
    std::error_code ec;
    if (known != m.verified.end() && known->second.first == f.size && fs::file_size(final_path, ec) == f.size && !ec) {
      report(f.size);
      continue;
    }

    // a finished file that was never recorded: check it from disk before trusting it
    if (fs::exists(final_path) && fs::file_size(final_path, ec) == f.size && !ec) {
      if (digest_of_file(final_path, f) == expected_digest(f)) {
        m.verified[f.path] = {f.size, expected_digest(f)};
        m.save(manifest_path);
        report(f.size);
        model_done += f.size;
        continue;
      }
      fs::remove(final_path, ec);
    }

    // resume from the partial file, hashing what it already holds
    uint64_t existing = fs::exists(part_path) ? fs::file_size(part_path, ec) : 0;
    if (ec || existing > f.size) {
      fs::remove(part_path, ec);
      existing = 0;
    }
    Verifier verifier(f);
    if (existing > 0) each_chunk(part_path, [&](const char* p, size_t n) { verifier.update(p, n); });

    const std::string url = std::string(kHub) + model.repo + "/resolve/" + model.revision + "/" + f.path;
    int attempt = 0;
    while (existing < f.size) {
      std::vector<std::string> headers;
      if (!token.empty()) headers.push_back(auth_header(token));
      if (existing > 0) headers.push_back("Range: bytes=" + std::to_string(existing) + "-");
      std::FILE* out = nullptr;
      uint64_t done = existing;
      bool first = true;
      bool abort_on_status = false;
      long status = 0;
      auto last_report = std::chrono::steady_clock::now();
      try {
        http::get(url, headers, [&](const char* data, size_t n, const http::Response& so_far) {
          if (first) {
            first = false;
            status = so_far.status;
            if (status == 200 && existing > 0) {
              // the server ignored the range: the file starts over
              verifier = Verifier(f);
              done = existing = 0;
            }
            if (status != 200 && status != 206) {
              abort_on_status = true;
              return false;
            }
            out = std::fopen(part_path.c_str(), existing > 0 ? "ab" : "wb");
            if (!out) throw std::runtime_error("cannot write " + part_path.string());
          }
          const size_t wrote = std::fwrite(data, 1, n, out);
          verifier.update(data, wrote);
          done += wrote;
          const auto now = std::chrono::steady_clock::now();
          if (now - last_report > std::chrono::milliseconds(100)) {
            last_report = now;
            if (progress) progress(Progress{f.path, done, f.size, i, model.files.size(), model_done + done, total});
          }
          return wrote == n;
        });
      } catch (const std::runtime_error&) {
        if (out) std::fclose(out);
        existing = fs::exists(part_path) ? fs::file_size(part_path, ec) : 0;
        if (++attempt > 5) throw;
        std::this_thread::sleep_for(std::chrono::seconds(1 << std::min(attempt, 4)));
        continue;
      }
      if (out) std::fclose(out);
      if (abort_on_status) {
        if (status == 416) {
          // the partial file is already whole (or wrong): check it below
          existing = f.size;
          break;
        }
        if (status == 401 || status == 403 || status == 404) refused(model.repo, status);
        if (status >= 500 && ++attempt <= 5) {
          std::this_thread::sleep_for(std::chrono::seconds(1 << std::min(attempt, 4)));
          existing = fs::exists(part_path) ? fs::file_size(part_path, ec) : 0;
          continue;
        }
        throw std::runtime_error("the hub answered " + std::to_string(status) + " for " + f.path);
      }
      existing = fs::exists(part_path) ? fs::file_size(part_path, ec) : 0;
      if (existing == f.size) break;
      // the connection closed early: go on from where it stopped
      if (++attempt > 5) throw std::runtime_error("the download of " + f.path + " keeps stopping short");
    }

    if (fs::file_size(part_path, ec) != f.size || ec) {
      fs::remove(part_path, ec);
      throw std::runtime_error("the download of " + f.path + " is the wrong size");
    }
    if (verifier.hex() != expected_digest(f)) {
      fs::remove(part_path, ec);
      throw std::runtime_error(f.path + " does not match the hub's checksum; the download was discarded, run pull again");
    }
    fs::rename(part_path, final_path);
    m.verified[f.path] = {f.size, expected_digest(f)};
    m.save(manifest_path);
    report(f.size);
    model_done += f.size;
  }

  m.complete = true;
  m.save(manifest_path);
}

}  // namespace lx::hub
