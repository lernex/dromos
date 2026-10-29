// Model names, the local cache, and downloads from the Hugging Face hub. A download is resumable,
// every file is verified (size, then SHA-256 for files the hub keeps in LFS, or the git blob id for
// the rest), and a model is only usable once all of its files have verified.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lx::hub {

// What a name means: an alias ("praxis", "logos:fp8") or a repository ("Lernex/Metis-1.6-Praxis").
struct ModelRef {
  std::string repo;    // owner/name on the hub
  std::string alias;   // the alias it came from, or empty for a plain repository
  std::string format;  // "bf16", "fp8", "nvfp4", or empty for a repository given directly
};

// Throws std::invalid_argument for a name that is neither an alias nor owner/name.
ModelRef resolve(const std::string& name);

struct Alias {
  std::string name;   // praxis, logos
  std::string title;  // Praxis, Logos
  std::string bf16;   // the repository for the default format
};
const std::vector<Alias>& aliases();

// The repository an alias names in a format ("bf16", "fp8", "nvfp4").
std::string alias_repo(const Alias& alias, const std::string& format);

std::filesystem::path cache_root();                  // DROMOS_MODELS, else ~/.cache/dromos/models
std::filesystem::path install_dir(const std::string& repo);
std::string hf_token();                              // HF_TOKEN, else the huggingface-cli token, else empty
bool installed(const std::filesystem::path& dir);    // every file of its manifest has verified

struct RemoteFile {
  std::string path;
  uint64_t size = 0;
  bool lfs = false;
  std::string sha256;   // LFS files
  std::string blob_id;  // git blob id, for the rest
};

struct RemoteModel {
  std::string repo;
  std::string revision;
  std::vector<RemoteFile> files;  // the files inference needs
  uint64_t total_bytes() const;
};

// The files the engine reads from a model folder (weights, configs, tokenizer, chat template).
bool needed_file(const std::string& path);

// Lists a repository from the hub API. Throws std::runtime_error, with the reason, when it cannot.
RemoteModel describe(const std::string& repo, const std::string& token);

struct Progress {
  std::string file;
  uint64_t done = 0, total = 0;  // this file
  size_t index = 0, count = 0;   // which file of the model, out of how many
  uint64_t model_done = 0, model_total = 0;
};
using ProgressFn = std::function<void(const Progress&)>;

// Downloads the model into dir (created if missing). Files already verified are kept. Throws
// std::runtime_error when a file cannot be fetched or does not verify.
void pull(const RemoteModel& model, const std::filesystem::path& dir, const std::string& token,
          const ProgressFn& progress);

}  // namespace lx::hub
