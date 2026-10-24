#include "cli/common.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <stdexcept>

#include <unistd.h>

#include "core/config.h"
#include "core/hub.h"
#include "nlohmann/json.hpp"

namespace lx::cli {
namespace fs = std::filesystem;

bool stderr_is_terminal() { return isatty(fileno(stderr)) != 0; }

bool looks_like_path(const std::string& name) {
  if (name.empty()) return false;
  if (name[0] == '/' || name[0] == '.' || name[0] == '~') return true;
  // owner/name is a repository unless a folder of that name exists here
  return name.find('/') != std::string::npos && fs::is_directory(name);
}

Model find_model(const std::string& name, bool download) {
  if (looks_like_path(name)) {
    fs::path dir = name;
    if (name.rfind("~", 0) == 0) {
      const char* home = std::getenv("HOME");
      dir = (home ? std::string(home) : std::string(".")) + name.substr(1);
    }
    if (!fs::is_directory(dir)) throw std::runtime_error("there is no model folder at " + dir.string());
    return Model{dir, name, ""};
  }

  const hub::ModelRef ref = hub::resolve(name);
  const fs::path dir = hub::install_dir(ref.repo);
  if (hub::installed(dir)) return Model{dir, name, ref.repo};
  if (!download) throw std::runtime_error(name + " is not installed yet: run `lernex pull " + name + "` first");

  const std::string token = hub::hf_token();
  std::fprintf(stderr, "%s is not installed yet. Pulling %s\n", name.c_str(), ref.repo.c_str());
  const hub::RemoteModel remote = hub::describe(ref.repo, token);
  std::fprintf(stderr, "%s, %zu files\n", human_bytes(remote.total_bytes()).c_str(), remote.files.size());
  const bool live = stderr_is_terminal();
  std::string printed_file;
  hub::pull(remote, dir, token, [&](const hub::Progress& p) {
    if (live) {
      std::fprintf(stderr, "\r  %-40.40s %6.1f%%   %s / %s   ", p.file.c_str(),
                   p.model_total ? 100.0 * static_cast<double>(p.model_done) / static_cast<double>(p.model_total) : 100.0,
                   human_bytes(p.model_done).c_str(), human_bytes(p.model_total).c_str());
    } else if (p.done >= p.total && p.file != printed_file) {
      std::fprintf(stderr, "  %s  %s\n", p.file.c_str(), human_bytes(p.total).c_str());
      printed_file = p.file;
    }
  });
  if (live) std::fprintf(stderr, "\r%80s\r", "");
  return Model{dir, name, ref.repo};
}

runtime::Plan plan_for(const Model& model, int sequences, int context, int weights) {
  const MetisConfig config = load_config(model.dir.string());
  runtime::PlanRequest request;
  request.sequences = sequences;
  request.context = context;
  request.weights = weights;
  return runtime::make_plan(runtime::query_device(0), config, runtime::CheckpointInfo{}, request);
}

runtime::SamplingParams recommended_sampling(const Model& model) {
  runtime::SamplingParams sp;
  std::ifstream in(model.dir / "generation_config.json");
  if (!in) return sp;
  try {
    const nlohmann::json j = nlohmann::json::parse(in);
    sp.temperature = j.value("temperature", sp.temperature);
    sp.top_p = j.value("top_p", sp.top_p);
    sp.top_k = j.value("top_k", sp.top_k);
  } catch (const std::exception&) {
  }
  return sp;
}

std::string human_bytes(uint64_t bytes) {
  char buf[32];
  const double b = static_cast<double>(bytes);
  if (b >= 1e9) std::snprintf(buf, sizeof buf, "%.1f GB", b / 1e9);
  else if (b >= 1e6) std::snprintf(buf, sizeof buf, "%.0f MB", b / 1e6);
  else if (b >= 1e3) std::snprintf(buf, sizeof buf, "%.0f KB", b / 1e3);
  else std::snprintf(buf, sizeof buf, "%llu B", static_cast<unsigned long long>(bytes));
  return buf;
}

std::string human_seconds(double seconds) {
  char buf[32];
  if (seconds < 1.0) std::snprintf(buf, sizeof buf, "%.0f ms", seconds * 1e3);
  else if (seconds < 90.0) std::snprintf(buf, sizeof buf, "%.1f s", seconds);
  else std::snprintf(buf, sizeof buf, "%.1f min", seconds / 60.0);
  return buf;
}

}  // namespace lx::cli
