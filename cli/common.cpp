#include "cli/common.h"

#include <algorithm>
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
  if (!download) throw std::runtime_error(name + " is not installed yet: run `dromos pull " + name + "` first");

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

const char* const kFormatUsage = "[--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x] [--fast-math]";

bool parse_format_flag(int argc, char** argv, int& i, FormatFlags& flags) {
  const std::string a = argv[i];
  if (a == "--no-bf16x") {
    flags.bf16x = false;
    return true;
  }
  if (a == "--fast-math") {
    flags.fast_math = true;
    return true;
  }
  if (a != "--weights") return false;
  if (i + 1 >= argc) throw std::runtime_error("--weights needs a value: bf16, int8, fp8, nvfp4 or bf16x");
  flags.weights = runtime::weights_from_flag(argv[++i]);
  return true;
}

namespace {

// A budget in GiB from the environment, in bytes, or 0 when it is not set.
size_t env_gib(const char* name) {
  const char* v = std::getenv(name);
  const double gib = v && *v ? std::atof(v) : 0.0;
  return gib > 0 ? static_cast<size_t>(gib * 1073741824.0) : 0;
}

}  // namespace

runtime::Plan plan_for(const Model& model, int sequences, int context, int weights, double memory_fraction,
                       bool bf16x, bool fast, bool speculative) {
  const MetisConfig config = load_config(model.dir.string());
  runtime::PlanRequest request;
  request.sequences = sequences;
  request.context = context;
  request.weights = weights;
  request.bf16x = bf16x;
  request.fast_math = fast;
  request.speculative = speculative;
  request.memory_fraction = memory_fraction;
  request.vram_budget = env_gib("LX_VRAM_BUDGET_GB");
  request.ram_budget = env_gib("LX_RAM_BUDGET_GB");
  const runtime::CheckpointInfo checkpoint = runtime::inspect_checkpoint(model.dir.string());
  return runtime::make_plan(runtime::query_device(0), config, checkpoint, request);
}

LoadedWeights load_weights(const fs::path& dir, const MetisConfig& config, const FormatFlags& flags) {
  const lx_wtype stored = ModelWeights::stored_format(dir);
  const lx_wtype target = flags.weights >= 0 ? static_cast<lx_wtype>(flags.weights) : stored;
  std::string how;
  if (!runtime::can_load(target, stored, config, &how))
    throw std::runtime_error(runtime::format_name(target) + " cannot be loaded: " + how);
  const runtime::DeviceInfo device = runtime::query_device(0);
  LoadedWeights out;
  out.format = target;
  out.formats = runtime::model_formats(target, device, flags.bf16x && runtime::bf16x_available(), flags.fast_math);
  LoadOptions options;
  options.target = target;
  out.weights = std::make_unique<ModelWeights>(ModelWeights::load(dir, config, options));
  auto math = [](lx_math m) {
    switch (m) {
      case LX_MATH_INT8: return "INT8";
      case LX_MATH_FP8: return "FP8";
      case LX_MATH_FP4: return "FP4";
      default: return "BF16";
    }
  };
  out.description = "weights " + runtime::format_name(target) + " (" + how + "), " +
                    human_bytes(out.weights->device_bytes()) + "; prefill math " + math(out.formats.prefill_math) +
                    ", experts " + math(out.formats.expert_math) +
                    (out.formats.bf16x_decode ? "; decode reads BF16X copies" : "");
  return out;
}

StartedEngine start_engine(const Model& model, int sequences, int context, int weights,
                           const runtime::EngineOptions& options, double memory_fraction, bool bf16x, bool fast) {
  double fraction = memory_fraction;
  int wanted = sequences;
  int format = weights;
  for (int attempt = 1;; ++attempt) {
    runtime::Plan plan = plan_for(model, wanted, context, format, fraction, bf16x, fast, options.speculate);
    try {
      auto engine = std::make_unique<runtime::Engine>(model.dir.string(), plan, options);
      return StartedEngine{std::move(plan), std::move(engine)};
    } catch (const std::runtime_error& e) {
      const std::string what = e.what();
      // Out of memory comes back from a device allocation, a kernel that allocates, or a copy; the message
      // carries the backend's own text for it.
      const bool memory = what.find("device allocation") != std::string::npos ||
                          what.find("out of memory") != std::string::npos;
      if (!memory || attempt >= 4) throw;
      // The retry keeps the format this attempt chose: re-planning could pick one the build cannot load.
      format = static_cast<int>(plan.weights);
      fraction = std::max(0.4, fraction * 0.8);
      wanted = std::max(1, (wanted > 0 ? wanted : plan.max_slots) / 2);
      std::fprintf(stderr,
                   "The memory the plan asked for was not free (%s).\n"
                   "Retrying with %d sequences and %.0f%% of the free memory.\n",
                   what.c_str(), wanted, fraction * 100.0);
    }
  }
}

runtime::SamplingParams recommended_sampling(const Model& model, bool thinking) {
  runtime::SamplingParams sp;
  std::ifstream in(model.dir / "generation_config.json");
  if (!in) return sp;
  const auto read = [&sp](const nlohmann::json& j) {
    sp.temperature = j.value("temperature", sp.temperature);
    sp.top_p = j.value("top_p", sp.top_p);
    sp.top_k = j.value("top_k", sp.top_k);
    sp.min_p = j.value("min_p", sp.min_p);
    sp.repetition_penalty = j.value("repetition_penalty", sp.repetition_penalty);
  };
  try {
    const nlohmann::json j = nlohmann::json::parse(in);
    read(j);
    if (!thinking && j.contains("direct") && j["direct"].is_object()) read(j["direct"]);
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
