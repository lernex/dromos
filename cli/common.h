// What the commands share: finding a model folder, the plan for this GPU, and small formatting helpers.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>

#include "core/checkpoint.h"
#include "runtime/engine.h"
#include "runtime/plan.h"

namespace lx::cli {

// A model the commands can run: a folder on disk, and the name it was asked for.
struct Model {
  std::filesystem::path dir;
  std::string name;  // what the user typed: "praxis:fp8", a repository, or a folder
  std::string repo;  // the hub repository, when it came from one
};

// Finds the model a name refers to. A folder path is used as it is. An alias or repository must be
// installed in the cache; when `download` is set and it is not, it is pulled first, with progress.
// Throws std::runtime_error with a plain message.
Model find_model(const std::string& name, bool download);

// The weight format a command was asked for: --weights bf16|int8|fp8|nvfp4|bf16x (-1, the default, is the format
// the file stores) and --no-bf16x (decode reads the plain BF16 tensors, for measuring).
struct FormatFlags {
  int weights = -1;
  bool bf16x = true;
  bool fast_math = false;  // --fast-math: the quantized tensor-core maths (native_math, fast)
};
// When argv[i] is --weights (with its value, which it consumes) or --no-bf16x, records it and returns true.
bool parse_format_flag(int argc, char** argv, int& i, FormatFlags& flags);
// "[--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x]", for usage lines.
extern const char* const kFormatUsage;

// The plan for this GPU, the model, and the user's wishes (0 and -1 for automatic). memory_fraction is
// the share of the free memory the plan may use. LX_VRAM_BUDGET_GB and LX_RAM_BUDGET_GB, when set, bound the
// GPU memory this process takes and the pinned RAM offloaded experts may take.
runtime::Plan plan_for(const Model& model, int sequences, int context, int weights = -1, double memory_fraction = 0.92,
                       bool bf16x = true, bool fast = false);

// A model folder loaded for a command that runs MetisModel itself (eval, prefill, spec, features): in the file's
// format or the one the flags name, under the plan's rules (runtime::can_load), with the formats and maths the plan
// gives that format on this GPU (runtime::model_formats). `description` is one line saying which.
struct LoadedWeights {
  lx_wtype format = LX_W_BF16;
  lx::ModelFormats formats;
  std::unique_ptr<lx::ModelWeights> weights;
  std::string description;
};
LoadedWeights load_weights(const std::filesystem::path& dir, const lx::MetisConfig& config, const FormatFlags& flags);

// The engine for a plan. When the memory the plan asked for is not there (another process took it
// after the plan was made), it plans again with fewer sequences and a smaller share of the free
// memory, and says so on stderr. Other failures are thrown as they are.
struct StartedEngine {
  runtime::Plan plan;
  std::unique_ptr<runtime::Engine> engine;
};
// Starts an engine with a plan for `sequences` at `context` (0 is the plan's own choice). The plan takes
// `memory_fraction` of the memory free now; if the engine's allocations still fail, it retries with fewer
// sequences and a smaller fraction, and says so on stderr.
StartedEngine start_engine(const Model& model, int sequences, int context, int weights,
                           const runtime::EngineOptions& options, double memory_fraction = 0.92, bool bf16x = true,
                           bool fast = false);

// The sampling a model recommends (generation_config.json), for a chat that asks for nothing else.
// With thinking off, its "direct" object overrides the top level.
runtime::SamplingParams recommended_sampling(const Model& model, bool thinking);

std::string human_bytes(uint64_t bytes);
std::string human_seconds(double seconds);

// Whether stderr is a terminal that can redraw a progress line.
bool stderr_is_terminal();

// Whether a name is a folder on disk rather than an alias or a repository.
bool looks_like_path(const std::string& name);

}  // namespace lx::cli
