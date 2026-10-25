// A Metis model folder, loaded for the engine in the format the plan asks for. Reads BF16
// checkpoints and the compressed-tensors FP8 and NVFP4 checkpoints, and converts between them at
// load (round-to-nearest) when asked for a format the file is not stored in. Conversion streams
// tensor by tensor from the memory-mapped file, so a model larger than the GPU in BF16 (Logos on
// 16 GB) still loads into a smaller format.
#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace lx {

struct LoadOptions {
  // The format the large projections are held in. Explicit only: nothing picks a lossy format for the
  // caller. INT8 is round-to-nearest from a BF16, FP8 or NVFP4 folder; FP8 keeps a stored FP8 folder as it is.
  lx_wtype target = LX_W_BF16;
  // Leave the routed expert stacks (layers.N.moe.local_experts.gate_up and .down, with their scales and
  // globals) off the device. The offload runtime holds them: their shapes and byte sizes stay readable.
  bool skip_experts = false;
};

// The loader's recipe, for the plan's accounting: the format a projection is held in for `target`, by its name
// without ".weight" ("layers.4.mixer.qkv"; "embedding" for the tied head). A folder loaded in its own format
// keeps each projection as the file stores it, which for the published folders is this same rule.
lx_wtype projection_format(lx_wtype target, std::string_view projection);
// Why NVFP4 cannot load from a folder stored as `stored`: the text ModelWeights::load throws, naming the
// model's GPTQ NVFP4 checkpoint when there is one.
std::string nvfp4_unavailable(lx_wtype stored, const MetisConfig& config);

class ModelWeights {
 public:
  // What the folder holds, without loading it: BF16, FP8 (F8_E4M3 projections) or NVFP4 (U8 packed
  // nibbles with e4m3 block scales and per-expert globals, as the GPTQ folder writes them).
  static lx_wtype stored_format(const std::filesystem::path& dir);

  // Throws std::runtime_error for a request the loader refuses (NVFP4 from a folder that is not NVFP4:
  // round-to-nearest fails its KL, so the message names the GPTQ NVFP4 checkpoint of that model).
  static ModelWeights load(const std::filesystem::path& dir, const MetisConfig& config, const LoadOptions& options);

  ModelWeights() = default;
  ModelWeights(ModelWeights&&) = default;
  ModelWeights& operator=(ModelWeights&&) = default;
  ModelWeights(const ModelWeights&) = delete;
  ModelWeights& operator=(const ModelWeights&) = delete;
  ~ModelWeights() = default;

  // A plain tensor (norms, embeddings, f32 routers, biases), bf16 or f32 as the model expects it.
  WeightView find(const std::string& name) const;
  // A projection [N, K] or an expert stack [E, N, K], in the format it is held in. Which ones stay
  // wider than the target (the Mamba-2 out_proj, the pass LoRA, the depth memory's key, value and
  // metadata write, and for NVFP4 the MoE latent_up) follows the recipe, so a converted model matches
  // the published file's choices.
  // Null when absent, and for an expert stack when LoadOptions::skip_experts left it off the device.
  const lx_weight* projection(const std::string& name) const;

  // The projections on the device ("layers.4.mixer.qkv", "lm_head"; the expert stacks only when loaded).
  struct Shape {
    int64_t E = 1, N = 0, K = 0;  // E experts of [N, K]; a matrix is E = 1
    bool stack = false;
  };
  std::vector<std::string> projection_names() const;
  // The shape of any projection in the folder, skipped expert stacks included.
  Shape projection_shape(const std::string& name) const;
  // Expert e of a loaded projection (0 for a matrix) as its own lx_weight: its rows, its scales and its
  // global. Throws for a stack that skip_experts left off the device.
  lx_weight projection_expert(const std::string& name, int64_t e) const;
  // The bytes one expert's data and its scales take in the target format (the device buffers, or the
  // size the offload runtime must hold for a skipped stack). Any projection in the folder.
  std::pair<size_t, size_t> expert_bytes(const std::string& name) const;

  lx_wtype format() const noexcept { return format_; }
  size_t device_bytes() const noexcept;
  // The most device memory the load held at once: the final buffers plus the conversion scratch.
  size_t peak_device_bytes() const noexcept { return peak_bytes_; }

 private:
  friend struct CheckpointBuilder;

  lx_wtype format_ = LX_W_BF16;
  std::vector<DeviceBuffer> buffers_;
  std::map<std::string, WeightView> plain_;
  std::map<std::string, lx_weight> projections_;  // loaded projections only
  std::map<std::string, Shape> shapes_;           // every projection, loaded or skipped
  std::map<std::string, lx_wtype> formats_;       // every projection's target format, loaded or skipped
  size_t peak_bytes_ = 0;
};

}  // namespace lx
