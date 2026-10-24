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
#include <utility>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace lx {

struct LoadOptions {
  lx_wtype target = LX_W_BF16;  // the format the large projections are held in
  bool skip_experts = false;    // leave the routed expert stacks out (the offload runtime holds them)
};

class ModelWeights {
 public:
  // What the folder holds, without loading it.
  static lx_wtype stored_format(const std::filesystem::path& dir);

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
  // wider than the target (the Mamba-2 out_proj, and for NVFP4 the MoE latent_up) follows the
  // checkpoints' recipe, so a converted model matches the published file's choices.
  const lx_weight* projection(const std::string& name) const;  // null when absent

  // Every projection's name ("layers.4.mixer.qkv", "lm_head"), and its shape as the folder states it.
  struct Shape {
    int64_t E = 1, N = 0, K = 0;  // E experts of [N, K]; a matrix is E = 1
    bool stack = false;
  };
  std::vector<std::string> projection_names() const;
  Shape projection_shape(const std::string& name) const;
  // Expert e of a projection (0 for a matrix) as its own lx_weight: its rows, its scales and its global.
  lx_weight projection_expert(const std::string& name, int64_t e) const;
  // The bytes one expert's data and its scales take in the device buffers.
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
  std::map<std::string, lx_weight> projections_;
  std::map<std::string, Shape> shapes_;
  size_t peak_bytes_ = 0;
};

}  // namespace lx
