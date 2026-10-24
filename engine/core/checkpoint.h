// A Metis model folder, loaded for the engine in the format the plan asks for. Reads BF16
// checkpoints and the compressed-tensors FP8 and NVFP4 checkpoints, and converts between them at
// load (round-to-nearest) when asked for a format the file is not stored in. Conversion streams
// tensor by tensor from the memory-mapped file, so a model larger than the GPU in BF16 (Logos on
// 16 GB) still loads into a smaller format.
#pragma once

#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace lx {

struct LoadOptions {
  lx_wtype target = LX_W_BF16;  // the format the large projections are held in
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
  lx_wtype format() const noexcept { return format_; }
  size_t device_bytes() const noexcept;

 private:
  lx_wtype format_ = LX_W_BF16;
  std::vector<DeviceBuffer> buffers_;
  std::map<std::string, WeightView> plain_;
  std::map<std::string, lx_weight> projections_;
};

}  // namespace lx
