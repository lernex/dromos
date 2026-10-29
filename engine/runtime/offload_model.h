// A Metis checkpoint loaded for running larger than the GPU: the dense weights in VRAM, loaded by
// ModelWeights in the plan's format with the routed experts left out, and the routed experts in an
// ExpertOffload (RAM, NVMe, and the VRAM slots it allocates last, from what the budget leaves), held in
// the plan's expert format. Each weight is held once, in one format: decode and prefill read the same
// copy, wherever it lives.
//
//   auto loaded = OffloadedModel::load(dir, config, options);
//   MetisModel model(config, loaded->weights(), formats, slots, kv, tokens, &loaded->experts());
//   loaded->finalize();   // the expert cache takes the VRAM the budget has left
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "core/safetensors.h"
#include "model/metis.h"
#include "runtime/offload.h"

namespace lx::offload {

struct OffloadLoadOptions {
  // The plan's formats. weights: the target the dense projections load in (ModelWeights' LoadOptions::target).
  // experts: what the routed experts are held in, LX_W_BF16, LX_W_INT8 or LX_W_FP8 (the formats the offload
  // kernels read); the checkpoint's are converted at load (round-to-nearest, or exact dequantization to BF16)
  // when it differs. -1 keeps the checkpoint's own format in either.
  int weights = -1;
  int experts = -1;
  int64_t vram_budget = 0;            // bytes for everything this process puts on the GPU, 0 for the card
  int64_t ram_budget = -1;            // bytes of pinned RAM for experts, -1 for all of them
  std::filesystem::path routing_stats;  // per-layer expert frequencies (dromos offload stats)
  OffloadOptions policy;              // the replacement and CPU lane policy (sizes are filled in)
};

class OffloadedModel {
 public:
  static std::unique_ptr<OffloadedModel> load(const std::filesystem::path& dir, const MetisConfig& config,
                                              const OffloadLoadOptions& options);
  ~OffloadedModel();

  // The dense weights, for MetisModel (with experts()).
  const ModelWeights& weights() const { return dense_; }
  ExpertOffload& experts() { return *offload_; }
  // Gives the expert cache whatever the budget leaves after everything allocated so far.
  void finalize();
  std::string describe() const;
  int64_t dense_vram_bytes() const { return dense_vram_; }

 private:
  OffloadedModel(SafetensorsSet set) : set_(std::move(set)) {}

  SafetensorsSet set_;
  OffloadLoadOptions options_;
  ModelWeights dense_;
  std::unique_ptr<ExpertOffload> offload_;
  RoutingStats stats_;
  int64_t dense_vram_ = 0;
  int64_t budget_ = 0;
  std::vector<std::string> notes_;
};

}  // namespace lx::offload
