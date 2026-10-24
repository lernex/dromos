// A Metis checkpoint loaded for running larger than the GPU: the dense tensors in VRAM (or, for the
// ones only a prefill reads, in pinned host memory: the model copies each into a VRAM buffer as a
// prefill product needs it), the routed experts in an ExpertOffload (RAM, NVMe, and the VRAM slots
// it allocates last, from what the budget leaves).
//
//   auto loaded = OffloadedModel::load(dir, config, options);
//   MetisModel model(config, loaded->lookup(), slots, kv, tokens, &loaded->experts());
//   loaded->finalize();   // the expert cache takes the VRAM the budget has left
#pragma once

#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/safetensors.h"
#include "model/metis.h"
#include "runtime/offload.h"

namespace lx::offload {

struct OffloadLoadOptions {
  int experts = -1;                   // lx_wtype of the experts; -1 keeps the checkpoint's format
  bool dense_prefill_on_host = false; // projections decode reads as int8 copies keep their bf16 in RAM
  int64_t vram_budget = 0;            // bytes for everything this process puts on the GPU, 0 for the card
  int64_t ram_budget = -1;            // bytes of pinned RAM for experts, -1 for all of them
  std::filesystem::path routing_stats;  // per-layer expert frequencies (lernex offload stats)
  OffloadOptions policy;              // the replacement and CPU lane policy (sizes are filled in)
  bool w8_decode = true;              // the model's decode reads int8 copies (LX_NO_W8 unset)
};

class OffloadedModel {
 public:
  static std::unique_ptr<OffloadedModel> load(const std::filesystem::path& dir, const MetisConfig& config,
                                              const OffloadLoadOptions& options);
  ~OffloadedModel();

  WeightLookup lookup() const;
  ExpertOffload& experts() { return *offload_; }
  // Gives the expert cache whatever the budget leaves after everything allocated so far.
  void finalize();
  std::string describe() const;
  int64_t dense_vram_bytes() const { return dense_vram_; }
  int64_t dense_host_bytes() const { return dense_host_; }

 private:
  OffloadedModel(SafetensorsSet set) : set_(std::move(set)) {}

  SafetensorsSet set_;
  OffloadLoadOptions options_;
  std::map<std::string, WeightView> views_;
  std::vector<void*> device_blocks_, host_blocks_;
  std::unique_ptr<ExpertOffload> offload_;
  RoutingStats stats_;
  int64_t dense_vram_ = 0, dense_host_ = 0;
  int64_t budget_ = 0;
  std::vector<std::string> notes_;
};

}  // namespace lx::offload
