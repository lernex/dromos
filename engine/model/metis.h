// The Metis forward pass (Metis-1.6 Logos and Praxis), built from the kernel library.
#pragma once

#include <cuda_runtime.h>

#include <functional>
#include <string>
#include <vector>

#include "core/config.h"

namespace lx {

namespace offload {
class ExpertOffload;
}

// A weight as the model needs it: a device pointer, its dtype and shape. The model checks both.
struct WeightView {
  const void* data = nullptr;
  int dtype = 0;  // lx_dtype: LX_F32 or LX_BF16
  std::vector<int64_t> shape;
};
using WeightLookup = std::function<WeightView(const std::string& name)>;

// One step: the new tokens of n_seq sequences, concatenated. Each sequence lives in a state
// slot; a slot's first step starts it from empty state, later steps continue it.
struct StepBatch {
  std::vector<int32_t> tokens;      // [T]
  std::vector<int32_t> cu_seqlens;  // [n_seq + 1], token offsets, starting at 0
  std::vector<int32_t> slots;       // [n_seq]
};

class MetisModel {
 public:
  // With `experts`, the routed experts are not looked up: every MoE layer runs them through the
  // offload runtime (engine/runtime/offload.h), wherever they live.
  MetisModel(const MetisConfig& config, const WeightLookup& weights, int max_slots, int kv_capacity,
             int max_tokens, offload::ExpertOffload* experts = nullptr);
  ~MetisModel();
  MetisModel(const MetisModel&) = delete;
  MetisModel& operator=(const MetisModel&) = delete;

  // Forgets a slot's state; its next step starts the sequence over.
  void reset_slot(int slot);
  int slot_length(int slot) const { return slot_len_[slot]; }

  // Runs one step. Writes f32 logits to `logits` (device): every token's row [T, V] when
  // all_rows, otherwise each sequence's last row [n_seq, V].
  void forward(const StepBatch& batch, float* logits, bool all_rows, cudaStream_t stream);

  int vocab_size() const { return cfg_.vocab_size; }
  const MetisConfig& config() const { return cfg_; }

  // The weights, state and scratch; defined in metis.cpp.
  struct Impl;

 private:
  void run(int T, int n_seq, bool all_rows, float* logits, cudaStream_t stream);

  MetisConfig cfg_;
  std::vector<int> slot_len_;
  Impl* impl_;
};

}  // namespace lx
