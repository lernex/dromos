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

// A point in the forward pass whose hidden state a drafter reads: the mean over the residual
// streams after block `layer` of pass `pass`, or with layer == -1 after the whole pass.
struct FeatureTap {
  int pass = 0;
  int layer = -1;
};

class MetisModel {
 public:
  // With `experts`, the routed experts are not looked up: every MoE layer runs them through the
  // offload runtime (engine/runtime/offload.h), wherever they live. `speculative` allocates the
  // pending-position history of every Mamba layer (kernels.h, "Speculative decoding"), which
  // forward_spec and truncate need.
  MetisModel(const MetisConfig& config, const WeightLookup& weights, int max_slots, int kv_capacity,
             int max_tokens, offload::ExpertOffload* experts = nullptr, bool speculative = false);
  ~MetisModel();
  MetisModel(const MetisModel&) = delete;
  MetisModel& operator=(const MetisModel&) = delete;

  // Forgets a slot's state; its next step starts the sequence over.
  void reset_slot(int slot);
  int slot_length(int slot) const { return slot_len_[slot]; }

  // A slot's complete state after its first `length` tokens, so a prefix can be restored without
  // re-running it. The Mamba state is a recurrence and cannot be cut back, so a snapshot is only
  // valid at the length it was taken. Layout, per (pass, layer): a Mamba layer's conv state then its
  // SSM state; an attention layer's K then V for the first `length` positions. `dst` and `src` are
  // device or pinned host memory; the copies run on `stream` in order with the model's steps.
  size_t slot_state_bytes(int length) const;
  void save_slot(int slot, void* dst, int length, cudaStream_t stream);
  void restore_slot(int slot, const void* src, int length, cudaStream_t stream);  // the slot now holds `length`

  // Runs one step. Writes f32 logits to `logits` (device): every token's row [T, V] when
  // all_rows, otherwise each sequence's last row [n_seq, V]. A slot with pending positions folds
  // them into its state first (it can no longer be truncated before this step).
  void forward(const StepBatch& batch, float* logits, bool all_rows, cudaStream_t stream);

  // ---- Speculative decoding ----------------------------------------------------------------
  // A verify step: like forward with all_rows, but each sequence's new tokens (at most
  // max_verify_tokens()) stay pending, so truncate can drop any suffix of them afterwards.
  // Steps of up to 8 tokens give every row the bits a one-token step gives it (the decode
  // numerics are batch invariant), so row j's logits are exactly what plain decoding computes
  // after the same tokens.
  void forward_spec(const StepBatch& batch, float* logits, cudaStream_t stream);
  // Drops the slot's positions from `length` on. length can go back as far as the slot's folded
  // length (pending positions only: at most the last LX_SPEC_MAX_PENDING).
  void truncate(int slot, int length);
  int folded_length(int slot) const { return folded_[slot]; }
  // The largest verify step that is still batch invariant. Attention splits a token's cache over a
  // cluster only while T * KVH < 64 (attention.cu kSplitBelow), and the split changes the order of
  // its sums: Praxis (8 kv heads) 7 tokens, Logos (10) 6.
  int max_verify_tokens() const;

  // Hidden states for drafters: after set_feature_taps, every step writes, for each of its tokens,
  // the taps' stream means side by side: features() is f32 [T, feature_dim()] on the device, valid
  // until the next step.
  void set_feature_taps(const std::vector<FeatureTap>& taps);
  int feature_dim() const;
  const float* features() const;

  // The tied embedding / LM head, for drafters that share it (a block drafter embeds its anchor
  // with it and reads its logits through it): bf16 [V, D], and the int8 decode copy with per-row
  // scales when the model made one (else null).
  struct HeadWeights {
    const void* bf16 = nullptr;
    const int8_t* int8 = nullptr;
    const float* scale = nullptr;
  };
  HeadWeights head_weights() const;

  int vocab_size() const { return cfg_.vocab_size; }
  const MetisConfig& config() const { return cfg_; }

  // The weights, state and scratch; defined in metis.cpp.
  struct Impl;

 private:
  void run(int T, int n_seq, bool all_rows, float* logits, bool spec, cudaStream_t stream);
  void stage(const StepBatch& b, bool spec, cudaStream_t stream);
  void launch(int T, int n_seq, bool all_rows, float* logits, bool spec, cudaStream_t stream);
  void fold_pending(const StepBatch& b, cudaStream_t stream);

  MetisConfig cfg_;
  std::vector<int> slot_len_;
  std::vector<int> folded_;   // positions in the base state
  std::vector<int> pending_;  // positions after it, held as inputs (speculative mode)
  Impl* impl_;
};

}  // namespace lx
