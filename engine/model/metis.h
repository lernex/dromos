// The Metis forward pass (Metis-1.6 Logos and Praxis), built from the kernel library.
#pragma once

#include <cstddef>
#include <functional>
#include <set>
#include <string>
#include <vector>

#include "core/config.h"
#include "cuda/kernels.h"
#include "model/traffic.h"

namespace lx {

namespace offload {
class ExpertOffload;
}
class ModelWeights;  // core/checkpoint.h

// A plain tensor as the model needs it: a device pointer, its dtype and shape. The model checks both.
struct WeightView {
  const void* data = nullptr;
  int dtype = 0;  // lx_dtype: LX_F32 or LX_BF16
  std::vector<int64_t> shape;
};

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

// The formats a model runs in, as the plan decides them: the format of each kind of tensor (WeightRecipe), the
// math each product multiplies in, and whether decode reads the lossless BF16X copy of the bf16 projections.
//
// Every projection and expert stack is one lx_weight, held in the format ModelWeights loaded it in, and decode
// and prefill read that one copy: decode-sized products (at most 8 rows) through the GEMV and expert decode
// kernels, weight-only; larger ones through the tensor-core GEMM and the grouped expert kernel, in
// prefill_math (dense) and expert_math (experts). A weight whose format the math cannot read (a BF16
// projection the recipe keeps wide, under INT8 or FP8 math) multiplies under BF16 math. dense, experts and
// latent_up are the formats the plan chose: a weight must be held in its kind's format or in BF16 (wider by the
// recipe), and the model refuses weights loaded in anything else.
//
// bf16x_decode: the model encodes a lossless BF16X copy (bf16x.cu) of every projection and stack held in BF16 at
// construction (in a BF16 model all of them; in a quantized one the wide tensors its recipe keeps), and decode reads
// it (the same logits to the bit at ~1.3x fewer bytes) while prefill keeps reading the BF16 copy. The plan budgets
// the second copy. Weights held as BF16X (loaded with that target) need no flag: decode reads them as they are, and
// prefill expands each one to a bf16 scratch first. bf16x_dropped names the copy units the plan left out because
// the room beside its sequences did not hold them (bf16x_copy_made says which are made).
struct ModelFormats {
  lx_wtype dense = LX_W_BF16, experts = LX_W_BF16, latent_up = LX_W_BF16;
  lx_math prefill_math = LX_MATH_BF16, expert_math = LX_MATH_BF16;
  bool bf16x_decode = false;
  std::set<std::string> bf16x_dropped;
};

// The BF16X copy of one unit is made: a projection by its checkpoint name, a layer's routed experts by
// "layers.N.moe.local_experts" (both stacks, which the decode kernel reads together), the LM head by "lm_head".
inline bool bf16x_copy_made(const ModelFormats& f, const std::string& unit) {
  return f.bf16x_decode && !f.bf16x_dropped.count(unit);
}

class MetisModel {
 public:
  // The model reads `weights` (which must outlive it) in the formats `formats` names. With `experts`, the
  // routed experts are not read from `weights`: every MoE layer runs them through the offload runtime
  // (engine/runtime/offload.h), wherever they live. `speculative` allocates the pending-position history of
  // every Mamba layer (kernels.h, "Speculative decoding"), which forward_spec and truncate need.
  //
  // LX_PREFILL=bf16|int8|fp8|fp4 overrides the prefill math of the dense products and the experts, for
  // measurement; it never changes a stored weight (a weight the math cannot read stays at BF16 math).
  MetisModel(const MetisConfig& config, const ModelWeights& weights, const ModelFormats& formats, int max_slots,
             int kv_capacity, int max_tokens, offload::ExpertOffload* experts = nullptr, bool speculative = false);
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
  void save_slot(int slot, void* dst, int length, void* stream);
  void restore_slot(int slot, const void* src, int length, void* stream);  // the slot now holds `length`

  // Runs one step. Writes f32 logits to `logits` (device): every token's row [T, V] when
  // all_rows, otherwise each sequence's last row [n_seq, V]. A slot with pending positions folds
  // them into its state first (it can no longer be truncated before this step).
  void forward(const StepBatch& batch, float* logits, bool all_rows, void* stream);

  // ---- Speculative decoding ----------------------------------------------------------------
  // A verify step: like forward with all_rows, but each sequence's new tokens (at most
  // max_verify_tokens()) stay pending, so truncate can drop any suffix of them afterwards.
  // Steps of up to 8 tokens give every row the bits a one-token step gives it (the decode
  // numerics are batch invariant), so row j's logits are exactly what plain decoding computes
  // after the same tokens.
  void forward_spec(const StepBatch& batch, float* logits, void* stream);
  // Drops the slot's positions from `length` on. length can go back as far as the slot's folded
  // length (pending positions only: at most the last LX_SPEC_MAX_PENDING).
  void truncate(int slot, int length);
  int folded_length(int slot) const { return folded_[slot]; }
  // The largest verify step that is still batch invariant. Attention splits a token's cache over a
  // cluster only while T * KVH < 64 (attention.cu kSplitBelow), and the split changes the order of
  // its sums: Praxis (8 kv heads) 7 tokens, Logos (10) 6.
  int max_verify_tokens() const;

  // Measurement (the early-exit drafter, cli/spec.cpp's pass1): while `logits` is set (a device f32 [V] or [T, V]
  // the caller owns), every step also reads the first pass's output, the residual streams after pass 0 through the
  // final norm and the tied head, into it: the logits an exit after one pass would give each row. That is one extra
  // head read per row, and steps run eagerly (no graphs, no decode program) while it is set. nullptr turns it off.
  void set_early_logits(float* logits);

  // Hidden states for drafters: after set_feature_taps, every step writes, for each of its tokens,
  // the taps' stream means side by side: features() is f32 [T, feature_dim()] on the device, valid
  // until the next step.
  void set_feature_taps(const std::vector<FeatureTap>& taps);
  int feature_dim() const;
  const float* features() const;

  // The tied embedding / LM head, for drafters that share it (a block drafter embeds its anchor
  // with it and reads its logits through it): the bf16 gather table [V, D] (bf16 in every format),
  // the head [V, D] as the model holds it (any format lx_gemv_w reads), and the BF16X copy decode
  // reads instead when the model made one (ModelFormats::bf16x_decode; else null).
  struct HeadWeights {
    const void* embedding = nullptr;
    lx_weight head{LX_W_BF16, nullptr, nullptr, 1.0f, nullptr};
    const void* head_bf16x = nullptr;
  };
  HeadWeights head_weights() const;

  int vocab_size() const { return cfg_.vocab_size; }

  // What one decode step moves through memory with `context` cache positions per stream (traffic.h): the matrices this
  // model loaded, in the formats and with the BF16X decode copies it reads, counted over the same events as decode.
  DecodeTraffic decode_traffic(int context) const;
  // The routing census behind decode_traffic's measured experts. While on, decode steps (one token per stream) run
  // eagerly and their routing is read back and counted; turning it on starts a fresh census.
  void set_expert_census(bool on);
  const MetisConfig& config() const { return cfg_; }

  // The weights, state and scratch; defined in metis.cpp.
  struct Impl;

 private:
  void run(int T, int n_seq, bool all_rows, float* logits, bool spec, void* stream);
  void stage(const StepBatch& b, bool spec, void* stream);
  void launch(int T, int n_seq, bool all_rows, float* logits, bool spec, void* stream);
  void fold_pending(const StepBatch& b, void* stream);

  MetisConfig cfg_;
  std::vector<int> slot_len_;
  std::vector<int> folded_;   // positions in the base state
  std::vector<int> pending_;  // positions after it, held as inputs (speculative mode)
  Impl* impl_;
};

}  // namespace lx
