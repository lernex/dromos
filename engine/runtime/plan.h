// What this GPU can do, and how to run a model on it. The user picks a model and, if they want,
// a weight format; everything else (which instructions multiply, how many sequences fit, how long
// they can be, which kernel variant runs each product) is decided here, from the hardware.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <string>
#include <vector>

#include "core/config.h"
#include "cuda/kernels.h"

namespace lx::runtime {

struct DeviceInfo {
  int ordinal = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;  // as the kernels see it (after LX_ARCH_OVERRIDE)
  int real_cc_major = 0, real_cc_minor = 0;
  int sms = 0;
  size_t total_bytes = 0, free_bytes = 0, l2_bytes = 0;
  double bandwidth_gbs = 0;  // peak, from the memory clock and bus width
  bool bf16_mma = false, int8_mma = false, fp8_mma = false, fp4_mma = false, clusters = false;
  bool built = false;  // this binary has kernels for the architecture the engine runs as
  int driver = 0, runtime = 0;  // CUDA versions as integers: 13020 for 13.2
  std::string driver_text, runtime_text;

  int cc() const noexcept { return cc_major * 10 + cc_minor; }
  int real_cc() const noexcept { return real_cc_major * 10 + real_cc_minor; }
  // A short, filesystem-safe identity for tuning tables: "NVIDIA_GeForce_RTX_5070_Ti_sm120".
  std::string slug() const;
};

// Queries device `ordinal`. LX_ARCH_OVERRIDE=80|86|89|90|100|120 (or lx_set_arch_override) makes the
// engine behave as that architecture: only lowering is allowed, so every fallback runs and is tested
// on one card.
DeviceInfo query_device(int ordinal = 0);

// Whether this binary has kernels for compute capability `cc` (80, 86, 89, 90, 100, 120, ...): the
// architectures listed in LX_ARCHS when it was built.
bool kernels_built_for(int cc);
std::vector<int> built_archs();

// The formats a checkpoint folder holds, read from its safetensors headers without loading it.
struct CheckpointInfo {
  lx_wtype stored = LX_W_BF16;  // the format of the large projections in the file
  size_t bytes = 0;             // the file's tensor bytes
  std::string description;      // plain language: "BF16 safetensors, 6.2 GB"
};
CheckpointInfo inspect_checkpoint(const std::string& model_dir);

// The user's wishes; anything left unset is decided from the hardware.
struct PlanRequest {
  int weights = -1;   // an lx_wtype, or -1 for automatic (the widest format that fits usefully)
  int context = 0;    // context each sequence is sized for, 0 for automatic
  int sequences = 0;  // concurrent sequences wanted, 0 for automatic
  double memory_fraction = 0.92;
};

// The format of each kind of tensor in a target format (the Formats recipe): dense projections, routed
// experts, and the two projections that stay wider by design.
struct WeightRecipe {
  lx_wtype dense = LX_W_BF16;   // attention, Mamba in_proj, depth memory, shared expert, latent_down, the head
  lx_wtype experts = LX_W_BF16; // routed expert stacks
  lx_wtype latent_up = LX_W_BF16;
  // Mamba-2's out_proj is BF16 in every format: it is the one projection int8 visibly hurts.
};
WeightRecipe recipe_for(lx_wtype target);

struct Plan {
  lx_wtype weights = LX_W_BF16;   // the target format (the headline: "BF16", "FP8", ...)
  WeightRecipe recipe;            // what each kind of tensor is held in
  bool convert_at_load = false;   // the file is in another format (round-to-nearest or exact dequant at load)
  lx_math prefill_math = LX_MATH_BF16;  // dense products, prefill and batched steps
  lx_math batch_math = LX_MATH_BF16;    // the same, for the rows of a batched decode step
  lx_math expert_math = LX_MATH_BF16;   // routed experts
  int max_slots = 1;          // concurrent sequences
  int kv_capacity = 0;        // cache positions per sequence
  int max_step_tokens = 0;    // tokens in one step (decode rows and prefill rows together)
  int prefill_chunk = 0;      // prompt tokens a sequence takes per step while it is being prefilled
  size_t weight_bytes = 0, state_bytes = 0, kv_bytes = 0, scratch_bytes = 0, budget_bytes = 0;
  double decode_ceiling_tps = 0;  // bandwidth / weight bytes: the most tokens per second one stream can make
  std::vector<std::string> notes;  // plain-language lines `lernex` prints (why this format, etc.)
};

// Throws std::runtime_error, saying what is missing, when no format fits the device for the model.
Plan make_plan(const DeviceInfo& device, const MetisConfig& config, const CheckpointInfo& checkpoint,
               const PlanRequest& request);

// Exact byte accounting, shared by the plan and its tests. Each weight is counted once, in the format
// the engine holds it, with the 256-byte alignment the upload uses.
struct WeightBytes {
  size_t plain = 0;     // bf16 and f32 tensors kept as they are (norms, biases, depth memory, routers)
  size_t dense = 0;     // dense projections, and the head's copy of the embedding table
  size_t latent_up = 0;
  size_t out_proj = 0;  // Mamba-2 out_proj, BF16 in every format
  size_t experts = 0;
  size_t total() const noexcept { return plain + dense + latent_up + out_proj + experts; }
};
WeightBytes weight_bytes(const MetisConfig& config, const WeightRecipe& recipe);
// Bytes of one R x K projection (or an expert stack of R = E * N rows) held in `format`.
size_t projection_bytes(lx_wtype format, int64_t rows, int64_t cols);
// Persistent state: the Mamba-2 conv and SSM state of one slot, and the attention KV cache of one slot
// at `capacity` positions. Both count every pass.
size_t state_bytes_per_slot(const MetisConfig& config);
size_t kv_bytes_per_position(const MetisConfig& config);
// The per-step scratch the model allocates for `tokens` tokens and `slots` slots (metis.cpp).
size_t scratch_bytes(const MetisConfig& config, int tokens, int slots);

}  // namespace lx::runtime
