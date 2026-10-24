// What this GPU can do, and how to run a model on it. The user picks a model and, if they want,
// a weight format; everything else (which instructions multiply, how many sequences fit, how long
// they can be, which kernel variant runs each product) is decided here, from the hardware.
#pragma once

#include <stddef.h>

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
  std::string driver, runtime;
};

// Queries a device. LX_ARCH_OVERRIDE=80|86|89|90|100|120 makes the engine behave as that
// architecture (only lowering is allowed), so every fallback path runs and is tested on one card.
DeviceInfo query_device(int ordinal = 0);

// What the checkpoint on disk holds.
struct CheckpointInfo {
  lx_wtype stored = LX_W_BF16;  // the format of the large projections in the file
  size_t bytes = 0;
};

// The user's wishes; anything left unset is decided from the hardware.
struct PlanRequest {
  int weights = -1;        // an lx_wtype, or -1 for automatic
  int context = 0;         // longest sequence wanted, 0 for automatic
  int sequences = 0;       // concurrent sequences wanted, 0 for automatic
  double memory_fraction = 0.92;
};

struct Plan {
  lx_wtype weights = LX_W_BF16;   // the format the projections are held in on the GPU
  bool convert_at_load = false;   // the file is in another format (round-to-nearest at load)
  lx_math prefill_math = LX_MATH_BF16, batch_math = LX_MATH_BF16, expert_math = LX_MATH_BF16;
  int max_slots = 1, kv_capacity = 0, max_step_tokens = 0, prefill_chunk = 0;
  size_t weight_bytes = 0, state_bytes = 0, kv_bytes = 0, scratch_bytes = 0;
  std::vector<std::string> notes;  // plain-language lines `lernex` prints (why this format, etc.)
};

Plan make_plan(const DeviceInfo& device, const MetisConfig& config, const CheckpointInfo& checkpoint,
               const PlanRequest& request);

}  // namespace lx::runtime
