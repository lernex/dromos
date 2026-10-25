// The plan for a model larger than the GPU, or than the VRAM budget the user set: make_plan's
// accounting for everything but the routed experts, and the tiers of runtime/offload.h for those.
//
// make_plan sizes a model that is resident; when its format does not fit the GPU usefully, it plans the same
// model, in the same format, with the experts offloaded (this), before anything else: one sequence, the dense
// weights, its state, KV cache and scratch in VRAM, and whatever is left as the expert cache. The format is the
// file's, or the one the request names; it is never narrowed to cache more (make_plan's notes name the narrower
// format that would fit instead). Measured on Logos FP8 (RTX 5070 Ti, PCIe 5), decode against fully resident:
// with 66%, 42% and 19% of the experts cached, 4%, 12% and 24% slower (int8 decode copies, bf16 dense on the
// host); with 57% and 33%, 5% and 13% slower (bf16 dense in VRAM). The cost grows smoothly; below kMinResident
// the notes say so.
#pragma once

#include "runtime/offload.h"
#include "runtime/plan.h"

namespace lx::runtime {

constexpr double kMinResident = 0.5;
constexpr int kOffloadStepTokens = 512;  // tokens per prefill step (the staged path syncs per layer)
constexpr size_t kOffloadMarginBytes = 384ull << 20;  // VRAM left free for the CUDA context's growth
// A CUDA context's own footprint (the plan's device numbers are taken with it already alive, so only a
// budget, which counts it, needs it): NVML charged 222 MiB to a lernex process on the RTX 5070 Ti.
constexpr size_t kCudaContextBytes = 256ull << 20;

// Throws std::runtime_error, saying what is missing, when not even one step's experts fit, or for a format the
// offload kernels do not read (they read BF16, INT8 and FP8 experts) or the file cannot load into.
// machine (may be null) sets the CPU lane's share from measured bandwidths.
Plan make_offload_plan(const DeviceInfo& device, const MetisConfig& config, const CheckpointInfo& checkpoint,
                       const PlanRequest& request, const offload::MachineProfile* machine = nullptr);

}  // namespace lx::runtime
