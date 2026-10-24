// The plan for a model larger than the GPU, or than the VRAM budget the user set: make_plan's
// accounting for everything but the routed experts, and the tiers of runtime/offload.h for those.
//
// make_plan sizes a model that is resident; when it throws (nothing fits), or the user asks for a
// format that does not fit, or for less VRAM than the model needs, this plans the same model with
// the experts offloaded: one sequence, the dense weights, its state, KV cache and scratch in VRAM,
// and whatever is left as the expert cache. Without a requested format it takes the widest one
// whose cache still holds kMinResident of the experts (the cache simulation and the Logos runs put
// the knee there: below it most steps miss), else the one whose cache holds the most.
#pragma once

#include "runtime/offload.h"
#include "runtime/plan.h"

namespace lx::runtime {

constexpr double kMinResident = 0.5;
constexpr int kOffloadStepTokens = 512;  // tokens per prefill step (the staged path syncs per layer)
constexpr size_t kOffloadMarginBytes = 384ull << 20;  // VRAM left free for the CUDA context's growth

// Throws std::runtime_error, saying what is missing, when not even one step's experts fit.
// machine (may be null) sets the CPU lane's share from measured bandwidths.
Plan make_offload_plan(const DeviceInfo& device, const MetisConfig& config, const CheckpointInfo& checkpoint,
                       const PlanRequest& request, const offload::MachineProfile* machine = nullptr);

}  // namespace lx::runtime
