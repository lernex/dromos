#include "runtime/offload_plan.h"

#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace lx::runtime {
namespace {

std::string gb(size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f GB", static_cast<double>(bytes) / 1e9);
  return buf;
}

const char* name_of(lx_wtype t) {
  switch (t) {
    case LX_W_BF16: return "BF16";
    case LX_W_INT8: return "INT8";
    case LX_W_FP8: return "FP8";
    default: return "NVFP4";
  }
}

}  // namespace

Plan make_offload_plan(const DeviceInfo& d, const MetisConfig& c, const CheckpointInfo& ck, const PlanRequest& req,
                       const offload::MachineProfile* machine) {
  // The VRAM this process may take: what is free (times the request's fraction), and no more than the
  // budget leaves after what the card already holds (the CUDA context, other processes).
  size_t budget = static_cast<size_t>(static_cast<double>(d.free_bytes) * req.memory_fraction);
  if (req.vram_budget > 0) {
    const size_t used = d.total_bytes > d.free_bytes ? d.total_bytes - d.free_bytes : 0;
    budget = std::min(budget, req.vram_budget > used ? req.vram_budget - used : 0);
  }
  budget = budget > kOffloadMarginBytes ? budget - kOffloadMarginBytes : 0;

  const int context_cap = c.max_position_embeddings > 0 ? c.max_position_embeddings : 8192;
  const int context = std::clamp(req.context > 0 ? req.context : 8192, 1, context_cap);
  const size_t state = state_bytes_per_slot(c) + kv_bytes_per_position(c) * static_cast<size_t>(context);
  const size_t scratch = scratch_bytes(c, kOffloadStepTokens, 1);
  const int64_t n_experts = static_cast<int64_t>(c.num_hidden_layers) * c.n_routed_experts;

  // The offload kernels read BF16, INT8 and FP8 experts; of the two 8-bit formats the one the GPU
  // multiplies natively goes first, as in make_plan.
  std::vector<lx_wtype> order = d.fp8_mma ? std::vector<lx_wtype>{LX_W_BF16, LX_W_FP8, LX_W_INT8}
                                          : std::vector<lx_wtype>{LX_W_BF16, LX_W_INT8, LX_W_FP8};
  if (req.weights >= 0) {
    if (req.weights == LX_W_NVFP4) throw std::runtime_error("offloaded experts run in BF16, INT8 or FP8, not NVFP4");
    order = {static_cast<lx_wtype>(req.weights)};
  }

  struct Fit {
    lx_wtype format = LX_W_BF16;
    WeightBytes wb;
    size_t fixed = 0, cache = 0;
    double resident = 0;
  };
  std::vector<Fit> fits;
  std::string why;
  for (const lx_wtype format : order) {
    Fit f;
    f.format = format;
    f.wb = weight_bytes(c, recipe_for(format));
    f.fixed = f.wb.total() - f.wb.experts + state + scratch;
    const size_t expert = f.wb.experts / static_cast<size_t>(n_experts);
    // At the least, every layer holds one step's experts (top_k slots).
    const size_t least = expert * static_cast<size_t>(c.num_hidden_layers) * static_cast<size_t>(c.num_experts_per_tok);
    if (budget < f.fixed + least) {
      why += std::string("\n  ") + name_of(format) + " needs " + gb(f.fixed + least) + " (dense weights, one sequence of " +
             std::to_string(context) + " tokens and one step of experts); " + gb(budget) + " is available";
      continue;
    }
    f.cache = std::min(budget - f.fixed, f.wb.experts) / expert * expert;
    f.resident = static_cast<double>(f.cache) / static_cast<double>(f.wb.experts);
    fits.push_back(f);
  }
  if (fits.empty()) throw std::runtime_error("the model does not run on this GPU even with its experts offloaded:" + why);
  // The widest format that caches kMinResident of its experts; failing that, the one that caches the
  // most (ties to the earlier in the order).
  const Fit* chosen = nullptr;
  for (const Fit& f : fits)
    if (f.resident >= kMinResident) {
      chosen = &f;
      break;
    }
  if (!chosen) {
    chosen = &fits.front();
    for (const Fit& f : fits)
      if (f.resident > chosen->resident) chosen = &f;
  }

  Plan p;
  p.weights = chosen->format;
  p.recipe = recipe_for(chosen->format);
  p.convert_at_load = ck.stored != chosen->format;
  p.max_slots = 1;
  p.kv_capacity = context;
  p.max_step_tokens = kOffloadStepTokens;
  p.prefill_chunk = kOffloadStepTokens - 1;
  p.weight_bytes = chosen->fixed - state - scratch + chosen->cache;
  p.state_bytes = state_bytes_per_slot(c);
  p.kv_bytes = kv_bytes_per_position(c) * static_cast<size_t>(context);
  p.scratch_bytes = scratch;
  p.budget_bytes = budget;
  p.decode_ceiling_tps = 0;  // set by the misses, which only a run measures

  Plan::Offload& o = p.offload;
  o.enabled = true;
  o.expert_cache_bytes = chosen->cache;
  o.resident_fraction = chosen->resident;
  o.ram_expert_bytes = req.ram_budget > 0 ? std::min(req.ram_budget, chosen->wb.experts) : chosen->wb.experts;
  o.disk_expert_bytes = chosen->wb.experts - o.ram_expert_bytes;

  offload::ExpertShape shape;
  shape.n_layers = c.num_hidden_layers;
  shape.n_experts = c.n_routed_experts;
  shape.top_k = c.num_experts_per_tok;
  shape.L = c.latent_dim;
  shape.I = c.expert_intermediate_size;
  shape.type = p.recipe.experts;
  const offload::Placement placement = offload::plan_placement(
      shape, static_cast<int64_t>(o.expert_cache_bytes),
      req.ram_budget > 0 ? static_cast<int64_t>(req.ram_budget) : -1, machine);
  o.cpu_fraction = placement.cpu_fraction;
  o.max_cpu_pairs = placement.max_cpu_pairs;
  o.prefetch = 0;

  char buf[256];
  std::snprintf(buf, sizeof buf, "%s experts offloaded: %.0f%% of them (%s) cached in %s of VRAM, all of them in %s",
                name_of(chosen->format), 100.0 * o.resident_fraction, gb(o.expert_cache_bytes).c_str(), gb(budget).c_str(),
                o.disk_expert_bytes ? "pinned RAM and on NVMe" : "pinned RAM");
  p.notes.push_back(buf);
  if (o.disk_expert_bytes) p.notes.push_back(gb(o.disk_expert_bytes) + " of experts are read from the checkpoint on NVMe");
  for (const std::string& n : placement.notes) p.notes.push_back(n);
  return p;
}

}  // namespace lx::runtime
