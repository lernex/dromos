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

std::string name_of(lx_wtype t) { return format_name(t); }

}  // namespace

Plan make_offload_plan(const DeviceInfo& d, const MetisConfig& c, const CheckpointInfo& ck, const PlanRequest& req,
                       const offload::MachineProfile* machine) {
  // The format: the file's own, or the one the request names, as make_plan takes it.
  const lx_wtype format = req.weights >= 0 ? static_cast<lx_wtype>(req.weights) : ck.stored;
  if (format != LX_W_BF16 && format != LX_W_INT8 && format != LX_W_FP8)
    throw std::runtime_error("offloaded experts run in BF16, INT8 or FP8, not " + format_name(format));
  std::string how;
  if (!can_load(format, ck.stored, c, &how)) throw std::runtime_error(format_name(format) + " cannot be planned: " + how);

  // The VRAM this process may take: what is free (times the request's fraction), and no more than the
  // budget, which counts this process's CUDA context too. Other processes only limit it through the
  // memory they leave free.
  size_t budget = static_cast<size_t>(static_cast<double>(d.free_bytes) * req.memory_fraction);
  if (req.vram_budget > 0)
    budget = std::min(budget, req.vram_budget > kCudaContextBytes ? req.vram_budget - kCudaContextBytes : 0);
  budget = budget > kOffloadMarginBytes ? budget - kOffloadMarginBytes : 0;

  const int context_cap = c.max_position_embeddings > 0 ? c.max_position_embeddings : 8192;
  const int context = std::clamp(req.context > 0 ? req.context : 8192, 1, context_cap);
  // The offloaded model's formats: the dense kind's native math, no BF16X decode copies (the VRAM goes to the
  // expert cache), and the experts multiplied weight-only by the offload kernels.
  lx::ModelFormats formats = model_formats(format, d, false);
  formats.expert_math = LX_MATH_BF16;
  const size_t state = state_bytes_per_slot(c) + kv_bytes_per_position(c) * static_cast<size_t>(context);
  const size_t scratch = scratch_bytes(c, kOffloadStepTokens, 1, formats);
  const int64_t n_experts = static_cast<int64_t>(c.num_hidden_layers) * c.n_routed_experts;

  const WeightBytes wb = weight_bytes(c, format);
  const size_t fixed = wb.total() - wb.experts + state + scratch;
  const size_t expert = wb.experts / static_cast<size_t>(n_experts);
  // At the least, every layer holds one step's experts (top_k slots).
  const size_t least = expert * static_cast<size_t>(c.num_hidden_layers) * static_cast<size_t>(c.num_experts_per_tok);
  if (budget < fixed + least)
    throw std::runtime_error("the model does not run on this GPU even with its experts offloaded: " + name_of(format) +
                             " needs " + gb(fixed + least) + " (dense weights, one sequence of " + std::to_string(context) +
                             " tokens and one step of experts); " + gb(budget) + " is available");
  const size_t cache = std::min(budget - fixed, wb.experts) / expert * expert;
  const double resident = static_cast<double>(cache) / static_cast<double>(wb.experts);

  Plan p;
  p.stored = ck.stored;
  p.weights = format;
  p.recipe = recipe_for(format);
  p.convert_at_load = ck.stored != format;
  p.prefill_math = p.batch_math = formats.prefill_math;
  p.expert_math = formats.expert_math;
  p.max_slots = 1;
  p.kv_capacity = context;
  p.max_step_tokens = kOffloadStepTokens;
  p.prefill_chunk = kOffloadStepTokens - 1;
  p.weight_bytes = fixed - state - scratch + cache;
  p.state_bytes = state_bytes_per_slot(c);
  p.kv_bytes = kv_bytes_per_position(c) * static_cast<size_t>(context);
  p.scratch_bytes = scratch;
  p.budget_bytes = budget;
  p.decode_ceiling_tps = 0;  // set by the misses, which only a run measures

  Plan::Offload& o = p.offload;
  o.enabled = true;
  o.expert_cache_bytes = cache;
  o.resident_fraction = resident;
  o.ram_expert_bytes = req.ram_budget > 0 ? std::min(req.ram_budget, wb.experts) : wb.experts;
  o.disk_expert_bytes = wb.experts - o.ram_expert_bytes;
  o.vram_budget = req.vram_budget;
  o.ram_budget = req.ram_budget;

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
                name_of(format).c_str(), 100.0 * o.resident_fraction, gb(o.expert_cache_bytes).c_str(),
                gb(budget).c_str(), o.disk_expert_bytes ? "pinned RAM and on NVMe" : "pinned RAM");
  p.notes.push_back(buf);
  if (o.resident_fraction < kMinResident)
    p.notes.push_back("less than half of the experts fit in VRAM, so most decode steps read some across PCIe");
  if (o.disk_expert_bytes) p.notes.push_back(gb(o.disk_expert_bytes) + " of experts are read from the checkpoint on NVMe");
  for (const std::string& n : placement.notes) p.notes.push_back(n);
  return p;
}

}  // namespace lx::runtime
