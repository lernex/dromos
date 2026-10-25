// Offload placement without a GPU: the plan for a model larger than the card (make_offload_plan), the
// VRAM slot layout (layout_slots) and the CPU lane's share (plan_placement). All three are pure
// functions of the device description, the config, the routing statistics and the measured machine.
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string>

#include "core/config.h"
#include "doctest.h"
#include "runtime/offload.h"
#include "runtime/offload_plan.h"
#include "runtime/plan.h"

using namespace lx;
using namespace lx::runtime;

namespace {

constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;

DeviceInfo card_16gb() {
  DeviceInfo d;
  d.name = "RTX 5070 Ti 16GB";
  d.cc_major = d.real_cc_major = 12;
  d.total_bytes = 16 * kGiB;
  d.free_bytes = d.total_bytes - 512ull * 1024 * 1024;
  d.bandwidth_gbs = 896;
  d.sms = 70;
  d.bf16_mma = d.int8_mma = d.fp8_mma = d.fp4_mma = true;
  d.built = true;
  return d;
}

MetisConfig logos() {
  MetisConfig c;
  c.vocab_size = 65536;
  c.hidden_size = 2560;
  c.num_hidden_layers = 20;
  c.attention_layer_indices = {4, 8, 12, 16};
  c.num_attention_heads = 40;
  c.num_key_value_heads = 10;
  c.head_dim = 64;
  c.attention_pass_lora_rank = 8;
  c.mamba_d_state = 128;
  c.mamba_d_conv = 4;
  c.mamba_expand = 2;
  c.mamba_head_dim = 64;
  c.mamba_n_groups = 10;
  c.n_streams = 4;
  c.mhc_pass_embedding_dim = 64;
  c.mhc_sinkhorn_iterations = 8;
  c.latent_dim = 1024;
  c.n_routed_experts = 192;
  c.num_experts_per_tok = 4;
  c.expert_intermediate_size = 768;
  c.shared_expert_intermediate_size = 768;
  c.route_feature_dim = 256;
  c.memory_dim = 256;
  c.num_passes = 2;
  c.max_passes = 5;
  c.max_position_embeddings = 131072;
  c.family = "logos";
  return c;
}

CheckpointInfo bf16_checkpoint() {
  CheckpointInfo ck;
  ck.stored = LX_W_BF16;
  return ck;
}

size_t used(const Plan& p) { return p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes; }

}  // namespace

TEST_CASE("Logos BF16 on a 16 GB card: the dense model resident, most of the experts cached") {
  PlanRequest req;
  req.weights = LX_W_BF16;
  const Plan p = make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req);
  CHECK(p.offload.enabled);
  CHECK(p.weights == LX_W_BF16);
  CHECK(p.max_slots == 1);
  CHECK(used(p) <= p.budget_bytes);
  // 18.1 GB of experts beside 2.4 GB of dense weights: the run on this card cached 72.7% of them
  CHECK(p.offload.resident_fraction > 0.6);
  CHECK(p.offload.resident_fraction < 0.8);
  CHECK(p.offload.ram_expert_bytes == weight_bytes(logos(), LX_W_BF16).experts);
  CHECK(p.offload.disk_expert_bytes == 0);
  CHECK_FALSE(p.convert_at_load);
}

TEST_CASE("the VRAM budget bounds this process; other processes only through the memory they leave") {
  DeviceInfo d = card_16gb();
  PlanRequest req;
  req.weights = LX_W_FP8;
  req.vram_budget = 8 * kGiB;
  const Plan p8 = make_offload_plan(d, logos(), bf16_checkpoint(), req);
  CHECK(used(p8) + kOffloadMarginBytes + kCudaContextBytes <= req.vram_budget);
  CHECK(p8.offload.resident_fraction > 0.4);
  CHECK(p8.offload.resident_fraction < 0.7);
  CHECK(p8.convert_at_load);
  // another process holding 3 GiB leaves more than the budget free: nothing changes
  d.free_bytes -= 3 * kGiB;
  CHECK(make_offload_plan(d, logos(), bf16_checkpoint(), req).offload.expert_cache_bytes ==
        p8.offload.expert_cache_bytes);
  // holding 10 GiB, it leaves less than the budget: the free memory bounds the cache
  d.free_bytes = card_16gb().free_bytes - 10 * kGiB;
  const Plan tight = make_offload_plan(d, logos(), bf16_checkpoint(), req);
  CHECK(tight.offload.expert_cache_bytes < p8.offload.expert_cache_bytes);
  CHECK(used(tight) + kOffloadMarginBytes <= static_cast<size_t>(d.free_bytes * req.memory_fraction));
  // a budget the whole FP8 model fits in caches every expert
  req.vram_budget = 14 * kGiB;
  const Plan p14 = make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req);
  CHECK(p14.offload.resident_fraction == doctest::Approx(1.0));
}

TEST_CASE("without a format the file's own is taken, however few experts it caches") {
  PlanRequest req;
  req.vram_budget = 16 * kGiB;
  CHECK(make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req).weights == LX_W_BF16);
  req.vram_budget = 8 * kGiB;  // BF16 caches about a quarter of its experts: it stays BF16, and says so
  const Plan p = make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req);
  CHECK(p.weights == LX_W_BF16);
  CHECK_FALSE(p.convert_at_load);
  CHECK(p.offload.resident_fraction < kMinResident);
  bool said = false;
  for (const std::string& n : p.notes) said |= n.find("less than half of the experts") != std::string::npos;
  CHECK(said);
  // an FP8 file offloads as FP8; NVFP4 does not offload, and a format wider than the file is refused
  CheckpointInfo fp8;
  fp8.stored = LX_W_FP8;
  const Plan q = make_offload_plan(card_16gb(), logos(), fp8, req);
  CHECK(q.weights == LX_W_FP8);
  CHECK_FALSE(q.convert_at_load);
  CHECK(q.offload.resident_fraction > p.offload.resident_fraction);
  req.weights = LX_W_BF16;
  CHECK_THROWS_AS(make_offload_plan(card_16gb(), logos(), fp8, req), std::runtime_error);
}

TEST_CASE("experts beyond the RAM budget are read from NVMe") {
  PlanRequest req;
  req.weights = LX_W_BF16;
  req.ram_budget = 10 * kGiB;
  const Plan p = make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req);
  CHECK(p.offload.ram_expert_bytes == 10 * kGiB);
  CHECK(p.offload.disk_expert_bytes == weight_bytes(logos(), LX_W_BF16).experts - 10 * kGiB);
}

TEST_CASE("a budget below one step of experts says what is missing") {
  PlanRequest req;
  req.vram_budget = 2 * kGiB;
  CHECK_THROWS_AS(make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req), std::runtime_error);
  req.weights = LX_W_NVFP4;
  req.vram_budget = 0;
  CHECK_THROWS_AS(make_offload_plan(card_16gb(), logos(), bf16_checkpoint(), req), std::runtime_error);
}

TEST_CASE("slot layout: everything resident, or an equal share without statistics") {
  offload::RoutingStats none;
  const offload::SlotLayout all = offload::layout_slots(3, 8, 24, none, 1.0);
  for (int l = 0; l < 3; ++l) {
    CHECK(all.slots[l] == 8);
    CHECK(all.pinned[l] == 8);
  }
  const offload::SlotLayout part = offload::layout_slots(3, 8, 10, none, 1.0);
  CHECK(part.slots == std::vector<int>{4, 3, 3});
  CHECK(part.pinned == std::vector<int>{0, 0, 0});
  CHECK(part.contents[1] == std::vector<int>{0, 1, 2});
  CHECK(std::accumulate(part.cpu_first.begin(), part.cpu_first.end(), 0) == 0);
}

TEST_CASE("slot layout with statistics: LRU slots start with each layer's hottest experts") {
  // two layers of 6 experts carrying the same routing mass; layer 0's is skewed, layer 1's flat
  offload::RoutingStats st;
  st.n_layers = 2;
  st.n_experts = 6;
  st.frequency = {50, 1, 30, 1, 10, 8, 17, 17, 17, 17, 16, 16};
  const offload::SlotLayout lru = offload::layout_slots(2, 6, 6, st, 1.0);
  CHECK(lru.pinned == std::vector<int>{0, 0});
  CHECK(lru.slots[0] + lru.slots[1] == 6);
  CHECK(lru.slots[0] == 3);
  CHECK(lru.contents[0] == std::vector<int>{0, 2, 4});
  // the experts left out start cold: their misses go to the CPU lane first
  CHECK(lru.cpu_first[0 * 6 + 1] == 1);
  CHECK(lru.cpu_first[0 * 6 + 0] == 0);
  CHECK(lru.cpu_first[1 * 6 + 5] == 1);

  // all slots pinned by frequency over both layers: the four hottest anywhere
  const offload::SlotLayout fixed = offload::layout_slots(2, 6, 4, st, 0.0);
  CHECK(fixed.pinned == std::vector<int>{2, 2});
  CHECK(fixed.contents[0] == std::vector<int>{0, 2});
  CHECK(fixed.contents[1] == std::vector<int>{0, 1});

  // half pinned, half LRU: dynamic slots follow the routing mass the pinned set leaves uncovered
  const offload::SlotLayout mixed = offload::layout_slots(2, 6, 6, st, 0.5);
  CHECK(mixed.pinned[0] + mixed.pinned[1] == 3);
  CHECK(mixed.slots[0] + mixed.slots[1] == 6);
  CHECK(mixed.slots[1] > mixed.pinned[1]);
  std::set<int> seen(mixed.contents[0].begin(), mixed.contents[0].end());
  CHECK(seen.size() == mixed.contents[0].size());  // no expert twice in a layer
}

TEST_CASE("the CPU lane's share follows the measured bandwidths") {
  offload::ExpertShape shape;
  shape.n_layers = 20;
  shape.n_experts = 192;
  shape.top_k = 4;
  shape.L = 1024;
  shape.I = 768;
  const int64_t all = 20ll * 192 * shape.expert_bytes();
  offload::MachineProfile m;
  m.cpu_expert_gbs = 79;
  m.zero_copy_gbs = 45;
  const offload::Placement resident = offload::plan_placement(shape, all, -1, &m);
  CHECK(resident.max_cpu_pairs == 0);
  // most experts cached: read-through fills beat the lane
  const offload::Placement half = offload::plan_placement(shape, all / 2, -1, &m);
  CHECK(half.vram_cache_bytes == all / 2);
  CHECK(half.max_cpu_pairs == 0);
  // few cached: the CPU's RAM bandwidth adds to the bus's
  const offload::Placement part = offload::plan_placement(shape, all / 5, -1, &m);
  CHECK(part.cpu_fraction > 0);
  CHECK(part.cpu_fraction <= 0.5);
  CHECK(part.max_cpu_pairs > 0);
  const offload::Placement unmeasured = offload::plan_placement(shape, all / 5, -1, nullptr);
  CHECK(unmeasured.max_cpu_pairs == 0);
  CHECK(offload::plan_placement(shape, all / 2, all / 3, &m).ram_bytes == all / 3);
}
