// The plan on simulated devices, and the exact byte accounting against the checkpoint on disk.
// No GPU is needed: make_plan is a pure function of the device description, the config and the
// checkpoint. The BF16 check reads LX_MODEL_DIR (a Praxis or Logos folder) and is skipped without it.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "core/config.h"
#include "core/safetensors.h"
#include "doctest.h"
#include "runtime/plan.h"

using namespace lx;
using namespace lx::runtime;

namespace {

constexpr size_t kGiB = 1024ull * 1024ull * 1024ull;

// A device described by its numbers, with the same capability flags query_device would set.
DeviceInfo simulated(const std::string& name, int cc, double gib, double bandwidth_gbs, int sms) {
  DeviceInfo d;
  d.name = name;
  d.cc_major = cc / 10;
  d.cc_minor = cc % 10;
  d.real_cc_major = d.cc_major;
  d.real_cc_minor = d.cc_minor;
  d.sms = sms;
  d.total_bytes = static_cast<size_t>(gib * kGiB);
  d.free_bytes = d.total_bytes - 512ull * 1024 * 1024;  // what a context leaves
  d.bandwidth_gbs = bandwidth_gbs;
  d.bf16_mma = cc >= 80;
  d.int8_mma = cc >= 80;
  d.fp8_mma = cc >= 89;
  d.fp4_mma = cc == 100 || cc == 120;
  d.clusters = cc >= 90;
  d.built = true;
  return d;
}

MetisConfig praxis() {
  MetisConfig c;
  c.vocab_size = 65536;
  c.hidden_size = 2048;
  c.num_hidden_layers = 12;
  c.attention_layer_indices = {4, 8};
  c.num_attention_heads = 32;
  c.num_key_value_heads = 8;
  c.head_dim = 64;
  c.attention_pass_lora_rank = 8;
  c.mamba_d_state = 128;
  c.mamba_d_conv = 4;
  c.mamba_expand = 2;
  c.mamba_head_dim = 64;
  c.mamba_n_groups = 8;
  c.n_streams = 4;
  c.mhc_pass_embedding_dim = 64;
  c.mhc_sinkhorn_iterations = 8;
  c.latent_dim = 1024;
  c.n_routed_experts = 128;
  c.num_experts_per_tok = 4;
  c.expert_intermediate_size = 512;
  c.shared_expert_intermediate_size = 512;
  c.route_feature_dim = 256;
  c.memory_dim = 256;
  c.num_passes = 2;
  c.max_passes = 5;
  c.max_position_embeddings = 131072;
  c.family = "praxis";
  return c;
}

MetisConfig logos() {
  MetisConfig c = praxis();
  c.hidden_size = 2560;
  c.num_hidden_layers = 20;
  c.attention_layer_indices = {4, 8, 12, 16};
  c.num_attention_heads = 40;
  c.num_key_value_heads = 10;
  c.mamba_n_groups = 10;
  c.n_routed_experts = 192;
  c.expert_intermediate_size = 768;
  c.shared_expert_intermediate_size = 768;
  c.family = "logos";
  return c;
}

CheckpointInfo bf16_checkpoint() {
  CheckpointInfo ck;
  ck.stored = LX_W_BF16;
  return ck;
}

// The GPUs the plan is judged on: the brief's list, with the numbers the vendors publish.
struct Gpu {
  const char* name;
  int cc;
  double gib;
  double bandwidth;
  int sms;
};
const Gpu kGpus[] = {
    {"RTX 3060 12GB", 86, 12, 360, 28},  {"RTX 3090 24GB", 86, 24, 936, 82},  {"RTX 4060 8GB", 89, 8, 272, 24},
    {"RTX 4090 24GB", 89, 24, 1008, 128}, {"RTX 5070 Ti 16GB", 120, 16, 896, 70}, {"RTX 5090 32GB", 120, 32, 1792, 170},
    {"A100 80GB", 80, 80, 2039, 108},     {"H100 80GB", 90, 80, 3350, 132},    {"B200", 100, 180, 8000, 148},
};

const Gpu& gpu(const char* name) {
  for (const Gpu& g : kGpus)
    if (std::string(g.name) == name) return g;
  throw std::runtime_error(std::string("no simulated GPU named ") + name);
}

}  // namespace

TEST_CASE("the recipe keeps Mamba out_proj wide, and NVFP4 keeps latent_up and dense FP8") {
  const WeightRecipe bf = recipe_for(LX_W_BF16);
  CHECK(bf.dense == LX_W_BF16);
  CHECK(bf.experts == LX_W_BF16);
  const WeightRecipe fp8 = recipe_for(LX_W_FP8);
  CHECK(fp8.dense == LX_W_FP8);
  CHECK(fp8.experts == LX_W_FP8);
  CHECK(fp8.latent_up == LX_W_FP8);
  const WeightRecipe fp4 = recipe_for(LX_W_NVFP4);
  CHECK(fp4.dense == LX_W_FP8);
  CHECK(fp4.experts == LX_W_NVFP4);
  CHECK(fp4.latent_up == LX_W_BF16);
}

TEST_CASE("projection bytes: data, scales and alignment") {
  CHECK(projection_bytes(LX_W_BF16, 2048, 2048) == 2048ull * 2048 * 2);
  // 8-bit: one byte per weight and one f32 scale per row
  CHECK(projection_bytes(LX_W_INT8, 2048, 2048) == 2048ull * 2048 + 2048ull * 4);
  CHECK(projection_bytes(LX_W_FP8, 2048, 2048) == projection_bytes(LX_W_INT8, 2048, 2048));
  // 4-bit: half a byte per weight, one e4m3 scale per 16, a global
  CHECK(projection_bytes(LX_W_NVFP4, 2048, 2048) == 2048ull * 2048 / 2 + 2048ull * 2048 / 16 + 256);
}

TEST_CASE("state: Mamba conv and SSM per slot, KV per position, both over every pass") {
  const MetisConfig c = praxis();
  // 10 Mamba layers, 2 passes: each with conv (3 x 6144 f32) and SSM (64 x 64 x 128 f32)
  const size_t conv = 3 * 6144 * 4, ssm = 64ull * 64 * 128 * 4;
  CHECK(state_bytes_per_slot(c) == 2 * 10 * (conv + ssm));
  // 2 attention layers, K and V of 8 heads x 64 dims in bf16, 2 passes
  CHECK(kv_bytes_per_position(c) == 2ull * 2 * (2 * 8 * 64 * 2));
}

TEST_CASE("the BF16 accounting matches the Praxis checkpoint on disk") {
  const char* dir = std::getenv("LX_MODEL_DIR");
  if (!dir) return;  // the server has the checkpoints; a laptop does not
  const MetisConfig c = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  // The engine reads every tensor the model uses, each once, in BF16. The file also holds what the
  // engine does not read: lm_head (the config ties it to the embedding), the k_router and k_budget
  // tensors, and depth_budget. Those are left out, and what is left must match the file within the
  // 256-byte alignment of each tensor.
  size_t used = 0;
  for (const auto& [name, t] : set.tensors()) {
    const bool unused = name.find("lm_head") != std::string::npos || name.find("k_router") != std::string::npos ||
                        name.find("k_budget") != std::string::npos || name.find("depth_budget") != std::string::npos;
    if (!unused) used += t.bytes;
  }
  const size_t accounted = weight_bytes(c, recipe_for(LX_W_BF16)).total();
  INFO("used " << used << " accounted " << accounted);
  CHECK(accounted >= used);
  CHECK(static_cast<double>(accounted - used) / static_cast<double>(used) < 0.002);
}

TEST_CASE("the Praxis plan on the development card runs BF16 with room to spare") {
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(p.weights == LX_W_BF16);
  CHECK_FALSE(p.convert_at_load);
  CHECK(p.prefill_math == LX_MATH_BF16);
  CHECK(p.expert_math == LX_MATH_BF16);
  CHECK(p.max_slots == 64);
  CHECK(p.kv_capacity == 8192);
  CHECK(p.max_step_tokens == 2048);
  CHECK(p.prefill_chunk == 2048 - 64);
  CHECK(p.decode_ceiling_tps > 100);
  CHECK(p.notes.size() >= 5);
}

TEST_CASE("the 8-bit format is FP8 on sm_89 and up, INT8 below, and quantized files compute with the math the GPU has") {
  // Praxis on a 6 GB card does not fit in BF16 (6.2 GB of weights), but the 8-bit formats do
  const Plan sm86 = make_plan(simulated("RTX 3060 6GB", 86, 6, 360, 28), praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(sm86.weights == LX_W_INT8);  // no FP8 tensor cores on sm_86: the IMMA format comes first
  CHECK(sm86.prefill_math == LX_MATH_INT8);
  CHECK(sm86.convert_at_load);
  const Plan sm89 = make_plan(simulated("RTX 4050 6GB", 89, 6, 272, 24), praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(sm89.weights == LX_W_FP8);
  CHECK(sm89.prefill_math == LX_MATH_FP8);
  CHECK(sm89.expert_math == LX_MATH_FP8);
  // Logos on a 16 GB card does not fit in BF16 (about 21 GB); its FP8 format does, with room
  const Plan logos16 = make_plan(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70), logos(), bf16_checkpoint(), PlanRequest{});
  CHECK(logos16.weights == LX_W_FP8);
  CHECK(logos16.expert_math == LX_MATH_FP8);
}

TEST_CASE("the 4-bit format: NVFP4 experts on sm_120 take FP4 math; below that they expand to BF16") {
  // Logos on 12 GB: BF16 and the 8-bit formats are too big for four sequences, NVFP4 is not
  const Plan twelve = make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), logos(), bf16_checkpoint(), PlanRequest{});
  CHECK(twelve.weights == LX_W_NVFP4);
  CHECK(twelve.recipe.dense == LX_W_FP8);
  CHECK(twelve.recipe.latent_up == LX_W_BF16);
  CHECK(twelve.prefill_math == LX_MATH_BF16);  // sm_86 has no FP8 tensor cores for the dense FP8 products
  CHECK(twelve.expert_math == LX_MATH_BF16);
  // on sm_120 the same format's experts multiply in FP4, the dense FP8 products in FP8
  const Plan blackwell = make_plan(simulated("RTX 5070 Ti 16GB", 120, 12, 896, 70), logos(), bf16_checkpoint(), PlanRequest{});
  CHECK(blackwell.weights == LX_W_NVFP4);
  CHECK(blackwell.expert_math == LX_MATH_FP4);
  CHECK(blackwell.prefill_math == LX_MATH_FP8);
  // a card too small for four sequences of Logos runs fewer, and the notes say the card is small
  for (double gib = 8.5; gib <= 10.0; gib += 0.25) {
    const Plan p = make_plan(simulated("Sized card", 89, gib, 272, 24), logos(), bf16_checkpoint(), PlanRequest{});
    bool said = false;
    for (const std::string& line : p.notes)
      if (line.find("small for the model") != std::string::npos) said = true;
    CHECK(p.max_slots >= 1);
    CHECK(said == (p.max_slots < 4));
    INFO("card " << gib << " GB: " << p.weights << " with " << p.max_slots << " sequences");
  }
  // 8 GB holds no Logos at all: the plan says what each format needs
  CHECK_THROWS_AS(make_plan(simulated("RTX 4060 8GB", 89, 8, 272, 24), logos(), bf16_checkpoint(), PlanRequest{}),
                  std::runtime_error);
}

TEST_CASE("the plan's choices across the brief's GPUs") {
  // Praxis: BF16 wherever it fits usefully (6.2 GB of weights), from 12 GB up
  const PlanRequest none{};
  CHECK(make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), praxis(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("RTX 4090 24GB", 89, 24, 1008, 128), praxis(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("H100 80GB", 90, 80, 3350, 132), praxis(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("B200", 100, 180, 8000, 148), praxis(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("RTX 5090 32GB", 120, 32, 1792, 170), praxis(), bf16_checkpoint(), none).max_slots == 64);
  CHECK(make_plan(simulated("A100 80GB", 80, 80, 2039, 108), praxis(), bf16_checkpoint(), none).expert_math == LX_MATH_BF16);
  // Logos: BF16 on the 24 GB and bigger cards, 4-bit on 12 GB
  CHECK(make_plan(simulated("RTX 4090 24GB", 89, 24, 1008, 128), logos(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("A100 80GB", 80, 80, 2039, 108), logos(), bf16_checkpoint(), none).weights == LX_W_BF16);
  CHECK(make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), logos(), bf16_checkpoint(), none).weights == LX_W_NVFP4);
  CHECK(make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), praxis(), bf16_checkpoint(), none).weights == LX_W_BF16);
}

TEST_CASE("a request for a format that cannot fit says why, and a plan never overflows the budget") {
  const DeviceInfo small = simulated("RTX 4060 8GB", 89, 8, 272, 24);
  CHECK_THROWS_AS(make_plan(small, logos(), bf16_checkpoint(), [] {
                    PlanRequest r;
                    r.weights = LX_W_BF16;
                    return r;
                  }()),
                  std::runtime_error);
  for (const Gpu& g : kGpus) {
    for (const MetisConfig& c : {praxis(), logos()}) {
      try {
        const Plan p = make_plan(simulated(g.name, g.cc, g.gib, g.bandwidth, g.sms), c, bf16_checkpoint(), PlanRequest{});
        const size_t used = p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes;
        INFO(g.name << " " << c.family);
        CHECK(used <= p.budget_bytes);
        CHECK(p.max_slots >= 1);
      } catch (const std::runtime_error&) {
        // a card that cannot hold the model at all: the message is the test
      }
    }
  }
}

TEST_CASE("a build without the card's kernels refuses the card") {
  DeviceInfo d = simulated("Hypothetical sm_121", 121, 16, 896, 70);
  d.built = false;
  CHECK_THROWS_AS(make_plan(d, praxis(), bf16_checkpoint(), PlanRequest{}), std::runtime_error);
}

TEST_CASE("the plan for each simulated GPU and model, written to LX_PLAN_REPORT when it is set") {
  const char* path = std::getenv("LX_PLAN_REPORT");
  if (!path) return;
  std::FILE* out = std::fopen(path, "w");
  REQUIRE(out != nullptr);
  for (const Gpu& g : kGpus) {
    for (const MetisConfig& c : {praxis(), logos()}) {
      std::fprintf(out, "== %s (sm_%d, %.0f GB, %.0f GB/s) - %s\n", g.name, g.cc, g.gib, g.bandwidth, c.family.c_str());
      try {
        const Plan p = make_plan(simulated(g.name, g.cc, g.gib, g.bandwidth, g.sms), c, bf16_checkpoint(), PlanRequest{});
        for (const std::string& note : p.notes) std::fprintf(out, "  %s\n", note.c_str());
      } catch (const std::runtime_error& e) {
        std::fprintf(out, "  does not run: %s\n", e.what());
      }
    }
  }
  std::fclose(out);
}
