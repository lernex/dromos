// The plan on simulated devices, and the exact byte accounting against the checkpoints on disk and the loader.
// make_plan is a pure function of the device description, the config and the checkpoint (the backend's
// workspace and BF16X row sizes aside, read through the dispatcher). The accounting checks read the folders:
//   LX_MODEL_DIR        a BF16 folder (Praxis or Logos): the BF16 total against the file
//   LX_CKPT_BF16_DIR    the Praxis BF16 folder   } the plan's bytes against the files' tensors, and (with a GPU)
//   LX_CKPT_FP8_DIR     the published FP8 folder } against what ModelWeights::load holds on the device, each
//   LX_CKPT_NVFP4_DIR   the GPTQ NVFP4 folder    } within 1%
// and each is skipped without its folder.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "core/safetensors.h"
#include "model/traffic.h"
#include "doctest.h"
#include "runtime/offload_plan.h"
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

// A CPU backend's device: host memory is the device's, so the free figure is the RAM the plan may fill, with no VRAM.
DeviceInfo simulated_cpu(const std::string& name, double free_gib) {
  DeviceInfo d;
  d.name = name;
  d.cpu = true;
  d.isa = "AVX2";
  d.total_bytes = static_cast<size_t>(free_gib * kGiB);
  d.free_bytes = d.total_bytes;
  d.bandwidth_gbs = 40;
  d.bandwidth_measured = true;
  d.built = true;
  return d;
}

// Every line a CPU plan says, its reason included, is free of the GPU's words: no VRAM, card, CUDA, graph or offload.
bool names_no_gpu(const Plan& p) {
  std::vector<std::string> lines = p.notes;
  lines.push_back(p.reason);
  lines.push_back(p.bf16x_reason);
  for (const std::string& line : lines)
    for (const char* word : {"VRAM", "GPU", "CUDA", "card", "offload", "Graph"})
      if (line.find(word) != std::string::npos) return false;
  return true;
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

CheckpointInfo checkpoint(lx_wtype stored) {
  CheckpointInfo ck;
  ck.stored = stored;
  return ck;
}
CheckpointInfo bf16_checkpoint() { return checkpoint(LX_W_BF16); }
CheckpointInfo fp8_checkpoint() { return checkpoint(LX_W_FP8); }
CheckpointInfo nvfp4_checkpoint() { return checkpoint(LX_W_NVFP4); }

PlanRequest weights(lx_wtype w) {
  PlanRequest r;
  r.weights = w;
  return r;
}

bool has_note(const Plan& p, const std::string& text) {
  for (const std::string& line : p.notes)
    if (line.find(text) != std::string::npos) return true;
  return false;
}

const char* folder(const char* variable) {
  const char* dir = std::getenv(variable);
  return dir && *dir ? dir : nullptr;
}

// What the loader holds for a folder, from its tensors alone: every tensor but the scales a projection does not
// keep (input_scale), the tied lm_head and n-gram tables the engine skips, and the plain tensors it never reads;
// plus the bf16 gather table a quantized embedding is dequantized into.
size_t folder_device_bytes(const std::string& dir) {
  const SafetensorsSet set = SafetensorsSet::open(dir);
  size_t bytes = 0;
  for (const auto& [name, t] : set.tensors()) {
    if (name.find("lm_head") != std::string::npos || name.find("ngram") != std::string::npos ||
        name.find("input_scale") != std::string::npos || name.find("k_router") != std::string::npos ||
        name.find("k_budget") != std::string::npos || name.find("depth_budget") != std::string::npos)
      continue;
    bytes += t.bytes;
    if (name == "embedding.weight" && t.dtype != DType::BF16)
      bytes += static_cast<size_t>(t.shape[0]) * static_cast<size_t>(t.shape[1]) * 2;
  }
  return bytes;
}

double relative(size_t a, size_t b) {
  return std::fabs(static_cast<double>(a) - static_cast<double>(b)) / static_cast<double>(b);
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

// The 8 GiB card the brief names, with 7.6 GiB free and nothing else on it: the free figure an idle card reports once
// this process's CUDA context is in (7.6 GiB is 0.4 GiB in use, of which the context is 0.25 GiB).
DeviceInfo idle_8gib() {
  DeviceInfo d = simulated("RTX 4060 8GB", 89, 8, 272, 24);
  d.free_bytes = static_cast<size_t>(7.6 * kGiB);
  return d;
}

}  // namespace

TEST_CASE("the recipe: kinds per format, the published FP8 and GPTQ NVFP4 choices") {
  const WeightRecipe bf = recipe_for(LX_W_BF16);
  CHECK(bf.dense == LX_W_BF16);
  CHECK(bf.experts == LX_W_BF16);
  const WeightRecipe fp8 = recipe_for(LX_W_FP8);
  CHECK(fp8.dense == LX_W_FP8);
  CHECK(fp8.experts == LX_W_FP8);
  CHECK(fp8.latent_up == LX_W_BF16);
  const WeightRecipe fp4 = recipe_for(LX_W_NVFP4);
  CHECK(fp4.dense == LX_W_FP8);
  CHECK(fp4.experts == LX_W_NVFP4);
  CHECK(fp4.latent_up == LX_W_BF16);
  // the loader's rule, which the accounting follows projection by projection
  CHECK(projection_format(LX_W_FP8, "layers.0.moe.latent_down") == LX_W_BF16);
  CHECK(projection_format(LX_W_FP8, "layers.0.moe.shared_expert.gate_up") == LX_W_BF16);
  CHECK(projection_format(LX_W_FP8, "depth_memory.query") == LX_W_BF16);
  CHECK(projection_format(LX_W_FP8, "embedding") == LX_W_BF16);
  CHECK(projection_format(LX_W_FP8, "layers.4.mixer.qkv") == LX_W_FP8);
  CHECK(projection_format(LX_W_FP8, "layers.0.mixer.impl.mixer.in_proj") == LX_W_FP8);
  CHECK(projection_format(LX_W_FP8, "layers.0.moe.local_experts.down") == LX_W_FP8);
  CHECK(projection_format(LX_W_NVFP4, "embedding") == LX_W_FP8);
  CHECK(projection_format(LX_W_NVFP4, "layers.0.moe.latent_up") == LX_W_BF16);
}

TEST_CASE("projection bytes: data, scales, an NVFP4 stack's globals, alignment") {
  CHECK(projection_bytes(LX_W_BF16, 2048, 2048) == 2048ull * 2048 * 2);
  // 8-bit: one byte per weight and one f32 scale per row
  CHECK(projection_bytes(LX_W_INT8, 2048, 2048) == 2048ull * 2048 + 2048ull * 4);
  CHECK(projection_bytes(LX_W_FP8, 2048, 2048) == projection_bytes(LX_W_INT8, 2048, 2048));
  // 4-bit: half a byte per weight, one e4m3 scale per 16; a stack of 128 experts adds their f32 globals
  CHECK(projection_bytes(LX_W_NVFP4, 2048, 2048) == 2048ull * 2048 / 2 + 2048ull * 2048 / 16);
  CHECK(projection_bytes(LX_W_NVFP4, 128 * 1024, 1024, 128) ==
        128ull * 1024 * 1024 / 2 + 128ull * 1024 * 1024 / 16 + 512);
}

TEST_CASE("state: Mamba conv and SSM per slot, KV per position, both over every pass") {
  const MetisConfig c = praxis();
  // 10 Mamba layers, 2 passes: each with conv (3 x 6144 f32) and SSM (64 x 64 x 128 f32)
  const size_t conv = 3 * 6144 * 4, ssm = 64ull * 64 * 128 * 4;
  CHECK(state_bytes_per_slot(c) == 2 * 10 * (conv + ssm));
  // 2 attention layers, K and V of 8 heads x 64 dims in bf16, 2 passes
  CHECK(kv_bytes_per_position(c) == 2ull * 2 * (2 * 8 * 64 * 2));
}

TEST_CASE("the default math is BF16 for every format on every device") {
  const DeviceInfo a100 = simulated("A100 80GB", 80, 80, 2039, 108), ada = simulated("RTX 4090 24GB", 89, 24, 1008, 128);
  const DeviceInfo b200 = simulated("B200", 100, 180, 8000, 148), rtx50 = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  // Chosen by KL on Praxis (plan.cpp); the quantized maths are LX_PREFILL's to select
  for (const DeviceInfo* d : {&a100, &ada, &b200, &rtx50}) {
    CHECK(native_math(LX_W_BF16, *d) == LX_MATH_BF16);
    CHECK(native_math(LX_W_INT8, *d) == LX_MATH_BF16);
    CHECK(native_math(LX_W_FP8, *d) == LX_MATH_BF16);
    CHECK(native_math(LX_W_NVFP4, *d) == LX_MATH_BF16);
  }
  // --fast-math: the quantized tensor-core maths where the device has them
  CHECK(native_math(LX_W_INT8, a100, true) == LX_MATH_INT8);
  CHECK(native_math(LX_W_FP8, a100, true) == LX_MATH_BF16);
  CHECK(native_math(LX_W_FP8, ada, true) == LX_MATH_FP8);
  CHECK(native_math(LX_W_NVFP4, ada, true) == LX_MATH_BF16);
  CHECK(native_math(LX_W_NVFP4, rtx50, true) == LX_MATH_FP4);
  PlanRequest fast = weights(LX_W_INT8);
  fast.fast_math = true;
  CHECK(make_plan(rtx50, praxis(), bf16_checkpoint(), fast).prefill_math == LX_MATH_INT8);
  CHECK(make_plan(rtx50, praxis(), bf16_checkpoint(), weights(LX_W_INT8)).prefill_math == LX_MATH_BF16);
  const lx::ModelFormats f = model_formats(LX_W_NVFP4, rtx50, true);
  CHECK(f.dense == LX_W_FP8);
  CHECK(f.experts == LX_W_NVFP4);
  CHECK(f.prefill_math == LX_MATH_BF16);
  CHECK(f.expert_math == LX_MATH_BF16);
  CHECK(f.bf16x_decode);  // the copies of the BF16 tensors NVFP4 keeps wide (latent_up, the Mamba out_proj, LoRA)
}

TEST_CASE("what each file loads into: never wider, NVFP4 only from the GPTQ folder") {
  const MetisConfig c = praxis();
  std::string how;
  CHECK(can_load(LX_W_BF16, LX_W_BF16, c, &how));
  CHECK(how == "as stored");
  CHECK(can_load(LX_W_FP8, LX_W_BF16, c));
  CHECK(can_load(LX_W_INT8, LX_W_BF16, c));
  CHECK(can_load(LX_W_INT8, LX_W_FP8, c));
  CHECK_FALSE(can_load(LX_W_BF16, LX_W_FP8, c, &how));
  CHECK(how.find("wider") != std::string::npos);
  CHECK_FALSE(can_load(LX_W_FP8, LX_W_NVFP4, c));
  CHECK_FALSE(can_load(LX_W_NVFP4, LX_W_BF16, c, &how));
  CHECK(how == "from its own checkpoint: dromos run Lernex/Metis-1.6-Praxis-NVFP4");
  CHECK(can_load(LX_W_NVFP4, LX_W_NVFP4, c));
  CHECK_FALSE(can_load(LX_W_BF16X, LX_W_FP8, c));
  // the plan refuses the same
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  CHECK_THROWS_AS(make_plan(d, c, fp8_checkpoint(), weights(LX_W_BF16)), std::runtime_error);
  CHECK_THROWS_AS(make_plan(d, c, bf16_checkpoint(), weights(LX_W_NVFP4)), std::runtime_error);
  CHECK(make_plan(d, c, fp8_checkpoint(), weights(LX_W_INT8)).weights == LX_W_INT8);
}

TEST_CASE("the BF16 accounting matches the checkpoint on disk") {
  const char* dir = folder("LX_MODEL_DIR");
  if (!dir) return;  // the server has the checkpoints; a laptop does not
  const MetisConfig c = load_config(dir);
  // The engine reads every tensor the model uses, each once, in BF16. The file also holds what the
  // engine does not read: lm_head (the config ties it to the embedding), the k_router and k_budget
  // tensors, and depth_budget. Those are left out, and what is left must match the file within the
  // 256-byte alignment of each tensor.
  const size_t used = folder_device_bytes(dir);
  const size_t accounted = weight_bytes(c, LX_W_BF16).loaded();
  INFO("used " << used << " accounted " << accounted);
  CHECK(accounted >= used);
  CHECK(static_cast<double>(accounted - used) / static_cast<double>(used) < 0.002);
}

TEST_CASE("the plan's bytes match each published folder, and what the loader holds, within 1%") {
  struct Case {
    const char* variable;
    lx_wtype format;
  };
  for (const Case& k : {Case{"LX_CKPT_BF16_DIR", LX_W_BF16}, Case{"LX_CKPT_FP8_DIR", LX_W_FP8},
                        Case{"LX_CKPT_NVFP4_DIR", LX_W_NVFP4}}) {
    const char* dir = folder(k.variable);
    if (!dir) continue;
    const MetisConfig c = load_config(dir);
    REQUIRE(ModelWeights::stored_format(dir) == k.format);
    const WeightBytes wb = weight_bytes(c, k.format);
    const size_t file = folder_device_bytes(dir);
    INFO(format_name(k.format) << ": the plan " << wb.loaded() << " bytes, the folder " << file);
    CHECK(relative(wb.loaded(), file) < 0.01);
    // On the device: the folder loaded in its own format, and without its experts.
    LoadOptions options;
    options.target = k.format;
    const ModelWeights m = ModelWeights::load(dir, c, options);
    std::printf("  %-6s plan %.4f GiB, loader %.4f GiB on the device (%.3f%%)\n", format_name(k.format).c_str(),
                static_cast<double>(wb.loaded()) / (1ull << 30), static_cast<double>(m.device_bytes()) / (1ull << 30),
                100.0 * relative(wb.loaded(), m.device_bytes()));
    CHECK(relative(wb.loaded(), m.device_bytes()) < 0.01);
    options.skip_experts = true;
    const ModelWeights dense = ModelWeights::load(dir, c, options);
    const size_t planned_dense = wb.loaded() - wb.experts;
    std::printf("  %-6s without experts: plan %.4f GiB, loader %.4f GiB (%.3f%%)\n", format_name(k.format).c_str(),
                static_cast<double>(planned_dense) / (1ull << 30), static_cast<double>(dense.device_bytes()) / (1ull << 30),
                100.0 * relative(planned_dense, dense.device_bytes()));
    CHECK(relative(planned_dense, dense.device_bytes()) < 0.01);
  }
}

TEST_CASE("the Praxis plan on the development card: BF16, the file's format, with room to spare") {
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  PlanRequest plain;
  plain.bf16x = false;
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), plain);
  CHECK(p.weights == LX_W_BF16);
  CHECK(p.stored == LX_W_BF16);
  CHECK_FALSE(p.convert_at_load);
  CHECK_FALSE(p.bf16x_decode);
  CHECK_FALSE(p.offload.enabled);
  CHECK(p.prefill_math == LX_MATH_BF16);
  CHECK(p.expert_math == LX_MATH_BF16);
  CHECK(p.max_slots == 64);
  CHECK(p.kv_capacity == 8192);
  CHECK(p.max_step_tokens == 2048);
  CHECK(p.prefill_chunk == 2048 - 64);
  CHECK(p.decode_ceiling_tps > 100);
  CHECK(p.notes.size() >= 5);
  // the faster format is named, not taken
  CHECK(has_note(p, "--weights int8 would decode at about"));
  // with the BF16X decode copies (when this build's backend has them): budgeted, and faster
  if (!bf16x_available()) return;
  const Plan x = make_plan(d, praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(x.weights == LX_W_BF16);
  CHECK(x.bf16x_decode);
  CHECK(x.bf16x_bytes > 0);
  CHECK(x.weight_bytes == p.weight_bytes + x.bf16x_bytes);
  CHECK(x.decode_ceiling_tps > p.decode_ceiling_tps * 1.1);  // measured: the copies decode about 15% faster
  CHECK(x.max_slots >= kUsefulSlots);
  CHECK(x.formats().bf16x_decode);
}

TEST_CASE("--weights bf16x: the lossless encoding as the one copy, from a BF16 file only") {
  CHECK(weights_from_flag("bf16x") == LX_W_BF16X);
  CHECK(weights_from_flag("nvfp4") == LX_W_NVFP4);
  CHECK_THROWS_AS(weights_from_flag("fp16"), std::runtime_error);
  if (!bf16x_available()) return;
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  const Plan x = make_plan(d, praxis(), bf16_checkpoint(), weights(LX_W_BF16X));
  CHECK(x.weights == LX_W_BF16X);
  CHECK(x.convert_at_load);
  CHECK_FALSE(x.bf16x_decode);  // the weights are the encoding: no second copy
  CHECK(x.weight_bytes < weight_bytes(praxis(), LX_W_BF16).total());
  CHECK(x.formats().dense == LX_W_BF16X);
  CHECK(x.prefill_math == LX_MATH_BF16);
  // prefill expands each weight into a bf16 scratch, which the plan counts
  CHECK(scratch_bytes(praxis(), 512, 4, x.formats()) > scratch_bytes(praxis(), 512, 4, model_formats(LX_W_BF16, d, false)));
  CHECK_THROWS_AS(make_plan(d, praxis(), fp8_checkpoint(), weights(LX_W_BF16X)), std::runtime_error);
}

TEST_CASE("BF16X decode copies never take a sequence; the ones left out are named") {
  if (!bf16x_available()) return;
  // Praxis BF16 on 8 GB asked for the useful count: the sequences and the step's scratch first, the copies in what is left
  const DeviceInfo d = simulated("RTX 4060 8GB", 89, 8, 272, 24);
  PlanRequest want;
  want.sequences = kUsefulSlots;
  PlanRequest want_plain = want;
  want_plain.bf16x = false;
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), want_plain);
  const Plan x = make_plan(d, praxis(), bf16_checkpoint(), want);
  CHECK(x.weights == LX_W_BF16);
  CHECK_FALSE(x.offload.enabled);
  CHECK(x.max_slots == p.max_slots);  // the copies take no sequence
  CHECK(x.weight_bytes + x.state_bytes + x.kv_bytes + x.scratch_bytes <= x.budget_bytes);
  CHECK(x.bf16x_bytes == weight_bytes(praxis(), LX_W_BF16, true, &x.bf16x_dropped).bf16x);
  if (!x.bf16x_dropped.empty()) CHECK(has_note(x, "Copies left out"));
}

TEST_CASE("the copies never take a sequence: Logos FP8 on 16 GB holds the 32 asked for") {
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  PlanRequest asked;
  asked.sequences = 32;
  asked.context = 784;  // the bench's context: the 32 held in full at this length
  PlanRequest plain = asked;
  plain.bf16x = false;
  const Plan p = make_plan(d, logos(), fp8_checkpoint(), plain);
  const Plan x = make_plan(d, logos(), fp8_checkpoint(), asked);
  CHECK(x.weights == LX_W_FP8);
  CHECK(p.max_slots == 32);
  CHECK(x.max_slots == 32);
  CHECK(x.weight_bytes + x.state_bytes + x.kv_bytes + x.scratch_bytes <= x.budget_bytes);
  if (!bf16x_available()) return;
  const size_t all = weight_bytes(logos(), LX_W_FP8, true).bf16x;
  CHECK(x.bf16x_bytes == weight_bytes(logos(), LX_W_FP8, true, &x.bf16x_dropped).bf16x);
  CHECK(x.bf16x_bytes <= all);
  if (x.bf16x_dropped.empty()) CHECK(x.bf16x_bytes == all);
  else CHECK(has_note(x, "Copies left out"));
  // The traffic count reads the copies the plan left out in BF16, as the loaded model does (bench checks the two
  // agree at 32 streams): the same dropped set goes to the config count and to the planner's bytes.
  const double with_plan = config_decode_traffic(logos(), LX_W_FP8, x.bf16x_decode, 784, &x.bf16x_dropped).bytes_per_token();
  const double all_copies = config_decode_traffic(logos(), LX_W_FP8, x.bf16x_decode, 784).bytes_per_token();
  CHECK(with_plan == decode_bytes_per_token(logos(), LX_W_FP8, x.bf16x_decode, 784, &x.bf16x_dropped));
  if (x.bf16x_dropped.empty()) CHECK(with_plan == all_copies);
  else CHECK(with_plan > all_copies);
  // A single stream keeps every copy: they fit beside it.
  PlanRequest one;
  one.sequences = 1;
  const Plan s = make_plan(d, logos(), fp8_checkpoint(), one);
  CHECK(s.max_slots == 1);
  CHECK(s.bf16x_decode);
  CHECK(s.bf16x_dropped.empty());
  CHECK(s.bf16x_bytes == all);
}

TEST_CASE("quantized targets: the BF16 tensors they keep wide decode from BF16X copies too") {
  if (!bf16x_available()) return;
  const DeviceInfo d = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  for (const lx_wtype target : {LX_W_FP8, LX_W_NVFP4}) {
    const CheckpointInfo ck = checkpoint(target);
    PlanRequest plain = weights(target);
    plain.bf16x = false;
    const Plan p = make_plan(d, praxis(), ck, plain);
    const Plan x = make_plan(d, praxis(), ck, weights(target));
    CHECK(x.weights == target);
    CHECK_FALSE(p.bf16x_decode);
    CHECK(x.bf16x_decode);
    CHECK(x.bf16x_bytes > 0);
    CHECK(x.bf16x_bytes == weight_bytes(praxis(), target, true).bf16x);
    CHECK(x.weight_bytes == p.weight_bytes + x.bf16x_bytes);
    CHECK(x.decode_ceiling_tps > p.decode_ceiling_tps);
    CHECK(x.formats().bf16x_decode);
    CHECK(has_note(x, "lossless BF16X copies of the BF16 tensors"));
  }
  // The lossless target is its own encoding: no second copy, whatever the flag says.
  CHECK(weight_bytes(praxis(), LX_W_BF16X, true).bf16x == 0);
  CHECK_FALSE(model_formats(LX_W_BF16X, d, true).bf16x_decode);
}

TEST_CASE("the default is the file's format: offloaded before anything narrower, never narrowed") {
  // Praxis BF16 on a 6 GB card does not fit usefully: its experts are offloaded, still BF16, and the notes name
  // the 8-bit format the GPU multiplies natively (IMMA on sm_86, FP8 on sm_89)
  const Plan sm86 = make_plan(simulated("RTX 3060 6GB", 86, 6, 360, 28), praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(sm86.weights == LX_W_BF16);
  CHECK(sm86.offload.enabled);
  CHECK_FALSE(sm86.bf16x_decode);
  CHECK(has_note(sm86, "--weights int8 would fit on the GPU"));
  const Plan sm89 = make_plan(simulated("RTX 4050 6GB", 89, 6, 272, 24), praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(sm89.weights == LX_W_BF16);
  CHECK(has_note(sm89, "--weights int8 would fit on the GPU"));
  // Logos BF16 (about 21 GB) on a 16 GB card: offloaded BF16
  const Plan logos16 = make_plan(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70), logos(), bf16_checkpoint(), PlanRequest{});
  CHECK(logos16.weights == LX_W_BF16);
  CHECK(logos16.offload.enabled);
  CHECK(logos16.offload.resident_fraction > 0.5);
  // the published FP8 file runs as FP8, resident; the GPTQ file as NVFP4
  const Plan fp8 = make_plan(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70), logos(), fp8_checkpoint(), PlanRequest{});
  CHECK(fp8.weights == LX_W_FP8);
  CHECK_FALSE(fp8.convert_at_load);
  CHECK_FALSE(fp8.offload.enabled);
  CHECK(fp8.prefill_math == LX_MATH_BF16);  // the default maths are BF16 (plan.cpp)
  CHECK(fp8.expert_math == LX_MATH_BF16);
  const Plan fp4 = make_plan(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70), praxis(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(fp4.weights == LX_W_NVFP4);
  CHECK(fp4.expert_math == LX_MATH_BF16);
  CHECK(fp4.prefill_math == LX_MATH_BF16);
  // an explicit narrower format is planned as asked, converted at load
  const Plan asked = make_plan(simulated("RTX 4050 6GB", 89, 6, 272, 24), praxis(), bf16_checkpoint(), weights(LX_W_FP8));
  CHECK(asked.weights == LX_W_FP8);
  CHECK(asked.convert_at_load);
  CHECK_FALSE(asked.offload.enabled);
  CHECK(asked.prefill_math == LX_MATH_BF16);
  const Plan int8 = make_plan(simulated("RTX 3060 6GB", 86, 6, 360, 28), praxis(), bf16_checkpoint(), weights(LX_W_INT8));
  CHECK(int8.prefill_math == LX_MATH_BF16);
  const Plan fp8_on_86 = make_plan(simulated("RTX 3060 6GB", 86, 6, 360, 28), praxis(), bf16_checkpoint(), weights(LX_W_FP8));
  CHECK(fp8_on_86.prefill_math == LX_MATH_BF16);  // no FP8 tensor cores: weight-only, BF16 math
  CHECK_FALSE(has_note(fp8_on_86, "no tensor-core instruction for FP8"));  // named only under --fast-math
  PlanRequest fp8_fast = weights(LX_W_FP8);
  fp8_fast.fast_math = true;
  const Plan fp8_fast_on_86 = make_plan(simulated("RTX 3060 6GB", 86, 6, 360, 28), praxis(), bf16_checkpoint(), fp8_fast);
  CHECK(fp8_fast_on_86.prefill_math == LX_MATH_BF16);
  CHECK(has_note(fp8_fast_on_86, "no tensor-core instruction for FP8"));
}

TEST_CASE("the 4-bit format from the GPTQ folder runs BF16 maths by default on every card") {
  const Plan twelve = make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), logos(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(twelve.weights == LX_W_NVFP4);
  CHECK(twelve.recipe.dense == LX_W_FP8);
  CHECK(twelve.recipe.latent_up == LX_W_BF16);
  CHECK(twelve.prefill_math == LX_MATH_BF16);
  CHECK(twelve.expert_math == LX_MATH_BF16);
  const Plan blackwell = make_plan(simulated("RTX 5070 Ti 16GB", 120, 12, 896, 70), logos(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(blackwell.expert_math == LX_MATH_BF16);
  CHECK(blackwell.prefill_math == LX_MATH_BF16);
  const Plan b200 = make_plan(simulated("B200", 100, 180, 8000, 148), logos(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(b200.expert_math == LX_MATH_BF16);
  CHECK(b200.prefill_math == LX_MATH_BF16);
  // NVFP4 experts are not offloaded, so a card too small for four sequences runs fewer, and says the card is small
  for (double gib = 8.5; gib <= 10.0; gib += 0.25) {
    const Plan p = make_plan(simulated("Sized card", 89, gib, 272, 24), logos(), nvfp4_checkpoint(), PlanRequest{});
    CHECK(p.max_slots >= 1);
    CHECK(has_note(p, "small for the model") == (p.max_slots < 4));
    INFO("card " << gib << " GB: " << p.max_slots << " sequences");
  }
}

TEST_CASE("the plan's choices across the brief's GPUs") {
  const PlanRequest none{};
  // Praxis: BF16 wherever it fits usefully (6.2 GB of weights), from 12 GB up
  for (const char* name : {"RTX 3060 12GB", "RTX 4090 24GB", "H100 80GB", "B200"}) {
    const Gpu& g = gpu(name);
    const Plan p = make_plan(simulated(g.name, g.cc, g.gib, g.bandwidth, g.sms), praxis(), bf16_checkpoint(), none);
    CHECK(p.weights == LX_W_BF16);
    CHECK_FALSE(p.offload.enabled);
  }
  CHECK(make_plan(simulated("RTX 5090 32GB", 120, 32, 1792, 170), praxis(), bf16_checkpoint(), none).max_slots == 64);
  CHECK(make_plan(simulated("A100 80GB", 80, 80, 2039, 108), praxis(), bf16_checkpoint(), none).expert_math == LX_MATH_BF16);
  // Logos: BF16 resident on the 24 GB and bigger cards, offloaded BF16 on 12 GB
  CHECK_FALSE(make_plan(simulated("RTX 4090 24GB", 89, 24, 1008, 128), logos(), bf16_checkpoint(), none).offload.enabled);
  CHECK_FALSE(make_plan(simulated("A100 80GB", 80, 80, 2039, 108), logos(), bf16_checkpoint(), none).offload.enabled);
  const Plan twelve = make_plan(simulated("RTX 3060 12GB", 86, 12, 360, 28), logos(), bf16_checkpoint(), none);
  CHECK(twelve.weights == LX_W_BF16);
  CHECK(twelve.offload.enabled);
}

TEST_CASE("a request that cannot fit says why, and a plan never overflows the budget") {
  // BF16 Logos does not run on 2 GB even offloaded; NVFP4 Logos (no offload for NVFP4) runs on an idle 8 GB card with
  // the fixed margin, and does not run on one with 6.6 GiB free, where the 8% share holds
  CHECK_THROWS_AS(make_plan(simulated("Tiny 2GB", 89, 2, 272, 24), logos(), bf16_checkpoint(), weights(LX_W_BF16)),
                  std::runtime_error);
  CHECK_NOTHROW(make_plan(simulated("RTX 4060 8GB", 89, 8, 272, 24), logos(), nvfp4_checkpoint(), PlanRequest{}));
  DeviceInfo busy8 = simulated("RTX 4060 8GB", 89, 8, 272, 24);
  busy8.free_bytes = static_cast<size_t>(6.6 * kGiB);
  CHECK_THROWS_AS(make_plan(busy8, logos(), nvfp4_checkpoint(), PlanRequest{}), std::runtime_error);
  for (const Gpu& g : kGpus) {
    for (const MetisConfig& c : {praxis(), logos()}) {
      for (const CheckpointInfo& ck : {bf16_checkpoint(), fp8_checkpoint(), nvfp4_checkpoint()}) {
        try {
          const Plan p = make_plan(simulated(g.name, g.cc, g.gib, g.bandwidth, g.sms), c, ck, PlanRequest{});
          const size_t used = p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes;
          INFO(g.name << " " << c.family << " " << format_name(ck.stored));
          CHECK(used <= p.budget_bytes);
          CHECK(p.max_slots >= 1);
          CHECK(p.weights == ck.stored);
        } catch (const std::runtime_error&) {
          // a card that cannot hold the model at all: the message is the test
        }
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

TEST_CASE("a CPU with the model's RAM free: resident in RAM, no expert offload, no GPU wording") {
  const DeviceInfo d = simulated_cpu("Windows VM CPU", 64);
  PlanRequest plain;
  plain.bf16x = false;
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), plain);
  CHECK(p.weights == LX_W_BF16);
  CHECK_FALSE(p.offload.enabled);
  CHECK(p.max_slots >= kUsefulSlots);
  CHECK(has_note(p, "Memory: "));
  CHECK(names_no_gpu(p));
  // The offload planner refuses a CPU outright: the experts are in RAM already.
  CHECK_THROWS_AS(make_offload_plan(d, praxis(), bf16_checkpoint(), plain), std::runtime_error);
}

TEST_CASE("a CPU with the model's size in RAM but less of it free: resident, with a note that the system pages") {
  // The 8.5 GB Windows VM that ran Praxis BF16 at 12 tok/s: 6.47 GB free at startup against 6.21 GB needed.
  DeviceInfo d = simulated_cpu("Windows VM CPU", 8.5);
  d.free_bytes = static_cast<size_t>(6.47 * 1e9);
  PlanRequest plain;
  plain.bf16x = false;
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), plain);
  CHECK(p.weights == LX_W_BF16);
  CHECK_FALSE(p.offload.enabled);
  CHECK(has_note(p, "pages other programs out"));
  CHECK(names_no_gpu(p));
}

TEST_CASE("a CPU makes no BF16X decode copies by default, and doctor's estimate reads the BF16 weights") {
  // The copies decoded 10% slower than the BF16 weights on an AVX2 desktop (plan.cpp, bf16x_copies_pay).
  const DeviceInfo d = simulated_cpu("Windows VM CPU", 64);
  CHECK_FALSE(bf16x_copies_pay(d));
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), PlanRequest{});
  CHECK(p.weights == LX_W_BF16);
  CHECK_FALSE(p.bf16x_decode);
  CHECK(p.bf16x_bytes == 0);
  if (bf16x_available()) CHECK(p.bf16x_reason.find("on a CPU") != std::string::npos);
  for (const FormatOption& o : p.options) CHECK(o.bf16x_bytes == 0);
  // The estimate is the CPU's measured share of its probe (0.76 for BF16), not the GPU's (0.795): about 20 tok/s on the
  // Windows VM, where 20 was measured.
  const double bytes = decode_bytes_per_token(praxis(), LX_W_BF16, false, p.kv_capacity);
  CHECK(p.decode_ceiling_tps == doctest::Approx(d.bandwidth_gbs * 1e9 * 0.76 / bytes).epsilon(0.01));
  CHECK(bf16x_copies_pay(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70)));
}

TEST_CASE("one sequence is said in the singular") {
  const DeviceInfo d = simulated_cpu("Windows VM CPU", 64);
  PlanRequest one;
  one.sequences = 1;
  const Plan p = make_plan(d, praxis(), bf16_checkpoint(), one);
  REQUIRE(p.max_slots == 1);
  CHECK(has_note(p, "One sequence, "));
  CHECK_FALSE(has_note(p, "Up to 1 sequences"));
}

TEST_CASE("a refusal lists what runs from this checkpoint, and NVFP4 with the command that loads it") {
  // The 8.5 GB Windows VM, with the budget held down so BF16 is refused and the narrower formats fit.
  DeviceInfo d = simulated_cpu("Windows VM CPU", 8.5);
  d.free_bytes = static_cast<size_t>(6.5 * 1e9);
  PlanRequest req = weights(LX_W_BF16);
  req.ram_budget = static_cast<size_t>(5.6 * kGiB);
  std::string why;
  try {
    make_plan(d, praxis(), bf16_checkpoint(), req);
  } catch (const std::runtime_error& e) {
    why = e.what();
  }
  INFO(why);
  REQUIRE_FALSE(why.empty());
  CHECK(why.find("BF16X fits") != std::string::npos);
  CHECK(why.find("INT8 fits") != std::string::npos);
  CHECK(why.find("NVFP4 would fit") != std::string::npos);
  CHECK(why.find("dromos run Lernex/Metis-1.6-Praxis-NVFP4") != std::string::npos);
  CHECK(why.find("NVFP4 fits") == std::string::npos);
  CHECK(why.find("/home/") == std::string::npos);
}

TEST_CASE("a CPU with less RAM free than the model: says so in RAM terms, no VRAM or offload") {
  const DeviceInfo d = simulated_cpu("Windows VM CPU", 3);
  PlanRequest plain;
  plain.bf16x = false;
  std::string why;
  bool threw = false;
  try {
    make_plan(d, praxis(), bf16_checkpoint(), plain);
  } catch (const std::runtime_error& e) {
    threw = true;
    why = e.what();
  }
  CHECK(threw);
  CHECK(why.find("this machine") != std::string::npos);
  CHECK(why.find("of RAM can hold the model") != std::string::npos);
  CHECK(why.find("VRAM") == std::string::npos);
  CHECK(why.find("GPU") == std::string::npos);
  CHECK(why.find("CUDA") == std::string::npos);
  CHECK(why.find("offload") == std::string::npos);
  // A narrower format that fits is named, as on a GPU: never taken for the user.
  CHECK((why.find("--weights") == std::string::npos || why.find("fits") != std::string::npos));
}


TEST_CASE("Logos NVFP4 on an 8 GiB card with 7.6 GiB free holds one 8192-token sequence") {
  // The brief's card: nothing else on it, so the default share is replaced by the fixed margin (plan.cpp).
  const DeviceInfo d = idle_8gib();
  PlanRequest one;
  one.context = 8192;
  one.sequences = 1;
  const Plan p = make_plan(d, logos(), nvfp4_checkpoint(), one);
  CHECK(p.weights == LX_W_NVFP4);
  CHECK_FALSE(p.offload.enabled);
  CHECK(p.max_slots == 1);
  CHECK(p.kv_capacity == 8192);
  CHECK(p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes <= p.budget_bytes);
  // The budget is the free memory less the fixed margin and the reserve: 256 MiB each, 7.1 GiB (7.62 GB) here.
  CHECK(p.budget_bytes == static_cast<size_t>(7.6 * kGiB) - 512 * 1024 * 1024);
  // The automatic request aims for 8192 tokens too, and fits one sequence of them.
  const Plan a = make_plan(d, logos(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(a.max_slots >= 1);
  CHECK(a.kv_capacity == 8192);
  CHECK(a.weight_bytes + a.state_bytes + a.kv_bytes + a.scratch_bytes <= a.budget_bytes);
}

TEST_CASE("the same request under LX_VRAM_BUDGET_GB=7.6 on a 16 GB card: the same budget, the same plan") {
  const DeviceInfo big = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  PlanRequest capped;
  capped.context = 8192;
  capped.sequences = 1;
  capped.vram_budget = static_cast<size_t>(7.6 * kGiB);  // the budget counts the CUDA context, and the reserve is kept
  const Plan p = make_plan(big, logos(), nvfp4_checkpoint(), capped);
  CHECK(p.max_slots == 1);
  CHECK(p.kv_capacity == 8192);
  CHECK(p.budget_bytes == static_cast<size_t>(7.6 * kGiB) - 512 * 1024 * 1024);
}

TEST_CASE("a card another process uses keeps the share; a fraction the user set is their cap") {
  constexpr size_t kReserve = 256ull * 1024 * 1024;
  // 6.6 GiB free of 8: 1.4 GiB in use beyond this process's context, so the 8% share holds and the model does not fit
  DeviceInfo shared = idle_8gib();
  shared.free_bytes = static_cast<size_t>(6.6 * kGiB);
  PlanRequest one;
  one.context = 8192;
  one.sequences = 1;
  CHECK_THROWS_AS(make_plan(shared, logos(), nvfp4_checkpoint(), one), std::runtime_error);
  // Praxis BF16 on a shared 16 GB card: the budget is the share, as it was before the margin existed
  DeviceInfo busy = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  busy.free_bytes = static_cast<size_t>(12.0 * kGiB);
  PlanRequest plain;
  plain.bf16x = false;
  const Plan s = make_plan(busy, praxis(), bf16_checkpoint(), plain);
  CHECK(s.budget_bytes == static_cast<size_t>(static_cast<double>(busy.free_bytes) * 0.92) - kReserve);
  // A fraction the user sets is a cap, on an idle card as on any other
  DeviceInfo idle = simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70);
  PlanRequest half;
  half.bf16x = false;
  half.memory_fraction = 0.5;
  const Plan h = make_plan(idle, praxis(), bf16_checkpoint(), half);
  CHECK(h.budget_bytes == (idle.free_bytes - static_cast<size_t>(static_cast<double>(idle.free_bytes) * 0.5)) - kReserve);
}

TEST_CASE("the fixed margin never shrinks a plan's budget on any simulated card") {
  // The margin replaces an 8% share with 256 MiB, so an idle card's budget can only grow, and a shared card's stays.
  constexpr size_t kReserve = 256ull * 1024 * 1024;
  for (const Gpu& g : kGpus) {
    DeviceInfo d = simulated(g.name, g.cc, g.gib, g.bandwidth, g.sms);
    const size_t share_budget = static_cast<size_t>(static_cast<double>(d.free_bytes) * 0.92) - kReserve;
    for (const MetisConfig& c : {praxis(), logos()}) {
      for (const CheckpointInfo& ck : {bf16_checkpoint(), fp8_checkpoint(), nvfp4_checkpoint()}) {
        try {
          const Plan p = make_plan(d, c, ck, PlanRequest{});
          // An offloaded plan's budget is the offload planner's (its own margin, runtime/offload_plan.cpp): not compared.
          if (p.offload.enabled) continue;
          INFO(g.name << " " << c.family << " " << format_name(ck.stored));
          CHECK(p.budget_bytes >= share_budget);
          CHECK(p.budget_bytes <= d.free_bytes - kReserve);
        } catch (const std::runtime_error&) {
          // a card that cannot hold the model at all
        }
      }
    }
  }
}

TEST_CASE("the prefix cache on unified memory: a quarter of what the plan leaves, at most 2 GiB, and no host tier") {
  // An M5 MacBook with 16 GB: Metal's working set is 12.7 GB, and the old shares (all the spare memory up to 8 GiB,
  // then 4 GiB of "pinned host" memory that is the same RAM) let a long chat's snapshots reach 13.9 GB of footprint.
  DeviceInfo mac = simulated("16 GB Mac, unified", 120, 12.71, 154, 10);
  mac.unified = true;
  DeviceInfo cpu = simulated_cpu("16 GB laptop CPU", 16);
  cpu.unified = true;
  for (const DeviceInfo& d : {mac, cpu}) {
    INFO(d.name);
    const Plan p = make_plan(d, praxis(), nvfp4_checkpoint(), PlanRequest{});
    const size_t planned = p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes;
    CHECK(p.prefix_host_bytes == 0);
    CHECK(p.prefix_device_bytes > 0);
    CHECK(p.prefix_device_bytes <= 2 * kGiB);
    CHECK(p.prefix_device_bytes <= (p.budget_bytes - planned) / 4);
  }
  // A discrete card keeps its spare VRAM for snapshots and spills to pinned RAM, which is memory of its own there.
  const Plan card = make_plan(simulated("RTX 5070 Ti 16GB", 120, 16, 896, 70), praxis(), nvfp4_checkpoint(), PlanRequest{});
  CHECK(card.prefix_host_bytes == 4 * kGiB);
  CHECK(card.prefix_device_bytes > 2 * kGiB);
}
