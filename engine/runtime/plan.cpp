// The plan: from the device, the model and its checkpoint, the weight format, the math each kind of
// product runs in, and how many sequences of what length fit. The byte accounting mirrors the
// allocations in engine/model/metis.cpp (its constructor is the list); the tests check the BF16 total
// against the checkpoint file it is read from.
#include "runtime/plan.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core/safetensors.h"
#include "lx_build_info.h"  // LX_BUILT_ARCHS, written by CMake


namespace lx::runtime {
namespace {

constexpr size_t kAlign = 256;  // Weights::upload and the model's allocations start on this boundary
constexpr size_t kMiB = 1024ull * 1024ull;

// Sized for the decisions the plan makes. A format must leave room for kUsefulSlots sequences of
// kUsefulContext positions to count as fitting. Automatic sizing aims for kDefaultContext and at most
// kMaxSlots sequences, and steps are at most kMaxStepTokens tokens.
constexpr int kUsefulSlots = 4;
constexpr int kUsefulContext = 4096;
constexpr int kLimitedContext = 2048;
constexpr int kDefaultContext = 8192;
constexpr int kMaxSlots = 64;
constexpr int kMinStepTokens = 512;
constexpr int kMaxStepTokens = 2048;
// What the plan keeps back beyond its own accounting: cuBLASLt's workspace and the allocator's slack.
constexpr size_t kReserveBytes = 256 * kMiB;
// The per-step expert workspace is sized for this many decode rows (metis.cpp, kGemvRows).
constexpr int kGemvRows = 8;

size_t aligned(size_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

std::string gb(size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.2f GB", static_cast<double>(bytes) / 1e9);
  return buf;
}
std::string mb(size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.0f MB", static_cast<double>(bytes) / 1e6);
  return buf;
}

std::string format_name(lx_wtype w) {
  switch (w) {
    case LX_W_BF16: return "BF16";
    case LX_W_INT8: return "INT8";
    case LX_W_FP8: return "FP8";
    case LX_W_NVFP4: return "NVFP4";
  }
  return "?";
}
std::string math_name(lx_math m) {
  switch (m) {
    case LX_MATH_BF16: return "BF16 tensor cores";
    case LX_MATH_INT8: return "INT8 tensor cores";
    case LX_MATH_FP8: return "FP8 tensor cores";
    case LX_MATH_FP4: return "FP4 tensor cores";
  }
  return "?";
}

// The math a product runs in, from the format it reads and the GPU. Quantized files run weight-only
// (expanded in registers, bf16 or f32 math) where the GPU has no instruction for their format.
lx_math dense_math(lx_wtype dense, const DeviceInfo& d) {
  switch (dense) {
    case LX_W_BF16: return LX_MATH_BF16;
    case LX_W_INT8: return LX_MATH_INT8;  // IMMA exists on every target (sm_80 and up)
    case LX_W_FP8:
    case LX_W_NVFP4: return d.fp8_mma ? LX_MATH_FP8 : LX_MATH_BF16;
  }
  return LX_MATH_BF16;
}
lx_math expert_math(lx_wtype experts, const DeviceInfo& d) {
  switch (experts) {
    case LX_W_BF16: return LX_MATH_BF16;
    case LX_W_INT8: return LX_MATH_INT8;
    case LX_W_FP8: return d.fp8_mma ? LX_MATH_FP8 : LX_MATH_BF16;
    // NVFP4 experts multiply in FP4 on sm_100 / sm_120; the mixed e2m1 x e4m3 path is the same class.
    case LX_W_NVFP4: return d.fp4_mma ? LX_MATH_FP4 : LX_MATH_BF16;
  }
  return LX_MATH_BF16;
}

// A convenience for the accounting: one bf16 or f32 tensor, aligned as the upload aligns it.
size_t bf16_bytes(int64_t n) { return aligned(static_cast<size_t>(n) * 2); }
size_t f32_bytes(int64_t n) { return aligned(static_cast<size_t>(n) * 4); }

// Attention layers of the config, and the Mamba layers that remain.
int attention_layers(const MetisConfig& c) { return static_cast<int>(c.attention_layer_indices.size()); }
int mamba_layers(const MetisConfig& c) { return c.num_hidden_layers - attention_layers(c); }

}  // namespace

bool kernels_built_for(int cc) {
  static constexpr int kBuilt[] = {LX_BUILT_ARCHS};
  return std::find(std::begin(kBuilt), std::end(kBuilt), cc) != std::end(kBuilt);
}

std::vector<int> built_archs() {
  static constexpr int kBuilt[] = {LX_BUILT_ARCHS};
  return {std::begin(kBuilt), std::end(kBuilt)};
}

WeightRecipe recipe_for(lx_wtype target) {
  WeightRecipe r;
  switch (target) {
    case LX_W_BF16:
      break;  // everything bf16
    case LX_W_INT8:
    case LX_W_FP8:
      r.dense = target;
      r.experts = target;
      r.latent_up = target;
      break;
    case LX_W_NVFP4:
      r.dense = LX_W_FP8;  // dense projections stay FP8; the experts are the 4-bit part
      r.experts = LX_W_NVFP4;
      r.latent_up = LX_W_BF16;
      break;
  }
  return r;
}

size_t projection_bytes(lx_wtype format, int64_t rows, int64_t cols) {
  switch (format) {
    case LX_W_BF16:
      return aligned(static_cast<size_t>(rows * cols * 2));
    case LX_W_INT8:
    case LX_W_FP8:
      return aligned(static_cast<size_t>(rows * cols)) + aligned(static_cast<size_t>(rows * 4));
    case LX_W_NVFP4:
      // two e2m1 per byte, one e4m3 block scale per 16 values, and one f32 global for the tensor
      return aligned(static_cast<size_t>(rows * cols / 2)) + aligned(static_cast<size_t>(rows * cols / 16)) + kAlign;
  }
  return 0;
}

WeightBytes weight_bytes(const MetisConfig& c, const WeightRecipe& r) {
  WeightBytes wb;
  const int64_t V = c.vocab_size, D = c.hidden_size, S = c.n_streams, Mm = c.memory_dim, R = c.route_feature_dim;
  const int64_t E = c.n_routed_experts, Lat = c.latent_dim, Iexp = c.expert_intermediate_size;
  const int64_t Ish = c.shared_expert_intermediate_size, P = c.num_passes, MP = c.max_passes;
  const int64_t I = c.mamba_intermediate(), H = c.mamba_heads(), Pd = c.mamba_head_dim, N = c.mamba_d_state;
  const int64_t G = c.mamba_n_groups, W = c.mamba_d_conv, conv_dim = I + 2 * G * N, proj = 2 * I + 2 * G * N + H;
  const int64_t AH = c.num_attention_heads, Dh = c.head_dim, KVH = c.num_key_value_heads;
  const int64_t qkv_width = (AH + 2 * KVH) * Dh, rank = c.attention_pass_lora_rank, A = attention_layers(c);
  const int64_t E_mhc = c.mhc_pass_embedding_dim, SS2 = S * S + 2 * S;
  const bool quant_dense = r.dense != LX_W_BF16;

  // Plain tensors: every format keeps these as they are.
  wb.plain += bf16_bytes(V * D);             // embedding table: the lookup needs bf16 rows
  wb.plain += bf16_bytes(S * D) + bf16_bytes(D);  // stream embeddings, final norm
  wb.plain += bf16_bytes(MP * E_mhc);        // pass embeddings
  // depth memory
  wb.plain += bf16_bytes(MP * Mm) + bf16_bytes((A + 1) * Mm);  // pass and anchor embeddings
  wb.plain += bf16_bytes(S * D) + bf16_bytes(S);               // stream gate and its bias
  wb.plain += bf16_bytes(Mm * (R + 4)) + bf16_bytes(Mm);       // metadata_write (never quantized) + bias
  wb.plain += bf16_bytes(Mm * Mm) + bf16_bytes(Mm);            // key (never quantized) + bias
  wb.plain += bf16_bytes(Mm * Mm) + bf16_bytes(Mm);            // value (never quantized) + bias
  wb.plain += bf16_bytes(Mm) + bf16_bytes(Mm);                 // state_write bias, query bias
  wb.plain += bf16_bytes(D) + bf16_bytes(R);                   // output bias, route_projection bias
  // the depth head: f32 hidden layer, its bias, the output row and its bias
  wb.plain += f32_bytes(R * (3 * D + R)) + f32_bytes(R) + f32_bytes(R) + f32_bytes(1);
  // per layer
  for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
    wb.plain += 2 * bf16_bytes(D);  // mixer and moe connection norms
    // The two hyper-connections: the controller's checkpoint tensors (bf16 or f32; counted as bf16),
    // and the per-pass f32 mix, read and write parameters the model computes at load.
    wb.plain += 2 * (bf16_bytes(SS2 * E_mhc) + bf16_bytes(SS2) + bf16_bytes(S * S) + 2 * bf16_bytes(S));
    wb.plain += 2 * f32_bytes(P * SS2);
    // MoE router and expert embeddings
    wb.plain += bf16_bytes(E * R) + f32_bytes(E * (Lat + R)) + f32_bytes(E) + f32_bytes(E);
    if (c.is_attention(layer)) {
      wb.plain += bf16_bytes(rank * D) + bf16_bytes(qkv_width * rank) + bf16_bytes(MP);  // pass LoRA, pass gates
      wb.dense += projection_bytes(r.dense, qkv_width, D) + projection_bytes(r.dense, D, AH * Dh);
    } else {
      wb.plain += bf16_bytes(conv_dim * W) + bf16_bytes(conv_dim) + bf16_bytes(I);  // conv, conv bias, gated norm
      wb.plain += 3 * f32_bytes(H);                                                  // A_log, D, dt_bias
      wb.dense += projection_bytes(r.dense, proj, D);                               // in_proj
      wb.out_proj += projection_bytes(LX_W_BF16, D, I);                             // out_proj stays bf16
    }
    wb.dense += projection_bytes(r.dense, Lat, D);                  // latent_down
    wb.latent_up += projection_bytes(r.latent_up, D, Lat);          // latent_up
    wb.dense += projection_bytes(r.dense, 2 * Ish, Lat) + projection_bytes(r.dense, Lat, Ish);  // shared expert
    wb.experts += projection_bytes(r.experts, E * 2 * Iexp, Lat);   // expert gate_up stack
    wb.experts += projection_bytes(r.experts, E * Lat, Iexp);       // expert down stack
  }
  // depth-memory projections
  wb.dense += projection_bytes(r.dense, Mm, D) + projection_bytes(r.dense, Mm, D);      // state_write, query
  wb.dense += projection_bytes(r.dense, D, Mm) + projection_bytes(r.dense, R, 3 * D + R);  // output, route
  // the tied head: the same weights as the embedding table, but as a projection in the dense format.
  // In BF16 the head reads the table itself.
  if (quant_dense) wb.dense += projection_bytes(r.dense, V, D);
  return wb;
}

size_t state_bytes_per_slot(const MetisConfig& c) {
  const int64_t I = c.mamba_intermediate(), H = c.mamba_heads(), Pd = c.mamba_head_dim, N = c.mamba_d_state;
  const int64_t G = c.mamba_n_groups, W = c.mamba_d_conv, conv_dim = I + 2 * G * N;
  const size_t per_layer = f32_bytes((W - 1) * conv_dim) + f32_bytes(H * Pd * N);  // conv then SSM state
  return static_cast<size_t>(c.num_passes) * static_cast<size_t>(mamba_layers(c)) * per_layer;
}

size_t kv_bytes_per_position(const MetisConfig& c) {
  // K and V, bf16, per attention layer and pass
  const size_t per_layer = 2 * static_cast<size_t>(c.num_key_value_heads) * c.head_dim * 2;
  return static_cast<size_t>(c.num_passes) * static_cast<size_t>(attention_layers(c)) * per_layer;
}

size_t scratch_bytes(const MetisConfig& c, int tokens, int slots) {
  const size_t T = static_cast<size_t>(tokens), TS = T * c.n_streams, S = static_cast<size_t>(slots);
  const size_t D = c.hidden_size, Mm = c.memory_dim, R = c.route_feature_dim, E = c.n_routed_experts;
  const size_t Lat = c.latent_dim, Iexp = c.expert_intermediate_size, Ish = c.shared_expert_intermediate_size;
  const size_t K = c.num_experts_per_tok;
  const size_t I = static_cast<size_t>(c.mamba_intermediate()), H = static_cast<size_t>(c.mamba_heads());
  const size_t G = c.mamba_n_groups, N = c.mamba_d_state;
  const size_t conv_dim = I + 2 * G * N, proj = 2 * I + 2 * G * N + H;
  const size_t qkv_width = static_cast<size_t>(c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim;
  const size_t rank = c.attention_pass_lora_rank, att = static_cast<size_t>(c.num_attention_heads) * c.head_dim;
  const size_t Emax = static_cast<size_t>(attention_layers(c) + 1) * c.num_passes;
  const size_t cast = std::max({3 * D + R, I, proj, D});

  size_t b = 0;
  auto add = [&](size_t bytes) { b += aligned(bytes); };
  // f32 and bf16 buffers, in the order metis.cpp allocates them
  add(TS * D * 4); add(TS * D * 4); add(TS * D * 2);                  // streams, mixed, streams_bf
  add(TS * Mm * 4); add(TS * Mm * 2);                                 // q, retrieved_bf
  add(TS * D * 4);                                                    // projected
  add(T * D * 4); add(T * D * 4); add(T * D * 4); add(T * D * 4);     // mem_summary, state, previous, difference
  add(T * R * 4); add(T * R * 4); add(T * (3 * D + R) * 2);           // history, features, feat_in_bf
  add(T * (3 * D + R) * 4); add(T * R * 4); add(T * 4); add(T * 4);  // cont_in, cont_hidden, cont_logit, continuation
  add(T * D * 2); add(T * D * 4);                                     // h_bf, update
  add(T * proj * 4); add(T * conv_dim * 4); add(T * I * 4);          // zxbcdt, xbc, ssm_y
  add(T * I * 2);                                                     // normed_bf
  add(T * qkv_width * 4); add(T * rank * 2); add(T * att * 2);       // qkv_buf, lora_bf, att_bf
  add(T * Lat * 4); add(T * Lat * 2);                                 // latent, latent_bf
  add(T * (Lat + R) * 4); add(T * E * 4);                             // router_in, router_logits
  add(T * K * 4); add(T * K * 4); add(T * 4); add(T * 4);            // route_ids, route_w, entropy, confidence
  add(T * Lat * 4); add(T * 2 * Ish * 4); add(T * Ish * 2); add(T * Lat * 2);  // routed, shared_gu, shared_bf, routed_bf
  add(T * R * 4);                                                     // summary
  add(T * Emax * Mm * 4); add(T * Emax * Mm * 4);                     // bank_k, bank_v
  add(T * Mm * 4); add(T * Mm * 2); add(T * (R + 4) * 2); add(T * D * 2);  // entry, entry_bf, meta_bf, out_bf
  add(S * D * 4);                                                     // last_rows
  add(TS * cast * 2);                                                 // cast_bf
  add(static_cast<size_t>(lx_moe_workspace_bytes(kGemvRows, static_cast<int>(Iexp), static_cast<int>(K))));
  add((T + (S + 1) + 3 * S) * 4);                                     // the step's int32 metadata
  return b;
}

CheckpointInfo inspect_checkpoint(const std::string& model_dir) {
  const SafetensorsSet set = SafetensorsSet::open(model_dir);
  CheckpointInfo info;
  info.bytes = set.total_bytes();
  bool fp8 = false, nvfp4 = false, int8 = false;
  for (const auto& [name, t] : set.tensors()) {
    if (t.dtype == DType::F8_E4M3 || t.dtype == DType::F8_E5M2) fp8 = true;
    if (t.dtype == DType::I8) int8 = true;
    if (t.dtype == DType::U8 && name.find("weight_packed") != std::string::npos) nvfp4 = true;
  }
  if (nvfp4) {
    info.stored = LX_W_NVFP4;
  } else if (fp8) {
    info.stored = LX_W_FP8;
  } else if (int8) {
    info.stored = LX_W_INT8;
  } else {
    info.stored = LX_W_BF16;
  }
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s checkpoint, %s on disk", format_name(info.stored).c_str(), gb(info.bytes).c_str());
  info.description = buf;
  return info;
}

Plan make_plan(const DeviceInfo& d, const MetisConfig& c, const CheckpointInfo& ck, const PlanRequest& req) {
  if (!d.built)
    throw std::runtime_error("this build has no kernels for sm_" + std::to_string(d.real_cc()) +
                             "; rebuild with -DLX_ARCHS=all to include every architecture the engine runs on");

  const size_t free_budget = static_cast<size_t>(static_cast<double>(d.free_bytes) * req.memory_fraction);
  const size_t budget = free_budget > kReserveBytes ? free_budget - kReserveBytes : 0;
  const size_t per_slot_state = state_bytes_per_slot(c);
  const size_t per_position = kv_bytes_per_position(c);
  const int context_cap = c.max_position_embeddings > 0 ? c.max_position_embeddings : kDefaultContext;
  const int slot_cap = req.sequences > 0 ? req.sequences : kMaxSlots;
  const int context_wanted = std::clamp(req.context > 0 ? req.context : kDefaultContext, 1, context_cap);

  // The formats to try, widest first. Of the two 8-bit formats, the one the GPU multiplies natively
  // goes first: FP8 tensor cores on sm_89 and up, IMMA on the rest.
  std::vector<lx_wtype> order = d.fp8_mma ? std::vector<lx_wtype>{LX_W_BF16, LX_W_FP8, LX_W_INT8, LX_W_NVFP4}
                                          : std::vector<lx_wtype>{LX_W_BF16, LX_W_INT8, LX_W_FP8, LX_W_NVFP4};
  if (req.weights >= 0) order = {static_cast<lx_wtype>(req.weights)};

  // One format at one tier: the largest context (from context_wanted, halving down to floor_ctx) at which
  // at least want_slots sequences fit beside the weights and the smallest step.
  struct Fit {
    lx_wtype format = LX_W_BF16;
    WeightBytes wb;
    size_t need = 0;  // the bytes the format needs for want_slots at floor_ctx
    int context = 0, slots = 0;
    bool ok = false;
  };
  auto attempt = [&](lx_wtype format, int want_slots, int floor_ctx) {
    Fit f;
    f.format = format;
    f.wb = weight_bytes(c, recipe_for(format));
    const size_t fixed = f.wb.total() + scratch_bytes(c, kMinStepTokens, 0);
    f.need = fixed + static_cast<size_t>(want_slots) * (per_slot_state + per_position * static_cast<size_t>(floor_ctx));
    const int need_slots = std::min(want_slots, slot_cap);
    for (int ctx = context_wanted;; ctx = std::max(floor_ctx, ctx / 2)) {
      const size_t per_slot = per_slot_state + per_position * static_cast<size_t>(ctx);
      if (budget > fixed && per_slot > 0) {
        int slots = static_cast<int>(std::min<size_t>((budget - fixed) / per_slot, static_cast<size_t>(slot_cap)));
        // the scratch grows with the slots' metadata too; take back any slot that no longer fits
        while (slots > 0 && fixed + scratch_bytes(c, kMinStepTokens, slots) - scratch_bytes(c, kMinStepTokens, 0) +
                                    static_cast<size_t>(slots) * per_slot > budget)
          --slots;
        if (slots >= need_slots) {
          f.context = ctx;
          f.slots = slots;
          f.ok = true;
          return f;
        }
      }
      if (req.context > 0 || ctx <= floor_ctx) break;  // an explicit context is kept, or nothing smaller helps
    }
    return f;
  };

  // Tier 1 is the useful fit: kUsefulSlots sequences of kUsefulContext positions. Tier 2 is what a GPU too
  // small for that still runs: one sequence of kLimitedContext positions, said so in the notes.
  const int useful_slots = req.sequences > 0 ? req.sequences : kUsefulSlots;
  std::vector<Fit> tried;
  const Fit* chosen = nullptr;
  bool limited = false;
  for (int tier = 0; tier < 2 && !chosen; ++tier) {
    const int want = tier == 0 ? useful_slots : (req.sequences > 0 ? req.sequences : 1);
    const int floor_ctx = tier == 0 ? kUsefulContext : kLimitedContext;
    tried.clear();
    for (const lx_wtype format : order) {
      tried.push_back(attempt(format, want, floor_ctx));
      if (tried.back().ok) {
        chosen = &tried.back();
        limited = tier == 1;
        break;
      }
    }
  }
  if (!chosen) {
    std::string why;
    for (const Fit& f : tried)
      why += "\n  " + format_name(f.format) + " needs " + gb(f.need) + " for one sequence of " +
             std::to_string(kLimitedContext) + " tokens; " + gb(budget) + " is free for the model";
    throw std::runtime_error("no weight format fits this GPU for this model:" + why);
  }

  Plan p;
  p.weights = chosen->format;
  p.recipe = recipe_for(chosen->format);
  p.convert_at_load = ck.stored != chosen->format;
  p.prefill_math = dense_math(p.recipe.dense, d);
  p.batch_math = p.prefill_math;
  p.expert_math = expert_math(p.recipe.experts, d);
  p.max_slots = chosen->slots;
  p.kv_capacity = chosen->context;
  p.weight_bytes = chosen->wb.total();
  p.state_bytes = per_slot_state * static_cast<size_t>(p.max_slots);
  p.kv_bytes = per_position * static_cast<size_t>(p.kv_capacity) * static_cast<size_t>(p.max_slots);

  // The step: as many tokens as the room left allows, up to kMaxStepTokens; prefill takes the rest
  // of the step beyond one row per sequence.
  int tokens = kMaxStepTokens;
  auto total_with = [&](int t) {
    return p.weight_bytes + p.state_bytes + p.kv_bytes + scratch_bytes(c, t, p.max_slots);
  };
  while (tokens > kMinStepTokens && total_with(tokens) > budget) tokens /= 2;
  p.max_step_tokens = tokens;
  p.scratch_bytes = scratch_bytes(c, tokens, p.max_slots);
  p.prefill_chunk = std::max(1, tokens - p.max_slots);
  p.budget_bytes = budget;

  // The decode ceiling: one stream reads the layers' weights and the routed experts it uses once per
  // pass (the stack runs num_passes times over the same weights), the LM head once, one embedding row
  // (nothing, next to the rest), and reads and writes its Mamba state. Bandwidth bounds how fast that
  // can go.
  const WeightBytes& wb = chosen->wb;
  const double head = static_cast<double>(projection_bytes(recipe_for(chosen->format).dense, c.vocab_size, c.hidden_size));
  const double embedding = static_cast<double>(projection_bytes(LX_W_BF16, c.vocab_size, c.hidden_size));
  const double experts_read = static_cast<double>(wb.experts) * c.num_experts_per_tok / c.n_routed_experts;
  // weight_bytes counts a quantized head among the dense projections; a BF16 head is the embedding itself.
  const bool head_in_dense = recipe_for(chosen->format).dense != LX_W_BF16;
  const double per_pass = static_cast<double>(wb.plain + wb.dense + wb.latent_up + wb.out_proj) - embedding -
                          (head_in_dense ? head : 0.0) + experts_read;
  const double active = per_pass * c.num_passes + head + 2.0 * static_cast<double>(state_bytes_per_slot(c));
  if (d.bandwidth_gbs > 0 && active > 0) p.decode_ceiling_tps = d.bandwidth_gbs * 1e9 / active;

  // Notes, in the order a person reads them.
  auto& n = p.notes;
  if (req.weights >= 0) {
    n.push_back("Holding the weights as " + format_name(p.weights) + " because you asked for it.");
  } else {
    n.push_back("Holding the weights as " + format_name(p.weights) + ": the widest format that leaves room for " +
                std::to_string(std::min(useful_slots, p.max_slots)) + "+ sequences at " +
                std::to_string(std::min(kUsefulContext, p.kv_capacity)) + "+ tokens of context on this GPU.");
  }
  if (limited)
    n.push_back("This GPU is small for the model: it runs " + std::to_string(p.max_slots) + " sequence" +
                (p.max_slots == 1 ? "" : "s") + " of up to " + std::to_string(p.kv_capacity) +
                " tokens, and the other formats do not leave room for more.");
  for (const Fit& f : tried) {
    if (f.ok) break;
    n.push_back("  " + format_name(f.format) + " does not fit " + (limited ? "even that" : "usefully") + ": it needs " +
                gb(f.need) + ", " + gb(budget) + " is free for the model.");
  }
  n.push_back("Weights " + gb(p.weight_bytes) + " (dense " + format_name(p.recipe.dense) + ", experts " +
              format_name(p.recipe.experts) + (p.recipe.latent_up != p.recipe.dense ? ", latent_up " +
                                                                                       format_name(p.recipe.latent_up)
                                                                                 : std::string()) +
              ", Mamba out_proj BF16).");
  if (p.convert_at_load)
    n.push_back("The file is " + format_name(ck.stored) + "; it is converted to " + format_name(p.weights) +
                " when it loads.");
  n.push_back("Dense products multiply in " + math_name(p.prefill_math) + "; routed experts in " +
              math_name(p.expert_math) + ".");
  if (p.prefill_math == LX_MATH_BF16 && p.recipe.dense != LX_W_BF16)
    n.push_back("This GPU has no tensor-core instruction for " + format_name(p.recipe.dense) +
                " math, so the weights are expanded in registers and multiplied in BF16.");
  n.push_back("Up to " + std::to_string(p.max_slots) + " sequences, " + std::to_string(p.kv_capacity) +
              " tokens each (" + gb(p.state_bytes + p.kv_bytes) + " of state and cache).");
  n.push_back("A step takes up to " + std::to_string(p.max_step_tokens) + " tokens; prompts are prefilled " +
              std::to_string(p.prefill_chunk) + " tokens at a time.");
  if (p.decode_ceiling_tps > 0) {
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "One stream reads about %s per token, so bandwidth caps it near %.0f tokens per second.",
                  mb(static_cast<size_t>(active)).c_str(), p.decode_ceiling_tps);
    n.push_back(buf);
  }
  n.push_back("Memory: " + gb(p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes) + " planned of " +
              gb(budget) + " available.");
  return p;
}

}  // namespace lx::runtime
