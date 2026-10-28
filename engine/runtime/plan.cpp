// The plan: from the device, the model and its checkpoint, the weight format, the math each kind of
// product runs in, and how many sequences of what length fit. The weight accounting follows the loader
// (core/checkpoint.cpp: one buffer per projection, in the format its recipe holds it) and the model's own
// allocations (engine/model/metis.cpp: its constructor is the list); the tests check both against the
// checkpoints on disk and the device memory a load takes.
#include "runtime/plan.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdio>
#include <stdexcept>

#include "core/checkpoint.h"
#include "core/safetensors.h"
#include "lx_build_info.h"  // LX_BUILT_ARCHS, written by CMake
#include "runtime/offload_plan.h"

namespace lx::runtime {
namespace {

constexpr size_t kAlign = 256;  // Weights::upload and the model's allocations start on this boundary
constexpr size_t kMiB = 1024ull * 1024ull;

// Sized for the decisions the plan makes (kUsefulSlots and kUsefulContext are in plan.h). Automatic sizing aims
// for kDefaultContext and at most kMaxSlots sequences, and steps are at most kMaxStepTokens tokens.
constexpr int kLimitedContext = 2048;
constexpr int kDefaultContext = 8192;
constexpr int kMaxSlots = 64;
constexpr int kMinStepTokens = 512;
constexpr int kMaxStepTokens = 2048;
// What the plan keeps back beyond its own accounting: cuBLASLt's workspace and the allocator's slack.
constexpr size_t kReserveBytes = 256 * kMiB;
// Free memory the plan leaves untouched at startup, so another process that grows (or an allocator
// that needs slack) does not turn a working plan into an out-of-memory failure.
constexpr size_t kStartupMarginBytes = 1024 * kMiB;
// The prefix cache's shares: device memory up to a cap, and pinned host memory spilled into.
constexpr size_t kPrefixDeviceCap = 8192 * kMiB;
constexpr size_t kPrefixHostBytes = 4096 * kMiB;
// Decode-sized products (at most this many rows) run as GEMVs, and the decode experts' workspace is sized
// for them (metis.cpp, kGemvRows).
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
std::string tps(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.0f", v);
  return buf;
}
std::string math_name(lx_math m, const DeviceInfo& d) {
  if (d.cpu) {
    const bool native = d.native_math & (1 << m);
    switch (m) {
      case LX_MATH_BF16: return native ? "BF16 dot products" : "f32 (BF16 widened)";
      case LX_MATH_INT8: return native ? "INT8 dot products" : "INT8 (16-bit multiplies)";
      case LX_MATH_FP8: return "FP8 (exact products, summed in double)";
      case LX_MATH_FP4: return "FP4 (exact products, summed in double)";
    }
    return "?";
  }
  switch (m) {
    case LX_MATH_BF16: return "BF16 tensor cores";
    case LX_MATH_INT8: return "INT8 tensor cores";
    case LX_MATH_FP8: return "FP8 tensor cores";
    case LX_MATH_FP4: return "FP4 tensor cores";
  }
  return "?";
}
// The device as the notes name it.
const char* this_device(const DeviceInfo& d) { return d.cpu ? "this machine" : "this GPU"; }

// Bits per stored value: what "narrower" and "wider" mean between formats.
int width_of(lx_wtype w) {
  switch (w) {
    case LX_W_NVFP4: return 4;
    case LX_W_INT8:
    case LX_W_FP8: return 8;
    default: return 16;
  }
}

// A convenience for the accounting: one bf16 or f32 tensor, aligned as the upload aligns it.
size_t bf16_bytes(int64_t n) { return aligned(static_cast<size_t>(n) * 2); }
size_t f32_bytes(int64_t n) { return aligned(static_cast<size_t>(n) * 4); }

// Attention layers of the config, and the Mamba layers that remain.
int attention_layers(const MetisConfig& c) { return static_cast<int>(c.attention_layer_indices.size()); }
int mamba_layers(const MetisConfig& c) { return c.num_hidden_layers - attention_layers(c); }

// The kinds of projection ModelFormats names: the head and the dense projections, latent_up, the routed experts.
enum class Kind { Dense, LatentUp, Experts };

// Every projection the loader holds, by the name it holds it under (checkpoint.cpp), with its shape: E experts of
// [N, K] (E = 1 for a matrix). The tied head ("embedding") is not among them; its own copy is counted apart.
template <typename Fn>
void each_projection(const MetisConfig& c, Fn&& fn) {
  const int64_t D = c.hidden_size, Mm = c.memory_dim, R = c.route_feature_dim, E = c.n_routed_experts;
  const int64_t Lat = c.latent_dim, Iexp = c.expert_intermediate_size, Ish = c.shared_expert_intermediate_size;
  const int64_t I = c.mamba_intermediate(), H = c.mamba_heads(), G = c.mamba_n_groups, N = c.mamba_d_state;
  const int64_t proj = 2 * I + 2 * G * N + H;
  const int64_t AH = c.num_attention_heads, Dh = c.head_dim, KVH = c.num_key_value_heads;
  const int64_t qkv_width = (AH + 2 * KVH) * Dh, rank = c.attention_pass_lora_rank;
  for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
    const std::string p = "layers." + std::to_string(layer) + ".";
    if (c.is_attention(layer)) {
      fn(p + "mixer.qkv", Kind::Dense, 1, qkv_width, D);
      fn(p + "mixer.out", Kind::Dense, 1, D, AH * Dh);
      fn(p + "mixer.pass_lora_down", Kind::Dense, 1, rank, D);
      fn(p + "mixer.pass_lora_up", Kind::Dense, 1, qkv_width, rank);
    } else {
      fn(p + "mixer.impl.mixer.in_proj", Kind::Dense, 1, proj, D);
      fn(p + "mixer.impl.mixer.out_proj", Kind::Dense, 1, D, I);
    }
    fn(p + "moe.latent_down", Kind::Dense, 1, Lat, D);
    fn(p + "moe.latent_up", Kind::LatentUp, 1, D, Lat);
    fn(p + "moe.shared_expert.gate_up", Kind::Dense, 1, 2 * Ish, Lat);
    fn(p + "moe.shared_expert.down", Kind::Dense, 1, Lat, Ish);
    fn(p + "moe.local_experts.gate_up", Kind::Experts, E, 2 * Iexp, Lat);
    fn(p + "moe.local_experts.down", Kind::Experts, E, Lat, Iexp);
  }
  const std::string dm = "depth_memory.";
  fn(dm + "query", Kind::Dense, 1, Mm, D);
  fn(dm + "state_write", Kind::Dense, 1, Mm, D);
  fn(dm + "output", Kind::Dense, 1, D, Mm);
  fn(dm + "route_projection", Kind::Dense, 1, R, 3 * D + R);
  fn(dm + "key", Kind::Dense, 1, Mm, Mm);
  fn(dm + "value", Kind::Dense, 1, Mm, Mm);
  fn(dm + "metadata_write", Kind::Dense, 1, Mm, R + 4);
}

// Whether the BF16X encoding of rows of K values is smaller than the bf16 rows (metis.cpp, bf16x_smaller).
bool bf16x_smaller(int64_t K) {
  const int64_t row = lx_bf16x_row_bytes(static_cast<int>(K));
  return row > 0 && row < 2 * K;
}

// Per projection, what the loader holds and what decode reads: the BF16X copy where the model makes one
// (metis.cpp: make_linear for a matrix held in BF16 whose encoding is smaller; expert_stack for the stacks held in
// BF16), the held weight otherwise. A quantized target's BF16 tensors (the ones its recipe keeps wide) get their
// copies too, so the kinds are not consulted: only the format each projection is held in.
struct Held {
  size_t held = 0, copy = 0;
  size_t decode() const { return copy ? copy : held; }
};
Held held_bytes(lx_wtype target, bool bf16x_decode, const std::string& key, Kind kind, int64_t E, int64_t N,
                int64_t K) {
  const lx_wtype format = projection_format(target, key);
  Held h;
  h.held = projection_bytes(format, E * N, K, E > 1 ? E : 0);
  if (bf16x_decode && format == LX_W_BF16 && (kind == Kind::Experts || bf16x_smaller(K)))
    h.copy = projection_bytes(LX_W_BF16X, E * N, K);
  return h;
}

// The loader's plain tensors (every bf16 or f32 tensor that is not a projection), the bf16 gather table among
// them, and the hyper-connections' per-pass parameters the model computes from them.
struct Plain {
  size_t plain = 0, gather = 0, derived = 0;
};
Plain plain_bytes(const MetisConfig& c) {
  const int64_t V = c.vocab_size, D = c.hidden_size, S = c.n_streams, Mm = c.memory_dim, R = c.route_feature_dim;
  const int64_t E = c.n_routed_experts, Lat = c.latent_dim, P = c.num_passes, MP = c.max_passes;
  const int64_t I = c.mamba_intermediate(), H = c.mamba_heads(), G = c.mamba_n_groups, N = c.mamba_d_state;
  const int64_t W = c.mamba_d_conv, conv_dim = I + 2 * G * N, A = attention_layers(c);
  const int64_t E_mhc = c.mhc_pass_embedding_dim, SS2 = S * S + 2 * S;
  Plain out;
  size_t& b = out.plain;
  out.gather = bf16_bytes(V * D);  // the gather table: bf16 in every format (dequantized from a quantized file)
  b += out.gather;
  b += bf16_bytes(S * D) + bf16_bytes(D);  // stream embeddings, final norm
  b += bf16_bytes(MP * E_mhc);             // pass embeddings
  // depth memory: pass and anchor embeddings, the stream gate and its bias, the projections' biases
  b += bf16_bytes(MP * Mm) + bf16_bytes((A + 1) * Mm) + bf16_bytes(S * D) + bf16_bytes(S);
  b += 5 * bf16_bytes(Mm) + bf16_bytes(D) + bf16_bytes(R);  // key, value, metadata_write, query, state_write; output; route
  // the depth head: f32 hidden layer, its bias, the output row and its bias
  b += f32_bytes(R * (3 * D + R)) + f32_bytes(R) + f32_bytes(R) + f32_bytes(1);
  for (int layer = 0; layer < c.num_hidden_layers; ++layer) {
    b += 2 * bf16_bytes(D);  // mixer and moe connection norms
    // the two hyper-connections' controllers (counted as bf16), and the per-pass f32 mix, read and write the
    // model computes at load
    b += 2 * (bf16_bytes(SS2 * E_mhc) + bf16_bytes(SS2) + bf16_bytes(S * S) + 2 * bf16_bytes(S));
    out.derived += 2 * f32_bytes(P * SS2);
    // MoE expert embeddings, router and selection bias
    b += bf16_bytes(E * R) + f32_bytes(E * (Lat + R)) + f32_bytes(E) + f32_bytes(E);
    if (c.is_attention(layer)) {
      b += bf16_bytes(MP);  // pass gates
    } else {
      b += bf16_bytes(conv_dim * W) + bf16_bytes(conv_dim) + bf16_bytes(I);  // conv, conv bias, gated norm
      b += 3 * f32_bytes(H);                                                  // A_log, D, dt_bias
    }
  }
  return out;
}

}  // namespace

bool kernels_built_for(int cc) {
  static constexpr int kBuilt[] = {LX_BUILT_ARCHS};
  return std::find(std::begin(kBuilt), std::end(kBuilt), cc) != std::end(kBuilt);
}

std::vector<int> built_archs() {
  static constexpr int kBuilt[] = {LX_BUILT_ARCHS};
  return {std::begin(kBuilt), std::end(kBuilt)};
}

std::string format_name(lx_wtype w) {
  switch (w) {
    case LX_W_BF16: return "BF16";
    case LX_W_INT8: return "INT8";
    case LX_W_FP8: return "FP8";
    case LX_W_NVFP4: return "NVFP4";
    case LX_W_BF16X: return "BF16X";
  }
  return "?";
}

std::string weights_flag(lx_wtype w) {
  std::string name = format_name(w);
  for (char& ch : name) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  return name;
}

int weights_from_flag(const std::string& name) {
  for (const lx_wtype w : {LX_W_BF16, LX_W_INT8, LX_W_FP8, LX_W_NVFP4, LX_W_BF16X})
    if (name == weights_flag(w)) return static_cast<int>(w);
  throw std::runtime_error("--weights takes bf16, int8, fp8, nvfp4 or bf16x, not " + name);
}

bool bf16x_available() { return lx_bf16x_row_bytes(1024) > 0; }

WeightRecipe recipe_for(lx_wtype target) {
  WeightRecipe r;
  switch (target) {
    case LX_W_BF16:
      break;  // everything bf16
    case LX_W_INT8:
      r.dense = r.experts = r.latent_up = LX_W_INT8;
      break;
    case LX_W_FP8:
      // the published FP8 folders: experts, attention qkv and out, Mamba in_proj FP8; latent_up stays BF16
      r.dense = r.experts = LX_W_FP8;
      break;
    case LX_W_NVFP4:
      r.dense = LX_W_FP8;  // the GPTQ folder: dense projections and the head FP8, the experts the 4-bit part
      r.experts = LX_W_NVFP4;
      break;
    case LX_W_BF16X:
      r.dense = r.experts = r.latent_up = LX_W_BF16X;
      break;
  }
  return r;
}

// The sm_120 family: the block-scaled FP4 and mixed e2m1 x e4m3 instructions the GEMM kernels use (gemm_w.cuh,
// kind_for) exist there and nowhere else in this build.
static bool sm120_family(const DeviceInfo& d) { return d.cc() >= 120 && d.cc() < 130; }

// The default maths are BF16 for every format, on every device. KL against the FP32 reference on Praxis (tools/compare.py,
// top-20, full prefill, chat_think / chat_direct_tools / prose): INT8 maths 0.0116 / 0.0308 / 0.0072 against 0.0034 /
// 0.0156 / 0.0031 in BF16; FP8 maths 0.0075 / 0.0385 / 0.0080 against 0.0047 / 0.0319 / 0.0048; NVFP4 with FP8 dense
// and FP4 experts 0.0441 / 0.1367 / 0.0199 against 0.0245 / 0.0874 / 0.0107. BF16 maths give up 1 to 7 percent of
// decode at 32 streams and 10 to 19 percent of prefill, and every format still beats BF16 on both.
lx_math native_math(lx_wtype format, const DeviceInfo& d, bool fast) {
  if (!fast) return LX_MATH_BF16;
  switch (format) {
    case LX_W_INT8: return d.int8_mma ? LX_MATH_INT8 : LX_MATH_BF16;
    case LX_W_FP8: return d.fp8_mma ? LX_MATH_FP8 : LX_MATH_BF16;
    case LX_W_NVFP4: return sm120_family(d) ? LX_MATH_FP4 : LX_MATH_BF16;
    default: return LX_MATH_BF16;
  }
}

lx::ModelFormats model_formats(lx_wtype target, const DeviceInfo& d, bool bf16x_decode, bool fast) {
  const WeightRecipe r = recipe_for(target);
  lx::ModelFormats f;
  f.dense = r.dense;
  f.experts = r.experts;
  f.latent_up = r.latent_up;
  f.prefill_math = native_math(r.dense, d, fast);
  f.expert_math = native_math(r.experts, d, fast);
  f.bf16x_decode = bf16x_decode && target != LX_W_BF16X;
  return f;
}

lx::ModelFormats Plan::formats() const {
  lx::ModelFormats f;
  f.dense = recipe.dense;
  f.experts = recipe.experts;
  f.latent_up = recipe.latent_up;
  f.prefill_math = prefill_math;
  f.expert_math = expert_math;
  f.bf16x_decode = bf16x_decode;
  return f;
}

bool can_load(lx_wtype target, lx_wtype stored, const MetisConfig& config, std::string* how) {
  std::string why;
  bool ok = true;
  const std::string file = "the " + format_name(stored) + " file";
  if (target == stored) {
    why = "as stored";
  } else if (target == LX_W_NVFP4) {
    why = nvfp4_unavailable(stored, config);
    ok = false;
  } else if (target == LX_W_BF16X) {
    if (stored != LX_W_BF16) {
      why = "BF16X encodes BF16 weights, and this is " + file;
      ok = false;
    } else if (!bf16x_available()) {
      why = "the loaded backend has no BF16X kernels";
      ok = false;
    } else {
      why = "lossless encoding of " + file;
    }
  } else if (width_of(target) > width_of(stored)) {
    why = "wider than " + file + ": the same values in more bytes";
    ok = false;
  } else {
    why = "round-to-nearest from " + file;
  }
  if (how) *how = why;
  return ok;
}

size_t projection_bytes(lx_wtype format, int64_t rows, int64_t cols, int64_t stack_experts) {
  const size_t r = static_cast<size_t>(rows), k = static_cast<size_t>(cols);
  switch (format) {
    case LX_W_BF16:
      return aligned(r * k * 2);
    case LX_W_INT8:
    case LX_W_FP8:
      return aligned(r * k) + aligned(r * 4);
    case LX_W_NVFP4:
      // two e2m1 per byte, one e4m3 block scale per 16 values, and a stack's per-expert f32 globals
      return aligned(r * k / 2) + aligned(r * k / 16) +
             (stack_experts > 0 ? aligned(static_cast<size_t>(stack_experts) * 4) : 0);
    case LX_W_BF16X: {
      const int64_t row = lx_bf16x_row_bytes(static_cast<int>(cols));
      return row > 0 ? aligned(r * static_cast<size_t>(row)) : 0;
    }
  }
  return 0;
}

WeightBytes weight_bytes(const MetisConfig& c, lx_wtype target, bool bf16x_decode) {
  const bool copies = bf16x_decode && target != LX_W_BF16X;  // the lossless target is the encoding: no second copy
  WeightBytes wb;
  const Plain plain = plain_bytes(c);
  wb.plain = plain.plain;
  wb.derived = plain.derived;
  each_projection(c, [&](const std::string& key, Kind kind, int64_t E, int64_t N, int64_t K) {
    const Held h = held_bytes(target, copies, key, kind, E, N, K);
    (kind == Kind::Experts ? wb.experts : wb.projections) += h.held;
    wb.bf16x += h.copy;
  });
  // The tied head: a BF16 head reads the gather table (and has a BF16X copy of it with the decode copies); any
  // other format is its own copy of the table.
  const Held head = held_bytes(target, copies, "embedding", Kind::Dense, 1, c.vocab_size, c.hidden_size);
  if (projection_format(target, "embedding") != LX_W_BF16) wb.head = head.held;
  wb.bf16x += head.copy;
  return wb;
}

double decode_bytes_per_token(const MetisConfig& c, lx_wtype target, bool bf16x_decode) {
  const bool copies = bf16x_decode && target != LX_W_BF16X;
  const Plain plain = plain_bytes(c);
  // Per pass: every plain tensor but the gather table (one row of it per token is nothing next to the rest), every
  // projection as decode reads it, and the routed experts' share a token's top_k choices read.
  double per_pass = static_cast<double>(plain.plain - plain.gather + plain.derived);
  double experts = 0;
  each_projection(c, [&](const std::string& key, Kind kind, int64_t E, int64_t N, int64_t K) {
    const double bytes = static_cast<double>(held_bytes(target, copies, key, kind, E, N, K).decode());
    (kind == Kind::Experts ? experts : per_pass) += bytes;
  });
  per_pass += experts * c.num_experts_per_tok / c.n_routed_experts;
  // The head once per token: its own copy, or the gather table (or its BF16X copy) when it is BF16.
  const Held head = held_bytes(target, copies, "embedding", Kind::Dense, 1, c.vocab_size, c.hidden_size);
  const double head_read = static_cast<double>(head.decode());
  return per_pass * c.num_passes + head_read + 2.0 * static_cast<double>(state_bytes_per_slot(c));
}

// The share of the card's bandwidth one stream's decode reaches on each path: the tokens per second the kernels measure
// (Praxis on an RTX 5070 Ti, bench at one stream, 2026-10-24) times the bytes a token reads, over the card's bandwidth.
// The decode estimates scale the card's bandwidth by these, so a note never promises more than the kernels deliver.
double decode_efficiency(lx_wtype target, bool bf16x_decode) {
  constexpr double kBF16 = 0.62, kBF16X = 0.56, kINT8 = 0.48, kFP8 = 0.53, kNVFP4 = 0.49;
  switch (target) {
    case LX_W_BF16: return bf16x_decode ? kBF16X : kBF16;
    case LX_W_BF16X: return kBF16X;
    case LX_W_INT8: return kINT8;
    case LX_W_FP8: return kFP8;
    case LX_W_NVFP4: return kNVFP4;
    default: return kBF16;
  }
}

double decode_ceiling_tps(const DeviceInfo& d, const MetisConfig& c, lx_wtype target, bool bf16x_decode) {
  const double bytes = decode_bytes_per_token(c, target, bf16x_decode);
  return d.bandwidth_gbs > 0 && bytes > 0 ? d.bandwidth_gbs * 1e9 * decode_efficiency(target, bf16x_decode) / bytes : 0.0;
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

size_t scratch_bytes(const MetisConfig& c, int tokens, int slots, const lx::ModelFormats& formats) {
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
  // Weights held as BF16X (the lossless target) have no GEMM: a prefill product expands its weight into a bf16
  // scratch first, sized for the largest one, or an expert layer's two stacks together (metis.cpp, unpack).
  if (formats.dense == LX_W_BF16X || formats.experts == LX_W_BF16X || formats.latent_up == LX_W_BF16X) {
    size_t unpack = 0, gu = 0, dn = 0;
    each_projection(c, [&](const std::string& key, Kind kind, int64_t e, int64_t n, int64_t k) {
      if (projection_format(LX_W_BF16X, key) != LX_W_BF16X) return;
      const size_t bytes = static_cast<size_t>(e * n * k) * 2;
      if (kind != Kind::Experts) unpack = std::max(unpack, bytes);
      else if (key.find("gate_up") != std::string::npos) gu = aligned(bytes);
      else dn = bytes;
    });
    if (projection_format(LX_W_BF16X, "embedding") == LX_W_BF16X)
      unpack = std::max(unpack, static_cast<size_t>(c.vocab_size) * c.hidden_size * 2);
    add(std::max(unpack, gu + dn));
  }
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
  // The decode experts' workspace, for every format (the BF16X kernels' and the others' share it).
  const int iexp = static_cast<int>(Iexp), top_k = static_cast<int>(K);
  add(static_cast<size_t>(std::max(lx_moe_decode_workspace_bytes(kGemvRows, iexp, top_k),
                                   lx_moe_bf16x_workspace_bytes(kGemvRows, iexp, top_k))));
  // The prefill products' workspaces, for every math a product can run in: the plan's, the one FP4 falls back to
  // (FP8, for FP8 weights under FP4 math), and BF16. Most products have T rows; the depth memory's query and
  // output have one row per stream. The grouped expert kernel's is laid out for the ordered sum.
  {
    const int kmax = static_cast<int>(std::max({3 * D + R, I, proj, D, Lat, Ish, att, Mm}));
    const int rows = static_cast<int>(T), stream_rows = static_cast<int>(TS);
    std::vector<lx_math> dense_maths = {LX_MATH_BF16, formats.prefill_math};
    std::vector<lx_math> expert_maths = {LX_MATH_BF16, formats.expert_math};
    if (formats.prefill_math == LX_MATH_FP4) dense_maths.push_back(LX_MATH_FP8);
    if (formats.expert_math == LX_MATH_FP4) expert_maths.push_back(LX_MATH_FP8);
    int64_t ws = 0, grouped = 0;
    for (const lx_math math : dense_maths)
      ws = std::max({ws, lx_gemm_workspace_bytes(rows, kmax, math),
                     lx_gemm_workspace_bytes(stream_rows, static_cast<int>(std::max(D, Mm)), math)});
    for (const lx_math math : expert_maths)
      grouped = std::max(grouped, lx_moe_grouped_workspace_bytes(rows, static_cast<int>(Lat), iexp,
                                                                 static_cast<int>(E), top_k, math));
    add(static_cast<size_t>(ws));
    add(static_cast<size_t>(grouped));
  }
  add((T + (S + 1) + 5 * S) * 4);                                     // the step's int32 metadata
  return b;
}

CheckpointInfo inspect_checkpoint(const std::string& model_dir) {
  const SafetensorsSet set = SafetensorsSet::open(model_dir);
  CheckpointInfo info;
  info.bytes = set.total_bytes();
  info.stored = ModelWeights::stored_format(model_dir);  // the loader's own reading of the folder
  char buf[160];
  std::snprintf(buf, sizeof buf, "%s checkpoint, %s on disk", format_name(info.stored).c_str(), gb(info.bytes).c_str());
  info.description = buf;
  return info;
}

namespace {

// The budget a resident plan may fill: the request's share of the free memory, no more than the VRAM budget
// (which counts this process's CUDA context too), less what the plan keeps back.
size_t resident_budget(const DeviceInfo& d, const PlanRequest& req) {
  size_t free_budget = static_cast<size_t>(static_cast<double>(d.free_bytes) * req.memory_fraction);
  if (req.vram_budget > 0)
    free_budget = std::min(free_budget, req.vram_budget > kCudaContextBytes ? req.vram_budget - kCudaContextBytes : 0);
  return free_budget > kReserveBytes ? free_budget - kReserveBytes : 0;
}

// What each kind holds in a target, for the notes.
std::string recipe_text(lx_wtype target) {
  switch (target) {
    case LX_W_BF16: return "every projection BF16";
    case LX_W_INT8: return "projections INT8; Mamba out_proj, the pass LoRA and three depth-memory projections BF16";
    case LX_W_FP8: return "experts, attention and Mamba in_proj FP8, the rest BF16 (the published FP8 recipe)";
    case LX_W_NVFP4: return "experts NVFP4, the dense projections and the head FP8, latent_up and a few small ones BF16";
    case LX_W_BF16X: return "projections in the lossless BF16X encoding, the few small ones BF16";
  }
  return "";
}

}  // namespace

// The products' maths as lernex doctor names them: one name, or the dense and the expert ones.
std::string maths_text(const lx::ModelFormats& f) {
  auto name = [](lx_math m) {
    return m == LX_MATH_INT8 ? "INT8" : m == LX_MATH_FP8 ? "FP8" : m == LX_MATH_FP4 ? "FP4" : "BF16";
  };
  if (f.prefill_math == f.expert_math) return name(f.prefill_math);
  return std::string("dense ") + name(f.prefill_math) + ", experts " + name(f.expert_math);
}

Plan make_plan(const DeviceInfo& d, const MetisConfig& c, const CheckpointInfo& ck, const PlanRequest& req) {
  if (!d.built)
    throw std::runtime_error("this build has no kernels for sm_" + std::to_string(d.real_cc()) +
                             "; rebuild with -DLX_ARCHS=all to include every architecture the engine runs on");

  // The format: the file's own, or the one the user named. Never wider than the file, and NVFP4 only from a
  // stored NVFP4 folder (can_load); nothing narrower is ever picked for the user.
  const lx_wtype stored = ck.stored;
  const bool asked = req.weights >= 0;
  const lx_wtype target = asked ? static_cast<lx_wtype>(req.weights) : stored;
  std::string how;
  if (!can_load(target, stored, c, &how)) throw std::runtime_error(format_name(target) + " cannot be planned: " + how);

  const size_t budget = resident_budget(d, req);
  const size_t per_slot_state = state_bytes_per_slot(c);
  const size_t per_position = kv_bytes_per_position(c);
  const int context_cap = c.max_position_embeddings > 0 ? c.max_position_embeddings : kDefaultContext;
  const int slot_cap = req.sequences > 0 ? req.sequences : kMaxSlots;
  const int context_wanted = std::clamp(req.context > 0 ? req.context : kDefaultContext, 1, context_cap);
  // Fitting usefully takes kUsefulSlots sequences (or as many as were asked for, when fewer); the plan then
  // gives as many as fit, up to what was asked.
  const int useful_slots = req.sequences > 0 ? std::min(req.sequences, kUsefulSlots) : kUsefulSlots;

  // One format at one tier: the largest context (from context_wanted, halving down to floor_ctx) at which
  // at least want_slots sequences fit beside the weights and a step. The full step (kMaxStepTokens) is kept
  // when the useful sequences fit beside it, so extra sequences never shrink prefill to its smallest step;
  // otherwise the smallest step (kMinStepTokens) is taken.
  struct Fit {
    lx_wtype format = LX_W_BF16;
    bool bf16x = false;
    lx::ModelFormats formats;
    WeightBytes wb;
    size_t need = 0;  // the bytes the format needs for want_slots at floor_ctx
    int context = 0, slots = 0;
    bool ok = false;
  };
  auto attempt = [&](lx_wtype format, bool bf16x, int want_slots, int floor_ctx) {
    Fit f;
    f.format = format;
    f.bf16x = bf16x;
    f.formats = model_formats(format, d, bf16x, req.fast_math);
    f.wb = weight_bytes(c, format, bf16x);
    f.need = f.wb.total() + scratch_bytes(c, kMinStepTokens, 0, f.formats) +
             static_cast<size_t>(want_slots) * (per_slot_state + per_position * static_cast<size_t>(floor_ctx));
    const int need_slots = std::min(want_slots, slot_cap);
    for (int ctx = context_wanted;; ctx = std::max(floor_ctx, ctx / 2)) {
      const size_t per_slot = per_slot_state + per_position * static_cast<size_t>(ctx);
      for (const int step : {kMaxStepTokens, kMinStepTokens}) {
        const size_t fixed = f.wb.total() + scratch_bytes(c, step, 0, f.formats);
        if (budget <= fixed || per_slot == 0) continue;
        int slots = static_cast<int>(std::min<size_t>((budget - fixed) / per_slot, static_cast<size_t>(slot_cap)));
        // the scratch grows with the slots' metadata too; take back any slot that no longer fits
        while (slots > 0 && fixed + scratch_bytes(c, step, slots, f.formats) - scratch_bytes(c, step, 0, f.formats) +
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

  // Every format, for lernex doctor and for the notes that name a faster one.
  const bool bf16x_kernels = bf16x_available();
  std::vector<FormatOption> options;
  for (const lx_wtype f : {LX_W_BF16, LX_W_BF16X, LX_W_INT8, LX_W_FP8, LX_W_NVFP4}) {
    FormatOption o;
    o.format = f;
    o.loadable = can_load(f, stored, c, &o.load);
    if (f == LX_W_BF16X && !o.loadable && stored != LX_W_BF16) continue;  // only a BF16 file has a BF16X form
    o.weight_bytes = weight_bytes(c, f).total();
    if (f != LX_W_BF16X && bf16x_kernels) o.bf16x_bytes = weight_bytes(c, f, true).bf16x;
    o.fits = attempt(f, false, useful_slots, kUsefulContext).ok;
    o.decode_ceiling_tps = decode_ceiling_tps(d, c, f, f != LX_W_BF16X && bf16x_kernels);
    o.maths = maths_text(model_formats(f, d, false));
    const std::string fast = maths_text(model_formats(f, d, false, true));
    o.fast_maths = fast == o.maths ? std::string("-") : fast;
    options.push_back(o);
  }
  // The narrower format to name when this plan is slower than it needs to be: of the ones this file loads into
  // and this GPU holds usefully, the 8-bit one the GPU multiplies natively (FP8 on sm_89 and up, INT8 below),
  // else any narrower that fits.
  auto faster = [&](double here) -> const FormatOption* {
    const FormatOption* best = nullptr;
    for (const FormatOption& o : options) {
      if (!o.loadable || !o.fits || width_of(o.format) >= width_of(target) || o.decode_ceiling_tps <= here * 1.1) continue;  // 10%: a note names a format only when it is clearly faster
      const bool native = o.format == (d.fp8_mma ? LX_W_FP8 : LX_W_INT8);
      if (!best || native) best = &o;
      if (native) break;
    }
    return best;
  };
  auto describe_load = [&]() {
    if (!asked) return std::string("the format the file stores");
    if (target == stored) return std::string("as asked, the format the file stores");
    return "as asked: " + how + " at load";
  };

  // Resident first, with the BF16X decode copies of the BF16 tensors the format holds (every BF16 weight of a BF16
  // model, the ones a quantized recipe keeps wide) when they fit beside the useful sequences.
  const size_t copies = target == LX_W_BF16X ? 0 : weight_bytes(c, target, true).bf16x;
  const bool copies_wanted = copies > 0 && req.bf16x && bf16x_kernels;
  std::string bf16x_reason;
  Fit chosen;
  Fit plain_fit;  // the same format without the copies, to say what they cost
  if (copies_wanted) chosen = attempt(target, true, useful_slots, kUsefulContext);
  if (copies_wanted) plain_fit = attempt(target, false, useful_slots, kUsefulContext);
  if (!chosen.ok) chosen = attempt(target, false, useful_slots, kUsefulContext);
  if (copies > 0) {
    const std::string kept =
        target == LX_W_BF16 ? "the BF16 weights" : "the BF16 tensors " + format_name(target) + " keeps wide";
    if (!req.bf16x)
      bf16x_reason = "BF16X decode copies off (--no-bf16x): decode reads the BF16 weights.";
    else if (!bf16x_kernels)
      bf16x_reason = "No BF16X decode copies: the loaded backend has no BF16X kernels.";
    else if (chosen.ok && chosen.bf16x)
      bf16x_reason = "Decode reads lossless BF16X copies of " + kept + " (" + gb(copies) +
                     " more): the same logits to the bit from about 1.3x fewer bytes. Prefill reads the BF16 copy.";
    else
      bf16x_reason = "BF16X decode copies dropped: their " + gb(copies) +
                     " does not fit beside the useful sequences; decode reads the BF16 weights.";
  }

  // Not resident usefully: offload the routed experts in the same format before anything narrower is considered.
  std::string offload_why;
  if (!chosen.ok) {
    if (target == LX_W_BF16 || target == LX_W_INT8 || target == LX_W_FP8) {
      try {
        PlanRequest same = req;
        same.weights = target;
        Plan p = make_offload_plan(d, c, ck, same);
        p.stored = stored;
        p.options = options;
        const FormatOption* self = nullptr;
        for (const FormatOption& o : options)
          if (o.format == target) self = &o;
        p.reason = describe_load();
        if (copies > 0)
          p.bf16x_reason = req.bf16x && bf16x_kernels
                               ? "No BF16X decode copies: offloaded, the VRAM goes to the expert cache."
                               : bf16x_reason;
        std::vector<std::string> notes;
        notes.push_back("Holding the weights as " + format_name(target) + ", " + p.reason + ".");
        notes.push_back("It does not fit this GPU usefully (" + gb(self ? self->weight_bytes : 0) + " of weights, " +
                        gb(budget) + " free for the model), so its routed experts are offloaded:");
        for (const std::string& n : p.notes) notes.push_back("  " + n);
        notes.push_back("Weights in VRAM " + gb(p.weight_bytes) + " (" + recipe_text(target) + "), one sequence of " +
                        std::to_string(p.kv_capacity) + " tokens.");
        if (!p.bf16x_reason.empty()) notes.push_back(p.bf16x_reason);
        notes.push_back("Dense products multiply in " + math_name(p.prefill_math, d) +
                        "; offloaded experts weight-only, in BF16 math.");
        if (const FormatOption* o = faster(0.0))
          notes.push_back("--weights " + weights_flag(o->format) + " would fit on the GPU (" +
                          gb(o->weight_bytes) + ") and decode at about " + tps(o->decode_ceiling_tps) +
                          " tok/s, with every expert resident (" + o->load + ").");
        p.notes = std::move(notes);
        return p;
      } catch (const std::runtime_error& e) {
        offload_why = e.what();
      }
    } else {
      offload_why = "offloaded experts run in BF16, INT8 or FP8, not " + format_name(target);
    }
  }
  // Then the limited tier: one sequence of kLimitedContext positions, said so in the notes.
  bool limited = false;
  if (!chosen.ok) {
    chosen = attempt(target, false, 1, kLimitedContext);
    limited = chosen.ok;
  }
  if (!chosen.ok) {
    std::string why = "\n  " + format_name(target) + " needs " + gb(chosen.need) + " for one sequence of " +
                      std::to_string(kLimitedContext) + " tokens; " + gb(budget) + " is free for the model";
    if (!offload_why.empty()) why += "\n  and offloading the experts does not help: " + offload_why;
    for (const FormatOption& o : options)
      if (o.fits && o.format != target)
        why += "\n  " + format_name(o.format) + " fits (" + gb(o.weight_bytes) + "): " +
               (o.loadable ? "--weights " + weights_flag(o.format) + ", " + o.load : o.load);
    throw std::runtime_error("the model does not run on " + std::string(this_device(d)) + " as " + format_name(target) + ":" + why);
  }

  Plan p;
  p.stored = stored;
  p.weights = target;
  p.recipe = recipe_for(target);
  p.convert_at_load = stored != target;
  p.bf16x_decode = chosen.bf16x;
  p.prefill_math = chosen.formats.prefill_math;
  p.batch_math = p.prefill_math;
  p.expert_math = chosen.formats.expert_math;
  p.max_slots = chosen.slots;
  p.kv_capacity = chosen.context;
  p.weight_bytes = chosen.wb.total();
  p.bf16x_bytes = chosen.wb.bf16x;
  p.state_bytes = per_slot_state * static_cast<size_t>(p.max_slots);
  p.kv_bytes = per_position * static_cast<size_t>(p.kv_capacity) * static_cast<size_t>(p.max_slots);
  p.options = options;
  p.offload.vram_budget = req.vram_budget;
  p.offload.ram_budget = req.ram_budget;

  // The step: as many tokens as the room left allows, up to kMaxStepTokens; prefill takes the rest
  // of the step beyond one row per sequence.
  int tokens = kMaxStepTokens;
  auto total_with = [&](int t) {
    return p.weight_bytes + p.state_bytes + p.kv_bytes + scratch_bytes(c, t, p.max_slots, chosen.formats);
  };
  while (tokens > kMinStepTokens && total_with(tokens) > budget) tokens /= 2;
  p.max_step_tokens = tokens;
  p.scratch_bytes = scratch_bytes(c, tokens, p.max_slots, chosen.formats);
  p.prefill_chunk = std::max(1, tokens - p.max_slots);
  p.budget_bytes = budget;
  {
    const size_t planned_bytes = total_with(tokens);
    const size_t spare = budget > planned_bytes + kStartupMarginBytes ? budget - planned_bytes - kStartupMarginBytes : 0;
    p.prefix_device_bytes = std::min(spare, kPrefixDeviceCap);
    p.prefix_host_bytes = kPrefixHostBytes;
  }
  const double active = decode_bytes_per_token(c, target, p.bf16x_decode);
  if (d.bandwidth_gbs > 0 && active > 0)
    p.decode_ceiling_tps = d.bandwidth_gbs * 1e9 * decode_efficiency(target, p.bf16x_decode) / active;

  // Notes, in the order a person reads them.
  p.reason = describe_load();
  p.bf16x_reason = bf16x_reason;
  auto& n = p.notes;
  n.push_back("Holding the weights as " + format_name(target) + ", " + p.reason + ".");
  if (limited)
    n.push_back(std::string(d.cpu ? "This machine's memory" : "This GPU") + " is small for the model: it runs " + std::to_string(p.max_slots) + " sequence" +
                (p.max_slots == 1 ? "" : "s") + " of up to " + std::to_string(p.kv_capacity) + " tokens" +
                (offload_why.empty() ? "" : " (offloading the experts does not apply: " + offload_why + ")") + ".");
  n.push_back("Weights " + gb(p.weight_bytes) + ": " + recipe_text(target) + ".");
  if (!p.bf16x_reason.empty()) n.push_back(p.bf16x_reason);
  if (p.bf16x_decode && plain_fit.ok && (plain_fit.slots > p.max_slots || plain_fit.context > p.kv_capacity))
    n.push_back("  The copies cost " + std::to_string(plain_fit.slots) + " sequences of " +
                std::to_string(plain_fit.context) + " tokens down to " + std::to_string(p.max_slots) + " of " +
                std::to_string(p.kv_capacity) + "; --no-bf16x keeps them.");
  n.push_back("Dense products multiply in " + math_name(p.prefill_math, d) + "; routed experts in " +
              math_name(p.expert_math, d) + ".");
  if (req.fast_math && p.prefill_math == LX_MATH_BF16 && p.recipe.dense != LX_W_BF16 && p.recipe.dense != LX_W_BF16X)
    n.push_back(std::string(d.cpu ? "This CPU has no instruction for " : "This GPU has no tensor-core instruction for ") +
                format_name(p.recipe.dense) + " math, so the weights are expanded in registers and multiplied in " +
                (d.cpu ? "f32." : "BF16."));
  n.push_back("Up to " + std::to_string(p.max_slots) + " sequences, " + std::to_string(p.kv_capacity) +
              " tokens each (" + gb(p.state_bytes + p.kv_bytes) + " of state and cache)" +
              (req.sequences > p.max_slots ? ": " + std::to_string(req.sequences) + " were asked for" : "") + ".");
  n.push_back("A step takes up to " + std::to_string(p.max_step_tokens) + " tokens; prompts are prefilled " +
              std::to_string(p.prefill_chunk) + " tokens at a time.");
  if (p.decode_ceiling_tps > 0) {
    char buf[200];
    std::snprintf(buf, sizeof buf,
                  "One stream reads about %s per token, so it decodes at about %.0f tokens per second.",
                  mb(static_cast<size_t>(active)).c_str(), p.decode_ceiling_tps);
    n.push_back(buf);
  }
  if (!req.fast_math) {
    const std::string fast = maths_text(model_formats(target, d, p.bf16x_decode, true));
    if (fast != maths_text(p.formats()))
      n.push_back("--fast-math multiplies in " + fast + " instead: faster prefill, less accurate than BF16.");
  }
  if (const FormatOption* o = faster(p.decode_ceiling_tps))
    n.push_back("--weights " + weights_flag(o->format) +
                " would decode at about " + tps(o->decode_ceiling_tps) + " tok/s against " +
                tps(p.decode_ceiling_tps) + " here (" + o->load + ").");
  n.push_back("Memory: " + gb(p.weight_bytes + p.state_bytes + p.kv_bytes + p.scratch_bytes) + " planned of " +
              gb(budget) + " available.");
  return p;
}

}  // namespace lx::runtime
