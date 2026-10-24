// The latent mixture of experts (kernels/moe.metal): routing, the summary, and the routed experts.
#include <algorithm>
#include <cstdio>

#include "cuda/tunable.h"
#include "metal/backend.h"
#include "metal/gemm.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kMaxTopK = 32;
constexpr int kRouteGroups = 4;
constexpr int kExpertRows = 4;
constexpr int kExpertGroups = 4;
constexpr int kDecodeTokens = 8;

// Variant index: format, then x vector loads / scalar (0: vector, unaligned x; 1: vector; 2: scalar).
std::string up_name(int i) {
  static const char* tail[3] = {"00", "10", "01"};
  return std::string("expert_up_") + std::to_string(i / 3) + "_" + tail[i % 3];
}
std::string down_name(int i) {
  static const char* tail[3] = {"00", "10", "01"};
  return std::string("expert_down_") + std::to_string(i / 3) + "_" + tail[i % 3];
}
KernelSet<12> g_up(up_name), g_down(down_name);

int values_per_chunk(lx_wtype t) { return t == LX_W_BF16 ? 8 : t == LX_W_NVFP4 ? 32 : 16; }
bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// The variant for rows of K values of w, against x rows at x of K bf16 values.
int expert_variant(const lx_weight* w, int K, const void* x) {
  const bool scalar = K % values_per_chunk(w->type) != 0 || !aligned16(w->data);
  if (scalar) return int(w->type) * 3 + 2;
  const bool xv = aligned16(x) && K % 8 == 0;
  return int(w->type) * 3 + (xv ? 1 : 0);
}

bool valid_stack(const lx_weight* w, int N, int K) {
  if (!w || !w->data || w->type < LX_W_BF16 || w->type > LX_W_NVFP4) return false;
  if (w->type != LX_W_BF16 && !w->scale) return false;
  if (w->type == LX_W_NVFP4 && K % 16 != 0) return false;
  return N > 0 && K > 0;
}

// A stack's weight as the kernels see it. The height of a stack (E) is not passed, so its extents are
// unknown: the registry finds the buffer from the base address.
WeightRef stack_ref(void* stream, const lx_weight* w) {
  return WeightRef{addr(stream, w->data, kUnknownExtent), addr(stream, w->scale, w->scale ? kUnknownExtent : 0),
                   addr(stream, w->expert_global, w->expert_global ? kUnknownExtent : 0),
                   w->type == LX_W_NVFP4 ? w->global : 1.0f, w->type};
}

int decode_experts(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                   const lx_weight* down, float* out, void* workspace, int T, int L, int I, int top_k,
                   void* stream) {
  static Kernel links_k("expert_links");
  const int P = T * top_k;
  const int64_t inter_bytes = round256(int64_t(P) * I * 2);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  void* inter = ws;
  int32_t* links = reinterpret_cast<int32_t*>(ws + inter_bytes);

  LinkArgs la{addr(stream, ids, size_t(P) * 4), addr_w(stream, links, size_t(P) * 8), P};
  if (int e = links_k.launch(stream, la, size1(ceil_div(P, 256)), size1(256))) return e;

  ExpertUpArgs ua{};
  ua.w = stack_ref(stream, gate_up);
  ua.latent = addr(stream, latent, size_t(T) * L * 2);
  ua.ids = addr(stream, ids, size_t(P) * 4);
  ua.links = addr(stream, links, size_t(P) * 8);
  ua.inter = addr_w(stream, inter, size_t(P) * I * 2);
  ua.L = L;
  ua.I = I;
  ua.top_k = top_k;
  ua.P = P;
  const int cols = kExpertRows * kExpertGroups;
  if (int e = g_up.launch(expert_variant(gate_up, L, latent), stream, ua, size2(ceil_div(I, cols), P),
                          size1(32 * kExpertGroups)))
    return e;

  ExpertDownArgs da{};
  da.w = stack_ref(stream, down);
  da.inter = addr(stream, inter, size_t(P) * I * 2);
  da.ids = addr(stream, ids, size_t(P) * 4);
  da.weights = addr(stream, weights, size_t(P) * 4);
  da.out = addr_w(stream, out, size_t(T) * L * 4);
  da.T = T;
  da.L = L;
  da.I = I;
  da.top_k = top_k;
  return g_down.launch(expert_variant(down, I, inter), stream, da, size2(ceil_div(L, cols), T),
                       size1(32 * kExpertGroups));
}

// Scratch for the entry points that take no workspace (lx_moe_experts), one per stream. Old buffers
// are kept when it grows: a recorded step may still point into them.
void* stream_scratch(void* stream, size_t bytes) {
  struct Scratch {
    void* ptr = nullptr;
    size_t bytes = 0;
    std::vector<void*> old;
  };
  static std::mutex mu;
  static std::unordered_map<Stream*, Scratch> scratch;
  std::lock_guard<std::mutex> g(mu);
  Scratch& s = scratch[stream_of(stream)];
  if (s.bytes < bytes) {
    if (s.ptr) s.old.push_back(s.ptr);
    void* p = nullptr;
    const size_t want = std::max<size_t>(bytes, size_t{1} << 20);
    if (Device::get().alloc(&p, want) != kOk) return nullptr;
    s.ptr = p;
    s.bytes = want;
  }
  return s.ptr;
}

// The grouped form's workspace: the grouping, the (quantized) latent, the intermediate and its
// quantized copy, and the per-pair down rows.
struct GroupedLayout {
  int P = 0, max_tiles = 0;
  int64_t counts = 0, order = 0, tiles = 0, latent_q = 0, inter = 0, inter_q = 0, rows = 0, bytes = 0;
  QuantizedRows lq, iq;
};
GroupedLayout grouped_layout(int T, int L, int I, int E, int top_k, lx_math math) {
  GroupedLayout g;
  g.P = T * top_k;
  g.max_tiles = (g.P + kTileM - 1) / kTileM + E;
  int64_t at = 0;
  g.counts = at;
  at += round256(int64_t(E + 1) * 4);
  g.order = at;
  at += round256(int64_t(g.P) * 4);
  g.tiles = at;
  at += round256(int64_t(1 + 3 * g.max_tiles) * 4);
  g.lq = quantized_rows(T, L, math);
  g.latent_q = at;
  at += g.lq.bytes;
  g.inter = at;
  at += round256(int64_t(g.P) * I * 2);
  g.iq = quantized_rows(g.P, I, math);
  g.inter_q = at;
  at += g.iq.bytes;
  g.rows = at;
  at += round256(int64_t(g.P) * L * 4);
  g.bytes = at;
  return g;
}

}  // namespace

extern "C" {

int lx_moe_route(const float* logits, const float* selection_bias, int32_t* ids, float* weights, float* entropy,
                 float* confidence, int T, int E, int top_k, void* stream) {
  static Kernel k("moe_route");
  if (T <= 0) return kOk;
  if (E <= 0 || top_k < 1 || top_k > E || top_k > kMaxTopK) return kInvalidValue;
  RouteArgs a{addr(stream, logits, size_t(T) * E * 4), addr(stream, selection_bias, selection_bias ? size_t(E) * 4 : 0),
              addr_w(stream, ids, size_t(T) * top_k * 4), addr_w(stream, weights, size_t(T) * top_k * 4),
              addr_w(stream, entropy, size_t(T) * 4), addr_w(stream, confidence, size_t(T) * 4), T, E, top_k};
  return k.launch(stream, a, size1(ceil_div(T, kRouteGroups)), size1(32 * kRouteGroups));
}

int lx_moe_summary(const int32_t* ids, const float* weights, const void* expert_emb, float* summary, int T, int R,
                   int top_k, void* stream) {
  static Kernel k("moe_summary");
  if (T <= 0 || R <= 0) return kOk;
  if (top_k < 1) return kInvalidValue;
  SummaryArgs a{addr(stream, ids, size_t(T) * top_k * 4), addr(stream, weights, size_t(T) * top_k * 4),
                addr(stream, expert_emb, kUnknownExtent), addr_w(stream, summary, size_t(T) * R * 4), T, R, top_k};
  return k.launch(stream, a, size2(ceil_div(R, 64), T), size2(64, 1));
}

int64_t lx_moe_decode_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t P = int64_t(T) * top_k;
  return round256(P * I * 2) + round256(P * 8);
}

int64_t lx_moe_workspace_bytes(int T, int I, int top_k) { return lx_moe_decode_workspace_bytes(T, I, top_k); }

int lx_moe_experts_decode(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                          const lx_weight* down, float* out, void* workspace, int T, int L, int I, int top_k,
                          void* stream) {
  if (T <= 0) return kOk;
  if (T > kDecodeTokens || top_k < 1 || !workspace) return kInvalidValue;
  if (!valid_stack(gate_up, 2 * I, L) || !valid_stack(down, L, I)) return kInvalidValue;
  return decode_experts(latent, ids, weights, gate_up, down, out, workspace, T, L, I, top_k, stream);
}

int lx_moe_experts_w8(const void* latent, const int32_t* ids, const float* weights, const int8_t* gate_up,
                      const float* gate_up_scale, const int8_t* down, const float* down_scale, float* out,
                      void* workspace, int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return kOk;
  if (top_k < 1 || L <= 0 || I <= 0 || !workspace || !gate_up_scale || !down_scale) return kInvalidValue;
  const lx_weight gu{LX_W_INT8, gate_up, gate_up_scale, 1.0f, nullptr};
  const lx_weight dn{LX_W_INT8, down, down_scale, 1.0f, nullptr};
  return decode_experts(latent, ids, weights, &gu, &dn, out, workspace, T, L, I, top_k, stream);
}

int lx_moe_experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                   const void* down, float* out, int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return kOk;
  if (top_k < 1 || L <= 0 || I <= 0) return kInvalidValue;
  void* ws = stream_scratch(stream, size_t(lx_moe_decode_workspace_bytes(T, I, top_k)));
  if (!ws) return kOutOfMemory;
  const lx_weight gu{LX_W_BF16, gate_up, nullptr, 1.0f, nullptr};
  const lx_weight dn{LX_W_BF16, down, nullptr, 1.0f, nullptr};
  return decode_experts(latent, ids, weights, &gu, &dn, out, ws, T, L, I, top_k, stream);
}

int64_t lx_moe_grouped_workspace_bytes(int T, int L, int I, int E, int top_k, lx_math math) {
  if (T <= 0 || L <= 0 || I <= 0 || E <= 0 || top_k <= 0) return 0;
  return grouped_layout(T, L, I, E, top_k, math).bytes;
}

int lx_moe_experts_grouped(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                           const lx_weight* down, lx_math math, float* out, void* workspace, int T, int L, int I,
                           int E, int top_k, void* stream) {
  static Kernel group_k("moe_group"), combine_k("moe_combine");
  if (T <= 0) return kOk;
  if (L <= 0 || I <= 0 || E <= 0 || top_k < 1 || top_k > E || !workspace) return kInvalidValue;
  if (!valid_stack(gate_up, 2 * I, L) || !valid_stack(down, L, I)) return kInvalidValue;
  if (math < LX_MATH_BF16 || math > LX_MATH_FP4) return kInvalidValue;
  // As on CUDA: the intermediate's width is a whole number of 8-wide steps.
  if (I % 8 != 0) return kNotSupported;
  if (!math_supported(gate_up->type, math) || !math_supported(down->type, math)) return kNotSupported;
  if (math == LX_MATH_FP4 && (L % 16 != 0 || I % 16 != 0)) return kNotSupported;
  if ((2 * E + 1024) * 4 > 32 * 1024) return kInvalidValue;

  const GroupedLayout g = grouped_layout(T, L, I, E, top_k, math);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  const int P = g.P;

  GroupArgs ga{addr(stream, ids, size_t(P) * 4), addr_w(stream, ws + g.counts, size_t(E + 1) * 4),
               addr_w(stream, ws + g.order, size_t(P) * 4), addr_w(stream, ws + g.tiles, size_t(1 + 3 * g.max_tiles) * 4),
               P, E, kTileM, g.max_tiles};
  if (int e = group_k.launch(stream, ga, size1(1), size1(1024), NSUInteger(2 * E + 1024) * 4)) return e;

  const bool quantized = math != LX_MATH_BF16;
  if (quantized) {
    uint8_t* q = ws + g.latent_q;
    if (int e = lx_quantize_act(latent, LX_BF16, L, math, q, g.lq.ld, reinterpret_cast<float*>(q + g.lq.scale_at),
                                q + g.lq.blocks_at, g.lq.ld_blocks, T, L, stream))
      return e;
  }
  const int af = activation_format(math, LX_BF16);

  // Stage 1: inter[pair] = bf16(silu(gate . x) * (up . x)) for every pair, by expert tiles.
  GemmArgs a{};
  a.w = stack_ref(stream, gate_up);
  if (quantized) {
    uint8_t* q = ws + g.latent_q;
    a.x = addr(stream, q, size_t(g.lq.scale_at));
    a.x_scale = addr(stream, q + g.lq.scale_at, size_t(T) * 4);
    a.x_block = math == LX_MATH_FP4 ? addr(stream, q + g.lq.blocks_at, size_t(T) * g.lq.ld_blocks) : 0;
    a.ld_x = g.lq.ld;
    a.ld_xb = g.lq.ld_blocks;
    a.x_vec = 1;
  } else {
    a.x = addr(stream, latent, size_t(T) * L * 2);
    a.ld_x = L;
    a.x_vec = (reinterpret_cast<uintptr_t>(latent) & 15) == 0 && L % 8 == 0;
  }
  a.y = addr_w(stream, ws + g.inter, size_t(P) * I * 2);
  a.order = addr(stream, ws + g.order, size_t(P) * 4);
  a.tiles = addr(stream, ws + g.tiles, size_t(1 + 3 * g.max_tiles) * 4);
  a.N = I;
  a.K = L;
  a.top_k = top_k;
  a.stack_n = 2 * I;
  if (int e = launch_gemm(af, gate_up->type, 1, stream, a, size2(ceil_div(I, kTileN / 2), NSUInteger(g.max_tiles))))
    return e;

  // Stage 2: rows[pair] = weight * (inter[pair] . down[l]), then the sum over each token's pairs.
  GemmArgs b{};
  b.w = stack_ref(stream, down);
  if (quantized) {
    uint8_t* q = ws + g.inter_q;
    if (int e = lx_quantize_act(ws + g.inter, LX_BF16, I, math, q, g.iq.ld, reinterpret_cast<float*>(q + g.iq.scale_at),
                                q + g.iq.blocks_at, g.iq.ld_blocks, P, I, stream))
      return e;
    b.x = addr(stream, q, size_t(g.iq.scale_at));
    b.x_scale = addr(stream, q + g.iq.scale_at, size_t(P) * 4);
    b.x_block = math == LX_MATH_FP4 ? addr(stream, q + g.iq.blocks_at, size_t(P) * g.iq.ld_blocks) : 0;
    b.ld_x = g.iq.ld;
    b.ld_xb = g.iq.ld_blocks;
  } else {
    b.x = addr(stream, ws + g.inter, size_t(P) * I * 2);
    b.ld_x = I;
  }
  b.x_vec = 1;  // workspace rows: I % 8 == 0 keeps bf16 rows 16-byte aligned, and quantized rows are padded
  b.y = addr_w(stream, ws + g.rows, size_t(P) * L * 4);
  b.order = addr(stream, ws + g.order, size_t(P) * 4);
  b.tiles = addr(stream, ws + g.tiles, size_t(1 + 3 * g.max_tiles) * 4);
  b.pair_w = addr(stream, weights, size_t(P) * 4);
  b.N = L;
  b.K = I;
  b.top_k = top_k;
  b.stack_n = L;
  if (int e = launch_gemm(af, down->type, 2, stream, b, size2(ceil_div(L, kTileN), NSUInteger(g.max_tiles))))
    return e;

  CombineArgs c{addr(stream, ws + g.rows, size_t(P) * L * 4), addr_w(stream, out, size_t(T) * L * 4), T, L, top_k};
  return combine_k.launch(stream, c, size2(ceil_div(L, 64), NSUInteger(T)), size2(64, 1));
}

}  // extern "C"
