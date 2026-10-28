// Speculative decoding on Metal (kernels/spec.metal): the kernels.h section of that name.
#include <cmath>

#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kGemvRows = 8;
constexpr int kMaxPending = LX_SPEC_MAX_PENDING;
constexpr int kScanChunk = 8;     // tokens staged at once (spec_ssm_scan, as ssm_scan)
constexpr int kThreads = 1024;    // sampling threadgroups
constexpr int kAttnThreads = 256; // block attention threadgroups
constexpr size_t kThreadgroupBytes = 32 * 1024;

int round4(int v) { return (v + 3) & ~3; }
int64_t hist_row_floats(int P, int N) { return round4(P) + round4(N) + 4; }

static_assert(sizeof(lx_spec_filter) == sizeof(SpecFilter), "SpecFilter mirrors lx_spec_filter");
static_assert(sizeof(lx_spec_rng) == sizeof(SpecRng), "SpecRng mirrors lx_spec_rng");

}  // namespace

extern "C" {

int64_t lx_spec_ssm_hist_floats(int H, int P, int N) { return int64_t(H) * kMaxPending * hist_row_floats(P, N); }

int lx_spec_conv1d(const float* x, int ld_x, const void* w, const void* bias, float* conv_state, float* conv_hist,
                   const int32_t* slot_ids, const int32_t* hist_ids, const int32_t* has_state, const int32_t* fold,
                   const int32_t* keep, const int32_t* cu_seqlens, int n_seq, float* out, int C, int W, void* stream) {
  static Kernel out_k("spec_conv_out"), state_k("spec_conv_state");
  if (n_seq <= 0) return kOk;
  if (C <= 0 || W < 1 || W > 8 || ld_x < C || n_seq > 65535) return kInvalidValue;
  // Token counts and slots are device data: unknown extents for x, out, the state and the history. hist_ids
  // is null for history 0 everywhere (addr maps null to 0, which the kernels read as "no array").
  SpecConvArgs a{addr(stream, x, kUnknownExtent), addr(stream, w, size_t(C) * W * 2), addr(stream, bias, size_t(C) * 2),
                 addr_w(stream, conv_state, kUnknownExtent), addr_w(stream, conv_hist, kUnknownExtent),
                 addr(stream, slot_ids, size_t(n_seq) * 4), addr(stream, hist_ids, size_t(n_seq) * 4),
                 addr(stream, has_state, size_t(n_seq) * 4),
                 addr(stream, fold, size_t(n_seq) * 4), addr(stream, keep, size_t(n_seq) * 4),
                 addr(stream, cu_seqlens, size_t(n_seq + 1) * 4), addr_w(stream, out, kUnknownExtent), ld_x, C, W};
  // The outputs read the old state and history, so they go first, and the state kernel waits for them.
  if (int e = out_k.launch(stream, a, size2(ceil_div(C, 32), n_seq), size2(32, 8))) return e;
  stream_of(stream)->again();
  return state_k.launch(stream, a, size2(ceil_div(C, 128), n_seq), size2(128, 1));
}

int lx_spec_ssm_scan(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B, const float* C, int ld_bc,
                     const float* A_log, const float* D, const float* dt_bias, float* ssm_state, float* ssm_hist,
                     const int32_t* slot_ids, const int32_t* hist_ids, const int32_t* has_state, const int32_t* fold,
                     const int32_t* keep, const int32_t* cu_seqlens, int n_seq, float* y, int H, int P, int G, int N,
                     void* stream) {
  static Kernel k1("spec_ssm_scan_1"), k2("spec_ssm_scan_2"), k4("spec_ssm_scan_4"), k8("spec_ssm_scan_8"),
      k16("spec_ssm_scan_16");
  if (n_seq <= 0) return kOk;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || P > 64 || N < 1 || N > 256 || n_seq > 65535) return kInvalidValue;
  if (ld_x < H * P || ld_dt < H || ld_bc < G * N) return kInvalidValue;
  SpecScanArgs a{addr(stream, x, kUnknownExtent),        addr(stream, dt_raw, kUnknownExtent),
                 addr(stream, B, kUnknownExtent),        addr(stream, C, kUnknownExtent),
                 addr(stream, A_log, size_t(H) * 4),     addr(stream, D, size_t(H) * 4),
                 addr(stream, dt_bias, size_t(H) * 4),   addr_w(stream, ssm_state, kUnknownExtent),
                 addr_w(stream, ssm_hist, kUnknownExtent), addr(stream, slot_ids, size_t(n_seq) * 4),
                 addr(stream, hist_ids, size_t(n_seq) * 4), addr(stream, has_state, size_t(n_seq) * 4),
                 addr(stream, fold, size_t(n_seq) * 4),
                 addr(stream, keep, size_t(n_seq) * 4),  addr(stream, cu_seqlens, size_t(n_seq + 1) * 4),
                 addr_w(stream, y, kUnknownExtent),      ld_x, ld_dt, ld_bc, H, P, G, N};
  // lx_ssm_scan_varlen's launch, plus room for the replayed history rows.
  const int threads = (4 * P + 31) / 32 * 32;
  const int ni = ((N + 3) / 4 + 3) / 4;
  Kernel& k = ni <= 1 ? k1 : ni <= 2 ? k2 : ni <= 4 ? k4 : ni <= 8 ? k8 : k16;
  const int nip = ni <= 1 ? 1 : ni <= 2 ? 2 : ni <= 4 ? 4 : ni <= 8 ? 8 : 16;
  const int row = round4(P) + 2 * 16 * nip + 4;
  const size_t bytes = (size_t(kScanChunk) * row + size_t(kMaxPending) * hist_row_floats(P, N)) * 4;
  return k.launch(stream, a, size2(H, n_seq), size1(threads), bytes);
}

int lx_spec_gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N,
                     int K, void* stream) {
  static Kernel k("spec_gemv_f32");
  if (M == 0 || N == 0) return kOk;
  if (M < 0 || M > kGemvRows || K <= 0 || ld_x < K || ld_y < N || !x || !w || !y) return kInvalidValue;
  SpecGemvArgs a{addr(stream, x, span(M, K, ld_x, 4)), addr(stream, w, size_t(N) * K * 4),
                 addr(stream, bias, size_t(N) * 4), addr_w(stream, y, span(M, N, ld_y, 4)), ld_x, ld_y, M, N, K};
  return k.launch(stream, a, size1(ceil_div(N, 4)), size1(128));
}

int lx_spec_penalize(float* logits, int V, lx_spec_penalty pen, const int32_t* tokens, int rows, const int32_t* seen,
                     const int32_t* out_counts, void* stream) {
  static Kernel k("spec_penalize");
  if (rows <= 0) return kOk;
  if (V <= 0 || !logits || !tokens || !seen || !out_counts) return kInvalidValue;
  SpecPenalizeArgs a{addr_w(stream, logits, size_t(rows) * V * 4), addr(stream, tokens, size_t(rows) * 4),
                     addr(stream, seen, size_t(V) * 4), addr(stream, out_counts, size_t(V) * 4),
                     pen.repetition_penalty, pen.presence_penalty, pen.frequency_penalty, V};
  return k.launch(stream, a, size1(NSUInteger(rows)), size1(kThreads));
}

int lx_spec_probs(const float* logits, int V, const lx_spec_filter* filters, int rows, float* probs, void* stream) {
  static Kernel k("spec_probs");
  if (rows <= 0) return kOk;
  if (V <= 0 || V > (1 << 24) || !logits || !filters || !probs) return kInvalidValue;
  SpecProbsArgs a{addr(stream, logits, size_t(rows) * V * 4), addr(stream, filters, size_t(rows) * sizeof(SpecFilter)),
                  addr_w(stream, probs, size_t(rows) * V * 4), V};
  return k.launch(stream, a, size1(NSUInteger(rows)), size1(kThreads));
}

int lx_spec_verify(const float* p, const float* q, int V, const int32_t* tokens, const int32_t* cu_seqlens,
                   const lx_spec_rng* rng, int n_seq, int32_t* accepted, int32_t* next, void* stream) {
  static Kernel k("spec_verify");
  if (n_seq <= 0) return kOk;
  if (V <= 0 || V > (1 << 24) || !p || !tokens || !cu_seqlens || !rng || !accepted || !next) return kInvalidValue;
  // The row count is cu_seqlens[n_seq], device data.
  SpecVerifyArgs a{addr(stream, p, kUnknownExtent),
                   q ? addr(stream, q, kUnknownExtent) : 0,
                   addr(stream, tokens, kUnknownExtent),
                   addr(stream, cu_seqlens, size_t(n_seq + 1) * 4),
                   addr(stream, rng, size_t(n_seq) * sizeof(SpecRng)),
                   addr_w(stream, accepted, size_t(n_seq) * 4),
                   addr_w(stream, next, size_t(n_seq) * 4),
                   V};
  return k.launch(stream, a, size1(NSUInteger(n_seq)), size1(kThreads));
}

int lx_spec_verify_greedy(const float* logits, int V, const int32_t* tokens, const int32_t* cu_seqlens, int n_seq,
                          int rows, int32_t* argmax, int32_t* accepted, int32_t* next, void* stream) {
  static Kernel arg_k("spec_argmax"), accept_k("spec_greedy_accept");
  if (n_seq <= 0 || rows <= 0) return kOk;
  if (V <= 0 || !logits || !tokens || !cu_seqlens || !argmax || !accepted || !next) return kInvalidValue;
  SpecGreedyArgs a{addr(stream, logits, size_t(rows) * V * 4),
                   addr(stream, tokens, size_t(rows) * 4),
                   addr(stream, cu_seqlens, size_t(n_seq + 1) * 4),
                   addr_w(stream, argmax, size_t(rows) * 4),
                   addr_w(stream, accepted, size_t(n_seq) * 4),
                   addr_w(stream, next, size_t(n_seq) * 4),
                   V,
                   n_seq};
  if (int e = arg_k.launch(stream, a, size1(NSUInteger(rows)), size1(kThreads))) return e;
  stream_of(stream)->again();
  return accept_k.launch(stream, a, size1(ceil_div(n_seq, 128)), size1(128));
}

int lx_spec_sample(const float* probs, int V, const int32_t* rows, const lx_spec_rng* rng, uint32_t domain, int n,
                   int32_t* tokens, void* stream) {
  static Kernel k("spec_sample");
  if (n <= 0) return kOk;
  if (V <= 0 || V > (1 << 24) || !probs || !rng || !tokens) return kInvalidValue;
  SpecSampleArgs a{rows ? addr(stream, probs, kUnknownExtent) : addr(stream, probs, size_t(n) * V * 4),
                   rows ? addr(stream, rows, size_t(n) * 4) : 0,
                   addr(stream, rng, size_t(n) * sizeof(SpecRng)),
                   addr_w(stream, tokens, size_t(n) * 4),
                   V,
                   domain};
  return k.launch(stream, a, size1(NSUInteger(n)), size1(kThreads));
}

int lx_spec_head_norm_rope(float* x, int ld_x, int rows, int heads, int head_dim, const void* norm_w, float eps,
                           int64_t pos0, double theta, void* stream) {
  static Kernel k("spec_head_norm_rope");
  if (rows <= 0 || heads <= 0) return kOk;
  if (!x || head_dim < 2 || head_dim % 2 || head_dim > 256 || ld_x < heads * head_dim || pos0 < 0 || !(theta > 1.0))
    return kInvalidValue;
  // The kernel takes the position as an exact float, and the angle in float pairs: about 48 bits, past
  // what positions below 2^24 need for a correctly rounded angle.
  if (pos0 + rows > (int64_t(1) << 24)) return kInvalidValue;
  SpecRopeArgs a{};
  a.x = addr_w(stream, x, span(rows, heads * head_dim, ld_x, 4));
  a.w = addr(stream, norm_w, size_t(head_dim) * 2);
  a.ld_x = ld_x;
  a.rows = rows;
  a.heads = heads;
  a.Dh = head_dim;
  a.eps = eps;
  a.pos0 = int(pos0);
  for (int i = 0; i < head_dim / 2; ++i) {
    const double inv = std::pow(theta, -2.0 * i / head_dim);
    a.inv_hi[i] = float(inv);
    a.inv_lo[i] = float(inv - double(a.inv_hi[i]));
  }
  const int64_t threads = int64_t(rows) * heads * 32;
  return k.launch(stream, a, size1(NSUInteger((threads + 255) / 256)), size1(256));
}

int lx_spec_block_attention(const float* q, int ld_q, const float* k_ctx, const float* v_ctx, int n_ctx,
                            const float* k_blk, const float* v_blk, int ld_blk, float* out, int ld_out, int B, int H,
                            int KVH, int head_dim, float scale, void* stream) {
  static Kernel k("spec_block_attention");
  if (B <= 0) return kOk;
  if (!q || !k_blk || !v_blk || !out || n_ctx < 0 || (n_ctx > 0 && (!k_ctx || !v_ctx)) || H <= 0 || KVH <= 0 ||
      H % KVH || head_dim <= 0 || ld_q < H * head_dim || ld_out < H * head_dim || ld_blk < KVH * head_dim || B > 65535)
    return kInvalidValue;
  // Keys are read as float4: head_dim, ld_blk and the key pointers in steps of 16 bytes.
  const auto misaligned = [](const void* p) { return reinterpret_cast<uintptr_t>(p) % 16 != 0; };
  if (head_dim % 4 || kAttnThreads % head_dim || ld_blk % 4 || misaligned(k_blk) || (n_ctx > 0 && misaligned(k_ctx)))
    return kInvalidValue;
  // Scores in chunks of what fits next to the query, the partial sums and the kernel's own 32 floats.
  const int fit = int(kThreadgroupBytes / 4) - head_dim - kAttnThreads - 32;
  const int n = n_ctx + B;
  const int chunk = std::min(n, fit);
  SpecBlockAttnArgs a{addr(stream, q, span(B, H * head_dim, ld_q, 4)),
                      n_ctx > 0 ? addr(stream, k_ctx, size_t(n_ctx) * KVH * head_dim * 4) : 0,
                      n_ctx > 0 ? addr(stream, v_ctx, size_t(n_ctx) * KVH * head_dim * 4) : 0,
                      addr(stream, k_blk, span(B, KVH * head_dim, ld_blk, 4)),
                      addr(stream, v_blk, span(B, KVH * head_dim, ld_blk, 4)),
                      addr_w(stream, out, span(B, H * head_dim, ld_out, 4)),
                      ld_q, n_ctx, ld_blk, ld_out, B, H, KVH, head_dim, chunk, scale};
  const size_t bytes = size_t(head_dim + kAttnThreads + chunk) * 4;
  return k.launch(stream, a, size2(H, B), size1(kAttnThreads), (bytes + 15) & ~size_t(15));
}

}  // extern "C"
