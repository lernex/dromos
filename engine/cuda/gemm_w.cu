// Tensor-core GEMM on any weight format and math (kernels.h, lx_gemm_w):
//   y[M, N] = alpha * (x[M, K] @ W[N, K]^T + bias[N]) + beta * y
//
// The activations are brought to the math's operand type first (lx_quantize_act into the workspace for
// INT8 / FP8 / FP4, a bf16 copy for an f32 x under BF16 math), then one gemm_tc.cuh kernel runs with
// the weight read as stored. Plain BF16 weights under BF16 math go to cuBLASLt (lx_gemm_bf16), which
// already runs at the bf16 tensor-core peak; every other pair is a kernel here:
//
//   math \ weight   BF16        INT8         FP8           NVFP4
//   BF16            cuBLASLt    W8A16        W8A16         W4A16          sm_80+
//   INT8            -           IMMA         -             -              sm_80+
//   FP8             -           -            FP8 MMA       W4A8           sm_89+ (W4A8: sm_120)
//   FP4             -           -            -             block-scaled   sm_120
//
// Kernel families by architecture (lx_arch, so an override runs an older family on a newer card):
//   sm_80  bf16 and int8 mma.sync, cp.async, ldmatrix; weight expansion through f32 and prmt
//   sm_89  adds the e4m3 mma.sync (FP8 math at the legacy rate)
//   sm_90  as sm_89 (wgmma is not used: it cannot be tested here; dense BF16 stays on cuBLASLt)
//   sm_120 FP8 through the block-scaled kind (twice the legacy rate), FP4 and W4A8, and the
//          single-instruction e4m3 -> bf16 conversion for W8A16 on FP8 weights
// Each family has tile variants registered with the tuner as "gemm_w.<kind>.sm<arch>"; lx_tune_get picks
// one per shape, and a heuristic on M stands in when the table has no entry.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "cuda/gemm_w.cuh"
#include "cuda/tunable.h"

namespace {

using namespace lx::tc;
using lx::gemmw::act_layout;
using lx::gemmw::act_math;
using lx::gemmw::ActLayout;
using lx::gemmw::DenseEpi;
using lx::gemmw::DenseProb;
using lx::gemmw::Family;
using lx::gemmw::k_ok;
using lx::gemmw::kind_for;
using lx::gemmw::Variant;
using lx::gemmw::weight_bits;

template <int kKind, int kArch>
struct Registered {
  Registered() {
    int n = 0;
    const Variant* v = Family<kKind, kArch>::variants(n);
    // TunableRegistration takes an initializer list; up to 8 variants per family.
    const char* names[8] = {};
    for (int i = 0; i < n && i < 8; ++i) names[i] = v[i].name;
    const char* const* m = names;
    switch (n) {
      case 1: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0]}); break;
      case 2: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1]}); break;
      case 3: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2]}); break;
      case 4: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3]}); break;
      case 5: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4]}); break;
      case 6: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4], m[5]}); break;
      case 7: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4], m[5], m[6]}); break;
      default: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7]});
    }
  }
};
const Registered<kBF16, 80> reg_bf16;
const Registered<kI8, 80> reg_i8;
const Registered<kF8, 89> reg_f8_89;
const Registered<kF8, 120> reg_f8_120;
const Registered<kF4, 120> reg_f4;
const Registered<kW8I8, 80> reg_w8i8;
const Registered<kW8F8, 80> reg_w8f8_80;
const Registered<kW8F8, 120> reg_w8f8_120;
const Registered<kW4A16, 80> reg_w4a16;
const Registered<kW4A8, 120> reg_w4a8;

template <int kKind, int kArch>
int run_family(const DenseProb::Params& pp, const DenseEpi::Params& ep, cudaStream_t s) {
  int n = 0;
  const Variant* v = Family<kKind, kArch>::variants(n);
  int pick = lx_tune_get(Family<kKind, kArch>::name(), pp.M, pp.N, pp.K);
  if (pick < 0 || pick >= n) {
    // Heuristic: the largest BM that M fills, then enough CTAs for the GPU (about two waves of 70 SMs).
    pick = 0;
    for (int i = 0; i < n; ++i)
      if (v[i].bm <= pp.M) pick = i;
    while (pick > 0) {
      const int64_t ctas = static_cast<int64_t>((pp.M + v[pick].bm - 1) / v[pick].bm) * ((pp.N + 127) / 128);
      if (ctas >= 140) break;
      --pick;
    }
  }
  return v[pick].fn(pp, ep, s);
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

}  // namespace

extern "C" int64_t lx_gemm_workspace_bytes(int M, int K, lx_math math) {
  if (M <= 0 || K <= 0) return 0;
  return act_layout(M, K, math).total;
}

extern "C" int lx_gemm_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, lx_math math, const void* bias,
                         lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                         float beta, void* workspace, void* stream) {
  if (M < 0 || N < 0 || K <= 0 || ld_x < K || ld_y < N || !w) return static_cast<int>(cudaErrorInvalidValue);
  if (M == 0 || N == 0) return 0;
  if (!x || !y || !w->data) return static_cast<int>(cudaErrorInvalidValue);
  if ((x_dtype != LX_F32 && x_dtype != LX_BF16) || (y_dtype != LX_F32 && y_dtype != LX_BF16) ||
      (bias && bias_dtype != LX_F32 && bias_dtype != LX_BF16))
    return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  const int arch = lx_arch();
  const bool sm120 = arch >= 120 && arch < 130;
  if (math < LX_MATH_BF16 || math > LX_MATH_FP4) return static_cast<int>(cudaErrorInvalidValue);
  const int kind = kind_for(math, w->type, arch);
  if (kind < 0) return static_cast<int>(cudaErrorNotSupported);
  if (w->type != LX_W_BF16 && !w->scale) return static_cast<int>(cudaErrorInvalidValue);

  // Plain BF16: cuBLASLt, at the bf16 peak, when its bias rules fit (bf16 bias); x converted if f32.
  const ActLayout lay = act_layout(M, K, math);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  if (kind == kBF16 && (!bias || bias_dtype == LX_BF16)) {
    if (x_dtype == LX_F32) {
      if (!ws) return static_cast<int>(cudaErrorInvalidValue);
      const int ld = static_cast<int>(lay.ld_q / 2);
      if (int e = lx_copy2d(x, LX_F32, ld_x, ws, LX_BF16, ld, M, K, stream)) return e;
      return lx_gemm_bf16(ws, ld, w->data, bias, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
    }
    return lx_gemm_bf16(x, ld_x, w->data, bias, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
  }

  // The tensor-core kernels read 16-byte chunks of rows: see k_ok.
  const int wbits = weight_bits(w->type);
  if (!k_ok(kind, w->type, K) || !aligned16(w->data)) return static_cast<int>(cudaErrorNotSupported);

  DenseProb::Params pp{};
  pp.M = M;
  pp.N = N;
  pp.K = K;
  pp.b = static_cast<const uint8_t*>(w->data);
  pp.ldb = static_cast<int64_t>(K) * wbits / 8;
  pp.b_global = w->type == LX_W_NVFP4 ? w->global : 1.0f;
  if (w->type == LX_W_INT8 || w->type == LX_W_FP8) pp.b_scale = static_cast<const float*>(w->scale);
  if (w->type == LX_W_NVFP4) {
    pp.b_sf = static_cast<const uint8_t*>(w->scale);
    pp.ld_bsf = K / 16;
  }

  // Operand A: bf16 as given when aligned, else a bf16 copy; otherwise quantized per token.
  const bool a_bf16 = kind == kBF16 || kind == kW8I8 || kind == kW8F8 || kind == kW4A16;
  if (a_bf16) {
    if (x_dtype == LX_BF16 && aligned16(x) && (static_cast<int64_t>(ld_x) * 2) % 16 == 0) {
      pp.a = static_cast<const uint8_t*>(x);
      pp.lda = static_cast<int64_t>(ld_x) * 2;
    } else {
      if (!ws) return static_cast<int>(cudaErrorInvalidValue);
      if (int e = lx_copy2d(x, x_dtype, ld_x, ws, LX_BF16, static_cast<int>(lay.ld_q / 2), M, K, stream)) return e;
      pp.a = ws;
      pp.lda = lay.ld_q;
    }
  } else {
    if (!ws) return static_cast<int>(cudaErrorInvalidValue);
    const lx_math qmath = act_math(kind);
    const ActLayout ql = act_layout(M, K, qmath);
    if (int e = lx_quantize_act(x, x_dtype, ld_x, qmath, ws + ql.q_off, static_cast<int>(ql.ld_q),
                                reinterpret_cast<float*>(ws + ql.scale_off), ws + ql.bs_off, static_cast<int>(ql.ld_bs),
                                M, K, stream))
      return e;
    pp.a = ws + ql.q_off;
    pp.lda = ql.ld_q;
    pp.a_scale = reinterpret_cast<const float*>(ws + ql.scale_off);
    pp.a_sf = ws + ql.bs_off;
    pp.ld_asf = ql.ld_bs;
  }

  DenseEpi::Params ep{};
  ep.y = y;
  ep.ld_y = ld_y;
  ep.y_bf16 = y_dtype == LX_BF16;
  ep.pair_ok = ld_y % 2 == 0 && (reinterpret_cast<uintptr_t>(y) & (ep.y_bf16 ? 3 : 7)) == 0;
  ep.bias = bias;
  ep.bias_bf16 = bias_dtype == LX_BF16;
  ep.alpha = alpha;
  ep.beta = beta;

  switch (kind) {
    case kBF16: return run_family<kBF16, 80>(pp, ep, s);
    case kI8: return run_family<kI8, 80>(pp, ep, s);
    case kF8: return sm120 ? run_family<kF8, 120>(pp, ep, s) : run_family<kF8, 89>(pp, ep, s);
    case kF4: return run_family<kF4, 120>(pp, ep, s);
    case kW8I8: return run_family<kW8I8, 80>(pp, ep, s);
    case kW8F8: return sm120 ? run_family<kW8F8, 120>(pp, ep, s) : run_family<kW8F8, 80>(pp, ep, s);
    case kW4A16: return run_family<kW4A16, 80>(pp, ep, s);
    default: return run_family<kW4A8, 120>(pp, ep, s);
  }
}
