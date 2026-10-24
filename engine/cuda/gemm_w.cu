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

#include "cuda/common.cuh"
#include "cuda/gemm_tc.cuh"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

namespace {

using lx::tc::Tile;

// ---- the dense problem ------------------------------------------------------------------------------

struct DenseProb {
  struct Params {
    const uint8_t* a;  // operand A rows, lda bytes apart
    int64_t lda;
    const float* a_scale;  // per row: INT8 / FP8 scale or the FP4 global
    const uint8_t* a_sf;   // FP4 block scales, ld_asf bytes per row
    int64_t ld_asf;
    const uint8_t* b;  // the weight, ldb bytes per row
    int64_t ldb;
    const float* b_scale;  // per row (INT8 / FP8)
    const uint8_t* b_sf;   // NVFP4 block scales, ld_bsf bytes per row
    int64_t ld_bsf;
    float b_global;
    int M, N, K;
    int tiles_m, tiles_n, group;  // raster: `group` M tiles walk across the N tiles together
  };
  template <int BM, int BN>
  static __device__ __forceinline__ bool tile(const Params& p, Tile& t) {
    const int per_group = p.group * p.tiles_n;
    const int g = blockIdx.x / per_group, in = blockIdx.x % per_group;
    const int gm = min(p.group, p.tiles_m - g * p.group);
    const int mt = g * p.group + in % gm, nt = in / gm;
    t.m0 = mt * BM;
    t.n0 = nt * BN;
    t.rows = min(BM, p.M - t.m0);
    t.cols = min(BN, p.N - t.n0);
    t.expert = 0;
    t.b = p.b;
    t.b_sf = p.b_sf;
    t.b_scale = p.b_scale;
    return true;
  }
  static __device__ __forceinline__ const uint8_t* a_row(const Params& p, const Tile& t, int r) {
    return p.a + (t.m0 + r) * p.lda;
  }
  static __device__ __forceinline__ const uint8_t* a_sf_row(const Params& p, const Tile& t, int r) {
    return p.a_sf + (t.m0 + r) * p.ld_asf;
  }
  static __device__ __forceinline__ float a_scale(const Params& p, const Tile& t, int r) { return p.a_scale[t.m0 + r]; }
  static __device__ __forceinline__ int64_t b_row(const Params&, const Tile& t, int r) { return t.n0 + r; }
};

// y = alpha * (v + bias) + beta * y, v already scaled. Pairs of columns, stored together when aligned.
struct DenseEpi {
  struct Params {
    void* y;
    int64_t ld_y;
    bool y_bf16, pair_ok;
    const void* bias;
    bool bias_bf16;
    float alpha, beta;
  };
  static __device__ __forceinline__ float bias_at(const Params& e, int n) {
    return e.bias_bf16 ? __bfloat162float(static_cast<const __nv_bfloat16*>(e.bias)[n]) : static_cast<const float*>(e.bias)[n];
  }
  static __device__ __forceinline__ void store2(const Params& e, const Tile& t, int r, int c, float v0, float v1) {
    const int64_t m = t.m0 + r;
    const int n = t.n0 + c;
    const bool two = c + 1 < t.cols;
    if (e.bias) {
      v0 += bias_at(e, n);
      if (two) v1 += bias_at(e, n + 1);
    }
    v0 *= e.alpha;
    v1 *= e.alpha;
    if (e.y_bf16) {
      __nv_bfloat16* y = static_cast<__nv_bfloat16*>(e.y) + m * e.ld_y + n;
      if (e.beta != 0.0f) {
        v0 += e.beta * __bfloat162float(y[0]);
        if (two) v1 += e.beta * __bfloat162float(y[1]);
      }
      if (two && e.pair_ok) {
        *reinterpret_cast<__nv_bfloat162*>(y) = __floats2bfloat162_rn(v0, v1);
      } else {
        y[0] = __float2bfloat16_rn(v0);
        if (two) y[1] = __float2bfloat16_rn(v1);
      }
    } else {
      float* y = static_cast<float*>(e.y) + m * e.ld_y + n;
      if (e.beta != 0.0f) {
        v0 += e.beta * y[0];
        if (two) v1 += e.beta * y[1];
      }
      if (two && e.pair_ok) {
        *reinterpret_cast<float2*>(y) = make_float2(v0, v1);
      } else {
        y[0] = v0;
        if (two) y[1] = v1;
      }
    }
  }
};

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB>
__global__ void __launch_bounds__(WM * WN * 32) dense_kernel(const DenseProb::Params pp, const DenseEpi::Params ep) {
  extern __shared__ __align__(128) uint8_t smem[];
  lx::tc::gemm_body<kKind, kArch, BM, BN, WM, WN, ST, RB, DenseProb, DenseEpi>(pp, ep, smem);
}

using LaunchFn = int (*)(DenseProb::Params, const DenseEpi::Params&, cudaStream_t);

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB>
int launch_dense(DenseProb::Params pp, const DenseEpi::Params& ep, cudaStream_t s) {
  using C = lx::tc::Config<kKind, kArch, BM, BN, WM, WN, ST, RB>;
  auto kernel = dense_kernel<kKind, kArch, BM, BN, WM, WN, ST, RB>;
  static const cudaError_t attr = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
  if (attr != cudaSuccess) return static_cast<int>(attr);
  pp.tiles_m = (pp.M + BM - 1) / BM;
  pp.tiles_n = (pp.N + BN - 1) / BN;
  pp.group = std::min(pp.tiles_m, 8);
  kernel<<<pp.tiles_m * pp.tiles_n, C::kThreads, C::kSmem, s>>>(pp, ep);
  return lx::launch_status();
}

struct Variant {
  const char* name;
  int bm;
  LaunchFn fn;
};

// Tile variants per (kind, family): BMxBN, warps along M x N, stages. Listed small to large; the
// heuristic takes the largest whose BM does not exceed what M fills.
#define LX_V(K, A, BM, BN, WM, WN, ST, RB) \
  Variant { #BM "x" #BN "_w" #WM "x" #WN "_s" #ST "_k" #RB, BM, launch_dense<K, A, BM, BN, WM, WN, ST, RB> }

template <int kKind, int kArch>
struct Family;

#define LX_FAMILY(K, A, NAME, ...)                                                       \
  template <>                                                                            \
  struct Family<K, A> {                                                                  \
    static constexpr const char* kName = NAME;                                           \
    static const Variant* variants(int& n) {                                             \
      static const Variant v[] = {__VA_ARGS__};                                          \
      n = static_cast<int>(sizeof(v) / sizeof(v[0]));                                    \
      return v;                                                                          \
    }                                                                                    \
  };

using namespace lx::tc;

LX_FAMILY(kBF16, 80, "gemm_w.bf16.sm80", LX_V(kBF16, 80, 32, 128, 1, 4, 4, 128), LX_V(kBF16, 80, 64, 128, 2, 4, 4, 128),
          LX_V(kBF16, 80, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kI8, 80, "gemm_w.i8.sm80", LX_V(kI8, 80, 32, 128, 1, 4, 4, 128), LX_V(kI8, 80, 64, 128, 2, 4, 4, 128),
          LX_V(kI8, 80, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kF8, 89, "gemm_w.f8.sm89", LX_V(kF8, 89, 32, 128, 1, 4, 4, 128), LX_V(kF8, 89, 64, 128, 2, 4, 4, 128),
          LX_V(kF8, 89, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kF8, 120, "gemm_w.f8.sm120", LX_V(kF8, 120, 32, 128, 1, 4, 4, 128), LX_V(kF8, 120, 64, 128, 2, 4, 4, 128),
          LX_V(kF8, 120, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kF4, 120, "gemm_w.f4.sm120", LX_V(kF4, 120, 32, 128, 1, 4, 4, 64), LX_V(kF4, 120, 64, 128, 2, 4, 4, 64),
          LX_V(kF4, 120, 128, 128, 2, 4, 3, 64))
LX_FAMILY(kW8I8, 80, "gemm_w.w8i8.sm80", LX_V(kW8I8, 80, 32, 128, 1, 4, 4, 128), LX_V(kW8I8, 80, 64, 128, 2, 4, 4, 128),
          LX_V(kW8I8, 80, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kW8F8, 80, "gemm_w.w8f8.sm80", LX_V(kW8F8, 80, 32, 128, 1, 4, 4, 128), LX_V(kW8F8, 80, 64, 128, 2, 4, 4, 128),
          LX_V(kW8F8, 80, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kW8F8, 120, "gemm_w.w8f8.sm120", LX_V(kW8F8, 120, 32, 128, 1, 4, 4, 128), LX_V(kW8F8, 120, 64, 128, 2, 4, 4, 128),
          LX_V(kW8F8, 120, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kW4A16, 80, "gemm_w.w4a16.sm80", LX_V(kW4A16, 80, 32, 128, 1, 4, 4, 128), LX_V(kW4A16, 80, 64, 128, 2, 4, 4, 128),
          LX_V(kW4A16, 80, 128, 128, 2, 4, 3, 128))
LX_FAMILY(kW4A8, 120, "gemm_w.w4a8.sm120", LX_V(kW4A8, 120, 32, 128, 1, 4, 4, 128), LX_V(kW4A8, 120, 64, 128, 2, 4, 4, 128),
          LX_V(kW4A8, 120, 128, 128, 2, 4, 3, 128))

template <int kKind, int kArch>
struct Registered {
  Registered() {
    int n = 0;
    const Variant* v = Family<kKind, kArch>::variants(n);
    // TunableRegistration takes an initializer list; up to 8 variants per family.
    const char* names[8] = {};
    for (int i = 0; i < n && i < 8; ++i) names[i] = v[i].name;
    switch (n) {
      case 1: lx::TunableRegistration(Family<kKind, kArch>::kName, {names[0]}); break;
      case 2: lx::TunableRegistration(Family<kKind, kArch>::kName, {names[0], names[1]}); break;
      case 3: lx::TunableRegistration(Family<kKind, kArch>::kName, {names[0], names[1], names[2]}); break;
      case 4: lx::TunableRegistration(Family<kKind, kArch>::kName, {names[0], names[1], names[2], names[3]}); break;
      default:
        lx::TunableRegistration(Family<kKind, kArch>::kName, {names[0], names[1], names[2], names[3], names[4]});
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
  int pick = lx_tune_get(Family<kKind, kArch>::kName, pp.M, pp.N, pp.K);
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

// ---- workspace layout -----------------------------------------------------------------------------

constexpr int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

struct ActLayout {
  int64_t ld_q = 0, ld_bs = 0;            // bytes per row of the quantized values / block scales
  int64_t q_off = 0, scale_off = 0, bs_off = 0, total = 0;
};

ActLayout act_layout(int M, int K, lx_math math) {
  ActLayout l;
  switch (math) {
    case LX_MATH_BF16:
      l.ld_q = round_up(static_cast<int64_t>(K) * 2, 16);
      break;
    case LX_MATH_INT8:
    case LX_MATH_FP8:
      l.ld_q = round_up(K, 16);
      break;
    case LX_MATH_FP4:
      l.ld_q = round_up((K + 1) / 2, 16);
      l.ld_bs = round_up((K + 15) / 16, 16);
      break;
  }
  l.q_off = 0;
  l.scale_off = round_up(l.q_off + l.ld_q * M, 256);
  l.bs_off = round_up(l.scale_off + 4 * static_cast<int64_t>(M), 256);
  l.total = round_up(l.bs_off + l.ld_bs * M, 256);
  return l;
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
  const bool sm89 = arch >= 89, sm120 = arch >= 120 && arch < 130;

  // Which kernel, or not supported here.
  int kind = -1;
  switch (math) {
    case LX_MATH_BF16:
      kind = w->type == LX_W_BF16 ? kBF16 : w->type == LX_W_INT8 ? kW8I8 : w->type == LX_W_FP8 ? kW8F8 : kW4A16;
      break;
    case LX_MATH_INT8:
      if (w->type == LX_W_INT8) kind = kI8;
      break;
    case LX_MATH_FP8:
      if (w->type == LX_W_FP8 && sm89) kind = kF8;
      else if (w->type == LX_W_NVFP4 && sm120) kind = kW4A8;
      break;
    case LX_MATH_FP4:
      if (w->type == LX_W_NVFP4 && sm120) kind = kF4;
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
  if (kind < 0 || arch < 80) return static_cast<int>(cudaErrorNotSupported);
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

  // The tensor-core kernels read 16-byte chunks: weight rows must be a multiple of 16 bytes, and NVFP4
  // scale rows a whole number of the kernel's K tiles.
  const int wbits = w->type == LX_W_BF16 ? 16 : w->type == LX_W_NVFP4 ? 4 : 8;
  if ((static_cast<int64_t>(K) * wbits / 8) % 16 != 0 || !aligned16(w->data))
    return static_cast<int>(cudaErrorNotSupported);
  if (w->type == LX_W_NVFP4) {
    const int ktile = kind == kF4 ? 256 : kind == kW4A8 ? 128 : 64;
    if (K % ktile != 0) return static_cast<int>(cudaErrorNotSupported);
  }

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
    const lx_math qmath = kind == kI8 ? LX_MATH_INT8 : kind == kF4 ? LX_MATH_FP4 : LX_MATH_FP8;
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
