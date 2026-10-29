// The dense problem and epilogue for lx_gemm_w (gemm_w.cu) and its tile-variant families. Each family's
// kernels are instantiated in their own translation unit (gemm_w_<kind>.cu) so they compile in parallel;
// gemm_w.cu dispatches to them through Family<kind, arch>::variants().
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "cuda/common.cuh"
#include "cuda/gemm_tc.cuh"
#include "cuda/kernels.h"

namespace lx::gemmw {

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
  static __device__ __forceinline__ bool tile(const Params& p, Tile& t, int cta) {
    const int per_group = p.group * p.tiles_n;
    const int g = cta / per_group, in = cta % per_group;
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
  static __device__ __forceinline__ float b_global(const Params& p, const Tile&) { return p.b_global; }
};

// y = alpha * (v + bias) + beta * y, v already scaled. Pairs of columns, stored together when aligned.
struct DenseEpi {
  static constexpr bool kGated = false;
  using Row = int;  // the tile row
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
  static __device__ __forceinline__ Row row(const Params&, const Tile&, int r) { return r; }
  static __device__ __forceinline__ void store2(const Params& e, const Tile& t, Row r, int c, float v0, float v1) {
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

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB, int kSplit>
__global__ void __launch_bounds__(WM * WN * 32, (lx::tc::Config<kKind, kArch, BM, BN, WM, WN, ST, RB>::kMinBlocks))
    dense_kernel(const DenseProb::Params pp, const DenseEpi::Params ep) {
  lx::pdl_trigger();
  lx::pdl_wait();
  extern __shared__ __align__(128) uint8_t smem[];
  lx::tc::gemm_body<kKind, kArch, BM, BN, WM, WN, ST, RB, DenseProb, DenseEpi, kSplit>(pp, ep, smem);
}

using LaunchFn = int (*)(DenseProb::Params, const DenseEpi::Params&, cudaStream_t);

// kSplit > 1: clusters of kSplit CTAs share each tile, split along K (gemm_body); sm_90+ only.
template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB, int kSplit = 1>
inline int launch_dense(DenseProb::Params pp, const DenseEpi::Params& ep, cudaStream_t s) {
  using C = lx::tc::Config<kKind, kArch, BM, BN, WM, WN, ST, RB>;
  auto kernel = dense_kernel<kKind, kArch, BM, BN, WM, WN, ST, RB, kSplit>;
  static const cudaError_t attr = [&] {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    if (e == cudaSuccess)
      e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    return e;
  }();
  if (attr != cudaSuccess) return static_cast<int>(attr);
  pp.tiles_m = (pp.M + BM - 1) / BM;
  pp.tiles_n = (pp.N + BN - 1) / BN;
  pp.group = std::min(pp.tiles_m, 8);
  const unsigned grid = static_cast<unsigned>(pp.tiles_m * pp.tiles_n * kSplit);
  if constexpr (kSplit == 1) {
    lx::launch_pdl(kernel, grid, C::kThreads, C::kSmem, s, pp, ep);
    return lx::launch_status();
  } else {
    cudaLaunchConfig_t cfg{};
    cfg.gridDim = dim3(grid);
    cfg.blockDim = dim3(C::kThreads);
    cfg.dynamicSmemBytes = C::kSmem;
    cfg.stream = s;
    cudaLaunchAttribute a[1];
    a[0].id = cudaLaunchAttributeClusterDimension;
    a[0].val.clusterDim.x = kSplit;
    a[0].val.clusterDim.y = 1;
    a[0].val.clusterDim.z = 1;
    cfg.attrs = a;
    cfg.numAttrs = 1;
    return static_cast<int>(cudaLaunchKernelEx(&cfg, kernel, pp, ep));
  }
}

struct Variant {
  const char* name;
  int bm;
  LaunchFn fn;
  int split;     // CTAs per tile along K (1: none)
  int min_arch;  // lx::kClusterArch for the split variants (thread block clusters)
};

template <int kKind, int kArch>
struct Family {
  static const char* name();
  static const Variant* variants(int& n);  // listed small to large
};

#define LX_GEMMW_FAMILY(K, A)                       \
  template <>                                       \
  const char* Family<lx::tc::K, A>::name();         \
  template <>                                       \
  const Variant* Family<lx::tc::K, A>::variants(int& n);
LX_GEMMW_FAMILY(kBF16, 80)
LX_GEMMW_FAMILY(kI8, 80)
LX_GEMMW_FAMILY(kF8, 89)
LX_GEMMW_FAMILY(kF8, 120)
LX_GEMMW_FAMILY(kF4, 120)
LX_GEMMW_FAMILY(kW8I8, 80)
LX_GEMMW_FAMILY(kW8F8, 80)
LX_GEMMW_FAMILY(kW8F8, 120)
LX_GEMMW_FAMILY(kW4A16, 80)
LX_GEMMW_FAMILY(kW4A8, 120)
#undef LX_GEMMW_FAMILY

// A family's definition: its tuner name and its variants, BMxBN, warps along M x N, stages, A bytes per
// row per k-tile.
#define LX_V(K, A, BM, BN, WM, WN, ST, RB) \
  Variant { #BM "x" #BN "_w" #WM "x" #WN "_s" #ST "_k" #RB, BM, launch_dense<lx::tc::K, A, BM, BN, WM, WN, ST, RB>, 1, 0 }
// A split-K variant: kSplit CTAs per tile along K (sm_90+).
#define LX_VS(K, A, BM, BN, WM, WN, ST, RB, S)                                                          \
  Variant {                                                                                            \
    #BM "x" #BN "_w" #WM "x" #WN "_s" #ST "_k" #RB "_x" #S, BM, launch_dense<lx::tc::K, A, BM, BN, WM, WN, ST, RB, S>, \
        S, lx::kClusterArch                                                                            \
  }
#define LX_FAMILY(K, A, NAME, ...)                                  \
  template <>                                                       \
  const char* Family<lx::tc::K, A>::name() {                        \
    return NAME;                                                    \
  }                                                                 \
  template <>                                                       \
  const Variant* Family<lx::tc::K, A>::variants(int& n) {           \
    static const Variant v[] = {__VA_ARGS__};                       \
    n = static_cast<int>(sizeof(v) / sizeof(v[0]));                 \
    return v;                                                       \
  }

// Where lx_quantize_act puts M rows of K activations for a math, inside a workspace: values, per-row
// scales, FP4 block scales, each 256-byte aligned. lx_gemm_w and lx_moe_experts_grouped share it.
constexpr int64_t round_up(int64_t v, int64_t a) { return (v + a - 1) / a * a; }

struct ActLayout {
  int64_t ld_q = 0, ld_bs = 0;            // bytes per row of the quantized values / block scales
  int64_t q_off = 0, scale_off = 0, bs_off = 0, total = 0;
};

inline ActLayout act_layout(int M, int K, lx_math math) {
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


// The kernel kind for a weight format under a math on an architecture, or -1 when the GPU cannot run
// that pair (the caller answers cudaErrorNotSupported). See the table in gemm_w.cu.
inline int kind_for(lx_math math, lx_wtype w, int arch) {
  using namespace lx::tc;
  if (arch < 80) return -1;
  const bool sm89 = arch >= 89, sm120 = arch >= 120 && arch < 130;
  switch (math) {
    case LX_MATH_BF16:
      return w == LX_W_BF16 ? kBF16 : w == LX_W_INT8 ? kW8I8 : w == LX_W_FP8 ? kW8F8 : kW4A16;
    case LX_MATH_INT8:
      return w == LX_W_INT8 ? kI8 : -1;
    case LX_MATH_FP8:
      return w == LX_W_FP8 && sm89 ? kF8 : w == LX_W_NVFP4 && sm120 ? kW4A8 : -1;
    case LX_MATH_FP4:
      return w == LX_W_NVFP4 && sm120 ? kF4 : -1;
  }
  return -1;
}

// Bits per stored weight value, and the K granularity the kernels need: weight rows a multiple of 16
// bytes, and NVFP4 scale rows a whole number of the kind's K tiles.
inline int weight_bits(lx_wtype w) { return w == LX_W_BF16 ? 16 : w == LX_W_NVFP4 ? 4 : 8; }
inline bool k_ok(int kind, lx_wtype w, int K) {
  if ((static_cast<int64_t>(K) * weight_bits(w) / 8) % 16 != 0) return false;
  if (w != LX_W_NVFP4) return true;
  const int ktile = kind == lx::tc::kF4 ? 256 : kind == lx::tc::kW4A8 ? 128 : 64;
  return K % ktile == 0;
}

// The math the activations are quantized to for a kind (BF16: not quantized).
inline lx_math act_math(int kind) {
  using namespace lx::tc;
  return kind == kI8 ? LX_MATH_INT8 : kind == kF4 ? LX_MATH_FP4 : (kind == kF8 || kind == kW4A8) ? LX_MATH_FP8 : LX_MATH_BF16;
}

}  // namespace lx::gemmw
