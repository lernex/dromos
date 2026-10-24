// Decode GEMV (kernels.h, "Decode GEMV" and lx_gemv_w): y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y.
//
// Decode is memory bound: a token streams the whole weight matrix once, so the lane mapping sets the
// cost. Every format moves one 8-byte word per lane per step (wformat.cuh): 4 bf16, 8 int8 or fp8, or
// 16 nvfp4 values, so a warp step covers 128, 256 or 512 k and always reads one contiguous 256 B run
// of weights. NVFP4 adds one e4m3 block scale per lane per step (the 16 values of its word are one
// block), read as a contiguous 32 B run per warp. The x loads for those k are contiguous too, and the
// RG rows a warp owns share them. A lane that owned a whole 16-byte chunk would scatter its x loads
// 64 B apart across lanes, up to 4x the L1 wavefronts for the same work. Weight loads are
// cache-streaming so that the x vector stays in L1. The loads for the next KU steps are in flight
// while the current ones are consumed, and they go before the programmatic-dependent-launch wait,
// since they do not depend on the previous kernel in the stream: the next GEMV starts streaming
// while this one finishes.
//
// Two kernels, for every format:
//   vec    K > 256 with aligned rows: the warps of a row group split the steps of each row.
//          When N is too small to fill the GPU, S warps split one row group's steps and combine in
//          shared memory in a fixed order, so split-K is deterministic without atomics.
//   small  K <= 256, or misaligned x or w: scalar loads. lpr lanes cover one row, and the 32 / lpr
//          rows of a warp reduce with xor shuffles inside their group.
// The per-row factor (the int8 / fp8 row scale, the nvfp4 global) is applied once per row:
// sum_k x * (q * s) = s * sum_k x * q.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <type_traits>

#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/tunable.h"
#include "cuda/wformat.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;  // vec kernel: 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kMaxM = 8;
constexpr int kSmallK = 256;       // rows this short are cheaper as scalar loads
constexpr int kWarpTarget = 2048;  // warps wanted before K is split across warps

template <int F>
constexpr int kVal = lx::WordFormat<F>::kValues;  // values per lane per step
template <int F>
constexpr int kStep = 32 * kVal<F>;  // k per warp step

// A lane's word for one step, and its block scale (NVFP4, decoded to f32; 1 for the other formats).
struct Lane {
  uint2 word;
  float scale;
};

// The lane's V x values at p as f32, V a multiple of 4. p is 16-byte aligned.
template <typename TX, int V>
__device__ __forceinline__ void load_x(const TX* p, float* f) {
  if constexpr (std::is_same_v<TX, float>) {
    const float4* q = reinterpret_cast<const float4*>(p);
#pragma unroll
    for (int i = 0; i < V / 4; ++i) {
      const float4 t = __ldg(q + i);
      f[4 * i] = t.x;
      f[4 * i + 1] = t.y;
      f[4 * i + 2] = t.z;
      f[4 * i + 3] = t.w;
    }
  } else {
    // bf16: four values per 8-byte load, the same unpacking the weights use.
    const uint2* q = reinterpret_cast<const uint2*>(p);
#pragma unroll
    for (int i = 0; i < V / 4; ++i) {
      const uint2 v = __ldg(q + i);
      f[4 * i] = __uint_as_float(v.x << 16);
      f[4 * i + 1] = __uint_as_float(v.x & 0xffff0000u);
      f[4 * i + 2] = __uint_as_float(v.y << 16);
      f[4 * i + 3] = __uint_as_float(v.y & 0xffff0000u);
    }
  }
}

__device__ __forceinline__ float bias_at(const void* bias, bool bias_bf16, int64_t n) {
  if (bias == nullptr) return 0.0f;
  return bias_bf16 ? lx::to_f32(static_cast<const bf16*>(bias)[n]) : static_cast<const float*>(bias)[n];
}

// Writes y = alpha * t + beta * y, where t already holds the scaled dot product plus bias. y is read
// only when beta != 0: a caller may pass an uninitialised buffer with beta = 0.
__device__ __forceinline__ void store_y(void* y, bool y_bf16, int64_t idx, float t, float alpha, float beta) {
  float v = alpha * t;
  if (beta != 0.0f) {
    const float old =
        y_bf16 ? lx::to_f32(static_cast<const bf16*>(y)[idx]) : static_cast<const float*>(y)[idx];
    v = fmaf(beta, old, v);
  }
  if (y_bf16) static_cast<bf16*>(y)[idx] = lx::from_f32<bf16>(v);
  else static_cast<float*>(y)[idx] = v;
}

// ---- vec kernel ------------------------------------------------------------------------------

// Lanes for the steps u_begin + u (u < KU) of rows first + r (r < RG). Steps past u_end, k past K and
// rows past N read as zero, so the consumer needs no load-side bounds handling.
template <int F, int KU, int RG>
__device__ __forceinline__ void load_steps(const lx_weight& w, int64_t first, int N, int K, int lane, int u_begin,
                                           int u_end, Lane (&raw)[KU][RG]) {
  constexpr int kV = kVal<F>;
  constexpr int kS = kStep<F>;
  const uint2* words = static_cast<const uint2*>(w.data);
#pragma unroll
  for (int u = 0; u < KU; ++u) {
    const int64_t k0 = static_cast<int64_t>(u_begin + u) * kS + kV * lane;
#pragma unroll
    for (int r = 0; r < RG; ++r) {
      raw[u][r] = Lane{make_uint2(0, 0), 0.0f};
      // Streamed once per token, so they go evict-first and leave L1 to the x vector they all reuse.
      if (u_begin + u < u_end && k0 < K && first + r < N) {
        raw[u][r].word = __ldcs(words + ((first + r) * K + k0) / kV);
        if constexpr (F == LX_W_NVFP4) {
          // The lane's word is one 16-value block: its e4m3 scale is the byte at block index k0 / 16.
          const uint8_t s = __ldcs(static_cast<const uint8_t*>(w.scale) + (first + r) * (K / 16) + k0 / 16);
          raw[u][r].scale = lx::e4m3_to_f32(s);
        } else {
          raw[u][r].scale = 1.0f;
        }
      }
    }
  }
}

// Block b owns kWarps / S row groups of RG rows each. Warp w is part `w % S` of group `w / S`.
template <int F, typename TX, int RG, int S, int KU>
__global__ void __launch_bounds__(kThreads)
gemv_vec_kernel(const TX* __restrict__ x, int ld_x, const lx_weight w, int K, const void* __restrict__ bias,
                bool bias_bf16, void* __restrict__ y, bool y_bf16, int ld_y, int M, int N, float alpha, float beta) {
  constexpr int kV = kVal<F>;
  constexpr int kS = kStep<F>;
  constexpr int kGroups = kWarps / S;
  __shared__ float partial[kWarps][RG * kMaxM];  // split-K partial sums, used when S > 1

  // Let the next kernel in the stream start its weight loads while this one runs.
  lx::pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = warp / S, part = warp % S;
  const int64_t first = (static_cast<int64_t>(blockIdx.x) * kGroups + group) * RG;
  // The S warps of a group take contiguous, evenly sized runs of each row's steps.
  const int steps = (K + kS - 1) / kS;
  const int u0 = (part * steps) / S;
  const int u1 = ((part + 1) * steps) / S;

  float acc[RG][kMaxM];
#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) acc[r][m] = 0.0f;

  // The first words go out before the dependency wait; x and y are read after it.
  Lane cur[KU][RG];
  load_steps<F, KU, RG>(w, first, N, K, lane, u0, u1, cur);
  lx::pdl_wait();

  // Warp-uniform guard: the whole warp takes the same branch, so the shuffles below stay full-warp.
  if (first < N) {
    for (int j = u0; j < u1; j += KU) {
      Lane nxt[KU][RG];  // the next KU steps are in flight while these are consumed
      load_steps<F, KU, RG>(w, first, N, K, lane, j + KU, u1, nxt);
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const int unit = j + u;
        const int64_t k0 = static_cast<int64_t>(unit) * kS + kV * lane;
        if (unit < u1 && k0 < K) {
          float wf[RG][kV];
#pragma unroll
          for (int r = 0; r < RG; ++r) lx::WordFormat<F>::decode(cur[u][r].word, cur[u][r].scale, wf[r]);
#pragma unroll
          for (int m = 0; m < kMaxM; ++m) {
            if (m < M) {
              float xf[kV];
              load_x<TX, kV>(x + static_cast<int64_t>(m) * ld_x + k0, xf);
#pragma unroll
              for (int r = 0; r < RG; ++r)
#pragma unroll
                for (int t = 0; t < kV; ++t) acc[r][m] = fmaf(wf[r][t], xf[t], acc[r][m]);
            }
          }
        }
      }
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int r = 0; r < RG; ++r) cur[u][r] = nxt[u][r];
    }
  }

#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      if (m < M) acc[r][m] = lx::warp_sum(acc[r][m]);

  if constexpr (S == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RG; ++r) {
        const int64_t n = first + r;
        if (n >= N) continue;
        const float rf = lx::WordFormat<F>::row_factor(w, n);
        const float b = bias_at(bias, bias_bf16, n);
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
          if (m < M) store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, rf * acc[r][m] + b, alpha, beta);
      }
    }
  } else {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RG; ++r)
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
          if (m < M) partial[warp][r * kMaxM + m] = acc[r][m];
    }
    __syncthreads();  // every thread reaches this: the early-out is per group, not per block
    // Each output is summed over the S parts in a fixed order, so the result never depends on timing.
    for (int o = threadIdx.x; o < kGroups * RG * M; o += kThreads) {
      const int g = o / (RG * M), rem = o % (RG * M), r = rem / M, m = rem % M;
      const int64_t n = (static_cast<int64_t>(blockIdx.x) * kGroups + g) * RG + r;
      if (n >= N) continue;
      float sum = 0.0f;
      for (int p = 0; p < S; ++p) sum += partial[g * S + p][r * kMaxM + m];
      const float rf = lx::WordFormat<F>::row_factor(w, n);
      store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, rf * sum + bias_at(bias, bias_bf16, n), alpha, beta);
    }
  }
}

// ---- small kernel ----------------------------------------------------------------------------

// One warp per block. The warp covers 32 / lpr rows: lane c of a row's group of lpr lanes reads the
// e consecutive inputs starting at c * e. Scalar loads, so any stride or alignment works.
template <int F, typename TX>
__global__ void gemv_small_kernel(const TX* __restrict__ x, int ld_x, const lx_weight w,
                                  const void* __restrict__ bias, bool bias_bf16, void* __restrict__ y, bool y_bf16,
                                  int ld_y, int M, int N, int K, int lpr, int e, float alpha, float beta) {
  lx::pdl_trigger();
  lx::pdl_wait();  // x and w may both be written by the previous kernel
  const int lane = threadIdx.x;
  const int slot = lane / lpr, c = lane % lpr;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * (32 / lpr) + slot;

  float acc[kMaxM];
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) acc[m] = 0.0f;
  if (n < N) {
    for (int i = 0; i < e; ++i) {
      const int k = c * e + i;
      if (k >= K) break;
      const float wv = lx::element<F>(w, n, k, K);
#pragma unroll
      for (int m = 0; m < kMaxM; ++m)
        if (m < M) acc[m] = fmaf(wv, lx::to_f32(x[static_cast<int64_t>(m) * ld_x + k]), acc[m]);
    }
  }

  // Sum the row's lpr lanes. The xor offsets stay inside the aligned group of lpr lanes.
  for (int off = lpr / 2; off > 0; off >>= 1)
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);

  if (c == 0 && n < N) {
    const float rf = lx::WordFormat<F>::row_factor(w, n);
    const float b = bias_at(bias, bias_bf16, n);
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      if (m < M) store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, rf * acc[m] + b, alpha, beta);
  }
}

// ---- quantizer -------------------------------------------------------------------------------

// One warp per row. The divisions are IEEE (__fdiv_rn), so q and scale match a reference that
// divides the same way.
__global__ void quantize_w8_kernel(const bf16* __restrict__ w, int8_t* __restrict__ q, float* __restrict__ scale,
                                   int64_t rows, int cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const bf16* src = w + row * cols;
  int8_t* dst = q + row * cols;

  float amax = 0.0f;
  for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(lx::to_f32(src[c])));
  amax = lx::warp_max(amax);
  // An all-zero row keeps scale 1, so it quantizes to exact zeros.
  const float s = amax > 0.0f ? __fdiv_rn(amax, 127.0f) : 1.0f;
  if (lane == 0) scale[row] = s;
  for (int c = lane; c < cols; c += 32) {
    const float r = rintf(__fdiv_rn(lx::to_f32(src[c]), s));  // rintf rounds half to even
    dst[c] = static_cast<int8_t>(static_cast<int>(fminf(fmaxf(r, -127.0f), 127.0f)));
  }
}

// ---- host side -------------------------------------------------------------------------------

struct Args {
  const void* x;
  int ld_x;
  lx_weight w;
  const void* bias;
  bool bias_bf16;
  void* y;
  bool y_bf16;
  int ld_y;
  int M, N, K;
  float alpha, beta;
  cudaStream_t stream;
  bool x_bf16;
};

// Programmatic dependent launch: this kernel may start while the previous one in the stream drains,
// and its own grid-dependency wait holds back everything that reads what that kernel wrote.
cudaLaunchConfig_t pdl_config(unsigned blocks, int threads, cudaStream_t stream, cudaLaunchAttribute& attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = lx::pdl_available() ? 1 : 0;  // below sm_90 the launch is an ordinary one, in stream order
  return cfg;
}

template <int F, typename TX>
cudaError_t launch_small(const Args& a) {
  // Lanes per row: the smallest power of two (at most 32) that gives each lane at most 8 inputs.
  int lpr = 1;
  while (lpr < 32 && lpr * 8 < a.K) lpr *= 2;
  const int e = (a.K + lpr - 1) / lpr;
  const int rows_per_warp = 32 / lpr;
  const unsigned blocks = static_cast<unsigned>((a.N + rows_per_warp - 1) / rows_per_warp);
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(blocks, 32, a.stream, attr);
  cudaLaunchKernelEx(&cfg, gemv_small_kernel<F, TX>, static_cast<const TX*>(a.x), a.ld_x, a.w, a.bias, a.bias_bf16,
                     a.y, a.y_bf16, a.ld_y, a.M, a.N, a.K, lpr, e, a.alpha, a.beta);
  return cudaGetLastError();
}

template <int F, typename TX, int RG, int S, int KU>
cudaError_t run_vec(const Args& a) {
  constexpr int kGroups = kWarps / S;
  const int64_t rows_per_block = static_cast<int64_t>(kGroups) * RG;
  const unsigned blocks = static_cast<unsigned>((a.N + rows_per_block - 1) / rows_per_block);
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(blocks, kThreads, a.stream, attr);
  cudaLaunchKernelEx(&cfg, gemv_vec_kernel<F, TX, RG, S, KU>, static_cast<const TX*>(a.x), a.ld_x, a.w, a.K, a.bias,
                     a.bias_bf16, a.y, a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  return cudaGetLastError();
}

// The vec kernel's shape for one launch: S warps split each row's steps, RG rows share a warp and KU
// steps' loads are in flight. The heuristic is from the sweep on the 5070 Ti.
struct VecConfig {
  int S;
  int RG;
  int KU;
};

// Every shape the vec kernel is built for. The autotuner's variant index picks one of these.
constexpr VecConfig kVariants[] = {{1, 1, 1}, {1, 1, 2}, {1, 1, 4}, {1, 2, 1}, {2, 1, 1},
                                   {2, 1, 2}, {2, 1, 4}, {4, 1, 1}, {4, 1, 2}, {4, 1, 4},
                                   {8, 1, 1}, {8, 1, 2}, {8, 1, 4}};
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

VecConfig choose_vec(int N, int K, int M, int F) {
  const int step = F == LX_W_BF16 ? 128 : (F == LX_W_NVFP4 ? 512 : 256);  // k per warp step
  const int steps = (K + step - 1) / step;
  int S = 1;
  while (S < 8 && static_cast<int64_t>(N) * S < kWarpTarget && steps / (2 * S) >= 1) S *= 2;
  // Rows per warp and steps in flight, from the sweep. At M = 1 a warp that owns the whole row takes
  // two rows, so one x load serves both; split rows already have enough warps in flight. The byte
  // formats (int8, fp8, nvfp4) share this rule: each moves 256 weight bytes per warp step.
  int RG = 1;
  int KU = 2;
  if (F != LX_W_BF16 && M == 1) {
    RG = S == 1 ? 2 : 1;
    KU = 1;
  }
  while (KU > 1 && KU > steps / S) KU /= 2;
  return VecConfig{S, RG, KU};
}

// Dispatch to the instantiation for one VecConfig. Only the variants in kVariants are built.
template <int F, typename TX>
cudaError_t dispatch_vec(const Args& a, const VecConfig& c) {
#define LX_VEC_CASE(S_, RG_, KU_) \
  if (c.S == S_ && c.RG == RG_ && c.KU == KU_) return run_vec<F, TX, RG_, S_, KU_>(a);
  LX_VEC_CASE(1, 1, 1) LX_VEC_CASE(1, 1, 2) LX_VEC_CASE(1, 1, 4) LX_VEC_CASE(1, 2, 1)
  LX_VEC_CASE(2, 1, 1) LX_VEC_CASE(2, 1, 2) LX_VEC_CASE(2, 1, 4)
  LX_VEC_CASE(4, 1, 1) LX_VEC_CASE(4, 1, 2) LX_VEC_CASE(4, 1, 4)
  LX_VEC_CASE(8, 1, 1) LX_VEC_CASE(8, 1, 2) LX_VEC_CASE(8, 1, 4)
#undef LX_VEC_CASE
  return cudaErrorInvalidValue;
}

// The tunable kernel name per format: the autotuner keys its table on (name, M bucket, N, K), and a
// format changes which variant is fastest, so each format has its own entry.
const char* tune_name(int F) {
  switch (F) {
    case LX_W_BF16: return "gemv_w_bf16";
    case LX_W_INT8: return "gemv_w_int8";
    case LX_W_FP8: return "gemv_w_fp8";
    default: return "gemv_w_nvfp4";
  }
}

// The variant names, in kVariants order, for each format.
#define LX_GEMV_VARIANT_NAMES                                                                          \
  "s1r1k1", "s1r1k2", "s1r1k4", "s1r2k1", "s2r1k1", "s2r1k2", "s2r1k4", "s4r1k1", "s4r1k2", "s4r1k4", \
      "s8r1k1", "s8r1k2", "s8r1k4"
const lx::TunableRegistration kRegBf16("gemv_w_bf16", {LX_GEMV_VARIANT_NAMES});
const lx::TunableRegistration kRegInt8("gemv_w_int8", {LX_GEMV_VARIANT_NAMES});
const lx::TunableRegistration kRegFp8("gemv_w_fp8", {LX_GEMV_VARIANT_NAMES});
const lx::TunableRegistration kRegNvfp4("gemv_w_nvfp4", {LX_GEMV_VARIANT_NAMES});

template <int F, typename TX>
cudaError_t launch(const Args& a) {
  // Vector loads need 16-byte alignment: the weight base, and the x base and row stride. K must be a
  // multiple of the lane's values per step, so each lane's word is whole within its row.
  const bool aligned = reinterpret_cast<uintptr_t>(a.w.data) % 16 == 0 &&
                       ((reinterpret_cast<uintptr_t>(a.x) | (static_cast<uint64_t>(a.ld_x) * sizeof(TX))) % 16 == 0);
  if (a.K <= kSmallK || a.K % kVal<F> != 0 || !aligned) return launch_small<F, TX>(a);
  const int tuned = lx_tune_get(tune_name(F), a.M, a.N, a.K);
  const VecConfig c = tuned >= 0 && tuned < kNumVariants ? kVariants[tuned] : choose_vec(a.N, a.K, a.M, F);
  return dispatch_vec<F, TX>(a, c);
}

template <int F>
cudaError_t launch_format(const Args& a) {
  return a.x_bf16 ? launch<F, bf16>(a) : launch<F, float>(a);
}

bool valid_dtype(lx_dtype d) { return d == LX_F32 || d == LX_BF16; }

}  // namespace

extern "C" int lx_gemv_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                         lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                         float beta, void* stream) {
  if (M == 0 || N == 0) return 0;
  if (M < 0 || M > kMaxM || K <= 0 || ld_x < K || ld_y < N) return static_cast<int>(cudaErrorInvalidValue);
  if (!valid_dtype(x_dtype) || !valid_dtype(y_dtype) || !valid_dtype(bias_dtype))
    return static_cast<int>(cudaErrorInvalidValue);
  if (!w || !w->data || !x || !y) return static_cast<int>(cudaErrorInvalidValue);
  switch (w->type) {
    case LX_W_BF16:
      break;
    case LX_W_INT8:
    case LX_W_FP8:
      if (!w->scale) return static_cast<int>(cudaErrorInvalidValue);
      break;
    case LX_W_NVFP4:
      if (!w->scale || K % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }

  const Args a{
      .x = x,
      .ld_x = ld_x,
      .w = *w,
      .bias = bias,
      .bias_bf16 = bias_dtype == LX_BF16,
      .y = y,
      .y_bf16 = y_dtype == LX_BF16,
      .ld_y = ld_y,
      .M = M,
      .N = N,
      .K = K,
      .alpha = alpha,
      .beta = beta,
      .stream = static_cast<cudaStream_t>(stream),
      .x_bf16 = x_dtype == LX_BF16,
  };
  cudaError_t e = cudaErrorInvalidValue;
  switch (w->type) {
    case LX_W_BF16: e = launch_format<LX_W_BF16>(a); break;
    case LX_W_INT8: e = launch_format<LX_W_INT8>(a); break;
    case LX_W_FP8: e = launch_format<LX_W_FP8>(a); break;
    case LX_W_NVFP4: e = launch_format<LX_W_NVFP4>(a); break;
  }
  return static_cast<int>(e);
}

// The original entry point: a bf16 or int8 weight with a per-row scale (lx_gemv_w with a lx_weight).
extern "C" int lx_gemv(const void* x, lx_dtype x_dtype, int ld_x, const void* w, lx_wtype w_type, const float* w_scale,
                       const void* bias, lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K,
                       float alpha, float beta, void* stream) {
  if (w_type != LX_W_BF16 && w_type != LX_W_INT8) return static_cast<int>(cudaErrorInvalidValue);
  lx_weight lw{};
  lw.type = w_type;
  lw.data = w;
  lw.scale = w_scale;
  lw.global = 1.0f;
  return lx_gemv_w(x, x_dtype, ld_x, &lw, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
}

extern "C" int lx_quantize_w8(const void* w, int8_t* q, float* scale, int64_t rows, int cols, void* stream) {
  if (rows <= 0) return 0;
  if (cols <= 0 || !w || !q || !scale) return static_cast<int>(cudaErrorInvalidValue);
  const int64_t blocks = (rows + kWarps - 1) / kWarps;
  quantize_w8_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const bf16*>(w), q, scale, rows, cols);
  return lx::launch_status();
}
