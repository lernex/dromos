// Per-token activation quantization for the low-precision GEMM maths (kernels.h, lx_quantize_act).
//
// One block per row. The row is read twice: once for its absmax, once to quantize; the second read
// comes from L1/L2 (a row is at most a few tens of KB). Each thread owns 16 consecutive values per
// step, one 32-byte read of bf16 or two of f32, which for FP4 is exactly one scale block, so the block
// absmax needs no reduction. The roundings are the spec's, written out so the PyTorch reference in
// tests/kernels/test_quant_act.py reproduces them bit for bit.
#include <cuda_fp8.h>

#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/mma.cuh"

namespace {

constexpr int kThreads = 128;

template <typename TX>
__device__ __forceinline__ void load16(const TX* p, float (&v)[16]) {
  if constexpr (sizeof(TX) == 4) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const float4 t = reinterpret_cast<const float4*>(p)[i];
      v[4 * i] = t.x;
      v[4 * i + 1] = t.y;
      v[4 * i + 2] = t.z;
      v[4 * i + 3] = t.w;
    }
  } else {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint4 t = reinterpret_cast<const uint4*>(p)[i];
      const uint32_t w[4] = {t.x, t.y, t.z, t.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        v[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
        v[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
      }
    }
  }
}

// Scalar fallback for a tail shorter than 16 or a misaligned row: missing values read as 0.
template <typename TX>
__device__ __forceinline__ void load16_tail(const TX* p, int n, float (&v)[16]) {
#pragma unroll
  for (int i = 0; i < 16; ++i) v[i] = i < n ? lx::to_f32(p[i]) : 0.0f;
}

// Round to the nearest e2m1 value, ties to even, saturating at 6. Returns the 4-bit code.
__device__ __forceinline__ uint32_t e2m1_rn(float v) {
  const float a = fabsf(v);
  uint32_t m;  // magnitude code: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  if (a <= 0.25f) m = 0;        // tie at 0.25 goes to 0 (even)
  else if (a < 0.75f) m = 1;
  else if (a <= 1.25f) m = 2;   // tie at 0.75 to 1.0, at 1.25 to 1.0
  else if (a < 1.75f) m = 3;
  else if (a <= 2.5f) m = 4;    // tie at 1.75 to 2.0, at 2.5 to 2.0
  else if (a < 3.5f) m = 5;
  else if (a <= 5.0f) m = 6;    // tie at 3.5 to 4.0, at 5.0 to 4.0
  else m = 7;
  return m | (v < 0.0f && m != 0 ? 8u : 0u);
}

__device__ __forceinline__ uint32_t e4m3_rn(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

template <typename TX, int kMath>
__global__ void __launch_bounds__(kThreads) quantize_act_kernel(const TX* __restrict__ x, int ld_x, uint8_t* __restrict__ q,
                                                                int ld_q, float* __restrict__ scale,
                                                                uint8_t* __restrict__ block_scale, int ld_bs, int K,
                                                                bool vec) {
  __shared__ float red[32];
  const int64_t row = blockIdx.x;
  const TX* xr = x + row * ld_x;
  float amax = 0.0f;
  for (int k = threadIdx.x * 16; k < K; k += kThreads * 16) {
    float v[16];
    if (vec && k + 16 <= K) load16(xr + k, v);
    else load16_tail(xr + k, K - k, v);
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
  }
  amax = lx::warp_max(amax);
  if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = amax;
  __syncthreads();
  const int lane = threadIdx.x & 31;
  amax = lane < kThreads / 32 ? red[lane] : 0.0f;
  amax = lx::warp_max(amax);  // every warp reduces the same kThreads / 32 partials

  uint8_t* qr = q + row * ld_q;
  if constexpr (kMath == LX_MATH_INT8 || kMath == LX_MATH_FP8) {
    constexpr float kMax = kMath == LX_MATH_INT8 ? 127.0f : 448.0f;
    const float inv = amax > 0.0f ? kMax / amax : 0.0f;
    if (threadIdx.x == 0) scale[row] = amax > 0.0f ? amax / kMax : 1.0f;
    for (int k = threadIdx.x * 16; k < K; k += kThreads * 16) {
      float v[16];
      if (vec && k + 16 <= K) load16(xr + k, v);
      else load16_tail(xr + k, K - k, v);
      uint32_t w[4];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        uint32_t b[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
          const float s = v[4 * j + i] * inv;
          if constexpr (kMath == LX_MATH_INT8) b[i] = static_cast<uint8_t>(static_cast<int8_t>(fminf(fmaxf(rintf(s), -127.0f), 127.0f)));
          else b[i] = e4m3_rn(s);
        }
        w[j] = b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24);
      }
      if (vec && k + 16 <= K) {
        *reinterpret_cast<uint4*>(qr + k) = make_uint4(w[0], w[1], w[2], w[3]);
      } else {
        for (int i = 0; i < 16 && k + i < K; ++i) qr[k + i] = static_cast<uint8_t>(w[i / 4] >> (8 * (i % 4)));
      }
    }
  } else {
    // NVFP4: per-token global, one e4m3 scale per 16 values, e2m1 values.
    const float global = amax > 0.0f ? amax / (448.0f * 6.0f) : 1.0f;
    if (threadIdx.x == 0) scale[row] = global;
    uint8_t* bsr = block_scale + row * ld_bs;
    for (int k = threadIdx.x * 16; k < K; k += kThreads * 16) {
      float v[16];
      if (vec && k + 16 <= K) load16(xr + k, v);
      else load16_tail(xr + k, K - k, v);
      float bmax = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
      const uint32_t s8 = e4m3_rn(bmax / (6.0f * global));
      const float s = lx::mma::e4m3_to_f32(s8);
      const float d = s * global;
      uint32_t w[2] = {0, 0};
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        const uint32_t c = s > 0.0f ? e2m1_rn(v[i] / d) : 0u;
        w[i / 8] |= c << (4 * (i % 8));
      }
      bsr[k / 16] = static_cast<uint8_t>(s8);
      if (vec && k + 16 <= K) {
        *reinterpret_cast<uint2*>(qr + k / 2) = make_uint2(w[0], w[1]);
      } else {
        for (int i = 0; i < 8 && k + 2 * i < K; ++i) qr[k / 2 + i] = static_cast<uint8_t>(w[i / 4] >> (8 * (i % 4)));
      }
    }
  }
}

template <typename TX, int kMath>
int launch(const void* x, int ld_x, void* q, int ld_q, float* scale, void* block_scale, int ld_bs, int M, int K,
           cudaStream_t s) {
  const uintptr_t align = reinterpret_cast<uintptr_t>(x) | (static_cast<uintptr_t>(ld_x) * sizeof(TX)) |
                          reinterpret_cast<uintptr_t>(q) | static_cast<uintptr_t>(ld_q);
  const bool vec = (align & 15) == 0 && K % 16 == 0;
  quantize_act_kernel<TX, kMath><<<M, kThreads, 0, s>>>(static_cast<const TX*>(x), ld_x, static_cast<uint8_t*>(q), ld_q,
                                                        scale, static_cast<uint8_t*>(block_scale), ld_bs, K, vec);
  return lx::launch_status();
}

}  // namespace

extern "C" int lx_quantize_act(const void* x, lx_dtype x_dtype, int ld_x, lx_math math, void* q, int ld_q, float* scale,
                               void* block_scale, int ld_bs, int M, int K, void* stream) {
  if (M <= 0) return 0;
  if (K <= 0 || ld_x < K || !x || !q || !scale) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  switch (math) {
    case LX_MATH_INT8:
    case LX_MATH_FP8:
      if (ld_q < K) return static_cast<int>(cudaErrorInvalidValue);
      break;
    case LX_MATH_FP4:
      if (K % 16 != 0 || ld_q < K / 2 || ld_bs < K / 16 || !block_scale) return static_cast<int>(cudaErrorInvalidValue);
      break;
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
  LX_DISPATCH_DTYPE(x_dtype, TX, {
    switch (math) {
      case LX_MATH_INT8: return launch<TX, LX_MATH_INT8>(x, ld_x, q, ld_q, scale, block_scale, ld_bs, M, K, s);
      case LX_MATH_FP8: return launch<TX, LX_MATH_FP8>(x, ld_x, q, ld_q, scale, block_scale, ld_bs, M, K, s);
      default: return launch<TX, LX_MATH_FP4>(x, ld_x, q, ld_q, scale, block_scale, ld_bs, M, K, s);
    }
  });
  return 0;
}
