// Per-token activation quantization for the low-precision GEMM maths (kernels.h, lx_quantize_act).
//
// Each thread owns 16 consecutive values per step, one 32-byte read of bf16 or two of f32, which for FP4
// is exactly one scale block, so the block absmax needs no reduction; consecutive threads read consecutive
// chunks. The row is read once and held in registers, as loaded, between the absmax and the quantization:
//   rows up to 4096 bf16 / 2048 f32 values   one warp per row, eight rows per block, no block barrier
//   longer rows, up to 16384 values           one 256-thread block per row, one barrier for the absmax
// Rows longer still, or not 16-byte aligned, are read twice (the second time from L1/L2). The roundings
// are the spec's, written out so the PyTorch reference in tests/kernels/test_quant_act.py reproduces them
// bit for bit; on sm_120 the e2m1 rounding is the hardware conversion, which rounds the same way.
#include <cuda_fp8.h>

#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/mma.cuh"

namespace {

constexpr int kRows = 8;  // warps (rows) per block

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

#ifndef LX_HAS_BLOCK_SCALE_MMA
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
#endif

__device__ __forceinline__ uint32_t e4m3_rn(float v) {
  return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
}

// Eight values times r to e2m1 codes, the even k in the low nibble. A value that rounds to zero is code 0
// with no sign, as the reference and the checkpoints store it (the hardware conversion keeps the sign).
__device__ __forceinline__ uint32_t e2m1x8_rn(const float* v, float r) {
  uint32_t w = 0;
#ifdef LX_HAS_BLOCK_SCALE_MMA
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    uint16_t b;
    asm("{\n .reg .b8 t;\n cvt.rn.satfinite.e2m1x2.f32 t, %1, %2;\n cvt.u16.u8 %0, t;\n}"
        : "=h"(b)
        : "f"(v[2 * i + 1] * r), "f"(v[2 * i] * r));
    w |= static_cast<uint32_t>(b) << (8 * i);
  }
  const uint32_t nz = (w | (w >> 1) | (w >> 2)) & 0x11111111u;  // a nibble's magnitude is not zero
  w = (w & 0x77777777u) | (w & (nz << 3));
#else
#pragma unroll
  for (int i = 0; i < 8; ++i) w |= e2m1_rn(v[i] * r) << (4 * i);
#endif
  return w;
}

// Quantizes one 16-value chunk at k of a row whose absmax is amax; the per-row factors come in precomputed.
template <int kMath>
struct RowQuant {
  float inv = 0.0f, global = 1.0f;  // INT8 / FP8: kMax / amax; FP4: the per-token global
  __device__ __forceinline__ RowQuant(float amax) {
    if constexpr (kMath == LX_MATH_FP4) {
      global = amax > 0.0f ? amax / (448.0f * 6.0f) : 1.0f;
    } else {
      constexpr float kMax = kMath == LX_MATH_INT8 ? 127.0f : 448.0f;
      inv = amax > 0.0f ? kMax / amax : 0.0f;
      global = amax > 0.0f ? amax / kMax : 1.0f;  // the row scale
    }
  }
  __device__ __forceinline__ void emit(const float (&v)[16], int k, int K, bool full, uint8_t* qr, uint8_t* bsr) const {
    if constexpr (kMath == LX_MATH_INT8 || kMath == LX_MATH_FP8) {
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
      if (full) {
        *reinterpret_cast<uint4*>(qr + k) = make_uint4(w[0], w[1], w[2], w[3]);
      } else {
        for (int i = 0; i < 16 && k + i < K; ++i) qr[k + i] = static_cast<uint8_t>(w[i / 4] >> (8 * (i % 4)));
      }
    } else {
      float bmax = 0.0f;
#pragma unroll
      for (int i = 0; i < 16; ++i) bmax = fmaxf(bmax, fabsf(v[i]));
      const uint32_t s8 = e4m3_rn(bmax / (6.0f * global));
      const float sb = lx::mma::e4m3_to_f32(s8);
      const float r = sb > 0.0f ? 1.0f / (sb * global) : 0.0f;  // a zero block: every code 0
      uint32_t w[2];
#pragma unroll
      for (int h = 0; h < 2; ++h) w[h] = e2m1x8_rn(v + 8 * h, r);
      bsr[k / 16] = static_cast<uint8_t>(s8);
      if (full) {
        *reinterpret_cast<uint2*>(qr + k / 2) = make_uint2(w[0], w[1]);
      } else {
        for (int i = 0; i < 8 && k + 2 * i < K; ++i) qr[k / 2 + i] = static_cast<uint8_t>(w[i / 4] >> (8 * (i % 4)));
      }
    }
  }
};

// A thread's chunk of 16 values, as loaded: bf16 in two 16-byte words, f32 in four.
template <typename TX>
struct Chunk {
  uint4 u[sizeof(TX) == 4 ? 4 : 2];
  __device__ __forceinline__ void load(const TX* p) {
#pragma unroll
    for (int i = 0; i < static_cast<int>(sizeof(u) / 16); ++i) u[i] = reinterpret_cast<const uint4*>(p)[i];
  }
  __device__ __forceinline__ void values(float (&v)[16]) const {
    if constexpr (sizeof(TX) == 4) {
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        v[4 * i] = __uint_as_float(u[i].x);
        v[4 * i + 1] = __uint_as_float(u[i].y);
        v[4 * i + 2] = __uint_as_float(u[i].z);
        v[4 * i + 3] = __uint_as_float(u[i].w);
      }
    } else {
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        const uint32_t w[4] = {u[i].x, u[i].y, u[i].z, u[i].w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          v[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
          v[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        }
      }
    }
  }
};

// W warps per row (1: eight rows per block; 8: a block per row). R > 0: each thread holds up to R chunks
// in registers (the row is at most R * 32 * W * 16 values, 16-byte aligned, K % 16 == 0); R == 0: the row is
// read twice, with a scalar path for tails and misaligned rows.
template <typename TX, int kMath, int W, int R>
__global__ void __launch_bounds__(kRows * 32) quantize_act_kernel(const TX* __restrict__ x, int ld_x,
                                                                  uint8_t* __restrict__ q, int ld_q,
                                                                  float* __restrict__ scale,
                                                                  uint8_t* __restrict__ block_scale, int ld_bs, int M,
                                                                  int K, bool vec) {
  constexpr int kTPR = 32 * W;  // threads per row
  __shared__ float red[kRows];
  const int warp = threadIdx.x >> 5;
  const int t = threadIdx.x % kTPR;
  const int64_t row = static_cast<int64_t>(blockIdx.x) * (kRows / W) + threadIdx.x / kTPR;
  const bool live = row < M;
  if (W == 1 && !live) return;  // no barrier in the warp-per-row form
  const TX* xr = x + (live ? row : 0) * ld_x;
  uint8_t* qr = q + row * ld_q;
  uint8_t* bsr = kMath == LX_MATH_FP4 ? block_scale + row * ld_bs : nullptr;

  float amax = 0.0f;
  Chunk<TX> c[R > 0 ? R : 1];
  if constexpr (R > 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int k = (r * kTPR + t) * 16;
      if (live && k < K) {
        c[r].load(xr + k);
        float v[16];
        c[r].values(v);
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
      }
    }
  } else if (live) {
    for (int k = t * 16; k < K; k += kTPR * 16) {
      float v[16];
      if (vec && k + 16 <= K) load16(xr + k, v);
      else load16_tail(xr + k, K - k, v);
#pragma unroll
      for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(v[i]));
    }
  }
  amax = lx::warp_max(amax);
  if constexpr (W > 1) {
    if ((threadIdx.x & 31) == 0) red[warp] = amax;
    __syncthreads();
    amax = 0.0f;
#pragma unroll
    for (int i = 0; i < W; ++i) amax = fmaxf(amax, red[(warp / W) * W + i]);
  }
  if (!live) return;
  const RowQuant<kMath> rq(amax);
  if (t == 0) scale[row] = rq.global;
  if constexpr (R > 0) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const int k = (r * kTPR + t) * 16;
      if (k < K) {
        float v[16];
        c[r].values(v);
        rq.emit(v, k, K, true, qr, bsr);
      }
    }
  } else {
    for (int k = t * 16; k < K; k += kTPR * 16) {
      float v[16];
      const bool full = vec && k + 16 <= K;
      if (full) load16(xr + k, v);
      else load16_tail(xr + k, K - k, v);
      rq.emit(v, k, K, full, qr, bsr);
    }
  }
}

template <typename TX, int kMath>
int launch(const void* x, int ld_x, void* q, int ld_q, float* scale, void* block_scale, int ld_bs, int M, int K,
           cudaStream_t s) {
  const uintptr_t align = reinterpret_cast<uintptr_t>(x) | (static_cast<uintptr_t>(ld_x) * sizeof(TX)) |
                          reinterpret_cast<uintptr_t>(q) | static_cast<uintptr_t>(ld_q);
  const bool vec = (align & 15) == 0 && K % 16 == 0;
  constexpr int kWarpK = sizeof(TX) == 4 ? 2048 : 4096;  // longest row a warp holds (64 registers of chunks)
  auto run = [&](auto kernel, int rows_per_block) {
    kernel<<<(M + rows_per_block - 1) / rows_per_block, kRows * 32, 0, s>>>(
        static_cast<const TX*>(x), ld_x, static_cast<uint8_t*>(q), ld_q, scale, static_cast<uint8_t*>(block_scale),
        ld_bs, M, K, vec);
  };
  if (!vec) run(quantize_act_kernel<TX, kMath, 1, 0>, kRows);
  else if (K <= 512) run(quantize_act_kernel<TX, kMath, 1, 1>, kRows);
  else if (K <= 1024) run(quantize_act_kernel<TX, kMath, 1, 2>, kRows);
  else if (K <= 2048) run(quantize_act_kernel<TX, kMath, 1, 4>, kRows);
  else if (K <= kWarpK) run(quantize_act_kernel<TX, kMath, 1, kWarpK / 512>, kRows);
  else if (K <= 4096 * 4) run(quantize_act_kernel<TX, kMath, kRows, 4>, 1);
  else run(quantize_act_kernel<TX, kMath, kRows, 0>, 1);
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
