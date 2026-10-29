// Decode GEMV (kernels.h, "Decode GEMV" and lx_gemv_w): y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y.
//
// Decode is memory bound: a token streams the whole weight matrix once, so what the kernel spends
// per weight byte (L1 wavefronts, issue slots) decides whether the weight stream keeps DRAM busy.
//
// Lane mapping. In every format a lane takes 8 consecutive k of a row per step (16 B of bf16, 8 B of
// int8 or fp8, 4 B of nvfp4 plus the byte of its e4m3 block scale), so a warp step is 256 k and one
// contiguous run of the row. The x values for those 8 k are one 16-byte load per lane, contiguous
// across the warp: x costs the minimum L1 wavefronts in every format, and RG rows share each load. (A
// lane that took 16 or 32 k would read x as 2 or 4 strided 16-byte pieces, which is what made NVFP4
// L1-bound at M = 8: 8 of every 32 sector bytes used, x re-read per row.)
//
// Math. With bf16 x the products run on the bf16 values directly: x stays packed as loaded, the weight
// is expanded to bf16 pairs (exact in every format: int8, e4m3, and e2m1 times its e4m3 block scale
// all fit bf16's 8 significant bits), and each product is one f32 fused multiply-add on bf16 operands
// (FHFMA.BF16 on sm_100 and later; older targets widen to f32 first, with the same result bit for bit).
// So x is never converted, and the decode cost per weight does not grow with M. With f32 x, the weight
// is expanded to f32 instead.
//
// Pipelining. Each warp keeps a ring of D steps of loads in flight: a slot is refilled with the step
// D ahead as soon as it is consumed. The first D steps go out before the programmatic-dependent-launch
// wait, since weights do not depend on the previous kernel in the stream: when D covers a warp's whole
// share of the row, every weight byte of the launch is requested before the previous kernel finishes.
// Weight loads are cache-streaming so that x stays in L1.
//
// Two kernels, for every format:
//   vec    K > 256 with aligned rows and K % 8 == 0. A warp owns RG rows, and S warps may split the
//          steps of one row group, combining in shared memory in a fixed order, so split-K is
//          deterministic without atomics. M is a template bucket (1, 2, 4, 8), so the accumulators are
//          only as many as the launch needs. Which (S, RG, D) runs is measured, not derived: see
//          choose_vec.
//   small  K <= 256, or misaligned x or w: scalar loads. lpr lanes cover one row, and the 32 / lpr
//          rows of a warp reduce with xor shuffles inside their group.
// The per-row factor (the int8 / fp8 row scale, the nvfp4 global) is applied once per row:
// sum_k x * (q * s) = s * sum_k x * q.

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstring>
#include <type_traits>

#include "cuda/common.cuh"
#include "cuda/gemv_class.cuh"
#include "cuda/kernels.h"
#include "cuda/tunable.h"
#include "cuda/wformat.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;  // vec kernel: 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kMaxM = 8;
constexpr int kSmallK = 256;         // rows this short are cheaper as scalar loads
constexpr int kLaneK = 8;            // k per lane per step, in every format
constexpr int kStepK = 32 * kLaneK;  // k per warp step

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

// ---- the lane's slice of a row ---------------------------------------------------------------

// The 8 values a lane reads from one row in one step, as stored.
template <int F>
struct Slice;
template <>
struct Slice<LX_W_BF16> {
  uint4 w;
};
template <>
struct Slice<LX_W_INT8> {
  uint2 w;
};
template <>
struct Slice<LX_W_FP8> {
  uint2 w;
};
template <>
struct Slice<LX_W_NVFP4> {
  uint32_t w;  // 8 e2m1 codes, the low nibble of each byte holding the even k
  uint32_t s;  // the e4m3 scale of the 16-value block the 8 belong to (k0 is a multiple of 8)
};

// Where a lane's slices are: the bytes of the warp's first row at this lane's slice of the current step,
// and for NVFP4 the scale byte of that slice's block. A lane keeps one pointer per stream and the loop
// advances it once per D steps. Everything else a load adds is warp-uniform and comes from the
// parameter bank (Offsets) or is an immediate: step d is d * kStepBytes<F> further on (d * kStepScales
// scale bytes), row r of the group is Offsets::w[r] (Offsets::s[r]).
template <int F>
constexpr int kSliceBytes = F == LX_W_BF16 ? 16 : (F == LX_W_NVFP4 ? 4 : 8);
template <int F>
constexpr int kStepBytes = 32 * kSliceBytes<F>;
constexpr int kStepScales = kStepK / 16;
constexpr int kMaxRG = 8;

// Byte offsets the loop adds to its pointers, computed on the host so they sit in the kernel's
// parameter bank: the address adds take them as constant operands, and they cost no registers.
struct Offsets {
  int64_t x[kMaxM];   // row m of x (rows past M repeat row M - 1, whose sums are never stored)
  int64_t w[kMaxRG];  // row r of the group, in weight bytes
  int64_t s[kMaxRG];  // row r of the group, in NVFP4 scale bytes
};

struct LaneSrc {
  const uint8_t* w;  // row `first`, this lane's slice, current step
  const uint8_t* s;  // NVFP4: the matching scale byte
  int rows;          // rows of the group that exist (RG, fewer in the last group, <= 0 past N)
};

template <int F>
__device__ __forceinline__ LaneSrc lane_src(const lx_weight& w, int64_t first, int N, int K, int lane, int unit) {
  const int64_t row_bytes = F == LX_W_BF16 ? 2 * static_cast<int64_t>(K) : (F == LX_W_NVFP4 ? K / 2 : K);
  LaneSrc p;
  p.rows = static_cast<int>(min(static_cast<int64_t>(N) - first, static_cast<int64_t>(kMaxRG)));
  p.w = static_cast<const uint8_t*>(w.data) + first * row_bytes + lane * kSliceBytes<F> +
        static_cast<int64_t>(unit) * kStepBytes<F>;
  p.s = F == LX_W_NVFP4 ? static_cast<const uint8_t*>(w.scale) + first * (K / 16) + lane / 2 +
                              static_cast<int64_t>(unit) * kStepScales
                        : nullptr;
  return p;
}

// Streamed once per token, so evict-first: L1 is left to the x vector every row reuses.
template <int F, int kAhead>
__device__ __forceinline__ void load_slice(const LaneSrc& p, const Offsets& off, int r, Slice<F>& v) {
  const uint8_t* q = p.w + off.w[r] + kAhead * kStepBytes<F>;
  if constexpr (F == LX_W_BF16) {
    v.w = __ldcs(reinterpret_cast<const uint4*>(q));
  } else if constexpr (F == LX_W_NVFP4) {
    v.w = __ldcs(reinterpret_cast<const unsigned int*>(q));
    v.s = __ldcs(p.s + off.s[r] + kAhead * kStepScales);
  } else {
    v.w = __ldcs(reinterpret_cast<const uint2*>(q));
  }
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  const __nv_bfloat162 p = __floats2bfloat162_rn(lo, hi);
  uint32_t r;
  memcpy(&r, &p, sizeof(r));
  return r;
}

__device__ __forceinline__ float2 e4m3x2_to_f32(uint32_t pair) {
  return __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(pair), __NV_E4M3)));
}

// The nvfp4 slice as 4 half pairs with the block scale applied. Exact: e2m1 has 2 significant bits and
// e4m3 4, so the product has at most 6 and lies in [2^-10, 2688], inside f16 and bf16 alike.
__device__ __forceinline__ void nvfp4_to_half2(const Slice<LX_W_NVFP4>& v, __half2 (&h)[4]) {
  const __half2 scale =
      __half2(__nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v.s * 0x0101u), __NV_E4M3));
#pragma unroll
  for (int p = 0; p < 4; ++p) {
    const __half2 q =
        __half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(v.w >> (8 * p)), __NV_E2M1));
    h[p] = __hmul2(q, scale);
  }
}

// One instruction from an e4m3 or e2m1 pair to a bf16 pair exists on the arch- and family-specific
// Blackwell targets (sm_100a, sm_120a, ...); elsewhere the pair goes through f16 and f32, exactly.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000 && defined(__CUDA_ARCH_FAMILY_SPECIFIC__)
#define LX_GEMV_BF16X2_CVT 1
#else
#define LX_GEMV_BF16X2_CVT 0
#endif

// The e4m3 pair in the low 16 bits of `pair` as a bf16 pair (low half = low byte).
__device__ __forceinline__ uint32_t e4m3x2_to_bf16x2(uint32_t pair) {
#if LX_GEMV_BF16X2_CVT
  uint32_t r;
  asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<unsigned short>(pair)));
  return r;
#else
  const float2 f = e4m3x2_to_f32(pair);
  return pack_bf16(f.x, f.y);
#endif
}

// The e2m1 pair in the low byte of `byte` as a bf16 pair (low half = low nibble).
__device__ __forceinline__ uint32_t e2m1x2_to_bf16x2(uint32_t byte) {
#if LX_GEMV_BF16X2_CVT
  uint32_t r;
  asm("{.reg .b8 b; cvt.u8.u16 b, %1; cvt.rn.bf16x2.e2m1x2 %0, b;}"
      : "=r"(r)
      : "h"(static_cast<unsigned short>(byte)));
  return r;
#else
  const float2 f =
      __half22float2(__half2(__nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(byte), __NV_E2M1)));
  return pack_bf16(f.x, f.y);
#endif
}

__device__ __forceinline__ uint32_t mul_bf16x2(uint32_t a, uint32_t b) {
  __nv_bfloat162 x, y;
  memcpy(&x, &a, sizeof(a));
  memcpy(&y, &b, sizeof(b));
  const __nv_bfloat162 z = __hmul2(x, y);
  uint32_t r;
  memcpy(&r, &z, sizeof(r));
  return r;
}

// The slice as 4 bf16 pairs (low half = lower k), without the per-row factor. Exact in every format.
template <int F>
__device__ __forceinline__ void slice_bf16x2(const Slice<F>& v, uint32_t (&p)[4]) {
  if constexpr (F == LX_W_BF16) {
    p[0] = v.w.x;
    p[1] = v.w.y;
    p[2] = v.w.z;
    p[3] = v.w.w;
  } else if constexpr (F == LX_W_INT8) {
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const uint32_t word = i == 0 ? v.w.x : v.w.y;
      p[2 * i] = pack_bf16(static_cast<float>(static_cast<int8_t>(word)), static_cast<float>(static_cast<int8_t>(word >> 8)));
      p[2 * i + 1] =
          pack_bf16(static_cast<float>(static_cast<int8_t>(word >> 16)), static_cast<float>(static_cast<int8_t>(word >> 24)));
    }
  } else if constexpr (F == LX_W_FP8) {
#pragma unroll
    for (int q = 0; q < 4; ++q) p[q] = e4m3x2_to_bf16x2((q < 2 ? v.w.x : v.w.y) >> (16 * (q & 1)));
  } else {
    // e2m1 times its block scale, multiplied in bf16: exact, as above. The conversion reads byte 0 or
    // byte 2 of a register, so bytes 1 and 3 come from one rotate of the word.
    const uint32_t scale = e4m3x2_to_bf16x2(__byte_perm(v.s, 0, 0));
    const uint32_t odd = __funnelshift_r(v.w, v.w, 8);
    p[0] = mul_bf16x2(e2m1x2_to_bf16x2(v.w), scale);
    p[1] = mul_bf16x2(e2m1x2_to_bf16x2(odd), scale);
    p[2] = mul_bf16x2(e2m1x2_to_bf16x2(v.w >> 16), scale);
    p[3] = mul_bf16x2(e2m1x2_to_bf16x2(odd >> 16), scale);
  }
}

// The slice as 8 f32, without the per-row factor (the f32-x path).
template <int F>
__device__ __forceinline__ void slice_f32(const Slice<F>& v, float (&f)[8]) {
  if constexpr (F == LX_W_BF16) {
    const uint32_t words[4] = {v.w.x, v.w.y, v.w.z, v.w.w};
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      f[2 * q] = __uint_as_float(words[q] << 16);
      f[2 * q + 1] = __uint_as_float(words[q] & 0xffff0000u);
    }
  } else if constexpr (F == LX_W_INT8) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[i] = static_cast<float>(static_cast<int8_t>(v.w.x >> (8 * i)));
      f[4 + i] = static_cast<float>(static_cast<int8_t>(v.w.y >> (8 * i)));
    }
  } else if constexpr (F == LX_W_FP8) {
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 t = e4m3x2_to_f32((q < 2 ? v.w.x : v.w.y) >> (16 * (q & 1)));
      f[2 * q] = t.x;
      f[2 * q + 1] = t.y;
    }
  } else {
    __half2 h[4];
    nvfp4_to_half2(v, h);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const float2 t = __half22float2(h[q]);
      f[2 * q] = t.x;
      f[2 * q + 1] = t.y;
    }
  }
}

// c + a.lo * b.lo + a.hi * b.hi for bf16 pairs a and b: two f32 fused multiply-adds in k order, each
// product exact and each sum rounded once. sm_100 and later take the bf16 halves as operands.
__device__ __forceinline__ float fma_bf16x2(uint32_t a, uint32_t b, float c) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 1000
  asm("fma.rn.f32.bf16 %0, %1, %2, %0;"
      : "+f"(c)
      : "h"(static_cast<unsigned short>(a)), "h"(static_cast<unsigned short>(b)));
  asm("fma.rn.f32.bf16 %0, %1, %2, %0;"
      : "+f"(c)
      : "h"(static_cast<unsigned short>(a >> 16)), "h"(static_cast<unsigned short>(b >> 16)));
#else
  c = fmaf(__uint_as_float(a << 16), __uint_as_float(b << 16), c);
  c = fmaf(__uint_as_float(a & 0xffff0000u), __uint_as_float(b & 0xffff0000u), c);
#endif
  return c;
}

// ---- vec kernel ------------------------------------------------------------------------------

// f(integral_constant<int, i>) for i = 0 .. N - 1, so i can be a template argument.
template <int N, int I = 0, typename Fn>
__device__ __forceinline__ void static_for(Fn&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    static_for<N, I + 1>(f);
  }
}

// The step kAhead steps past the pointers' current one, for the RG rows, into s, if this lane reads it.
// A slot that is not loaded keeps what it held, and is not consumed (rows past N keep their zeros).
template <int F, int RG, int kAhead>
__device__ __forceinline__ void load_step(const LaneSrc& src, const Offsets& off, bool live, Slice<F> (&s)[RG]) {
  if (live) {
#pragma unroll
    for (int r = 0; r < RG; ++r)
      if (r < src.rows) load_slice<F, kAhead>(src, off, r, s[r]);
  }
}

// One step: the RG slices against the lane's 8 x values of each of the MT rows of x, kAhead steps past
// xl (rows past M repeat row M - 1, whose sums are never stored, so the loop has no branch per row).
// The x loads all go out before the weights are expanded.
template <int F, typename TX, int MT, int RG, int kAhead>
__device__ __forceinline__ void consume(const Slice<F> (&s)[RG], const TX* xl, const Offsets& off,
                                        float (&acc)[RG][MT]) {
  const char* xb = reinterpret_cast<const char*>(xl) + kAhead * kStepK * static_cast<int>(sizeof(TX));
  if constexpr (std::is_same_v<TX, bf16>) {
    uint4 xv[MT];
#pragma unroll
    for (int m = 0; m < MT; ++m)
      xv[m] = __ldg(reinterpret_cast<const uint4*>(xb + off.x[m]));
    uint32_t wv[RG][4];
#pragma unroll
    for (int r = 0; r < RG; ++r) slice_bf16x2<F>(s[r], wv[r]);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
#pragma unroll
      for (int r = 0; r < RG; ++r) {
        float a = acc[r][m];
        a = fma_bf16x2(xv[m].x, wv[r][0], a);
        a = fma_bf16x2(xv[m].y, wv[r][1], a);
        a = fma_bf16x2(xv[m].z, wv[r][2], a);
        a = fma_bf16x2(xv[m].w, wv[r][3], a);
        acc[r][m] = a;
      }
    }
  } else {
    float4 xv[MT][2];
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float4* q = reinterpret_cast<const float4*>(xb + off.x[m]);
      xv[m][0] = __ldg(q);
      xv[m][1] = __ldg(q + 1);
    }
    float wf[RG][8];
#pragma unroll
    for (int r = 0; r < RG; ++r) slice_f32<F>(s[r], wf[r]);
#pragma unroll
    for (int m = 0; m < MT; ++m) {
      const float xf[8] = {xv[m][0].x, xv[m][0].y, xv[m][0].z, xv[m][0].w,
                           xv[m][1].x, xv[m][1].y, xv[m][1].z, xv[m][1].w};
#pragma unroll
      for (int r = 0; r < RG; ++r)
#pragma unroll
        for (int t = 0; t < 8; ++t) acc[r][m] = fmaf(wf[r][t], xf[t], acc[r][m]);
    }
  }
}

// Sums N values over the warp, leaving each lane with whole sums: each xor stage trades half the
// values a lane still holds for its partner's other half, so N values take N - 1 shuffles (plus one per
// stage left once a lane is down to one value) instead of 5 per value. After it, lane l holds in v[i]
// the sum of value out_index<N>(l, i).
template <int N, int OFF>
__device__ __forceinline__ void reduce_scatter(float* v, int lane) {
  if constexpr (OFF >= 1) {
    if constexpr (N > 1) {
      const bool upper = (lane & OFF) != 0;
#pragma unroll
      for (int i = 0; i < N / 2; ++i) {
        const float send = upper ? v[i] : v[i + N / 2];
        const float keep = upper ? v[i + N / 2] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, OFF);
      }
      reduce_scatter<N / 2, OFF / 2>(v, lane);
    } else {
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], OFF);
      reduce_scatter<1, OFF / 2>(v, lane);
    }
  }
}

template <int N>
__device__ __forceinline__ int out_index(int lane, int i) {
  int idx = i, n = N;
#pragma unroll
  for (int off = 16; off >= 1; off >>= 1) {
    if (n > 1) {
      n >>= 1;
      if (lane & off) idx += n;
    }
  }
  return idx;
}

// Block b owns kWarps / S row groups of RG rows each. Warp w is part `w % S` of group `w / S`. MT is the
// M bucket (M <= MT). G picks the loop: 0 runs whole rounds of D steps and then the last few, 1 checks
// the bound at every step. The round loop schedules better and wins on most shapes, but at M = 8 with
// RG >= 2 it costs registers: the compiler then crosses the line of 2 blocks per SM at RG = 4 (and of 3
// at RG = 2) that the checked loop stays under, and a grid that needs that residency runs slower.
template <int F, typename TX, int MT, int RG, int S, int D, int G>
__global__ void __launch_bounds__(kThreads)
gemv_vec_kernel(const TX* __restrict__ x, const __grid_constant__ Offsets off, const lx_weight w, int K,
                const void* __restrict__ bias, bool bias_bf16, void* __restrict__ y, bool y_bf16, int ld_y, int M, int N,
                float alpha, float beta) {
  constexpr int kGroups = kWarps / S;
  constexpr int kOut = RG * MT;                          // sums per warp
  constexpr int kHeld = kOut > 32 ? kOut / 32 : 1;       // whole sums per lane after the reduction
  constexpr int kSpread = kOut >= 32 ? 1 : 32 / kOut;    // lanes that end up holding each sum
  __shared__ float partial[S > 1 ? kWarps : 1][kOut];  // split-K partial sums, used when S > 1

  // Let the next kernel in the stream start its weight loads while this one runs.
  lx::pdl_trigger();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = warp / S, part = warp % S;
  const int64_t first = (static_cast<int64_t>(blockIdx.x) * kGroups + group) * RG;
  // The S warps of a group take contiguous, evenly sized runs of each row's steps.
  const int steps = (K + kStepK - 1) / kStepK;
  const int u0 = (part * steps) / S;
  const int u1 = ((part + 1) * steps) / S;
  // This lane reads steps u0 <= u < lim: its 8 k of step u start at u * kStepK + lane_k, so it may stop
  // a step early at the end of the row, and a group past N reads nothing.
  const int lane_k = kLaneK * lane;
  int lim = K > lane_k ? min(u1, (K - lane_k + kStepK - 1) / kStepK) : u0;
  if (first >= N) lim = u0;

  LaneSrc src = lane_src<F>(w, first, N, K, lane, u0);

  // The first D steps go out before the dependency wait; x and y are read after it.
  Slice<F> ring[D][RG];
  memset(&ring, 0, sizeof(ring));
  static_for<D>([&](auto d) { load_step<F, RG, d>(src, off, u0 + d < lim, ring[d]); });

  const TX* xl = x + lane_k + u0 * kStepK;
  lx::pdl_wait();

  float acc[RG][MT];
#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;

  if constexpr (G == 0) {
    // Whole rounds of D steps: slot d holds step j + d, and is refilled with step j + d + D once used.
    int j = u0;
    for (; j + D <= lim; j += D) {
      static_for<D>([&](auto d) {
        consume<F, TX, MT, RG, d>(ring[d], xl, off, acc);
        load_step<F, RG, d + D>(src, off, j + d + D < lim, ring[d]);
      });
      xl += D * kStepK;
      src.w += D * kStepBytes<F>;
      if constexpr (F == LX_W_NVFP4) src.s += D * kStepScales;
    }
    // The last lim - j < D steps are already in slots 0 .. lim - j - 1.
    static_for<D - 1>([&](auto d) {
      if (d < lim - j) consume<F, TX, MT, RG, d>(ring[d], xl, off, acc);
    });
  } else {
    for (int j = u0; j < lim; j += D) {
      static_for<D>([&](auto d) {
        if (j + d < lim) {
          consume<F, TX, MT, RG, d>(ring[d], xl, off, acc);
          load_step<F, RG, d + D>(src, off, j + d + D < lim, ring[d]);
        }
      });
      xl += D * kStepK;
      src.w += D * kStepBytes<F>;
      if constexpr (F == LX_W_NVFP4) src.s += D * kStepScales;
    }
  }

  float v[kOut];
#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) v[r * MT + m] = acc[r][m];
  reduce_scatter<kOut, 16>(v, lane);
  const bool holder = lane % kSpread == 0;

  if constexpr (S == 1) {
    if (holder) {
#pragma unroll
      for (int i = 0; i < kHeld; ++i) {
        const int idx = out_index<kOut>(lane, i);
        const int r = idx / MT, m = idx % MT;
        const int64_t n = first + r;
        if (m < M && n < N) {
          const float rf = lx::WordFormat<F>::row_factor(w, n);
          store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, rf * v[i] + bias_at(bias, bias_bf16, n), alpha,
                  beta);
        }
      }
    }
  } else {
    if (holder) {
#pragma unroll
      for (int i = 0; i < kHeld; ++i) partial[warp][out_index<kOut>(lane, i)] = v[i];
    }
    __syncthreads();  // every thread reaches this: the early-out is per group, not per block
    // Each output is summed over the S parts in a fixed order, so the result never depends on timing.
    for (int o = threadIdx.x; o < kGroups * RG * M; o += kThreads) {
      const int g = o / (RG * M), rem = o % (RG * M), r = rem / M, m = rem % M;
      const int64_t n = (static_cast<int64_t>(blockIdx.x) * kGroups + g) * RG + r;
      if (n >= N) continue;
      float sum = 0.0f;
      for (int p = 0; p < S; ++p) sum += partial[g * S + p][r * MT + m];
      const float rf = lx::WordFormat<F>::row_factor(w, n);
      store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, rf * sum + bias_at(bias, bias_bf16, n), alpha, beta);
    }
  }
}

// ---- small kernel ----------------------------------------------------------------------------

// One warp per block. The warp covers 32 / lpr rows: lane c of a row's group of lpr lanes reads the
// e consecutive inputs starting at c * e. Scalar loads, so any stride or alignment works.
//
// kVec: e == 8 (K a power of two, or any K that is 8 * lpr), K % 8 == 0, aligned rows. Then a lane's 8 weights are
// one 8- or 16-byte load, fetched before the dependency wait (weights never depend on the previous kernel), and each
// x row is one vector load (16 bytes of bf16, or two float4 of f32). The products and their fma order are exactly
// the scalar path's (k ascending within the lane, then the same xor tree over the lpr lanes), so the bits are the same.
template <int F>
__device__ __forceinline__ Slice<F> load_row8(const lx_weight& w, int64_t n, int K, int c) {
  Slice<F> s;
  if constexpr (F == LX_W_BF16) {
    s.w = __ldcs(reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(w.data) + n * K + c * 8));
  } else {
    s.w = __ldcs(reinterpret_cast<const uint2*>(static_cast<const uint8_t*>(w.data) + n * K + c * 8));
  }
  return s;
}

// The 8 x values at p as f32, in k order.
template <typename TX>
__device__ __forceinline__ void load_x8(const TX* p, float (&f)[8]);
template <>
__device__ __forceinline__ void load_x8<bf16>(const bf16* p, float (&f)[8]) {
  const uint4 v = __ldg(reinterpret_cast<const uint4*>(p));
  const uint32_t words[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
  for (int q = 0; q < 4; ++q) {
    f[2 * q] = __uint_as_float(words[q] << 16);
    f[2 * q + 1] = __uint_as_float(words[q] & 0xffff0000u);
  }
}
template <>
__device__ __forceinline__ void load_x8<float>(const float* p, float (&f)[8]) {
  const float4 a = __ldg(reinterpret_cast<const float4*>(p));
  const float4 b = __ldg(reinterpret_cast<const float4*>(p) + 1);
  f[0] = a.x, f[1] = a.y, f[2] = a.z, f[3] = a.w;
  f[4] = b.x, f[5] = b.y, f[6] = b.z, f[7] = b.w;
}

template <int F, typename TX, bool kVec>
__global__ void gemv_small_kernel(const TX* __restrict__ x, int ld_x, const lx_weight w,
                                  const void* __restrict__ bias, bool bias_bf16, void* __restrict__ y, bool y_bf16,
                                  int ld_y, int M, int N, int K, int lpr, int e, float alpha, float beta) {
  const int lane = threadIdx.x;
  const int slot = lane / lpr, c = lane % lpr;
  const int64_t n = static_cast<int64_t>(blockIdx.x) * (32 / lpr) + slot;

  Slice<F> row{};
  if constexpr (kVec) {
    if (n < N) row = load_row8<F>(w, n, K, c);  // before the wait: weights do not depend on the previous kernel
  }
  lx::pdl_trigger();
  lx::pdl_wait();  // x may be written by the previous kernel

  float acc[kMaxM];
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) acc[m] = 0.0f;
  if (n < N) {
    if constexpr (kVec) {
      float wf[8];
      slice_f32<F>(row, wf);
      for (int m = 0; m < M; ++m) {
        float xf[8];
        load_x8<TX>(x + static_cast<int64_t>(m) * ld_x + c * 8, xf);
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[m] = fmaf(wf[i], xf[i], acc[m]);
      }
    } else {
      for (int i = 0; i < e; ++i) {
        const int k = c * e + i;
        if (k >= K) break;
        const float wv = lx::element<F>(w, n, k, K);
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
          if (m < M) acc[m] = fmaf(wv, lx::to_f32(x[static_cast<int64_t>(m) * ld_x + k]), acc[m]);
      }
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
  lx::pdl_trigger();
  lx::pdl_wait();
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
  cfg.numAttrs = lx::pdl_old_on() ? 1 : 0;  // below sm_90 the launch is an ordinary one, in stream order
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
  // The vector path (gemv_small_kernel's kVec): 8 values a lane, a row whose x and weights are aligned for vector loads.
  const bool aligned = reinterpret_cast<uintptr_t>(a.w.data) % 16 == 0 &&
                       ((reinterpret_cast<uintptr_t>(a.x) | (static_cast<uint64_t>(a.ld_x) * sizeof(TX))) % 16 == 0);
  if constexpr (F != LX_W_NVFP4) {  // NVFP4 keeps its scalar path (its 8 values share two scale bytes)
    if (aligned && e == kLaneK && a.K % kLaneK == 0) {
      cudaLaunchKernelEx(&cfg, gemv_small_kernel<F, TX, true>, static_cast<const TX*>(a.x), a.ld_x, a.w, a.bias,
                         a.bias_bf16, a.y, a.y_bf16, a.ld_y, a.M, a.N, a.K, lpr, e, a.alpha, a.beta);
      return cudaGetLastError();
    }
  }
  cudaLaunchKernelEx(&cfg, gemv_small_kernel<F, TX, false>, static_cast<const TX*>(a.x), a.ld_x, a.w, a.bias,
                     a.bias_bf16, a.y, a.y_bf16, a.ld_y, a.M, a.N, a.K, lpr, e, a.alpha, a.beta);
  return cudaGetLastError();
}

template <int F, typename TX, int MT, int RG, int S, int D, int G>
cudaError_t run_vec(const Args& a) {
  constexpr int kGroups = kWarps / S;
  const int64_t rows_per_block = static_cast<int64_t>(kGroups) * RG;
  const unsigned blocks = static_cast<unsigned>((a.N + rows_per_block - 1) / rows_per_block);
  const int64_t row_bytes = F == LX_W_BF16 ? 2 * static_cast<int64_t>(a.K) : (F == LX_W_NVFP4 ? a.K / 2 : a.K);
  Offsets off{};
  for (int m = 0; m < kMaxM; ++m)
    off.x[m] = static_cast<int64_t>(m < a.M ? m : a.M - 1) * a.ld_x * static_cast<int64_t>(sizeof(TX));
  for (int r = 0; r < kMaxRG; ++r) {
    off.w[r] = r * row_bytes;
    off.s[r] = r * static_cast<int64_t>(a.K / 16);
  }
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(blocks, kThreads, a.stream, attr);
  cudaLaunchKernelEx(&cfg, gemv_vec_kernel<F, TX, MT, RG, S, D, G>, static_cast<const TX*>(a.x), off, a.w, a.K, a.bias,
                     a.bias_bf16, a.y, a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  return cudaGetLastError();
}

// The vec kernel's shape for one launch: S warps split each row's steps, a warp owns RG rows, D steps
// of loads are in flight per warp, and G picks the loop (see gemv_vec_kernel).
struct VecConfig {
  int S;
  int RG;
  int D;
  int G;
};

// Every shape the vec kernel is built for (with bf16 x), as X(S, RG, D, G). The autotuner's variant
// index picks one of these; the order is the variant numbering. The checked loop (G = 1) is only built
// for the M = 8 bucket, where it can win; below that its variants run the round loop.
#define LX_GEMV_VARIANTS(X)                                                                                \
  X(1, 1, 2, 0) X(1, 1, 4, 0) X(1, 1, 8, 0) X(1, 2, 2, 0) X(1, 2, 4, 0) X(1, 4, 2, 0) X(1, 4, 4, 0)        \
  X(1, 8, 2, 0) X(2, 1, 2, 0) X(2, 1, 4, 0) X(2, 1, 8, 0) X(2, 2, 2, 0) X(2, 2, 4, 0) X(2, 4, 2, 0)        \
  X(4, 1, 2, 0) X(4, 1, 4, 0) X(4, 2, 2, 0) X(4, 2, 4, 0) X(8, 1, 2, 0) X(8, 1, 4, 0) X(8, 2, 2, 0)        \
  X(1, 2, 3, 0) X(1, 4, 3, 0) X(2, 2, 3, 0) X(2, 4, 3, 0) X(4, 2, 3, 0) X(1, 2, 2, 1) X(1, 2, 3, 1)        \
  X(1, 2, 4, 1) X(2, 2, 2, 1) X(1, 4, 2, 1) X(1, 4, 3, 1) X(2, 4, 2, 1) X(2, 4, 3, 1)

#define LX_VARIANT_ENTRY(S_, RG_, D_, G_) VecConfig{S_, RG_, D_, G_},
constexpr VecConfig kVariants[] = {LX_GEMV_VARIANTS(LX_VARIANT_ENTRY)};
#undef LX_VARIANT_ENTRY
constexpr int kNumVariants = sizeof(kVariants) / sizeof(kVariants[0]);

// The default variant per format, shape and M bucket, from the autotuner (bench/sweep_gemv.py). Each
// (format, shape) has one order class, its split S, and every M bucket of the shape carries that same S: the
// class is the split whose worst loss over M = 1, 2, 4 and 8 is smallest, except that a shape whose M = 1 cost
// of that split is over 1% against the split fastest at M = 1 keeps that M = 1 class (the single-stream guard).
// Each bucket takes its fastest variant of its class. The class is what keeps a row's bits the same at every
// batch size (admits<F>, and the static assert on classes_agree). A shape not in the table takes the entry
// nearest to it in (log2 N, log2 K) for its format and M bucket.
struct Pick {
  int F;
  int MT;
  int N;
  int K;
  VecConfig c;
};

// clang-format off
constexpr Pick kPicks[] = {
    {LX_W_BF16, 1, 10304, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 2, 10304, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 4, 10304, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 8, 10304, 2048, {1, 2, 4, 1}},
    {LX_W_BF16, 1, 2048, 4096, {2, 2, 4, 0}},
    {LX_W_BF16, 2, 2048, 4096, {2, 1, 2, 0}},
    {LX_W_BF16, 4, 2048, 4096, {2, 4, 2, 0}},
    {LX_W_BF16, 8, 2048, 4096, {2, 4, 3, 0}},
    {LX_W_BF16, 1, 3072, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 2, 3072, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 4, 3072, 2048, {1, 8, 2, 0}},
    {LX_W_BF16, 8, 3072, 2048, {1, 2, 2, 1}},
    {LX_W_BF16, 1, 2048, 2048, {1, 1, 2, 0}},
    {LX_W_BF16, 2, 2048, 2048, {1, 1, 2, 0}},
    {LX_W_BF16, 4, 2048, 2048, {1, 4, 4, 0}},
    {LX_W_BF16, 8, 2048, 2048, {1, 4, 2, 0}},
    {LX_W_BF16, 1, 1024, 2048, {2, 1, 2, 0}},
    {LX_W_BF16, 2, 1024, 2048, {2, 1, 2, 0}},
    {LX_W_BF16, 4, 1024, 2048, {2, 2, 3, 0}},
    {LX_W_BF16, 8, 1024, 2048, {2, 4, 2, 0}},
    {LX_W_BF16, 1, 2048, 1024, {1, 4, 4, 0}},
    {LX_W_BF16, 2, 2048, 1024, {1, 2, 3, 0}},
    {LX_W_BF16, 4, 2048, 1024, {1, 2, 3, 0}},
    {LX_W_BF16, 8, 2048, 1024, {1, 4, 2, 0}},
    {LX_W_BF16, 1, 65536, 2048, {2, 4, 3, 0}},
    {LX_W_BF16, 2, 65536, 2048, {2, 4, 3, 0}},
    {LX_W_BF16, 4, 65536, 2048, {2, 2, 3, 0}},
    {LX_W_BF16, 8, 65536, 2048, {2, 4, 2, 1}},
    {LX_W_BF16, 1, 1024, 1024, {2, 2, 4, 0}},
    {LX_W_BF16, 2, 1024, 1024, {2, 4, 3, 0}},
    {LX_W_BF16, 4, 1024, 1024, {2, 2, 4, 0}},
    {LX_W_BF16, 8, 1024, 1024, {2, 2, 2, 0}},
    {LX_W_BF16, 1, 1024, 512, {1, 1, 8, 0}},
    {LX_W_BF16, 2, 1024, 512, {1, 1, 4, 0}},
    {LX_W_BF16, 4, 1024, 512, {1, 4, 4, 0}},
    {LX_W_BF16, 8, 1024, 512, {1, 4, 2, 0}},
    {LX_W_BF16, 1, 256, 2048, {8, 1, 2, 0}},
    {LX_W_BF16, 2, 256, 2048, {8, 2, 2, 0}},
    {LX_W_BF16, 4, 256, 2048, {8, 1, 2, 0}},
    {LX_W_BF16, 8, 256, 2048, {8, 2, 2, 0}},
    {LX_W_BF16, 1, 256, 6400, {8, 1, 4, 0}},
    {LX_W_BF16, 2, 256, 6400, {8, 1, 4, 0}},
    {LX_W_BF16, 4, 256, 6400, {8, 1, 2, 0}},
    {LX_W_BF16, 8, 256, 6400, {8, 2, 2, 0}},
    {LX_W_BF16, 1, 8, 2048, {4, 1, 4, 0}},
    {LX_W_BF16, 2, 8, 2048, {4, 1, 2, 0}},
    {LX_W_BF16, 4, 8, 2048, {4, 1, 4, 0}},
    {LX_W_BF16, 8, 8, 2048, {4, 1, 4, 0}},
    {LX_W_INT8, 1, 10304, 2048, {2, 4, 3, 0}},
    {LX_W_INT8, 2, 10304, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 4, 10304, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 8, 10304, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 1, 2048, 4096, {2, 2, 2, 0}},
    {LX_W_INT8, 2, 2048, 4096, {2, 2, 2, 0}},
    {LX_W_INT8, 4, 2048, 4096, {2, 4, 3, 0}},
    {LX_W_INT8, 8, 2048, 4096, {2, 4, 3, 1}},
    {LX_W_INT8, 1, 3072, 2048, {1, 1, 2, 0}},
    {LX_W_INT8, 2, 3072, 2048, {1, 1, 2, 0}},
    {LX_W_INT8, 4, 3072, 2048, {1, 2, 3, 0}},
    {LX_W_INT8, 8, 3072, 2048, {1, 2, 3, 1}},
    {LX_W_INT8, 1, 2048, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 2, 2048, 2048, {2, 2, 3, 0}},
    {LX_W_INT8, 4, 2048, 2048, {2, 4, 3, 0}},
    {LX_W_INT8, 8, 2048, 2048, {2, 4, 2, 1}},
    {LX_W_INT8, 1, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_INT8, 2, 1024, 2048, {4, 2, 4, 0}},
    {LX_W_INT8, 4, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_INT8, 8, 1024, 2048, {4, 2, 4, 0}},
    {LX_W_INT8, 1, 2048, 1024, {2, 2, 2, 0}},
    {LX_W_INT8, 2, 2048, 1024, {2, 4, 3, 0}},
    {LX_W_INT8, 4, 2048, 1024, {2, 4, 3, 0}},
    {LX_W_INT8, 8, 2048, 1024, {2, 4, 3, 1}},
    {LX_W_INT8, 1, 65536, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 2, 65536, 2048, {2, 2, 2, 0}},
    {LX_W_INT8, 4, 65536, 2048, {2, 1, 2, 0}},
    {LX_W_INT8, 8, 65536, 2048, {2, 4, 3, 1}},
    {LX_W_INT8, 1, 1024, 1024, {2, 1, 4, 0}},
    {LX_W_INT8, 2, 1024, 1024, {2, 1, 2, 0}},
    {LX_W_INT8, 4, 1024, 1024, {2, 1, 2, 0}},
    {LX_W_INT8, 8, 1024, 1024, {2, 2, 2, 0}},
    {LX_W_INT8, 1, 1024, 512, {1, 4, 4, 0}},
    {LX_W_INT8, 2, 1024, 512, {1, 1, 8, 0}},
    {LX_W_INT8, 4, 1024, 512, {1, 1, 8, 0}},
    {LX_W_INT8, 8, 1024, 512, {1, 4, 2, 0}},
    {LX_W_INT8, 1, 256, 2048, {8, 1, 2, 0}},
    {LX_W_INT8, 2, 256, 2048, {8, 1, 2, 0}},
    {LX_W_INT8, 4, 256, 2048, {8, 1, 2, 0}},
    {LX_W_INT8, 8, 256, 2048, {8, 2, 2, 0}},
    {LX_W_INT8, 1, 256, 6400, {8, 1, 4, 0}},
    {LX_W_INT8, 2, 256, 6400, {8, 1, 4, 0}},
    {LX_W_INT8, 4, 256, 6400, {8, 1, 2, 0}},
    {LX_W_INT8, 8, 256, 6400, {8, 2, 2, 0}},
    {LX_W_INT8, 1, 8, 2048, {8, 2, 2, 0}},
    {LX_W_INT8, 2, 8, 2048, {8, 1, 4, 0}},
    {LX_W_INT8, 4, 8, 2048, {8, 1, 2, 0}},
    {LX_W_INT8, 8, 8, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 1, 10304, 2048, {2, 4, 2, 0}},
    {LX_W_FP8, 2, 10304, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 4, 10304, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 8, 10304, 2048, {2, 4, 2, 0}},
    {LX_W_FP8, 1, 2048, 4096, {2, 2, 2, 0}},
    {LX_W_FP8, 2, 2048, 4096, {2, 2, 2, 0}},
    {LX_W_FP8, 4, 2048, 4096, {2, 2, 2, 0}},
    {LX_W_FP8, 8, 2048, 4096, {2, 4, 2, 0}},
    {LX_W_FP8, 1, 3072, 2048, {1, 2, 3, 0}},
    {LX_W_FP8, 2, 3072, 2048, {1, 2, 3, 0}},
    {LX_W_FP8, 4, 3072, 2048, {1, 2, 3, 0}},
    {LX_W_FP8, 8, 3072, 2048, {1, 2, 3, 1}},
    {LX_W_FP8, 1, 2048, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 2, 2048, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 4, 2048, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 8, 2048, 2048, {2, 4, 2, 0}},
    {LX_W_FP8, 1, 1024, 2048, {2, 1, 4, 0}},
    {LX_W_FP8, 2, 1024, 2048, {2, 1, 4, 0}},
    {LX_W_FP8, 4, 1024, 2048, {2, 2, 3, 0}},
    {LX_W_FP8, 8, 1024, 2048, {2, 4, 3, 0}},
    {LX_W_FP8, 1, 2048, 1024, {2, 2, 2, 0}},
    {LX_W_FP8, 2, 2048, 1024, {2, 4, 3, 0}},
    {LX_W_FP8, 4, 2048, 1024, {2, 4, 2, 0}},
    {LX_W_FP8, 8, 2048, 1024, {2, 4, 2, 0}},
    {LX_W_FP8, 1, 65536, 2048, {2, 2, 3, 0}},
    {LX_W_FP8, 2, 65536, 2048, {2, 2, 2, 0}},
    {LX_W_FP8, 4, 65536, 2048, {2, 4, 3, 0}},
    {LX_W_FP8, 8, 65536, 2048, {2, 4, 2, 0}},
    {LX_W_FP8, 1, 1024, 1024, {2, 1, 4, 0}},
    {LX_W_FP8, 2, 1024, 1024, {2, 1, 2, 0}},
    {LX_W_FP8, 4, 1024, 1024, {2, 2, 3, 0}},
    {LX_W_FP8, 8, 1024, 1024, {2, 2, 2, 0}},
    {LX_W_FP8, 1, 1024, 512, {1, 1, 2, 0}},
    {LX_W_FP8, 2, 1024, 512, {1, 4, 4, 0}},
    {LX_W_FP8, 4, 1024, 512, {1, 1, 8, 0}},
    {LX_W_FP8, 8, 1024, 512, {1, 4, 2, 0}},
    {LX_W_FP8, 1, 256, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 2, 256, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 4, 256, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 8, 256, 2048, {8, 2, 2, 0}},
    {LX_W_FP8, 1, 256, 6400, {8, 1, 4, 0}},
    {LX_W_FP8, 2, 256, 6400, {8, 1, 4, 0}},
    {LX_W_FP8, 4, 256, 6400, {8, 1, 2, 0}},
    {LX_W_FP8, 8, 256, 6400, {8, 2, 2, 0}},
    {LX_W_FP8, 1, 8, 2048, {8, 2, 2, 0}},
    {LX_W_FP8, 2, 8, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 4, 8, 2048, {8, 1, 2, 0}},
    {LX_W_FP8, 8, 8, 2048, {8, 1, 2, 0}},
    {LX_W_NVFP4, 1, 10304, 2048, {1, 4, 3, 0}},
    {LX_W_NVFP4, 2, 10304, 2048, {1, 4, 2, 0}},
    {LX_W_NVFP4, 4, 10304, 2048, {1, 4, 3, 0}},
    {LX_W_NVFP4, 8, 10304, 2048, {1, 4, 2, 1}},
    {LX_W_NVFP4, 1, 2048, 4096, {2, 2, 4, 0}},
    {LX_W_NVFP4, 2, 2048, 4096, {2, 2, 4, 0}},
    {LX_W_NVFP4, 4, 2048, 4096, {2, 4, 3, 0}},
    {LX_W_NVFP4, 8, 2048, 4096, {2, 4, 3, 1}},
    {LX_W_NVFP4, 1, 3072, 2048, {2, 2, 2, 0}},
    {LX_W_NVFP4, 2, 3072, 2048, {2, 4, 2, 0}},
    {LX_W_NVFP4, 4, 3072, 2048, {2, 2, 2, 0}},
    {LX_W_NVFP4, 8, 3072, 2048, {2, 2, 2, 1}},
    {LX_W_NVFP4, 1, 2048, 2048, {1, 1, 4, 0}},
    {LX_W_NVFP4, 2, 2048, 2048, {1, 1, 2, 0}},
    {LX_W_NVFP4, 4, 2048, 2048, {1, 1, 4, 0}},
    {LX_W_NVFP4, 8, 2048, 2048, {1, 2, 4, 0}},
    {LX_W_NVFP4, 1, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_NVFP4, 2, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_NVFP4, 4, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_NVFP4, 8, 1024, 2048, {4, 2, 2, 0}},
    {LX_W_NVFP4, 1, 2048, 1024, {2, 4, 3, 0}},
    {LX_W_NVFP4, 2, 2048, 1024, {2, 4, 2, 0}},
    {LX_W_NVFP4, 4, 2048, 1024, {2, 2, 2, 0}},
    {LX_W_NVFP4, 8, 2048, 1024, {2, 4, 2, 0}},
    {LX_W_NVFP4, 1, 65536, 2048, {1, 2, 3, 0}},
    {LX_W_NVFP4, 2, 65536, 2048, {1, 2, 2, 0}},
    {LX_W_NVFP4, 4, 65536, 2048, {1, 4, 3, 0}},
    {LX_W_NVFP4, 8, 65536, 2048, {1, 4, 2, 1}},
    {LX_W_NVFP4, 1, 1024, 1024, {2, 2, 3, 0}},
    {LX_W_NVFP4, 2, 1024, 1024, {2, 1, 4, 0}},
    {LX_W_NVFP4, 4, 1024, 1024, {2, 2, 3, 0}},
    {LX_W_NVFP4, 8, 1024, 1024, {2, 2, 4, 0}},
    {LX_W_NVFP4, 1, 1024, 512, {1, 4, 3, 0}},
    {LX_W_NVFP4, 2, 1024, 512, {1, 4, 2, 0}},
    {LX_W_NVFP4, 4, 1024, 512, {1, 4, 3, 0}},
    {LX_W_NVFP4, 8, 1024, 512, {1, 1, 8, 0}},
    {LX_W_NVFP4, 1, 256, 2048, {4, 2, 3, 0}},
    {LX_W_NVFP4, 2, 256, 2048, {4, 1, 4, 0}},
    {LX_W_NVFP4, 4, 256, 2048, {4, 1, 4, 0}},
    {LX_W_NVFP4, 8, 256, 2048, {4, 1, 2, 0}},
    {LX_W_NVFP4, 1, 256, 6400, {8, 1, 4, 0}},
    {LX_W_NVFP4, 2, 256, 6400, {8, 2, 2, 0}},
    {LX_W_NVFP4, 4, 256, 6400, {8, 1, 2, 0}},
    {LX_W_NVFP4, 8, 256, 6400, {8, 2, 2, 0}},
    {LX_W_NVFP4, 1, 8, 2048, {8, 2, 2, 0}},
    {LX_W_NVFP4, 2, 8, 2048, {8, 1, 2, 0}},
    {LX_W_NVFP4, 4, 8, 2048, {8, 1, 2, 0}},
    {LX_W_NVFP4, 8, 8, 2048, {8, 1, 4, 0}},
};
// clang-format on

constexpr bool built(const VecConfig& c) {
  for (const VecConfig& v : kVariants)
    if (v.S == c.S && v.RG == c.RG && v.D == c.D && v.G == c.G) return true;
  return false;
}

constexpr bool all_built() {
  for (const Pick& p : kPicks)
    if (!built(p.c)) return false;
  return true;
}
static_assert(all_built(), "every default pick must be a built variant");

// Every M bucket of a shape carries the same split (its order class): the table is one class per shape.
constexpr bool classes_agree() {
  for (const Pick& p : kPicks)
    for (const Pick& q : kPicks)
      if (q.F == p.F && q.N == p.N && q.K == p.K && q.c.S != p.c.S) return false;
  return true;
}
static_assert(classes_agree(), "every M bucket of a shape must share one order class");

// The pick nearest a shape in (log2 N, log2 K) for a format and M bucket. Every format has the same shapes in
// every bucket, so the nearest shape is the same whatever the bucket.
const Pick* nearest_pick(int F, int N, int K, int mt) {
  const double ln = std::log2(static_cast<double>(N)), lk = std::log2(static_cast<double>(K));
  const Pick* best = nullptr;
  double best_d = 0.0;
  for (const Pick& p : kPicks) {
    if (p.F != F || p.MT != mt) continue;
    const double dn = ln - std::log2(static_cast<double>(p.N)), dk = lk - std::log2(static_cast<double>(p.K));
    const double d = dn * dn + dk * dk;
    if (!best || d < best_d) best = &p, best_d = d;
  }
  return best;
}

// The order class of a product: the split S of its shape. A row sums its steps in an order set by S alone:
// RG, D and G change how the loads are issued and the steps batched, not the order the sums run in, and the
// cross-lane reduction gives each output the same tree whatever its warp holds. So a class fixes every row's
// bits at every batch size, and the class depends on (F, N, K) only.
int order_class(int F, int N, int K) {
  const Pick* p = nearest_pick(F, N, K, 1);
  return p ? p->c.S : 1;
}

VecConfig choose_vec(int N, int K, int M, int F) {
  const int mt = M <= 1 ? 1 : M <= 2 ? 2 : M <= 4 ? 4 : 8;
  const Pick* p = nearest_pick(F, N, K, mt);
  return p ? p->c : VecConfig{1, 1, 4, 0};
}

// The admission rule the tuner registry asks: a variant may be a shape's choice only when its split is the
// shape's order class. M does not enter, so the admitted set is the same at every batch size.
template <int F>
bool admits(int variant, int M, int N, int K) {
  (void)M;
  return kVariants[variant].S == order_class(F, N, K);
}

template <int F, int MT>
cudaError_t dispatch_vec_bf16(const Args& a, const VecConfig& c) {
#define LX_VEC_CASE(S_, RG_, D_, G_)                       \
  if (c.S == S_ && c.RG == RG_ && c.D == D_ && c.G == G_) \
    return run_vec<F, bf16, MT, RG_, S_, D_, MT == 8 ? G_ : 0>(a);
  LX_GEMV_VARIANTS(LX_VEC_CASE)
#undef LX_VEC_CASE
  return cudaErrorInvalidValue;
}

// f32 x: one row per warp and a ring of 4; the split follows the variant.
template <int F, int MT>
cudaError_t dispatch_vec_f32(const Args& a, const VecConfig& c) {
  switch (c.S) {
    case 1: return run_vec<F, float, MT, 1, 1, 4, 0>(a);
    case 2: return run_vec<F, float, MT, 1, 2, 4, 0>(a);
    case 4: return run_vec<F, float, MT, 1, 4, 4, 0>(a);
    default: return run_vec<F, float, MT, 1, 8, 4, 0>(a);
  }
}

template <int F, typename TX, int MT>
cudaError_t dispatch_vec(const Args& a, const VecConfig& c) {
  if constexpr (std::is_same_v<TX, bf16>) return dispatch_vec_bf16<F, MT>(a, c);
  else return dispatch_vec_f32<F, MT>(a, c);
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

// The variant names, in kVariants order, for each format: s<S>r<RG>d<D>, and a trailing g for the
// checked loop.
#define LX_G_SUFFIX_0 ""
#define LX_G_SUFFIX_1 "g"
#define LX_VARIANT_NAME(S_, RG_, D_, G_) "s" #S_ "r" #RG_ "d" #D_ LX_G_SUFFIX_##G_,
const lx::TunableRegistration kRegBf16("gemv_w_bf16", {LX_GEMV_VARIANTS(LX_VARIANT_NAME)}, &admits<LX_W_BF16>);
const lx::TunableRegistration kRegInt8("gemv_w_int8", {LX_GEMV_VARIANTS(LX_VARIANT_NAME)}, &admits<LX_W_INT8>);
const lx::TunableRegistration kRegFp8("gemv_w_fp8", {LX_GEMV_VARIANTS(LX_VARIANT_NAME)}, &admits<LX_W_FP8>);
const lx::TunableRegistration kRegNvfp4("gemv_w_nvfp4", {LX_GEMV_VARIANTS(LX_VARIANT_NAME)}, &admits<LX_W_NVFP4>);
#undef LX_VARIANT_NAME
#undef LX_G_SUFFIX_0
#undef LX_G_SUFFIX_1

template <int F, typename TX>
cudaError_t launch(const Args& a) {
  // Vector loads need 16-byte alignment: the weight base, and the x base and row stride. K must be a
  // multiple of 8, so each lane's 8 values are whole within their row.
  const bool aligned = reinterpret_cast<uintptr_t>(a.w.data) % 16 == 0 &&
                       ((reinterpret_cast<uintptr_t>(a.x) | (static_cast<uint64_t>(a.ld_x) * sizeof(TX))) % 16 == 0);
  if (a.K <= kSmallK || a.K % kLaneK != 0 || !aligned) return launch_small<F, TX>(a);
  const int tuned = lx_tune_get(tune_name(F), a.M, a.N, a.K);
  const VecConfig c = tuned >= 0 && tuned < kNumVariants ? kVariants[tuned] : choose_vec(a.N, a.K, a.M, F);
  if (a.M <= 1) return dispatch_vec<F, TX, 1>(a, c);
  if (a.M <= 2) return dispatch_vec<F, TX, 2>(a, c);
  if (a.M <= 4) return dispatch_vec<F, TX, 4>(a, c);
  return dispatch_vec<F, TX, 8>(a, c);
}

template <int F>
cudaError_t launch_format(const Args& a) {
  return a.x_bf16 ? launch<F, bf16>(a) : launch<F, float>(a);
}

bool valid_dtype(lx_dtype d) { return d == LX_F32 || d == LX_BF16; }

}  // namespace

int lx::gemv_order_class(int format, int N, int K) { return order_class(format, N, K); }

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
    case LX_W_BF16X:  // no scale: the encoding carries every value
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

  if (w->type == LX_W_BF16X) {
    // Lossless bf16 (bf16x.cu): the same contract, y equal bit for bit to the bf16 weights it encodes.
    return lx_gemv_bf16x(x, x_dtype, ld_x, w->data, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
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
  lx::launch_pdl(quantize_w8_kernel, static_cast<unsigned int>(blocks), kThreads, 0, static_cast<cudaStream_t>(stream), static_cast<const bf16*>(w), q, scale, rows, cols);
  return lx::launch_status();
}
