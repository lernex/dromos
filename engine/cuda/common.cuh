// Shared helpers for every kernel. Kernels read and write row-major, contiguous buffers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "cuda/kernels.h"

namespace lx {

__device__ __forceinline__ float to_f32(float v) { return v; }
__device__ __forceinline__ float to_f32(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T from_f32(float v);
template <>
__device__ __forceinline__ float from_f32<float>(float v) { return v; }
template <>
__device__ __forceinline__ __nv_bfloat16 from_f32<__nv_bfloat16>(float v) { return __float2bfloat16_rn(v); }

__device__ __forceinline__ float silu(float v) { return v / (1.0f + __expf(-v)); }
__device__ __forceinline__ float softplus(float v) { return v > 20.0f ? v : log1pf(__expf(v)); }

// Sum over a warp, every lane gets the result.
__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
  return v;
}
__device__ __forceinline__ float warp_max(float v) {
  for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
  return v;
}

// Sum over a block of up to 1024 threads; every thread gets the result. `scratch` holds 32 floats.
__device__ __forceinline__ float block_sum(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = warp_sum(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  v = lane < warps ? scratch[lane] : 0.0f;
  return warp_sum(v);
}

inline int launch_status() { return static_cast<int>(cudaGetLastError()); }

}  // namespace lx

// Dispatch on an lx_dtype for a templated launch. Use inside a function returning int.
#define LX_DISPATCH_DTYPE(dtype, T, ...)                \
  switch (dtype) {                                      \
    case LX_F32: { using T = float; __VA_ARGS__; break; } \
    case LX_BF16: { using T = __nv_bfloat16; __VA_ARGS__; break; } \
    default: return static_cast<int>(cudaErrorInvalidValue); \
  }
