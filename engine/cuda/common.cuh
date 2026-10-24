// Shared helpers for every kernel. Kernels read and write row-major, contiguous buffers.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include "cuda/kernels.h"
#include "cuda/tunable.h"

// The device code of this library is built for every target in LX_ARCHS, and some of it uses
// hardware that only exists from sm_90 on: thread-block clusters with distributed shared memory, and
// the programmatic dependent launch wait. LX_SM90_DEVICE is 1 in device code compiled for sm_90 or
// newer and 0 everywhere else. The helpers below compile to nothing on the older targets, and the
// host never launches a cluster kernel or asks for a dependent launch there: it checks lx_arch()
// against lx::kClusterArch and takes the one-block path instead (the kernels' own fallbacks).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
#define LX_SM90_DEVICE 1
#define LX_CLUSTER_DIMS(n) __cluster_dims__(n, 1, 1)
#else
#define LX_SM90_DEVICE 0
#define LX_CLUSTER_DIMS(n)
#endif

namespace lx {

#if LX_SM90_DEVICE
__device__ __forceinline__ unsigned cluster_block_rank() { return cooperative_groups::this_cluster().block_rank(); }
__device__ __forceinline__ void cluster_sync() { cooperative_groups::this_cluster().sync(); }
// The same shared-memory address in block `rank` of this block's cluster.
template <typename T>
__device__ __forceinline__ T* cluster_peer(T* p, unsigned rank) {
  return cooperative_groups::this_cluster().map_shared_rank(p, rank);
}
__device__ __forceinline__ void pdl_trigger() { cudaTriggerProgrammaticLaunchCompletion(); }
__device__ __forceinline__ void pdl_wait() { cudaGridDependencySynchronize(); }
#else
__device__ __forceinline__ unsigned cluster_block_rank() { return 0; }
__device__ __forceinline__ void cluster_sync() {}
template <typename T>
__device__ __forceinline__ T* cluster_peer(T* p, unsigned) {
  return p;
}
__device__ __forceinline__ void pdl_trigger() {}
__device__ __forceinline__ void pdl_wait() {}
#endif

// Whether a launch may use a programmatic dependent launch on this device (host code).
inline bool pdl_available() { return lx_arch() >= kClusterArch; }

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
