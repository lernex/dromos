// The portable layer under the HIP port of the kernel library. Every translated source includes this
// instead of the CUDA runtime (translate.py maps the include). It holds the few things the NVIDIA
// sources use that HIP does not spell the same way, each as the portable form the port runs:
//
//   LX_WARP            lanes per wavefront, known at compile time on the device: 32 on RDNA (wave32),
//                      64 on CDNA (wave64). Host code asks lx_warp_size() instead.
//   LX_FULL_MASK       the all-lanes mask for the shuffles (64 bits on a wave64 target).
//   __floats2bfloat162_rn  the bf16 pair conversion CUDA has and ROCm does not.
//   lx_ldg, lx_ldcs    plain loads. The read-only and streaming cache hints are a tuning item.
//   lx_pipeline_*      cp.async, as a synchronous copy: every reader of a staged tile is behind a
//                      barrier, so the copy need not overlap the math to be correct.
//   lx_pdl_*, lx_launch_kernel_ex
//                      programmatic dependent launch. Without it the stream orders the kernels, which
//                      is what the port relies on; the attributes are accepted and ignored.
//
// The NVIDIA test build compiles these same sources against engine/hip/nvidia, a HIP-named shim over
// the CUDA runtime that defines LX_NVIDIA_SHIM. The AMD build uses the ROCm headers.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <utility>

#include <hip/hip_runtime.h>  // the ROCm headers, or nvidia/hip (LX_NVIDIA_SHIM) in the test build
#include <hip/hip_bf16.h>

// The wavefront size is a compile-time property of the target. clang deprecates the macro that exposes
// it, and the port needs it as a constant for array sizes and loop steps, so its warning is silenced.
#if defined(__clang__)
#pragma clang diagnostic ignored "-Wdeprecated-pragma"
#endif
#if defined(__AMDGCN_WAVEFRONT_SIZE__)
#define LX_WARP __AMDGCN_WAVEFRONT_SIZE__
#else
#define LX_WARP 32
#endif

#if LX_WARP == 64
#define LX_FULL_MASK 0xffffffffffffffffull
#else
#define LX_FULL_MASK 0xffffffffull
#endif

// ---- bf16 pairs -------------------------------------------------------------------------------------

#if !defined(LX_NVIDIA_SHIM)
// ROCm builds the pair from halves; the NVIDIA shim gets CUDA's own overload.
__device__ __forceinline__ __hip_bfloat162 __floats2bfloat162_rn(float lo, float hi) {
  return __halves2bfloat162(__float2bfloat16(lo), __float2bfloat16(hi));
}
#endif

// ---- Loads -----------------------------------------------------------------------------------------

template <typename T>
__device__ __forceinline__ T lx_ldg(const T* p) {
  return *p;
}

template <typename T>
__device__ __forceinline__ T lx_ldcs(const T* p) {
  return *p;
}

// ---- cp.async ---------------------------------------------------------------------------------------

__device__ __forceinline__ void lx_pipeline_memcpy_async(void* dst, const void* src, size_t bytes) {
  if (bytes == 16) {
    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
    return;
  }
  const unsigned char* s = static_cast<const unsigned char*>(src);
  unsigned char* d = static_cast<unsigned char*>(dst);
  for (size_t i = 0; i < bytes; ++i) d[i] = s[i];
}
__device__ __forceinline__ void lx_pipeline_commit() {}
__device__ __forceinline__ void lx_pipeline_wait_prior(int) {}

// ---- Programmatic dependent launch ------------------------------------------------------------------

__device__ __forceinline__ void lx_pdl_trigger() {}
__device__ __forceinline__ void lx_pdl_wait() {}

enum { LX_LAUNCH_ATTR_PDL = 1 };
struct lx_launch_attr {
  int id;
  union {
    int programmaticStreamSerializationAllowed;
  } val;
};
struct lx_launch_config {
  dim3 gridDim;
  dim3 blockDim;
  size_t dynamicSmemBytes;
  hipStream_t stream;
  lx_launch_attr* attrs;
  unsigned numAttrs;
};

#if defined(LX_NVIDIA_SHIM)
template <typename... KArgs, typename... Args>
inline hipError_t lx_launch_kernel_ex(const lx_launch_config* cfg, void (*k)(KArgs...), Args&&... args) {
  cudaLaunchConfig_t c{};
  c.gridDim = cfg->gridDim;
  c.blockDim = cfg->blockDim;
  c.dynamicSmemBytes = cfg->dynamicSmemBytes;
  c.stream = cfg->stream;
  return cudaLaunchKernelEx(&c, k, std::forward<Args>(args)...);
}
#else
template <typename... KArgs, typename... Args>
inline hipError_t lx_launch_kernel_ex(const lx_launch_config* cfg, void (*k)(KArgs...), Args&&... args) {
  hipLaunchKernelGGL(k, cfg->gridDim, cfg->blockDim, cfg->dynamicSmemBytes, cfg->stream,
                     std::forward<Args>(args)...);
  return hipGetLastError();
}
#endif

// ---- Host ------------------------------------------------------------------------------------------

// Lanes per wavefront on the current device. Host code sizes blocks and grids with this; the device
// code it launches was compiled for the same target, so the two agree.
inline int lx_warp_size() {
#if defined(LX_NVIDIA_SHIM)
  return 32;
#else
  int dev = 0, warp = 0;
  if (hipGetDevice(&dev) != hipSuccess) return 32;
  if (hipDeviceGetAttribute(&warp, hipDeviceAttributeWarpSize, dev) != hipSuccess || warp <= 0) return 32;
  return warp;
#endif
}
