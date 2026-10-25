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

// HIP takes a 64-bit warp mask on every AMD target, wave32 included (it truncates to the wave), so the
// AMD mask is 64 bits. The NVIDIA shim keeps CUDA's 32-bit mask.
#if defined(LX_NVIDIA_SHIM)
#define LX_FULL_MASK 0xffffffffu
using lx_mask_t = unsigned int;
#else
#define LX_FULL_MASK 0xffffffffffffffffull
using lx_mask_t = unsigned long long;
#endif

// __grid_constant__ marks a kernel parameter read-only in CUDA's parameter space. ROCm keeps kernel
// arguments in the kernarg segment already, so the qualifier has no meaning there.
#ifndef __grid_constant__
#define __grid_constant__
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

// ---- Clock and system-scope loads --------------------------------------------------------------------

// A nanosecond clock across the block: the SM globaltimer on NVIDIA, the 100 MHz real-time counter
// (10 ns ticks) on AMD.
__device__ __forceinline__ uint64_t lx_global_ns() {
#if defined(LX_NVIDIA_SHIM)
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
#else
  // The constant-rate wall clock (HIP's wall_clock64): s_memrealtime on gfx9, the realtime message on RDNA.
  // The device reports its rate as hipDeviceAttributeWallClockRate, 100 MHz on the targets this port
  // builds, so each tick is 10 ns.
  return static_cast<uint64_t>(__builtin_readsteadycounter()) * 10u;
#endif
}

// A load that orders the reads after it: the poll loops that wait on another device's write.
__device__ __forceinline__ uint32_t lx_load_acquire_sys(const volatile uint32_t* p) {
  const uint32_t v = *p;
  __threadfence_system();
  return v;
}

// ---- Shuffles over a 32-lane warp -------------------------------------------------------------------
// The port keeps CUDA's warp: 32 lanes, addressed as threadIdx.x & 31 and threadIdx.x >> 5. A wave64
// holds two such warps, so every shuffle is taken within its own 32-lane group (width 32). The mask
// is passed through: the NVIDIA shim checks it, and ROCm ignores it.

template <typename T, typename M>
__device__ __forceinline__ T lx_shfl_sync(M mask, T v, int src, int width = 32) {
  return __shfl_sync(static_cast<lx_mask_t>(mask), v, src, width);
}
template <typename T, typename M>
__device__ __forceinline__ T lx_shfl_xor_sync(M mask, T v, int lane_mask, int width = 32) {
  return __shfl_xor_sync(static_cast<lx_mask_t>(mask), v, lane_mask, width);
}
template <typename T, typename M>
__device__ __forceinline__ T lx_shfl_down_sync(M mask, T v, unsigned delta, int width = 32) {
  return __shfl_down_sync(static_cast<lx_mask_t>(mask), v, delta, width);
}
template <typename T, typename M>
__device__ __forceinline__ T lx_shfl_up_sync(M mask, T v, unsigned delta, int width = 32) {
  return __shfl_up_sync(static_cast<lx_mask_t>(mask), v, delta, width);
}

// The most LDS one workgroup may use on the current device. On AMD that is the hardware limit (the
// SDK compiler reports 65536 bytes on gfx942, gfx1100 and gfx1201). The NVIDIA shim keeps its opt-in
// sizes, so it has no cap here.
#if defined(LX_NVIDIA_SHIM)
inline int lx_lds_limit() { return 0x7fffffff; }
#else
inline int lx_lds_limit() {
  int dev = 0, v = 0;
  hipGetDevice(&dev);
  if (hipDeviceGetAttribute(&v, hipDeviceAttributeMaxSharedMemoryPerBlock, dev) != hipSuccess || v <= 0)
    return 65536;
  return v;
}
#endif

// cudaFuncSetAttribute takes the kernel itself, a template instance; hipFuncSetAttribute takes a
// const void*. The carveout preference splits L1 against shared memory on NVIDIA. An AMD part has a fixed
// LDS, so there the preference is accepted and changes nothing.
#if !defined(LX_NVIDIA_SHIM)
#ifndef hipSharedmemCarveoutMaxShared
#define hipSharedmemCarveoutMaxShared 100
#endif
#endif

template <typename F>
inline hipError_t lx_func_set_attribute(F func, hipFuncAttribute attr, int value) {
#if !defined(LX_NVIDIA_SHIM)
  if (attr == hipFuncAttributePreferredSharedMemoryCarveout) return hipSuccess;
#endif
  return hipFuncSetAttribute(reinterpret_cast<const void*>(func), attr, value);
}

#if !defined(LX_NVIDIA_SHIM)
// CUDA's cache-volatile load (ld.global.cv): a volatile load reads memory, not a cached copy.
template <typename T>
__device__ __forceinline__ T __ldcv(const T* p) {
  return *static_cast<const volatile T*>(p);
}
// The 32-bit shared-window address of an LDS pointer is the low 32 bits of the flat pointer, the same
// convention the port's smem_addr uses.
__device__ __forceinline__ uint32_t __cvta_generic_to_shared(const void* p) {
  return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p));
}
#endif

// ---- Warp functions over a 32-lane group -----------------------------------------------------------
// The CUDA intrinsics with a 32-bit full mask, as the port runs them. Each reduces or matches within
// its own 32 lanes (a wave64 holds two such groups), which is the group every caller addresses with
// the 0xffffffff mask.

template <typename T>
__device__ __forceinline__ T lx_reduce_max_sync(unsigned, T v) {
  for (int o = 16; o > 0; o >>= 1) {
    const T w = static_cast<T>(__shfl_xor_sync(LX_FULL_MASK, v, o, 32));
    v = v > w ? v : w;
  }
  return v;
}

template <typename T>
__device__ __forceinline__ T lx_reduce_add_sync(unsigned, T v) {
  for (int o = 16; o > 0; o >>= 1) v += static_cast<T>(__shfl_xor_sync(LX_FULL_MASK, v, o, 32));
  return v;
}

// The lanes of the group whose value equals this lane's, as a bitmask over group lanes 0..31.
__device__ __forceinline__ unsigned lx_match_any_sync(unsigned, int v) {
  unsigned m = 0;
  for (int i = 0; i < 32; ++i) {
    const int w = __shfl_sync(LX_FULL_MASK, v, i, 32);
    if (w == v) m |= 1u << i;
  }
  return m;
}

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
