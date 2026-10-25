// Native matrix instructions for the AMD targets, each behind a target check.
//
// Three families, one per hardware generation the port builds for:
//   gfx942  (CDNA3)  v_mfma_f32_16x16x16_bf16 and v_mfma_f32_16x16x32_fp8_fp8, wave64
//   gfx1100 (RDNA3)  v_wmma_f32_16x16x16_bf16, wave32
//   gfx1201 (RDNA4)  v_wmma_f32_16x16x16_bf16 with the gfx12 operand layout, wave32
//
// The portable path (lx_mma.cuh and lx_fp8.h) is the verified one, and the kernels run on it. These
// primitives are compile-checked for their targets (tests/lane_check.hip builds them for all three); they
// have not run on AMD hardware yet. The lane layouts below are what that test checks on the day a card is
// available, and a mismatch there is a bug in this header, not in the kernels.
//
// Lane layouts (l is the lane, 0..63 on wave64, 0..31 on wave32):
//   MFMA 16x16x16 bf16:  A row l%16, K 4*(l/16)..+3; B column l%16, the same K slice.
//                        D column l%16, rows 4*(l/16)..+3.
//   WMMA 16x16x16 bf16 (gfx1100): A row l%16 with all 16 K, repeated in lanes 16..31; B column l%16 likewise.
//                        D column l%16, rows 2*i + l/16 for i = 0..7.
//   WMMA 16x16x16 bf16 (gfx1201): A row l%16, K 8*(l/16)..+7; B column l%16, the same K slice.
//                        D column l%16, rows 2*i + l/16 for i = 0..7 (as gfx1100; not yet confirmed).
//   MFMA 16x16x32 fp8 (e4m3 x e4m3): A row l%16, K 8*(l/16)..+7, one byte each, packed in a uint64;
//                        B column l%16 likewise; D as the bf16 MFMA.
//
// Operands are raw bit patterns (bf16 and fp8 are passed as integers), and accumulation is f32.
#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>

// ---- Target checks (device compile only; the host never sees these) --------------------------------

#if defined(__HIP_DEVICE_COMPILE__) && defined(__gfx942__)
#define LX_NATIVE_MFMA 1
#endif
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__))
#define LX_NATIVE_WMMA_GFX11 1
#endif
#if defined(__HIP_DEVICE_COMPILE__) && (defined(__gfx1200__) || defined(__gfx1201__))
#define LX_NATIVE_WMMA_GFX12 1
#endif

namespace lx::native {

typedef short bf16x4_t __attribute__((ext_vector_type(4)));
typedef short bf16x8_t __attribute__((ext_vector_type(8)));
typedef __bf16 bf16x16_t __attribute__((ext_vector_type(16)));
typedef float f32x4_t __attribute__((ext_vector_type(4)));
typedef float f32x8_t __attribute__((ext_vector_type(8)));

// Which family a device runs, from its gcnArchName ("gfx942:sramecc+:xnack-" and so on). Host side, so
// the caller can pick a kernel before launching it.
enum class Family { none, mfma_cdna3, wmma_rdna3, wmma_rdna4 };

inline Family family_of(const hipDeviceProp_t& prop) {
  const char* a = prop.gcnArchName;
  auto is = [a](const char* name) {
    size_t n = 0;
    while (name[n]) ++n;
    for (size_t i = 0; i < n; ++i)
      if (a[i] != name[i]) return false;
    return true;
  };
  if (is("gfx942")) return Family::mfma_cdna3;
  if (is("gfx1100") || is("gfx1101") || is("gfx1102") || is("gfx1103")) return Family::wmma_rdna3;
  if (is("gfx1200") || is("gfx1201")) return Family::wmma_rdna4;
  return Family::none;
}

#if defined(LX_NATIVE_MFMA)
// D = A * B + C on one 16x16x16 tile, bf16 raw bits, f32 accumulate. wave64.
__device__ __forceinline__ f32x4_t mfma_16x16x16_bf16(bf16x4_t a, bf16x4_t b, f32x4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x16bf16_1k(a, b, c, 0, 0, 0);
}
// D = A * B + C on one 16x16x32 tile, e4m3 bytes packed 8 per lane. wave64.
__device__ __forceinline__ f32x4_t mfma_16x16x32_fp8(uint64_t a, uint64_t b, f32x4_t c) {
  return __builtin_amdgcn_mfma_f32_16x16x32_fp8_fp8(static_cast<long>(a), static_cast<long>(b), c, 0, 0, 0);
}
#endif

#if defined(LX_NATIVE_WMMA_GFX11)
// D = A * B + C on one 16x16x16 tile, bf16, f32 accumulate. wave32, RDNA3 operand layout.
__device__ __forceinline__ f32x8_t wmma_16x16x16_bf16(bf16x16_t a, bf16x16_t b, f32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
}
#endif

#if defined(LX_NATIVE_WMMA_GFX12)
// The same tile on RDNA4: each lane supplies 8 bf16 (a K half of its row or column) rather than 16.
__device__ __forceinline__ f32x8_t wmma_16x16x16_bf16(bf16x8_t a, bf16x8_t b, f32x8_t c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32_gfx12(a, b, c);
}
#endif

}  // namespace lx::native
