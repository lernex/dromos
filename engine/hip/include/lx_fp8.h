// The FP8 and FP4 conversions the CUDA sources use (cuda_fp8.h, cuda_fp4.h), in portable form. The
// names and enumerators are CUDA's, so the translated sources compile unchanged.
//
// Every conversion here is exact in the direction the kernels use it (e4m3, e5m2 and e2m1 values are
// all representable in half), and the encoders round to nearest even, saturating when asked to, as the
// CUDA intrinsics do. The matrix units consume these formats directly through lx_mma.cuh, so this
// header only serves the element-wise decode and quantization paths.
#pragma once

#include <stdint.h>

#include <hip/hip_fp16.h>

#include "lx_hip.h"

typedef uint8_t __nv_fp8_storage_t;
typedef uint16_t __nv_fp8x2_storage_t;
typedef uint8_t __nv_fp4x2_storage_t;

enum __nv_saturation_t { __NV_NOSAT, __NV_SATFINITE };
enum __nv_fp8_interpretation_t { __NV_E4M3, __NV_E5M2 };
enum __nv_fp4_interpretation_t { __NV_E2M1 };

namespace lx::fp8 {

// f32 to IEEE half bits, round to nearest even, overflow to infinity.
__device__ __forceinline__ uint16_t f32_to_f16_bits(float f) {
  const uint32_t u = __float_as_uint(f);
  const uint32_t sign = (u >> 16) & 0x8000u;
  const uint32_t absu = u & 0x7fffffffu;
  if (absu >= 0x7f800000u) return static_cast<uint16_t>(sign | (absu > 0x7f800000u ? 0x7e00u : 0x7c00u));
  if (absu >= 0x477ff000u) return static_cast<uint16_t>(sign | 0x7c00u);  // 65520 and up round to infinity
  if (absu < 0x38800000u) {                                                 // below 2^-14: subnormal half
    const float t = __uint_as_float(absu) * 16777216.0f;                    // exact: value in units of 2^-24
    return static_cast<uint16_t>(sign | __float2uint_rn(t));
  }
  // Normal: rebias the exponent (127 -> 15) and round the 13 dropped mantissa bits to nearest even.
  const uint32_t odd = (absu >> 13) & 1u;
  const uint32_t m = (absu - (112u << 23)) + 0xfffu + odd;
  return static_cast<uint16_t>(sign | (m >> 13));
}

// e4m3 bits to f32 (exact): the magnitude placed under an f32 exponent, rescaled by 2^120.
__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  const float m = __uint_as_float(((b & 0x7fu) << 20) | ((b & 0x80u) << 24));
  return m * 0x1p120f;
}

// e2m1 nibble to f32 (exact).
__device__ __forceinline__ float e2m1_to_f32(uint32_t n) {
  const float mag = (n & 4u) ? ((n & 2u) ? ((n & 1u) ? 6.0f : 4.0f) : ((n & 1u) ? 3.0f : 2.0f))
                             : ((n & 2u) ? ((n & 1u) ? 1.5f : 1.0f) : ((n & 1u) ? 0.5f : 0.0f));
  return (n & 8u) ? -mag : mag;
}

// f32 to e4m3 (bias 7, max 448, 0x7f is NaN). Saturating mode clamps to 448; otherwise overflow is NaN.
__device__ __forceinline__ uint8_t f32_to_e4m3(float v, bool saturate) {
  const uint32_t u = __float_as_uint(v);
  const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80u);
  const float a = fabsf(v);
  if (isnan(v)) return 0x7fu;
  if (a >= 464.0f) return saturate ? static_cast<uint8_t>(sign | 0x7eu) : static_cast<uint8_t>(sign | 0x7fu);
  if (a < 0.015625f) {  // below 2^-6: subnormal e4m3, units of 2^-9, rounded to nearest even
    return static_cast<uint8_t>(sign | static_cast<uint8_t>(__float2uint_rn(a * 512.0f)));
  }
  const uint32_t au = __float_as_uint(a);
  int exp = static_cast<int>(au >> 23) - 127 + 7;  // biased e4m3 exponent before rounding
  uint32_t m3 = (au >> 20) & 7u;
  const uint32_t rem = au & 0xfffffu;
  if (rem > 0x80000u || (rem == 0x80000u && (m3 & 1u))) {
    if (++m3 == 8) {
      m3 = 0;
      ++exp;
    }
  }
  if (exp > 15 || (exp == 15 && m3 == 7)) {
    return saturate ? static_cast<uint8_t>(sign | 0x7eu) : static_cast<uint8_t>(sign | 0x7fu);
  }
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(exp) << 3) | m3);
}

// f32 to e5m2 (bias 15, max 57344), rounded once from the f32 bits (a half intermediate would round twice).
__device__ __forceinline__ uint8_t f32_to_e5m2(float v, bool saturate) {
  const uint32_t u = __float_as_uint(v);
  const uint32_t sign = (u >> 24) & 0x80u;
  const uint32_t a = u & 0x7fffffffu;
  if (a > 0x7f800000u) return static_cast<uint8_t>(sign | 0x7fu);                     // NaN
  if (a == 0x7f800000u) return static_cast<uint8_t>(sign | (saturate ? 0x7bu : 0x7cu));  // infinity
  if (a < 0x38800000u) {  // below 2^-14: subnormal, units of 2^-16, nearest even (4 units is the first normal)
    return static_cast<uint8_t>(sign | __float2uint_rn(__uint_as_float(a) * 65536.0f));
  }
  // Normal: the exponent rebiased to 15 sits above the 23 mantissa bits; keep 2 of them, round at bit 21.
  uint32_t m = a - (112u << 23);
  m += (1u << 20) - 1u + ((m >> 21) & 1u);  // nearest, ties to even
  m >>= 21;
  if (m >= 0x7cu) return static_cast<uint8_t>(sign | (saturate ? 0x7bu : 0x7cu));  // overflow
  return static_cast<uint8_t>(sign | m);
}

}  // namespace lx::fp8

// ---- CUDA names -----------------------------------------------------------------------------------

__device__ __forceinline__ __nv_fp8_storage_t __nv_cvt_float_to_fp8(float v, __nv_saturation_t sat,
                                                                    __nv_fp8_interpretation_t interp) {
  const bool saturate = sat == __NV_SATFINITE;
  return interp == __NV_E4M3 ? lx::fp8::f32_to_e4m3(v, saturate) : lx::fp8::f32_to_e5m2(v, saturate);
}

// A __half2_raw is one 32-bit word of two half bit patterns. Its members are unsigned shorts on NVIDIA and
// half-precision unions on AMD, so the word is written whole.
__device__ __forceinline__ __half2_raw lx_half2_raw(uint16_t lo, uint16_t hi) {
  const uint32_t packed = static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
  __half2_raw r;
  static_assert(sizeof(r) == sizeof(packed), "__half2_raw is one 32-bit word");
  __builtin_memcpy(&r, &packed, sizeof r);
  return r;
}

__device__ __forceinline__ __half2_raw __nv_cvt_fp8x2_to_halfraw2(__nv_fp8x2_storage_t x,
                                                                  __nv_fp8_interpretation_t interp) {
  if (interp == __NV_E4M3) {
    return lx_half2_raw(lx::fp8::f32_to_f16_bits(lx::fp8::e4m3_to_f32(x & 0xffu)),
                        lx::fp8::f32_to_f16_bits(lx::fp8::e4m3_to_f32((x >> 8) & 0xffu)));
  }
  // e5m2 is the top byte of a half: the bits move up by 8 and are exact
  return lx_half2_raw(static_cast<uint16_t>((x & 0xffu) << 8), static_cast<uint16_t>((x >> 8) << 8));
}

__device__ __forceinline__ __half2_raw __nv_cvt_fp4x2_to_halfraw2(__nv_fp4x2_storage_t x,
                                                                  __nv_fp4_interpretation_t) {
  return lx_half2_raw(lx::fp8::f32_to_f16_bits(lx::fp8::e2m1_to_f32(x & 0xfu)),
                      lx::fp8::f32_to_f16_bits(lx::fp8::e2m1_to_f32((x >> 4) & 0xfu)));
}
