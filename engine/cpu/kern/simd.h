// The vector layer of the per-ISA kernels (kern/isa.cpp). Exactly one section is compiled per level:
// LX_ISA_CODE picks it (tunable.h's lx_arch codes on the CPU, engine/cpu/isa.cpp). Everything here is
// `static inline` inside the level's own namespace, so no two levels share a symbol.
//
//   V       a vector of W floats
//   vload / vstore / vset1 / vzero / vfma(a, b, c) = a * b + c / vadd / vsub / vmul / vmax / vmin
//   vhsum   the horizontal sum, in one fixed order per level (results are deterministic)
//   load_bf16(p)  W bf16 values as f32; load_i8(p) W int8; load_e4m3(p) W e4m3; load_e2m1(p, s) W e2m1
//                 codes from W / 2 bytes (low nibble first) times s
//   vexp    exp, within 2 ulp on [-87.3, 88.3], clamped outside
#pragma once

#include <stdint.h>
#include <string.h>

#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 50
#include <immintrin.h>
#elif LX_ISA_CODE >= 50
#include <arm_neon.h>
#endif

#define LX_JOIN2(a, b) a##b
#define LX_JOIN(a, b) LX_JOIN2(a, b)

namespace lx::cpu::LX_ISA {

static inline float bits_f32(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
static inline float bf16f(uint16_t v) { return bits_f32(static_cast<uint32_t>(v) << 16); }
static inline uint16_t f32_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
static inline float e4m3f(uint8_t b) {
  const uint32_t mag = b & 0x7fu;
  float v = mag >= 8 ? bits_f32((mag << 20) + (120u << 23)) : static_cast<float>(mag) * 0x1p-9f;
  return (b & 0x80u) ? -v : v;
}
static inline float e2m1f(uint32_t code) {
  static const float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kGrid[code & 7u];
  return (code & 8u) ? -v : v;
}
template <class T>
static inline T lmin(T a, T b) {
  return a < b ? a : b;
}
template <class T>
static inline T lmax(T a, T b) {
  return a < b ? b : a;
}

// Cephes expf constants, shared by every level.
constexpr float kExpHi = 88.3762626647949f, kExpLo = -87.3365447504f;
constexpr float kLog2e = 1.44269504088896341f, kC1 = 0.693359375f, kC2 = -2.12194440e-4f;
constexpr float kP0 = 1.9875691500e-4f, kP1 = 1.3981999507e-3f, kP2 = 8.3334519073e-3f, kP3 = 4.1665795894e-2f,
                kP4 = 1.6666665459e-1f, kP5 = 5.0000001201e-1f;

static inline float sexp(float x) {
  x = lmin(lmax(x, kExpLo), kExpHi);
  const float fx = __builtin_rintf(x * kLog2e);
  float r = x - fx * kC1;
  r = r - fx * kC2;
  float p = kP0;
  p = p * r + kP1;
  p = p * r + kP2;
  p = p * r + kP3;
  p = p * r + kP4;
  p = p * r + kP5;
  const float y = p * r * r + r + 1.0f;
  return y * bits_f32(static_cast<uint32_t>(static_cast<int32_t>(fx) + 127) << 23);
}

#if LX_ISA_CODE == 1
// ---- Generic: plain C++, four lanes, auto-vectorized by the compiler for the baseline ----------------
constexpr int W = 4;
struct V {
  float v[4];
};
static inline V vzero() { return V{{0, 0, 0, 0}}; }
static inline V vset1(float f) { return V{{f, f, f, f}}; }
static inline V vload(const float* p) {
  V r;
  for (int i = 0; i < 4; ++i) r.v[i] = p[i];
  return r;
}
static inline void vstore(float* p, V a) {
  for (int i = 0; i < 4; ++i) p[i] = a.v[i];
}
static inline V vfma(V a, V b, V c) {
  for (int i = 0; i < 4; ++i) c.v[i] = c.v[i] + a.v[i] * b.v[i];
  return c;
}
#define LX_GEN_BIN(name, expr)                    \
  static inline V name(V a, V b) {                \
    for (int i = 0; i < 4; ++i) a.v[i] = (expr); \
    return a;                                     \
  }
LX_GEN_BIN(vadd, a.v[i] + b.v[i])
LX_GEN_BIN(vsub, a.v[i] - b.v[i])
LX_GEN_BIN(vmul, a.v[i] * b.v[i])
LX_GEN_BIN(vmax, lmax(a.v[i], b.v[i]))
LX_GEN_BIN(vmin, lmin(a.v[i], b.v[i]))
#undef LX_GEN_BIN
static inline float vhsum(V a) { return (a.v[0] + a.v[2]) + (a.v[1] + a.v[3]); }
static inline float vhmax(V a) { return lmax(lmax(a.v[0], a.v[2]), lmax(a.v[1], a.v[3])); }
static inline V load_bf16(const uint16_t* p) {
  V r;
  for (int i = 0; i < 4; ++i) r.v[i] = bf16f(p[i]);
  return r;
}
static inline V load_i8(const int8_t* p) {
  V r;
  for (int i = 0; i < 4; ++i) r.v[i] = static_cast<float>(p[i]);
  return r;
}
static inline V load_e4m3(const uint8_t* p) {
  V r;
  for (int i = 0; i < 4; ++i) r.v[i] = e4m3f(p[i]);
  return r;
}
static inline V load_e2m1(const uint8_t* p, float s) {
  V r;
  for (int i = 0; i < 2; ++i) {
    r.v[2 * i] = e2m1f(p[i] & 15u) * s;
    r.v[2 * i + 1] = e2m1f(p[i] >> 4) * s;
  }
  return r;
}
static inline V vexp(V a) {
  for (int i = 0; i < 4; ++i) a.v[i] = sexp(a.v[i]);
  return a;
}

#elif LX_ISA_CODE >= 10 && LX_ISA_CODE < 20
// ---- AVX2 + FMA (and AVX-VNNI, which only changes the int8 kernels) ----------------------------------
constexpr int W = 8;
using V = __m256;
static inline V vzero() { return _mm256_setzero_ps(); }
static inline V vset1(float f) { return _mm256_set1_ps(f); }
static inline V vload(const float* p) { return _mm256_loadu_ps(p); }
static inline void vstore(float* p, V a) { _mm256_storeu_ps(p, a); }
static inline V vfma(V a, V b, V c) { return _mm256_fmadd_ps(a, b, c); }
static inline V vadd(V a, V b) { return _mm256_add_ps(a, b); }
static inline V vsub(V a, V b) { return _mm256_sub_ps(a, b); }
static inline V vmul(V a, V b) { return _mm256_mul_ps(a, b); }
static inline V vmax(V a, V b) { return _mm256_max_ps(a, b); }
static inline V vmin(V a, V b) { return _mm256_min_ps(a, b); }
static inline float hsum128(__m128 s) {
  __m128 sh = _mm_movehdup_ps(s);
  s = _mm_add_ps(s, sh);
  sh = _mm_movehl_ps(sh, s);
  return _mm_cvtss_f32(_mm_add_ss(s, sh));
}
static inline float vhsum(V a) { return hsum128(_mm_add_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1))); }
static inline float vhmax(V a) {
  __m128 s = _mm_max_ps(_mm256_castps256_ps128(a), _mm256_extractf128_ps(a, 1));
  s = _mm_max_ps(s, _mm_movehl_ps(s, s));
  s = _mm_max_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}
static inline V load_bf16(const uint16_t* p) {
  const __m256i w = _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  return _mm256_castsi256_ps(_mm256_slli_epi32(w, 16));
}
static inline V load_i8(const int8_t* p) {
  return _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
// e4m3 without touching f32 denormals: normal codes rebase the exponent with an integer add (bias 7 to
// 127), subnormal codes (magnitude < 8) are m * 2^-9 from an integer conversion.
static inline V e4m3_lanes(__m256i b) {
  const __m256i mag = _mm256_and_si256(b, _mm256_set1_epi32(0x7f));
  const __m256i sign = _mm256_slli_epi32(_mm256_and_si256(b, _mm256_set1_epi32(0x80)), 24);
  const __m256i normal = _mm256_add_epi32(_mm256_slli_epi32(mag, 20), _mm256_set1_epi32(120 << 23));
  const __m256 sub = _mm256_mul_ps(_mm256_cvtepi32_ps(mag), _mm256_set1_ps(0x1p-9f));
  const __m256i is_sub = _mm256_cmpgt_epi32(_mm256_set1_epi32(8), mag);
  const __m256 v = _mm256_blendv_ps(_mm256_castsi256_ps(normal), sub, _mm256_castsi256_ps(is_sub));
  return _mm256_or_ps(v, _mm256_castsi256_ps(sign));
}
static inline V load_e4m3(const uint8_t* p) {
  return e4m3_lanes(_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p))));
}
// Twice the e2m1 value of each code, as int8: a 16-entry table for pshufb.
static inline __m128i e2m1_table() { return _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12); }
static inline V load_e2m1(const uint8_t* p, float s) {
  uint32_t raw;
  memcpy(&raw, p, 4);
  const __m128i b = _mm_cvtsi32_si128(static_cast<int>(raw));
  const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(15));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(15));
  const __m128i codes = _mm_unpacklo_epi8(lo, hi);
  const __m128i v2 = _mm_shuffle_epi8(e2m1_table(), codes);
  return _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v2)), _mm256_set1_ps(0.5f * s));
}
static inline V vexp(V x) {
  x = _mm256_min_ps(_mm256_max_ps(x, _mm256_set1_ps(kExpLo)), _mm256_set1_ps(kExpHi));
  const __m256 fx = _mm256_round_ps(_mm256_mul_ps(x, _mm256_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m256 r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(kC1), x);
  r = _mm256_fnmadd_ps(fx, _mm256_set1_ps(kC2), r);
  __m256 p = _mm256_set1_ps(kP0);
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP1));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP2));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP3));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP4));
  p = _mm256_fmadd_ps(p, r, _mm256_set1_ps(kP5));
  const __m256 rr = _mm256_mul_ps(r, r);
  const __m256 y = _mm256_add_ps(_mm256_fmadd_ps(p, rr, r), _mm256_set1_ps(1.0f));
  const __m256i e = _mm256_slli_epi32(_mm256_add_epi32(_mm256_cvtps_epi32(fx), _mm256_set1_epi32(127)), 23);
  return _mm256_mul_ps(y, _mm256_castsi256_ps(e));
}

#elif LX_ISA_CODE >= 20 && LX_ISA_CODE < 50
// ---- AVX-512 (F, BW, DQ, VL; with VNNI and BF16 at the higher levels) --------------------------------
constexpr int W = 16;
using V = __m512;
static inline V vzero() { return _mm512_setzero_ps(); }
static inline V vset1(float f) { return _mm512_set1_ps(f); }
static inline V vload(const float* p) { return _mm512_loadu_ps(p); }
static inline void vstore(float* p, V a) { _mm512_storeu_ps(p, a); }
static inline V vfma(V a, V b, V c) { return _mm512_fmadd_ps(a, b, c); }
static inline V vadd(V a, V b) { return _mm512_add_ps(a, b); }
static inline V vsub(V a, V b) { return _mm512_sub_ps(a, b); }
static inline V vmul(V a, V b) { return _mm512_mul_ps(a, b); }
static inline V vmax(V a, V b) { return _mm512_max_ps(a, b); }
static inline V vmin(V a, V b) { return _mm512_min_ps(a, b); }
static inline float hsum128(__m128 s) {
  __m128 sh = _mm_movehdup_ps(s);
  s = _mm_add_ps(s, sh);
  sh = _mm_movehl_ps(sh, s);
  return _mm_cvtss_f32(_mm_add_ss(s, sh));
}
static inline float vhsum(V a) {
  const __m256 h = _mm256_add_ps(_mm512_castps512_ps256(a), _mm512_extractf32x8_ps(a, 1));
  return hsum128(_mm_add_ps(_mm256_castps256_ps128(h), _mm256_extractf128_ps(h, 1)));
}
static inline float vhmax(V a) { return _mm512_reduce_max_ps(a); }
static inline V load_bf16(const uint16_t* p) {
  const __m512i w = _mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)));
  return _mm512_castsi512_ps(_mm512_slli_epi32(w, 16));
}
static inline V load_i8(const int8_t* p) {
  return _mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p))));
}
static inline V load_e4m3(const uint8_t* p) {
  const __m512i b = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  const __m512i mag = _mm512_and_si512(b, _mm512_set1_epi32(0x7f));
  const __m512i sign = _mm512_slli_epi32(_mm512_and_si512(b, _mm512_set1_epi32(0x80)), 24);
  const __m512i normal = _mm512_add_epi32(_mm512_slli_epi32(mag, 20), _mm512_set1_epi32(120 << 23));
  const __m512 sub = _mm512_mul_ps(_mm512_cvtepi32_ps(mag), _mm512_set1_ps(0x1p-9f));
  const __mmask16 is_sub = _mm512_cmplt_epi32_mask(mag, _mm512_set1_epi32(8));
  const __m512 v = _mm512_mask_blend_ps(is_sub, _mm512_castsi512_ps(normal), sub);
  return _mm512_or_ps(v, _mm512_castsi512_ps(sign));
}
static inline __m128i e2m1_table() { return _mm_setr_epi8(0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12); }
static inline V load_e2m1(const uint8_t* p, float s) {
  const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(15));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(15));
  const __m128i codes = _mm_unpacklo_epi8(lo, hi);
  const __m128i v2 = _mm_shuffle_epi8(e2m1_table(), codes);
  return _mm512_mul_ps(_mm512_cvtepi32_ps(_mm512_cvtepi8_epi32(v2)), _mm512_set1_ps(0.5f * s));
}
static inline V vexp(V x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(kExpLo)), _mm512_set1_ps(kExpHi));
  const __m512 fx = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(kLog2e)), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(kC1), x);
  r = _mm512_fnmadd_ps(fx, _mm512_set1_ps(kC2), r);
  __m512 p = _mm512_set1_ps(kP0);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP1));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP2));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP3));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP4));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(kP5));
  const __m512 rr = _mm512_mul_ps(r, r);
  const __m512 y = _mm512_add_ps(_mm512_fmadd_ps(p, rr, r), _mm512_set1_ps(1.0f));
  const __m512i e = _mm512_slli_epi32(_mm512_add_epi32(_mm512_cvtps_epi32(fx), _mm512_set1_epi32(127)), 23);
  return _mm512_mul_ps(y, _mm512_castsi512_ps(e));
}

#elif LX_ISA_CODE >= 50
// ---- NEON (with the dot product, and I8MM + BF16, at the higher levels) ------------------------------
constexpr int W = 4;
using V = float32x4_t;
static inline V vzero() { return vdupq_n_f32(0.0f); }
static inline V vset1(float f) { return vdupq_n_f32(f); }
static inline V vload(const float* p) { return vld1q_f32(p); }
static inline void vstore(float* p, V a) { vst1q_f32(p, a); }
static inline V vfma(V a, V b, V c) { return vfmaq_f32(c, a, b); }
static inline V vadd(V a, V b) { return vaddq_f32(a, b); }
static inline V vsub(V a, V b) { return vsubq_f32(a, b); }
static inline V vmul(V a, V b) { return vmulq_f32(a, b); }
static inline V vmax(V a, V b) { return vmaxq_f32(a, b); }
static inline V vmin(V a, V b) { return vminq_f32(a, b); }
static inline float vhsum(V a) {
  const float32x2_t s = vadd_f32(vget_low_f32(a), vget_high_f32(a));  // (0 + 2, 1 + 3)
  return vget_lane_f32(s, 0) + vget_lane_f32(s, 1);
}
static inline float vhmax(V a) { return vmaxvq_f32(a); }
static inline V load_bf16(const uint16_t* p) { return vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(p), 16)); }
static inline V load_i8(const int8_t* p) {
  int32_t raw;
  memcpy(&raw, p, 4);
  const int16x8_t w = vmovl_s8(vreinterpret_s8_s32(vdup_n_s32(raw)));
  return vcvtq_f32_s32(vmovl_s16(vget_low_s16(w)));
}
// e4m3 through f16: the byte's exponent and mantissa placed in an f16 (bias 15) give 2^-8 of the value,
// subnormals included, and the f16 to f32 conversion is exact.
static inline V e4m3_half4(uint16x4_t b) {
  const uint16x4_t h = vorr_u16(vshl_n_u16(vand_u16(b, vdup_n_u16(0x7f)), 7), vshl_n_u16(vand_u16(b, vdup_n_u16(0x80)), 8));
  return vmulq_n_f32(vcvt_f32_f16(vreinterpret_f16_u16(h)), 256.0f);
}
static inline V load_e4m3(const uint8_t* p) {
  int32_t raw;
  memcpy(&raw, p, 4);
  const uint16x8_t b = vmovl_u8(vreinterpret_u8_s32(vdup_n_s32(raw)));
  return e4m3_half4(vget_low_u16(b));
}
static inline int8x16_t e2m1_table() {
  static const int8_t t[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
  return vld1q_s8(t);
}
static inline V load_e2m1(const uint8_t* p, float s) {
  uint16_t raw;
  memcpy(&raw, p, 2);
  const uint8x8_t b = vreinterpret_u8_u16(vdup_n_u16(raw));
  const uint8x8_t codes = vzip1_u8(vand_u8(b, vdup_n_u8(15)), vshr_n_u8(b, 4));
  const int8x8_t v2 = vqtbl1_s8(e2m1_table(), codes);
  return vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(vmovl_s8(v2)))), 0.5f * s);
}
static inline V vexp(V x) {
  x = vminq_f32(vmaxq_f32(x, vdupq_n_f32(kExpLo)), vdupq_n_f32(kExpHi));
  const V fx = vrndnq_f32(vmulq_n_f32(x, kLog2e));
  V r = vfmsq_n_f32(x, fx, kC1);
  r = vfmsq_n_f32(r, fx, kC2);
  V p = vdupq_n_f32(kP0);
  p = vfmaq_f32(vdupq_n_f32(kP1), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP2), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP3), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP4), p, r);
  p = vfmaq_f32(vdupq_n_f32(kP5), p, r);
  const V rr = vmulq_f32(r, r);
  const V y = vaddq_f32(vfmaq_f32(r, p, rr), vdupq_n_f32(1.0f));
  const int32x4_t e = vshlq_n_s32(vaddq_s32(vcvtq_s32_f32(fx), vdupq_n_s32(127)), 23);
  return vmulq_f32(y, vreinterpretq_f32_s32(e));
}
#endif

}  // namespace lx::cpu::LX_ISA
