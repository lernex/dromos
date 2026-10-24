// The vector layer of the per-ISA kernels (kern/isa.cpp). Exactly one section is compiled per level:
// LX_ISA_CODE picks it (tunable.h's lx_arch codes on the CPU, engine/cpu/isa.cpp). Everything here is
// `static inline` inside the level's own namespace, so no two levels share a symbol.
//
//   V       a vector of W floats
//   vload / vstore / vset1 / vzero / vfma(a, b, c) = a * b + c / vadd / vsub / vmul / vmax / vmin
//   vhsum   the horizontal sum, in one fixed order per level (results are deterministic)
//   load_bf16(p)  W bf16 values as f32; load_i8(p) W int8; load_e4m3(p) W e4m3 times 1 / kE4m3Post (an
//                 exact power of two the caller multiplies a whole sum by); load_e2m1(p, s) W e2m1 codes
//                 from W / 2 bytes (low nibble first) times s
//   span_*(p, out)  kSpan* values (a multiple of W) at once into kSpan* / W vectors, the same values as the
//                 load_* of each W, decoded together where the ISA converts wider than one vector
//   kE4m3Tab      the 256 e4m3 values (a compile-time table: block scales cost one load)
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
// e4m3 as e4m3f decodes it (the codes the "fn" format calls NaN decode as 480 here; no valid weight
// or scale holds one), built at compile time so no code runs at load in a per-ISA object.
constexpr float e4m3_value(int b) {
  const int e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? static_cast<float>(m) / 512.0f : 1.0f + static_cast<float>(m) / 8.0f;
  for (int i = 7; e > 0 && i < e; ++i) v *= 2.0f;
  for (int i = e; e > 0 && i < 7; ++i) v *= 0.5f;
  return (b & 0x80) ? -v : v;
}
struct E4m3Table {
  float v[256];
};
constexpr E4m3Table make_e4m3_table() {
  E4m3Table t{};
  for (int i = 0; i < 256; ++i) t.v[i] = e4m3_value(i);
  return t;
}
alignas(64) static constexpr E4m3Table kE4m3Tab = make_e4m3_table();

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
constexpr float kE4m3Post = 1.0f;
constexpr int kSpanBF16 = W, kSpanI8 = W, kSpanE4m3 = W, kSpanE2m1 = W;
static inline void span_bf16(const uint16_t* p, V* out) { out[0] = load_bf16(p); }
static inline void span_i8(const int8_t* p, V* out) { out[0] = load_i8(p); }
static inline void span_e4m3(const uint8_t* p, V* out) { out[0] = load_e4m3(p); }
static inline void span_e2m1(const uint8_t* p, float s, V* out) { out[0] = load_e2m1(p, s); }

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
// e4m3 through f16 (F16C): the byte's exponent and mantissa placed in an f16 (bias 15 against 7) give
// 2^-8 of the value, subnormals included, and the conversion is exact. kE4m3Post puts the 2^8 back,
// once per sum. 16-bit lanes hold the bytes.
constexpr float kE4m3Post = 256.0f;
static inline __m128i e4m3_half(__m128i b) {
  return _mm_or_si128(_mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(0x7f)), 7),
                      _mm_slli_epi16(_mm_and_si128(b, _mm_set1_epi16(0x80)), 8));
}
static inline V load_e4m3(const uint8_t* p) {
  return _mm256_cvtph_ps(e4m3_half(_mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i*>(p)))));
}
constexpr int kSpanBF16 = W, kSpanI8 = W, kSpanE4m3 = 2 * W, kSpanE2m1 = 2 * W;
static inline void span_bf16(const uint16_t* p, V* out) { out[0] = load_bf16(p); }
static inline void span_i8(const int8_t* p, V* out) { out[0] = load_i8(p); }
static inline void span_e4m3(const uint8_t* p, V* out) {
  const __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  const __m256i h = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), 7),
                                    _mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x80)), 8));
  out[0] = _mm256_cvtph_ps(_mm256_castsi256_si128(h));
  out[1] = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1));
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
// One 16-value scale block from 8 bytes.
static inline void span_e2m1(const uint8_t* p, float s, V* out) {
  const __m128i b = _mm_loadl_epi64(reinterpret_cast<const __m128i*>(p));
  const __m128i lo = _mm_and_si128(b, _mm_set1_epi8(15));
  const __m128i hi = _mm_and_si128(_mm_srli_epi16(b, 4), _mm_set1_epi8(15));
  const __m128i v2 = _mm_shuffle_epi8(e2m1_table(), _mm_unpacklo_epi8(lo, hi));
  const __m256 hs = _mm256_set1_ps(0.5f * s);
  out[0] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(v2)), hs);
  out[1] = _mm256_mul_ps(_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_unpackhi_epi64(v2, v2))), hs);
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
// e4m3 through f16, as on AVX2: 2^-8 of the value, exact; kE4m3Post restores it once per sum.
constexpr float kE4m3Post = 256.0f;
static inline V load_e4m3(const uint8_t* p) {
  const __m256i b = _mm256_cvtepu8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
  const __m256i h = _mm256_or_si256(_mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x7f)), 7),
                                    _mm256_slli_epi16(_mm256_and_si256(b, _mm256_set1_epi16(0x80)), 8));
  return _mm512_cvtph_ps(h);
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
constexpr int kSpanBF16 = W, kSpanI8 = W, kSpanE4m3 = W, kSpanE2m1 = W;
static inline void span_bf16(const uint16_t* p, V* out) { out[0] = load_bf16(p); }
static inline void span_i8(const int8_t* p, V* out) { out[0] = load_i8(p); }
static inline void span_e4m3(const uint8_t* p, V* out) { out[0] = load_e4m3(p); }
static inline void span_e2m1(const uint8_t* p, float s, V* out) { out[0] = load_e2m1(p, s); }
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
// subnormals included, and the f16 to f32 conversion is exact. kE4m3Post restores the 2^8 once per sum.
constexpr float kE4m3Post = 256.0f;
static inline V e4m3_half4(uint16x4_t b) {
  const uint16x4_t h = vorr_u16(vshl_n_u16(vand_u16(b, vdup_n_u16(0x7f)), 7), vshl_n_u16(vand_u16(b, vdup_n_u16(0x80)), 8));
  return vcvt_f32_f16(vreinterpret_f16_u16(h));
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
// Sixteen values at a time: one load and wide conversions instead of four narrow ones.
constexpr int kSpanBF16 = 4 * W, kSpanI8 = 4 * W, kSpanE4m3 = 4 * W, kSpanE2m1 = 4 * W;
static inline void span_bf16(const uint16_t* p, V* out) {
  const uint16x8_t a = vld1q_u16(p), b = vld1q_u16(p + 8);
  out[0] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(a), 16));
  out[1] = vreinterpretq_f32_u32(vshll_high_n_u16(a, 16));
  out[2] = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(b), 16));
  out[3] = vreinterpretq_f32_u32(vshll_high_n_u16(b, 16));
}
static inline void span_i8(const int8_t* p, V* out) {
  const int8x16_t v = vld1q_s8(p);
  const int16x8_t lo = vmovl_s8(vget_low_s8(v)), hi = vmovl_high_s8(v);
  out[0] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(lo)));
  out[1] = vcvtq_f32_s32(vmovl_high_s16(lo));
  out[2] = vcvtq_f32_s32(vmovl_s16(vget_low_s16(hi)));
  out[3] = vcvtq_f32_s32(vmovl_high_s16(hi));
}
static inline float16x8_t e4m3_half8(uint16x8_t b) {
  return vreinterpretq_f16_u16(
      vorrq_u16(vshlq_n_u16(vandq_u16(b, vdupq_n_u16(0x7f)), 7), vshlq_n_u16(vandq_u16(b, vdupq_n_u16(0x80)), 8)));
}
static inline void span_e4m3(const uint8_t* p, V* out) {
  const uint8x16_t v = vld1q_u8(p);
  const float16x8_t lo = e4m3_half8(vmovl_u8(vget_low_u8(v))), hi = e4m3_half8(vmovl_high_u8(v));
  out[0] = vcvt_f32_f16(vget_low_f16(lo));
  out[1] = vcvt_high_f32_f16(lo);
  out[2] = vcvt_f32_f16(vget_low_f16(hi));
  out[3] = vcvt_high_f32_f16(hi);
}
// One 16-value scale block from 8 bytes.
static inline void span_e2m1(const uint8_t* p, float s, V* out) {
  const uint8x8_t b = vld1_u8(p);
  const uint8x8_t lo = vand_u8(b, vdup_n_u8(15)), hi = vshr_n_u8(b, 4);
  const int8x16_t v2 = vqtbl1q_s8(e2m1_table(), vcombine_u8(vzip1_u8(lo, hi), vzip2_u8(lo, hi)));
  const int16x8_t a = vmovl_s8(vget_low_s8(v2)), c = vmovl_high_s8(v2);
  const float hs = 0.5f * s;
  out[0] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(a))), hs);
  out[1] = vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(a)), hs);
  out[2] = vmulq_n_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(c))), hs);
  out[3] = vmulq_n_f32(vcvtq_f32_s32(vmovl_high_s16(c)), hs);
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

// a * b + c rounded once, for the scalar tails of the dot tiles: written out, so whether a tail is fused
// never depends on how the compiler inlined the weight's decoder (the formats must agree to the bit).
#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 50
static inline float sfma(float a, float b, float c) {
  return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
#else
static inline float sfma(float a, float b, float c) { return __builtin_fmaf(a, b, c); }
#endif

// Double accumulators for the exact dots of the quantized maths (kern/isa.cpp, tile_exact): a vector of W
// f32 products, each exact, is widened in two halves and added to two double vectors. Every partial sum is
// exact too, so neither the order nor the width changes the result.
#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 20
using VD = __m256d;
static inline VD vd_zero() { return _mm256_setzero_pd(); }
static inline void vd_acc(VD& lo, VD& hi, V p) {
  lo = _mm256_add_pd(lo, _mm256_cvtps_pd(_mm256_castps256_ps128(p)));
  hi = _mm256_add_pd(hi, _mm256_cvtps_pd(_mm256_extractf128_ps(p, 1)));
}
static inline double vd_hsum(VD lo, VD hi) {
  const __m256d s = _mm256_add_pd(lo, hi);
  const __m128d h = _mm_add_pd(_mm256_castpd256_pd128(s), _mm256_extractf128_pd(s, 1));
  return _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
}
#elif LX_ISA_CODE >= 20 && LX_ISA_CODE < 50
using VD = __m512d;
static inline VD vd_zero() { return _mm512_setzero_pd(); }
static inline void vd_acc(VD& lo, VD& hi, V p) {
  lo = _mm512_add_pd(lo, _mm512_cvtps_pd(_mm512_castps512_ps256(p)));
  hi = _mm512_add_pd(hi, _mm512_cvtps_pd(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(p), 1))));
}
static inline double vd_hsum(VD lo, VD hi) {
  const __m512d s = _mm512_add_pd(lo, hi);
  const __m256d q = _mm256_add_pd(_mm512_castpd512_pd256(s), _mm512_extractf64x4_pd(s, 1));
  const __m128d h = _mm_add_pd(_mm256_castpd256_pd128(q), _mm256_extractf128_pd(q, 1));
  return _mm_cvtsd_f64(_mm_add_sd(h, _mm_unpackhi_pd(h, h)));
}
#elif LX_ISA_CODE >= 50
using VD = float64x2_t;
static inline VD vd_zero() { return vdupq_n_f64(0.0); }
static inline void vd_acc(VD& lo, VD& hi, V p) {
  lo = vaddq_f64(lo, vcvt_f64_f32(vget_low_f32(p)));
  hi = vaddq_f64(hi, vcvt_high_f64_f32(p));
}
static inline double vd_hsum(VD lo, VD hi) { return vaddvq_f64(vaddq_f64(lo, hi)); }
#else
struct VD {
  double v[2];
};
static inline VD vd_zero() { return VD{{0.0, 0.0}}; }
static inline void vd_acc(VD& lo, VD& hi, V p) {
  lo.v[0] += p.v[0];
  lo.v[1] += p.v[1];
  hi.v[0] += p.v[2];
  hi.v[1] += p.v[3];
}
static inline double vd_hsum(VD lo, VD hi) { return (lo.v[0] + hi.v[0]) + (lo.v[1] + hi.v[1]); }
#endif

// ---- BF16X ---------------------------------------------------------------------------------------------
// 32 consecutive values of one 128-value block of an LX_W_BF16X row (layout: kern/table.h, bf16x) to their
// bf16 bits, in k order: sm the values' 32 sign|mantissa bytes, codes the 16 code bytes that cover them,
// base the block's largest exponent. A bf16's high byte is sign | exponent >> 1 and its low byte
// (exponent & 1) << 7 | mantissa, so each half is two masks and a shift of the bytes. Patched values come
// out with the base exponent (their code is 0); the caller writes their exponents in.
#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 50
static inline void bf16x_span32(const uint8_t* sm, const uint8_t* codes, uint32_t base, uint16_t* out) {
  // Code word g holds values 8g .. 8g + 3 in its bytes' low nibbles and 8g + 4 .. 8g + 7 in the high
  // ones: interleaving the low and high nibble words puts the codes in k order.
  const __m128i c = _mm_loadu_si128(reinterpret_cast<const __m128i*>(codes));
  const __m128i nib = _mm_set1_epi8(15);
  const __m128i lo = _mm_and_si128(c, nib), hi = _mm_and_si128(_mm_srli_epi16(c, 4), nib);
  const __m256i code = _mm256_set_m128i(_mm_unpackhi_epi32(lo, hi), _mm_unpacklo_epi32(lo, hi));
  const __m256i e = _mm256_sub_epi8(_mm256_set1_epi8(static_cast<char>(base)), code);
  const __m256i v = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(sm));
  const __m256i m80 = _mm256_set1_epi8(static_cast<char>(0x80)), m7f = _mm256_set1_epi8(0x7f);
  const __m256i hb = _mm256_or_si256(_mm256_and_si256(v, m80), _mm256_and_si256(_mm256_srli_epi16(e, 1), m7f));
  const __m256i lb = _mm256_or_si256(_mm256_and_si256(v, m7f), _mm256_and_si256(_mm256_slli_epi16(e, 7), m80));
  const __m256i a = _mm256_unpacklo_epi8(lb, hb);  // values 0-7 | 16-23
  const __m256i b = _mm256_unpackhi_epi8(lb, hb);  // values 8-15 | 24-31
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out), _mm256_permute2x128_si256(a, b, 0x20));
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(out + 16), _mm256_permute2x128_si256(a, b, 0x31));
}
#elif LX_ISA_CODE >= 50
static inline void bf16x_span32(const uint8_t* sm, const uint8_t* codes, uint32_t base, uint16_t* out) {
  const uint8x16_t c = vld1q_u8(codes);
  const uint32x4_t lo = vreinterpretq_u32_u8(vandq_u8(c, vdupq_n_u8(15)));
  const uint32x4_t hi = vreinterpretq_u32_u8(vshrq_n_u8(c, 4));
  const uint8x16_t code[2] = {vreinterpretq_u8_u32(vzip1q_u32(lo, hi)), vreinterpretq_u8_u32(vzip2q_u32(lo, hi))};
  const uint8x16_t b = vdupq_n_u8(static_cast<uint8_t>(base));
  for (int h = 0; h < 2; ++h) {
    const uint8x16_t v = vld1q_u8(sm + 16 * h), e = vsubq_u8(b, code[h]);
    const uint8x16_t hb = vorrq_u8(vandq_u8(v, vdupq_n_u8(0x80)), vshrq_n_u8(e, 1));
    const uint8x16_t lb = vorrq_u8(vandq_u8(v, vdupq_n_u8(0x7f)), vshlq_n_u8(e, 7));
    vst1q_u8(reinterpret_cast<uint8_t*>(out + 16 * h), vzip1q_u8(lb, hb));
    vst1q_u8(reinterpret_cast<uint8_t*>(out + 16 * h + 8), vzip2q_u8(lb, hb));
  }
}
#else
static inline void bf16x_span32(const uint8_t* sm, const uint8_t* codes, uint32_t base, uint16_t* out) {
  for (int i = 0; i < 32; ++i) {
    const uint32_t c = codes[(i >> 3) * 4 + (i & 3)];
    const uint32_t e = (base - ((i & 4) ? c >> 4 : c & 15u)) & 0xffu;
    const uint32_t v = sm[i];
    out[i] = static_cast<uint16_t>(((v & 0x80u) << 8) | (e << 7) | (v & 0x7fu));
  }
}
#endif

}  // namespace lx::cpu::LX_ISA
