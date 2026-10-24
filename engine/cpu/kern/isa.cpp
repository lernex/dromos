// The hot kernels, compiled once per ISA level (engine/cpu/CMakeLists.txt: LX_ISA names the level's
// namespace, LX_ISA_CODE its lx_arch code). Everything is written once against kern/simd.h; the int8 and
// bf16 dot products, which are where the ISAs differ most, have a section per instruction set.
//
// Products are in dot form: both operands are K-contiguous rows (activations [M, K], weights [N, K] as
// the checkpoint stores them), and a tile of MR x NR outputs keeps one vector accumulator per output,
// summed over k in order and reduced once at the end. That form reads the weights where they are (no
// transposed copy, so a memory-mapped checkpoint is used as it is), maps directly onto the dot-product
// instructions (VNNI, SDOT, BFDOT), and makes every output's arithmetic independent of the tile shape,
// the batch size and the thread that computes it.
#include <stdint.h>
#include <string.h>

#include "cpu/kern/simd.h"
#include "cpu/kern/table.h"

namespace lx::cpu::LX_ISA {
namespace {

// ---- Weight rows -------------------------------------------------------------------------------------
// One row of a weight in its format: load(k) gives the W values at k (a multiple of W) as f32, with any
// block scale applied; at(k) one value, for the tail of a row whose length is not a multiple of W.

template <int F>
struct Row;

template <>
struct Row<kW_F32> {
  const float* p;
  void init(const WeightRef& w, int64_t n, int64_t K) { p = static_cast<const float*>(w.data) + n * K; }
  void init_panel(const float* row) { p = row; }
  V load(int64_t k) const { return vload(p + k); }
  float at(int64_t k) const { return p[k]; }
};

template <>
struct Row<kW_BF16> {
  const uint16_t* p;
  void init(const WeightRef& w, int64_t n, int64_t K) { p = static_cast<const uint16_t*>(w.data) + n * K; }
  V load(int64_t k) const { return load_bf16(p + k); }
  float at(int64_t k) const { return bf16f(p[k]); }
};

template <>
struct Row<kW_INT8> {
  const int8_t* p;
  void init(const WeightRef& w, int64_t n, int64_t K) { p = static_cast<const int8_t*>(w.data) + n * K; }
  V load(int64_t k) const { return load_i8(p + k); }
  float at(int64_t k) const { return static_cast<float>(p[k]); }
};

template <>
struct Row<kW_FP8> {
  const uint8_t* p;
  void init(const WeightRef& w, int64_t n, int64_t K) { p = static_cast<const uint8_t*>(w.data) + n * K; }
  V load(int64_t k) const { return load_e4m3(p + k); }
  float at(int64_t k) const { return e4m3f(p[k]); }
};

// K is a multiple of 16 and W divides 16, so a vector never straddles two scale blocks.
template <>
struct Row<kW_NVFP4> {
  const uint8_t* p;
  const uint8_t* s;
  void init(const WeightRef& w, int64_t n, int64_t K) {
    p = static_cast<const uint8_t*>(w.data) + n * (K / 2);
    s = static_cast<const uint8_t*>(w.scale) + n * (K / 16);
  }
  V load(int64_t k) const { return load_e2m1(p + k / 2, e4m3f(s[k >> 4])); }
  float at(int64_t k) const {
    const uint8_t b = p[k >> 1];
    return e2m1f((k & 1) ? (b >> 4) : (b & 15u)) * e4m3f(s[k >> 4]);
  }
};

// ---- f32 dot tiles -----------------------------------------------------------------------------------

#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 20
// 16 vector registers.
constexpr int kBigMR = 4, kBigNR = 3;
constexpr int small_nr(int mr) { return mr == 1 ? 6 : mr == 2 ? 4 : mr == 3 ? 3 : mr <= 6 ? 2 : 1; }
#else
// 32 vector registers (AVX-512, NEON; the generic level lets the compiler decide).
constexpr int kBigMR = 6, kBigNR = 4;
constexpr int small_nr(int mr) { return mr == 1 ? 8 : mr == 2 ? 6 : mr == 3 ? 5 : mr <= 5 ? 4 : 3; }
#endif
constexpr int kSmallM = 8;  // up to this many rows the weights are streamed in their format

// out[m * ld_out + r] = sum_k x[m, k] * row_r[k] for an MR x NR tile.
template <int F, int MR, int NR>
inline void tile(const float* x, int64_t ld_x, const Row<F>* rows, int64_t K, float* out, int64_t ld_out) {
  V acc[MR][NR];
  for (int m = 0; m < MR; ++m)
    for (int r = 0; r < NR; ++r) acc[m][r] = vzero();
  int64_t k = 0;
  for (; k + W <= K; k += W) {
    V wv[NR];
    for (int r = 0; r < NR; ++r) wv[r] = rows[r].load(k);
    for (int m = 0; m < MR; ++m) {
      const V xv = vload(x + m * ld_x + k);
      for (int r = 0; r < NR; ++r) acc[m][r] = vfma(xv, wv[r], acc[m][r]);
    }
  }
  for (int m = 0; m < MR; ++m) {
    for (int r = 0; r < NR; ++r) {
      float s = vhsum(acc[m][r]);
      for (int64_t kk = k; kk < K; ++kk) s += x[m * ld_x + kk] * rows[r].at(kk);
      out[m * ld_out + r] = s;
    }
  }
}

template <int F, int MR, int NR>
inline void tile_n(int nr, const float* x, int64_t ld_x, const Row<F>* rows, int64_t K, float* out, int64_t ld_out) {
  if constexpr (NR > 1) {
    if (nr < NR) {
      tile_n<F, MR, NR - 1>(nr, x, ld_x, rows, K, out, ld_out);
      return;
    }
  }
  tile<F, MR, NR>(x, ld_x, rows, K, out, ld_out);
}

// Rows [n0, n1) of the weight against x rows [m0, m1), m1 - m0 <= kSmallM, streamed in their format.
// out points at the tile's first output (row m, column n0).
template <int F, int MR>
inline void small_rows(const DotF32& a, int64_t m, int64_t n0, int64_t n1, float* out) {
  constexpr int NR = small_nr(MR);
  Row<F> rows[NR];
  for (int64_t n = n0; n < n1; n += NR) {
    const int nr = static_cast<int>(lmin<int64_t>(NR, n1 - n));
    for (int r = 0; r < nr; ++r) rows[r].init(a.w, n + r, a.K);
    tile_n<F, MR, NR>(nr, a.x + m * a.ld_x, a.ld_x, rows, a.K, out + (n - n0), a.ld_out);
  }
}

template <int F>
inline void small_block(const DotF32& a, int64_t m, int mr, int64_t n0, int64_t n1, float* out) {
  switch (mr) {
    case 1: small_rows<F, 1>(a, m, n0, n1, out); break;
    case 2: small_rows<F, 2>(a, m, n0, n1, out); break;
    case 3: small_rows<F, 3>(a, m, n0, n1, out); break;
    case 4: small_rows<F, 4>(a, m, n0, n1, out); break;
    case 5: small_rows<F, 5>(a, m, n0, n1, out); break;
    case 6: small_rows<F, 6>(a, m, n0, n1, out); break;
    case 7: small_rows<F, 7>(a, m, n0, n1, out); break;
    default: small_rows<F, 8>(a, m, n0, n1, out); break;
  }
}

// An MR x NR tile over f32 panel rows.
template <int MR>
inline void panel_tile(int nr, const float* x, int64_t ld_x, const float* panel, int64_t K, float* out, int64_t ld_out) {
  Row<kW_F32> rows[kBigNR];
  for (int r = 0; r < nr; ++r) rows[r].init_panel(panel + r * K);
  tile_n<kW_F32, MR, kBigNR>(nr, x, ld_x, rows, K, out, ld_out);
}

inline void panel_mtile(int mr, int nr, const float* x, int64_t ld_x, const float* panel, int64_t K, float* out,
                        int64_t ld_out) {
  if constexpr (kBigMR == 6) {
    switch (mr) {
      case 6: panel_tile<6>(nr, x, ld_x, panel, K, out, ld_out); return;
      case 5: panel_tile<5>(nr, x, ld_x, panel, K, out, ld_out); return;
      default: break;
    }
  }
  switch (mr) {
    case 4: panel_tile<4>(nr, x, ld_x, panel, K, out, ld_out); return;
    case 3: panel_tile<3>(nr, x, ld_x, panel, K, out, ld_out); return;
    case 2: panel_tile<2>(nr, x, ld_x, panel, K, out, ld_out); return;
    default: panel_tile<1>(nr, x, ld_x, panel, K, out, ld_out); return;
  }
}

// Decodes rows [n, n + count) of a weight to f32 rows of K.
template <int F>
inline void decode_rows(const WeightRef& w, int64_t n, int64_t count, int64_t K, float* dst) {
  for (int64_t r = 0; r < count; ++r) {
    Row<F> row;
    row.init(w, n + r, K);
    float* d = dst + r * K;
    int64_t k = 0;
    for (; k + W <= K; k += W) vstore(d + k, row.load(k));
    for (; k < K; ++k) d[k] = row.at(k);
  }
}

template <int F>
void dot_f32_fmt(const DotF32& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
  const int64_t K = a.K;
  const bool panel = F != kW_F32 && m1 - m0 > kSmallM && a.panel && a.panel_floats >= kBigNR * K;
  if (!panel) {
    // Streamed: each weight row is read once per kSmallM activation rows.
    for (int64_t m = m0; m < m1; m += kSmallM)
      small_block<F>(a, m, static_cast<int>(lmin<int64_t>(kSmallM, m1 - m)), n0, n1, a.out + (m - m0) * a.ld_out);
    return;
  }
  // Many rows: decode a block of weight rows to f32 once, then run every activation row over it. The
  // arithmetic is the same as the streamed path's (the decode is exact).
  int64_t nb = a.panel_floats / K;
  nb = lmax<int64_t>(kBigNR, nb - nb % kBigNR);
  for (int64_t n = n0; n < n1; n += nb) {
    const int64_t rows = lmin(nb, n1 - n);
    decode_rows<F>(a.w, n, rows, K, a.panel);
    for (int64_t m = m0; m < m1; m += kBigMR) {
      const int mr = static_cast<int>(lmin<int64_t>(kBigMR, m1 - m));
      for (int64_t r = 0; r < rows; r += kBigNR) {
        const int nr = static_cast<int>(lmin<int64_t>(kBigNR, rows - r));
        panel_mtile(mr, nr, a.x + m * a.ld_x, a.ld_x, a.panel + r * K, K, a.out + (m - m0) * a.ld_out + (n - n0) + r,
                    a.ld_out);
      }
    }
  }
}

void dot_f32(const DotF32& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
  switch (a.w.type) {
    case kW_BF16: dot_f32_fmt<kW_BF16>(a, m0, m1, n0, n1); break;
    case kW_INT8: dot_f32_fmt<kW_INT8>(a, m0, m1, n0, n1); break;
    case kW_FP8: dot_f32_fmt<kW_FP8>(a, m0, m1, n0, n1); break;
    case kW_NVFP4: dot_f32_fmt<kW_NVFP4>(a, m0, m1, n0, n1); break;
    default: dot_f32_fmt<kW_F32>(a, m0, m1, n0, n1); break;
  }
}

// ---- int8 dot tiles ----------------------------------------------------------------------------------
// sum_k x[m, k] * w[n, k] in int32. Activations and weights are in [-127, 127] (both quantizers clamp
// there), which the sign tricks below rely on.

#if LX_ISA_CODE == 11 || LX_ISA_CODE == 21 || LX_ISA_CODE == 22
#define LX_I8_VNNI 1
#endif

#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 20
constexpr int kI8Step = 32;
using VI = __m256i;
inline VI i8_zero() { return _mm256_setzero_si256(); }
inline VI i8_load(const int8_t* p) { return _mm256_loadu_si256(reinterpret_cast<const __m256i*>(p)); }
inline int32_t i8_hsum(VI v) {
  __m128i s = _mm_add_epi32(_mm256_castsi256_si128(v), _mm256_extracti128_si256(v, 1));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0x4e));
  s = _mm_add_epi32(s, _mm_shuffle_epi32(s, 0xb1));
  return _mm_cvtsi128_si32(s);
}
#if defined(LX_I8_VNNI)
// u8 x s8 products: the activation is biased to unsigned (x + 128, an xor of the sign bit) and the bias
// is taken back out with 128 * sum(w) per weight row.
constexpr bool kI8Biased = true;
inline VI i8_prep(VI x) { return _mm256_xor_si256(x, _mm256_set1_epi8(static_cast<char>(0x80))); }
inline VI i8_dot(VI acc, VI xu, VI w) { return _mm256_dpbusd_avx_epi32(acc, xu, w); }
inline VI i8_wsum(VI acc, VI w) { return _mm256_dpbusd_avx_epi32(acc, _mm256_set1_epi8(1), w); }
#else
// |x| (unsigned) times w with x's sign: pairs of products stay within int16 (2 * 127 * 127).
constexpr bool kI8Biased = false;
inline VI i8_prep(VI x) { return x; }
inline VI i8_dot(VI acc, VI x, VI w) {
  const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(x, x), _mm256_sign_epi8(w, x));
  return _mm256_add_epi32(acc, _mm256_madd_epi16(p, _mm256_set1_epi16(1)));
}
inline VI i8_wsum(VI acc, VI) { return acc; }
#endif

#elif LX_ISA_CODE >= 20 && LX_ISA_CODE < 50
constexpr int kI8Step = 64;
using VI = __m512i;
inline VI i8_zero() { return _mm512_setzero_si512(); }
inline VI i8_load(const int8_t* p) { return _mm512_loadu_si512(p); }
inline int32_t i8_hsum(VI v) { return _mm512_reduce_add_epi32(v); }
#if defined(LX_I8_VNNI)
constexpr bool kI8Biased = true;
inline VI i8_prep(VI x) { return _mm512_xor_si512(x, _mm512_set1_epi8(static_cast<char>(0x80))); }
inline VI i8_dot(VI acc, VI xu, VI w) { return _mm512_dpbusd_epi32(acc, xu, w); }
inline VI i8_wsum(VI acc, VI w) { return _mm512_dpbusd_epi32(acc, _mm512_set1_epi8(1), w); }
#else
constexpr bool kI8Biased = false;
inline VI i8_prep(VI x) { return x; }
inline VI i8_dot(VI acc, VI x, VI w) {
  const __mmask64 neg = _mm512_movepi8_mask(x);
  const __m512i sw = _mm512_mask_sub_epi8(w, neg, _mm512_setzero_si512(), w);
  const __m512i p = _mm512_maddubs_epi16(_mm512_abs_epi8(x), sw);
  return _mm512_add_epi32(acc, _mm512_madd_epi16(p, _mm512_set1_epi16(1)));
}
inline VI i8_wsum(VI acc, VI) { return acc; }
#endif

#elif LX_ISA_CODE >= 50
constexpr int kI8Step = 16;
using VI = int32x4_t;
constexpr bool kI8Biased = false;
inline VI i8_zero() { return vdupq_n_s32(0); }
inline int8x16_t i8_load(const int8_t* p) { return vld1q_s8(p); }
inline int32_t i8_hsum(VI v) { return vaddvq_s32(v); }
inline int8x16_t i8_prep(int8x16_t x) { return x; }
#if LX_ISA_CODE >= 51
inline VI i8_dot(VI acc, int8x16_t x, int8x16_t w) { return vdotq_s32(acc, x, w); }
#else
inline VI i8_dot(VI acc, int8x16_t x, int8x16_t w) {
  int16x8_t p = vmull_s8(vget_low_s8(x), vget_low_s8(w));
  p = vmlal_high_s8(p, x, w);
  return vpadalq_s16(acc, p);
}
#endif
inline VI i8_wsum(VI acc, int8x16_t) { return acc; }

#else
// Generic: the int32 sums of 16 bytes at a time, in plain C++.
constexpr int kI8Step = 16;
struct VI {
  int32_t v[4];
};
struct VB {
  int8_t v[16];
};
constexpr bool kI8Biased = false;
inline VI i8_zero() { return VI{{0, 0, 0, 0}}; }
inline VB i8_load(const int8_t* p) {
  VB b;
  memcpy(b.v, p, 16);
  return b;
}
inline int32_t i8_hsum(VI v) { return (v.v[0] + v.v[2]) + (v.v[1] + v.v[3]); }
inline VB i8_prep(VB x) { return x; }
inline VI i8_dot(VI acc, VB x, VB w) {
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc.v[i] += static_cast<int32_t>(x.v[4 * i + j]) * w.v[4 * i + j];
  return acc;
}
inline VI i8_wsum(VI acc, VB) { return acc; }
#endif

#if LX_ISA_CODE >= 10 && LX_ISA_CODE < 20
constexpr int kI8MR = 4, kI8NR = 3;
#else
constexpr int kI8MR = 4, kI8NR = 4;
#endif

template <int MR, int NR>
inline void i8_tile(const int8_t* x, int64_t ld_x, const int8_t* const* w, const int32_t* wsum, int64_t K, float* out,
                    int64_t ld_out) {
  VI acc[MR][NR];
  for (int m = 0; m < MR; ++m)
    for (int r = 0; r < NR; ++r) acc[m][r] = i8_zero();
  int64_t k = 0;
  for (; k + kI8Step <= K; k += kI8Step) {
    decltype(i8_load(x)) wv[NR];
    for (int r = 0; r < NR; ++r) wv[r] = i8_load(w[r] + k);
    for (int m = 0; m < MR; ++m) {
      const auto xv = i8_prep(i8_load(x + m * ld_x + k));
      for (int r = 0; r < NR; ++r) acc[m][r] = i8_dot(acc[m][r], xv, wv[r]);
    }
  }
  for (int m = 0; m < MR; ++m) {
    for (int r = 0; r < NR; ++r) {
      int32_t s = i8_hsum(acc[m][r]);
      if (kI8Biased) s -= 128 * wsum[r];
      for (int64_t kk = k; kk < K; ++kk) s += static_cast<int32_t>(x[m * ld_x + kk]) * w[r][kk];
      out[m * ld_out + r] = static_cast<float>(s);
    }
  }
}

template <int MR, int NR>
inline void i8_tile_n(int nr, const int8_t* x, int64_t ld_x, const int8_t* const* w, const int32_t* wsum, int64_t K,
                      float* out, int64_t ld_out) {
  if constexpr (NR > 1) {
    if (nr < NR) {
      i8_tile_n<MR, NR - 1>(nr, x, ld_x, w, wsum, K, out, ld_out);
      return;
    }
  }
  i8_tile<MR, NR>(x, ld_x, w, wsum, K, out, ld_out);
}

template <int MR>
inline void i8_mtile(int nr, const int8_t* x, int64_t ld_x, const int8_t* const* w, const int32_t* wsum, int64_t K,
                     float* out, int64_t ld_out) {
  i8_tile_n<MR, kI8NR>(nr, x, ld_x, w, wsum, K, out, ld_out);
}

// The sum of a weight row over the whole-vector part of K (the part the biased products cover).
inline int32_t i8_row_sum(const int8_t* w, int64_t K) {
  VI acc = i8_zero();
  int64_t k = 0;
  for (; k + kI8Step <= K; k += kI8Step) acc = i8_wsum(acc, i8_load(w + k));
  return kI8Biased ? i8_hsum(acc) : 0;
}

void dot_i8(const DotI8& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
  const int64_t K = a.K;
  for (int64_t n = n0; n < n1; n += kI8NR) {
    const int nr = static_cast<int>(lmin<int64_t>(kI8NR, n1 - n));
    const int8_t* w[kI8NR];
    int32_t wsum[kI8NR];
    for (int r = 0; r < nr; ++r) {
      w[r] = a.w + (n + r) * K;
      wsum[r] = i8_row_sum(w[r], K);
    }
    for (int64_t m = m0; m < m1; m += kI8MR) {
      const int mr = static_cast<int>(lmin<int64_t>(kI8MR, m1 - m));
      const int8_t* x = a.x + m * a.ld_x;
      float* out = a.out + (m - m0) * a.ld_out + (n - n0);
      switch (mr) {
        case 4: i8_mtile<4>(nr, x, a.ld_x, w, wsum, K, out, a.ld_out); break;
        case 3: i8_mtile<3>(nr, x, a.ld_x, w, wsum, K, out, a.ld_out); break;
        case 2: i8_mtile<2>(nr, x, a.ld_x, w, wsum, K, out, a.ld_out); break;
        default: i8_mtile<1>(nr, x, a.ld_x, w, wsum, K, out, a.ld_out); break;
      }
    }
  }
}

// ---- bf16 dot tiles (AVX-512 BF16, Arm BF16) -----------------------------------------------------------

#if LX_ISA_CODE == 22 || LX_ISA_CODE == 52
#define LX_HAVE_BF16_DOT 1

#if LX_ISA_CODE == 22
constexpr int kBStep = 32;  // bf16 per vector
constexpr int kBMR = 4, kBNR = 6;
using VB16 = __m512i;
inline VB16 b_load(const uint16_t* p) { return _mm512_loadu_si512(p); }
inline V b_dot(V acc, VB16 x, VB16 w) { return _mm512_dpbf16_ps(acc, (__m512bh)x, (__m512bh)w); }
#else
constexpr int kBStep = 8;
constexpr int kBMR = 4, kBNR = 6;
using VB16 = bfloat16x8_t;
inline VB16 b_load(const uint16_t* p) { return vreinterpretq_bf16_u16(vld1q_u16(p)); }
inline V b_dot(V acc, VB16 x, VB16 w) { return vbfdotq_f32(acc, x, w); }
#endif

template <int MR, int NR>
inline void b_tile(const uint16_t* x, int64_t ld_x, const uint16_t* const* w, int64_t K, float* out, int64_t ld_out) {
  V acc[MR][NR];
  for (int m = 0; m < MR; ++m)
    for (int r = 0; r < NR; ++r) acc[m][r] = vzero();
  int64_t k = 0;
  for (; k + kBStep <= K; k += kBStep) {
    VB16 wv[NR];
    for (int r = 0; r < NR; ++r) wv[r] = b_load(w[r] + k);
    for (int m = 0; m < MR; ++m) {
      const VB16 xv = b_load(x + m * ld_x + k);
      for (int r = 0; r < NR; ++r) acc[m][r] = b_dot(acc[m][r], xv, wv[r]);
    }
  }
  for (int m = 0; m < MR; ++m) {
    for (int r = 0; r < NR; ++r) {
      float s = vhsum(acc[m][r]);
      for (int64_t kk = k; kk < K; ++kk) s += bf16f(x[m * ld_x + kk]) * bf16f(w[r][kk]);
      out[m * ld_out + r] = s;
    }
  }
}

template <int MR, int NR>
inline void b_tile_n(int nr, const uint16_t* x, int64_t ld_x, const uint16_t* const* w, int64_t K, float* out,
                     int64_t ld_out) {
  if constexpr (NR > 1) {
    if (nr < NR) {
      b_tile_n<MR, NR - 1>(nr, x, ld_x, w, K, out, ld_out);
      return;
    }
  }
  b_tile<MR, NR>(x, ld_x, w, K, out, ld_out);
}

// Rows of any format to bf16 (exact: every format's values have at most 8 significant bits).
template <int F>
inline void decode_rows_bf16(const WeightRef& wr, int64_t n, int64_t count, int64_t K, uint16_t* dst) {
  for (int64_t r = 0; r < count; ++r) {
    Row<F> row;
    row.init(wr, n + r, K);
    uint16_t* d = dst + r * K;
    float tmp[W];
    int64_t k = 0;
    for (; k + W <= K; k += W) {
      vstore(tmp, row.load(k));
      for (int i = 0; i < W; ++i) d[k + i] = f32_bf16(tmp[i]);
    }
    for (; k < K; ++k) d[k] = f32_bf16(row.at(k));
  }
}

void dot_bf16(const DotBF16& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1) {
  const int64_t K = a.K;
  const bool direct = a.w.type == kW_BF16;
  int64_t nb = direct ? (n1 - n0) : (a.panel_bytes / (2 * K));
  if (!direct) nb = lmax<int64_t>(1, lmin<int64_t>(nb, n1 - n0));
  for (int64_t nbase = n0; nbase < n1; nbase += nb) {
    const int64_t rows = lmin(nb, n1 - nbase);
    const uint16_t* base;
    if (direct) {
      base = static_cast<const uint16_t*>(a.w.data) + nbase * K;
    } else {
      uint16_t* p = static_cast<uint16_t*>(a.panel);
      switch (a.w.type) {
        case kW_INT8: decode_rows_bf16<kW_INT8>(a.w, nbase, rows, K, p); break;
        case kW_FP8: decode_rows_bf16<kW_FP8>(a.w, nbase, rows, K, p); break;
        default: decode_rows_bf16<kW_NVFP4>(a.w, nbase, rows, K, p); break;
      }
      base = p;
    }
    for (int64_t m = m0; m < m1; m += kBMR) {
      const int mr = static_cast<int>(lmin<int64_t>(kBMR, m1 - m));
      for (int64_t r = 0; r < rows; r += kBNR) {
        const int nr = static_cast<int>(lmin<int64_t>(kBNR, rows - r));
        const uint16_t* w[kBNR];
        for (int i = 0; i < nr; ++i) w[i] = base + (r + i) * K;
        const uint16_t* x = a.x + m * a.ld_x;
        float* out = a.out + (m - m0) * a.ld_out + (nbase - n0) + r;
        switch (mr) {
          case 4: b_tile_n<4, kBNR>(nr, x, a.ld_x, w, K, out, a.ld_out); break;
          case 3: b_tile_n<3, kBNR>(nr, x, a.ld_x, w, K, out, a.ld_out); break;
          case 2: b_tile_n<2, kBNR>(nr, x, a.ld_x, w, K, out, a.ld_out); break;
          default: b_tile_n<1, kBNR>(nr, x, a.ld_x, w, K, out, a.ld_out); break;
        }
      }
    }
  }
}
#endif

// ---- exp ---------------------------------------------------------------------------------------------

void exp_f32(const float* x, float* y, int64_t n) {
  int64_t i = 0;
  for (; i + W <= n; i += W) vstore(y + i, vexp(vload(x + i)));
  for (; i < n; ++i) y[i] = sexp(x[i]);
}

// ---- Attention -----------------------------------------------------------------------------------------
// One (sequence, kv head) and a range of its new tokens. For each token, the G query heads of the kv head
// are served from one pass over the visible cache: scores for every visible position (keys converted
// once per position for all G heads), an exact softmax over them, then the value pass.

constexpr int kMaxDh = 128;
constexpr int kMaxG = 16;

inline float dot_row(const float* a, const float* b, int n) {
  V acc = vzero();
  int i = 0;
  for (; i + W <= n; i += W) acc = vfma(vload(a + i), vload(b + i), acc);
  float s = vhsum(acc);
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline void bf16_row(const uint16_t* src, float* dst, int n) {
  int i = 0;
  for (; i + W <= n; i += W) vstore(dst + i, load_bf16(src + i));
  for (; i < n; ++i) dst[i] = bf16f(src[i]);
}

void attention(const AttnItem& a) {
  const int G = a.H / a.KVH, Dh = a.Dh;
  const int64_t row_stride = static_cast<int64_t>(a.KVH) * Dh;
  float q[kMaxG][kMaxDh], acc[kMaxG][kMaxDh], kv[kMaxDh];
  // Positions go in blocks of kBlock with an online softmax across blocks, so any context length fits
  // the stack.
  constexpr int kBlock = 256;
  float s[kMaxG][kBlock];
  for (int j = a.q0; j < a.q1; ++j) {
    const float* qrow = a.qkv + j * a.ld_qkv + static_cast<int64_t>(a.kvh) * G * Dh;
    for (int g = 0; g < G; ++g) {
      for (int d = 0; d < Dh; ++d) {
        q[g][d] = qrow[g * Dh + d] * a.scale;
        acc[g][d] = 0.0f;
      }
    }
    float mx[kMaxG], sum[kMaxG];
    for (int g = 0; g < G; ++g) {
      mx[g] = -__builtin_inff();
      sum[g] = 0.0f;
    }
    const int L = a.start + j + 1;
    for (int p0 = 0; p0 < L; p0 += kBlock) {
      const int n = lmin(kBlock, L - p0);
      for (int p = 0; p < n; ++p) {
        bf16_row(a.k + (p0 + p) * row_stride + a.kvh * Dh, kv, Dh);
        for (int g = 0; g < G; ++g) s[g][p] = dot_row(q[g], kv, Dh);
      }
      // Online softmax: rescale what was accumulated when the running max rises.
      for (int g = 0; g < G; ++g) {
        float bm = s[g][0];
        for (int p = 1; p < n; ++p) bm = lmax(bm, s[g][p]);
        const float nm = lmax(mx[g], bm);
        const float corr = mx[g] == -__builtin_inff() ? 0.0f : sexp(mx[g] - nm);
        for (int p = 0; p < n; ++p) s[g][p] -= nm;
        exp_f32(s[g], s[g], n);
        float bs = 0.0f;
        for (int p = 0; p < n; ++p) bs += s[g][p];
        sum[g] = sum[g] * corr + bs;
        if (corr != 1.0f)
          for (int d = 0; d < Dh; ++d) acc[g][d] *= corr;
        mx[g] = nm;
      }
      for (int p = 0; p < n; ++p) {
        bf16_row(a.v + (p0 + p) * row_stride + a.kvh * Dh, kv, Dh);
        for (int g = 0; g < G; ++g) {
          const V w = vset1(s[g][p]);
          int d = 0;
          for (; d + W <= Dh; d += W) vstore(acc[g] + d, vfma(w, vload(kv + d), vload(acc[g] + d)));
          for (; d < Dh; ++d) acc[g][d] += s[g][p] * kv[d];
        }
      }
    }
    uint16_t* orow = a.out + static_cast<int64_t>(j) * a.H * Dh + static_cast<int64_t>(a.kvh) * G * Dh;
    for (int g = 0; g < G; ++g) {
      const float inv = 1.0f / sum[g];
      for (int d = 0; d < Dh; ++d) orow[g * Dh + d] = f32_bf16(acc[g][d] * inv);
    }
  }
}

// ---- Selective scan --------------------------------------------------------------------------------------

inline float softplus_f(float v) { return v > 20.0f ? v : __builtin_log1pf(sexp(v)); }

void scan(const ScanItem& a) {
  const int P = a.P, N = a.N;
  float* S = a.state;
  if (!a.has_state)
    for (int i = 0; i < P * N; ++i) S[i] = 0.0f;
  for (int t = 0; t < a.len; ++t) {
    const float* xr = a.x + t * a.ld_x + a.h * P;
    const float* Br = a.B + t * a.ld_bc + a.g * N;
    const float* Cr = a.C + t * a.ld_bc + a.g * N;
    const float dt = softplus_f(a.dt[t * a.ld_dt + a.h] + a.dt_bias);
    const float decay = sexp(dt * a.negA);
    const V vdecay = vset1(decay);
    float* yr = a.y + t * a.ld_y + a.h * P;
    for (int p = 0; p < P; ++p) {
      float* Sp = S + p * N;
      const float dtx = dt * xr[p];
      const V vdtx = vset1(dtx);
      V acc = vzero();
      int n = 0;
      for (; n + W <= N; n += W) {
        const V sv = vfma(vdecay, vload(Sp + n), vmul(vdtx, vload(Br + n)));
        vstore(Sp + n, sv);
        acc = vfma(sv, vload(Cr + n), acc);
      }
      float yv = vhsum(acc);
      for (; n < N; ++n) {
        Sp[n] = decay * Sp[n] + dtx * Br[n];
        yv += Sp[n] * Cr[n];
      }
      yr[p] = yv + a.D * xr[p];
    }
  }
}

}  // namespace

#define LX_STR2(x) #x
#define LX_STR(x) LX_STR2(x)

const Kern kTable = {
    LX_STR(LX_ISA),
    LX_ISA_CODE,
    W,
    dot_f32,
    dot_i8,
#if defined(LX_HAVE_BF16_DOT)
    dot_bf16,
#else
    nullptr,
#endif
    attention,
    scan,
    exp_f32,
};

}  // namespace lx::cpu::LX_ISA

namespace lx::cpu {
const Kern* LX_JOIN(kern_, LX_ISA)() { return &LX_ISA::kTable; }
}  // namespace lx::cpu
