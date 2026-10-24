// The weight formats' number types, the weight quantizers (lx_quantize_w8, lx_quantize_weight,
// lx_dequantize_weight) and the per-token activation quantizer (lx_quantize_act). Every rounding follows
// the CUDA kernels and the PyTorch references bit for bit: divisions are IEEE (never a multiply by a
// reciprocal unless the spec says so), and ties go to even.
#include <math.h>

#include "cpu/cpu.h"

namespace lx::cpu {

float e4m3_to_f32(uint8_t b) {
  const uint32_t mag = b & 0x7fu;
  // Normal codes rebase the exponent (bias 7 to 127) with an integer add; subnormals are m * 2^-9.
  const float v = mag >= 8 ? bits_f32((mag << 20) + (120u << 23)) : static_cast<float>(mag) * 0x1p-9f;
  return (b & 0x80u) ? -v : v;
}

uint8_t f32_to_e4m3(float f) {
  uint32_t u = f32_bits(f);
  const uint8_t sign = static_cast<uint8_t>((u >> 24) & 0x80u);
  u &= 0x7fffffffu;
  if (u > 0x7f800000u) return static_cast<uint8_t>(sign | 0x7fu);  // NaN
  const float a = bits_f32(u);
  if (a >= 448.0f) return static_cast<uint8_t>(sign | 0x7eu);  // saturate (448 is the largest finite)
  if (a < 0x1p-6f) {
    // Subnormal range: m * 2^-9, m rounded to nearest even; m = 8 is the smallest normal, code 0x08.
    const int m = static_cast<int>(rintf(a * 512.0f));
    return static_cast<uint8_t>(sign | m);
  }
  u += 0x7ffffu + ((u >> 20) & 1u);  // round the mantissa to 3 bits, ties to even
  u &= 0xfff00000u;
  const int e = static_cast<int>(u >> 23) - 127 + 7;
  const uint32_t m = (u >> 20) & 7u;
  if (e > 15 || (e == 15 && m == 7)) return static_cast<uint8_t>(sign | 0x7eu);
  return static_cast<uint8_t>(sign | (static_cast<uint32_t>(e) << 3) | m);
}

float e2m1_to_f32(uint32_t code) {
  static const float kGrid[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
  const float v = kGrid[code & 7u];
  return (code & 8u) ? -v : v;
}

uint32_t f32_to_e2m1(float x) {
  const float mag = fminf(fabsf(x), 6.0f);
  static const float kMids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  uint32_t idx = 0;
  for (int i = 0; i < 7; ++i) idx += mag > kMids[i];
  // On the midpoint above code idx: go to the even code of the two.
  if (idx < 7 && mag == kMids[idx] && (idx & 1u)) ++idx;
  const uint32_t sign = (x < 0.0f && idx > 0) ? 8u : 0u;
  return idx | sign;
}

namespace {

const uint16_t* bf16_ptr(const void* p) { return static_cast<const uint16_t*>(p); }

// max |w| over a bf16 row.
float row_absmax(const uint16_t* w, int cols) {
  float m = 0.0f;
  for (int c = 0; c < cols; ++c) m = fmaxf(m, fabsf(bf16_to_f32(w[c])));
  return m;
}

int quantize_int8(const uint16_t* w, int8_t* q, float* scale, int64_t rows, int cols) {
  parallel_for(rows, grain_for(rows, 16), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const uint16_t* src = w + r * cols;
      const float amax = row_absmax(src, cols);
      // An all-zero row keeps scale 1, so it quantizes to exact zeros.
      const float s = amax > 0.0f ? amax / 127.0f : 1.0f;
      scale[r] = s;
      int8_t* dst = q + r * cols;
      for (int c = 0; c < cols; ++c) {
        const float v = rintf(bf16_to_f32(src[c]) / s);
        dst[c] = static_cast<int8_t>(static_cast<int>(fminf(fmaxf(v, -127.0f), 127.0f)));
      }
    }
  });
  return kOk;
}

int quantize_fp8(const uint16_t* w, uint8_t* q, float* scale, int64_t rows, int cols) {
  parallel_for(rows, grain_for(rows, 16), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const uint16_t* src = w + r * cols;
      const float amax = row_absmax(src, cols);
      const float s = amax > 0.0f ? amax / 448.0f : 1.0f;
      scale[r] = s;
      uint8_t* dst = q + r * cols;
      for (int c = 0; c < cols; ++c) dst[c] = f32_to_e4m3(fminf(fmaxf(bf16_to_f32(src[c]) / s, -448.0f), 448.0f));
    }
  });
  return kOk;
}

// One 16-value block: its e4m3 scale (absmax / (6 * global)) and its codes, w / (scale * global).
void nvfp4_block(const float* v, float global, uint8_t* scale_out, uint8_t* codes_out) {
  float amax = 0.0f;
  for (int j = 0; j < 16; ++j) amax = fmaxf(amax, fabsf(v[j]));
  const uint8_t sb = f32_to_e4m3(amax / (6.0f * global));
  *scale_out = sb;
  const float eff = e4m3_to_f32(sb) * global;
  for (int p = 0; p < 8; ++p) {
    uint32_t lo = 0, hi = 0;
    if (eff > 0.0f) {
      lo = f32_to_e2m1(v[2 * p] / eff);
      hi = f32_to_e2m1(v[2 * p + 1] / eff);
    }
    codes_out[p] = static_cast<uint8_t>(lo | (hi << 4));  // the low nibble is the even k
  }
}

int quantize_nvfp4(const uint16_t* w, uint8_t* q, uint8_t* scale, float* global_out, int64_t rows, int cols) {
  // The global: max |w| of the whole matrix / (448 * 6).
  std::atomic<uint32_t> amax_bits{0};
  parallel_for(rows, grain_for(rows, 16), [&](int64_t r0, int64_t r1) {
    float m = 0.0f;
    for (int64_t r = r0; r < r1; ++r) m = fmaxf(m, row_absmax(w + r * cols, cols));
    const uint32_t b = f32_bits(m);  // non-negative: the bits order like the values
    uint32_t cur = amax_bits.load(std::memory_order_relaxed);
    while (b > cur && !amax_bits.compare_exchange_weak(cur, b, std::memory_order_relaxed)) {
    }
  });
  const float global = bits_f32(amax_bits.load()) / 2688.0f;
  *global_out = global;
  const int blocks = cols / 16;
  parallel_for(rows, grain_for(rows, 16), [&](int64_t r0, int64_t r1) {
    float v[16];
    for (int64_t r = r0; r < r1; ++r) {
      for (int b = 0; b < blocks; ++b) {
        const uint16_t* src = w + r * cols + 16 * b;
        for (int j = 0; j < 16; ++j) v[j] = bf16_to_f32(src[j]);
        nvfp4_block(v, global, scale + r * blocks + b, q + r * (cols / 2) + 8 * b);
      }
    }
  });
  return kOk;
}

// Element (n, k) of a weight in any format, without its row factor.
float element(const lx_weight& w, int64_t n, int64_t k, int64_t K) {
  switch (w.type) {
    case LX_W_BF16: return bf16_to_f32(static_cast<const uint16_t*>(w.data)[n * K + k]);
    case LX_W_INT8: return static_cast<float>(static_cast<const int8_t*>(w.data)[n * K + k]);
    case LX_W_FP8: return e4m3_to_f32(static_cast<const uint8_t*>(w.data)[n * K + k]);
    default: {
      const uint8_t b = static_cast<const uint8_t*>(w.data)[n * (K / 2) + k / 2];
      return e2m1_to_f32((k & 1) ? (b >> 4) : (b & 15u)) *
             e4m3_to_f32(static_cast<const uint8_t*>(w.scale)[n * (K / 16) + k / 16]);
    }
  }
}

}  // namespace

// The per-token quantizer of lx_quantize_act, one row. Shared with the GEMM and the grouped experts.
void quantize_act_row(const void* x, int x_dtype, int64_t row, int64_t ld_x, int math, int K, uint8_t* q,
                      float* scale, uint8_t* block_scale) {
  const int64_t base = row * ld_x;
  float a = 0.0f;
  for (int k = 0; k < K; ++k) a = fmaxf(a, fabsf(load_as_f32(x, x_dtype, base + k)));
  if (math == LX_MATH_INT8 || math == LX_MATH_FP8) {
    const float qmax = math == LX_MATH_INT8 ? 127.0f : 448.0f;
    const float inv = a > 0.0f ? qmax / a : 0.0f;
    if (math == LX_MATH_INT8) {
      int8_t* d = reinterpret_cast<int8_t*>(q);
      for (int k = 0; k < K; ++k) {
        const float v = rintf(load_as_f32(x, x_dtype, base + k) * inv);
        d[k] = static_cast<int8_t>(static_cast<int>(fminf(fmaxf(v, -127.0f), 127.0f)));
      }
    } else {
      for (int k = 0; k < K; ++k) q[k] = f32_to_e4m3(load_as_f32(x, x_dtype, base + k) * inv);
    }
    *scale = a > 0.0f ? a / qmax : 1.0f;
    return;
  }
  // FP4: a per-token global, an e4m3 scale per 16 values; a zero row gets global 1.
  const float g = a > 0.0f ? a / 2688.0f : 1.0f;
  *scale = g;
  float v[16];
  for (int b = 0; b < K / 16; ++b) {
    float amax = 0.0f;
    for (int j = 0; j < 16; ++j) {
      v[j] = load_as_f32(x, x_dtype, base + 16 * b + j);
      amax = fmaxf(amax, fabsf(v[j]));
    }
    const uint8_t s8 = f32_to_e4m3(amax / (6.0f * g));
    block_scale[b] = s8;
    const float d = e4m3_to_f32(s8) * g;
    for (int p = 0; p < 8; ++p) {
      uint32_t lo = 0, hi = 0;
      if (d > 0.0f) {
        lo = f32_to_e2m1(v[2 * p] / d);
        hi = f32_to_e2m1(v[2 * p + 1] / d);
      }
      q[8 * b + p] = static_cast<uint8_t>(lo | (hi << 4));
    }
  }
}

}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_quantize_w8(const void* w, int8_t* q, float* scale, int64_t rows, int cols, void* stream) {
  (void)stream;
  if (rows <= 0) return kOk;
  if (cols <= 0 || !w || !q || !scale) return fail(kInvalid);
  return quantize_int8(bf16_ptr(w), q, scale, rows, cols);
}

int lx_quantize_weight(const void* w, lx_weight* out, int64_t rows, int cols, void* stream) {
  (void)stream;
  if (!out || !w) return fail(kInvalid);
  if (rows <= 0) return kOk;
  if (cols <= 0) return fail(kInvalid);
  void* data = const_cast<void*>(out->data);
  void* scale = const_cast<void*>(out->scale);
  switch (out->type) {
    case LX_W_INT8:
      if (!data || !scale) return fail(kInvalid);
      return quantize_int8(bf16_ptr(w), static_cast<int8_t*>(data), static_cast<float*>(scale), rows, cols);
    case LX_W_FP8:
      if (!data || !scale) return fail(kInvalid);
      return quantize_fp8(bf16_ptr(w), static_cast<uint8_t*>(data), static_cast<float*>(scale), rows, cols);
    case LX_W_NVFP4:
      if (!data || !scale || cols % 16 != 0) return fail(kInvalid);
      return quantize_nvfp4(bf16_ptr(w), static_cast<uint8_t*>(data), static_cast<uint8_t*>(scale), &out->global, rows,
                            cols);
    default:
      return fail(kInvalid);  // BF16: nothing to quantize, the bf16 weight is the result
  }
}

int lx_dequantize_weight(const lx_weight* w, void* out, int64_t rows, int cols, void* stream) {
  if (!w || !w->data || !out) return fail(kInvalid);
  if (rows <= 0) return kOk;
  if (cols <= 0) return fail(kInvalid);
  if (w->type != LX_W_BF16 && w->type != LX_W_INT8 && w->type != LX_W_FP8 && w->type != LX_W_NVFP4)
    return fail(kInvalid);
  if (w->type != LX_W_BF16 && !w->scale) return fail(kInvalid);
  if (w->type == LX_W_NVFP4 && cols % 16 != 0) return fail(kInvalid);
  if (w->type == LX_W_BF16) return lx_rt_copy(out, w->data, 2 * rows * cols, LX_COPY_D2D, stream);
  uint16_t* o = static_cast<uint16_t*>(out);
  const lx_weight wt = *w;
  parallel_for(rows, grain_for(rows, 16), [&](int64_t r0, int64_t r1) {
    for (int64_t n = r0; n < r1; ++n) {
      const float f = wt.type == LX_W_NVFP4 ? wt.global : static_cast<const float*>(wt.scale)[n];
      for (int64_t k = 0; k < cols; ++k) o[n * cols + k] = f32_to_bf16(element(wt, n, k, cols) * f);
    }
  });
  return kOk;
}

int lx_quantize_act(const void* x, lx_dtype x_dtype, int ld_x, lx_math math, void* q, int ld_q, float* scale,
                    void* block_scale, int ld_bs, int M, int K, void* stream) {
  (void)stream;
  if (M <= 0) return kOk;
  if (!x || !q || !scale || K <= 0 || !valid_dtype(x_dtype) || (M > 1 && ld_x < K)) return fail(kInvalid);
  if (math != LX_MATH_INT8 && math != LX_MATH_FP8 && math != LX_MATH_FP4) return fail(kInvalid);
  if (math == LX_MATH_FP4) {
    if (K % 16 != 0 || !block_scale || (M > 1 && (ld_q < K / 2 || ld_bs < K / 16))) return fail(kInvalid);
  } else if (M > 1 && ld_q < K) {
    return fail(kInvalid);
  }
  uint8_t* qb = static_cast<uint8_t*>(q);
  uint8_t* bs = static_cast<uint8_t*>(block_scale);
  parallel_for(M, grain_for(M, 4), [&](int64_t m0, int64_t m1) {
    for (int64_t m = m0; m < m1; ++m)
      quantize_act_row(x, x_dtype, m, ld_x, math, K, qb + m * ld_q, scale + m, bs ? bs + m * ld_bs : nullptr);
  });
  return kOk;
}

}  // extern "C"
