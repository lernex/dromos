// Dense products: lx_gemm_bf16, lx_gemm_f32, lx_gemv, lx_gemv_w and lx_gemm_w, through one driver.
//
// A product is: prepare the activations for the math (f32 rows, int8 rows with per-token scales, or bf16
// rows), split the output into (row block, column block) items for the pool, run the level's dot kernel
// on each item into a small per-thread buffer, and apply the epilogue there:
//   y[m, n] = alpha * (row_factor[n] * x_scale[m] * dot[m, n] + bias[n]) + beta * y[m, n]
// (y is read only when beta != 0). Decode-sized calls (M <= 8) stream each weight row once for all rows of
// x; larger ones decode a block of weight rows once and run every activation row over it.
//
// Maths on the CPU (kernels.h, lx_math): BF16 runs on the level's bf16 dot instructions where it has them
// and in f32 otherwise (bf16 values multiply exactly in f32); INT8 on the int8 dot instructions (VNNI,
// SDOT); FP8 and FP4 quantize the activations exactly as a GPU would, then multiply the quantized values,
// which bf16 and f32 hold exactly, so every math gives the GPU's numbers up to summation order.
#include <algorithm>

#include "cpu/cpu.h"
#include "cpu/product.h"

namespace lx::cpu {

void Epilogue::apply(const float* sums, int64_t ld_s, int64_t m0, int64_t m1, int64_t n0, int64_t n1) const {
  for (int64_t m = m0; m < m1; ++m) {
    const float* srow = sums + (m - m0) * ld_s;
    const float xs = x_scale ? x_scale[m] : 1.0f;
    for (int64_t n = n0; n < n1; ++n) {
      const float rf = row_factor ? row_factor[n] : row_factor_scalar;
      float t = rf * xs * srow[n - n0];
      if (bias) t += load_as_f32(bias, bias_dtype, n);
      float v = alpha * t;
      if (beta != 0.0f) v = beta * load_as_f32(y, y_dtype, m * ld_y + n) + v;
      store_from_f32(y, y_dtype, m * ld_y + n, v);
    }
  }
}

void Epilogue::apply_d(const double* sums, int64_t ld_s, int64_t m0, int64_t m1, int64_t n0, int64_t n1) const {
  for (int64_t m = m0; m < m1; ++m) {
    const double* srow = sums + (m - m0) * ld_s;
    const double xs = x_scale ? x_scale[m] : 1.0;
    for (int64_t n = n0; n < n1; ++n) {
      const double rf = row_factor ? row_factor[n] : row_factor_scalar;
      float t = static_cast<float>(rf * xs * srow[n - n0]);
      if (bias) t += load_as_f32(bias, bias_dtype, n);
      float v = alpha * t;
      if (beta != 0.0f) v = beta * load_as_f32(y, y_dtype, m * ld_y + n) + v;
      store_from_f32(y, y_dtype, m * ld_y + n, v);
    }
  }
}

namespace {

// Item shape: columns per item, and rows per item, so the pool has a few items per thread.
void item_shape(int64_t M, int64_t N, int64_t* mc, int64_t* nc) {
  const int64_t target = static_cast<int64_t>(threads()) * 4;
  if (M <= 8) {
    *mc = M;
    int64_t c = (N + target - 1) / target;
    c = std::max<int64_t>(8, (c + 3) / 4 * 4);
    *nc = c;
    return;
  }
  int64_t c = (N + target - 1) / target;
  c = std::max<int64_t>(16, (c + 15) / 16 * 16);
  const int64_t n_items = (N + c - 1) / c;
  int64_t m_items = 1;
  if (n_items < target) m_items = std::min<int64_t>((M + 63) / 64, (target + n_items - 1) / n_items);
  *mc = (M + m_items - 1) / m_items;
  *nc = c;
}

}  // namespace

int product(const Product& p) {
  if (p.M <= 0 || p.N <= 0) return kOk;
  const Kern& k = kern();
  int64_t mc, nc;
  item_shape(p.M, p.N, &mc, &nc);
  const int64_t m_items = (p.M + mc - 1) / mc, n_items = (p.N + nc - 1) / nc;
  const int64_t items = m_items * n_items;
  const int64_t K = p.K;
  // Weight rows decoded per item: a few hundred KB, so the block stays in the core's L2.
  const int64_t panel_floats = std::min<int64_t>(nc * K, std::max<int64_t>(128 * 1024, 4 * K));
  parallel_for(items, 1, [&](int64_t i0, int64_t i1) {
    float* sums = static_cast<float*>(scratch(sizeof(double) * mc * nc, 2));
    for (int64_t it = i0; it < i1; ++it) {
      const int64_t m0 = (it / n_items) * mc, n0 = (it % n_items) * nc;
      const int64_t m1 = std::min(p.M, m0 + mc), n1 = std::min(p.N, n0 + nc);
      switch (p.path) {
        case kPathF32: {
          DotF32 a{p.xf, p.ld_x, p.w, K, sums, nc, nullptr, 0};
          if (m1 - m0 > 8 && p.w.type != kW_F32) {
            a.panel = static_cast<float*>(scratch(sizeof(float) * panel_floats, 3));
            a.panel_floats = panel_floats;
          }
          if (p.exact) {
            a.out_d = reinterpret_cast<double*>(sums);
            k.dot_f32(a, m0, m1, n0, n1);
            p.ep.apply_d(a.out_d, nc, m0, m1, n0, n1);
            continue;
          }
          k.dot_f32(a, m0, m1, n0, n1);
          break;
        }
        case kPathI8: {
          DotI8 a{p.xi, p.ld_x, static_cast<const int8_t*>(p.w.data), K, sums, nc, nullptr, 0};
          k.dot_i8(a, m0, m1, n0, n1);
          break;
        }
        default: {
          DotBF16 a{p.xb, p.ld_x, p.w, K, sums, nc, nullptr, 0};
          if (p.w.type != kW_BF16) {
            a.panel = scratch(2 * panel_floats, 3);
            a.panel_bytes = 2 * panel_floats;
          }
          k.dot_bf16(a, m0, m1, n0, n1);
          break;
        }
      }
      p.ep.apply(sums, nc, m0, m1, n0, n1);
    }
  });
  return kOk;
}

namespace {

// Activations as f32 rows: the caller's buffer when it is f32 already (no copy), else converted into dst.
const float* acts_f32(const void* x, int dtype, int64_t ld_x, int64_t M, int64_t K, float* dst, int64_t* ld_out) {
  if (dtype == LX_F32) {
    *ld_out = ld_x;
    return static_cast<const float*>(x);
  }
  const uint16_t* xb = static_cast<const uint16_t*>(x);
  for (int64_t m = 0; m < M; ++m)
    for (int64_t kk = 0; kk < K; ++kk) dst[m * K + kk] = bf16_to_f32(xb[m * ld_x + kk]);
  *ld_out = K;
  return dst;
}

bool valid_weight(const lx_weight* w, int64_t K) {
  if (!w || !w->data) return false;
  switch (w->type) {
    case LX_W_BF16: return true;
    case LX_W_INT8:
    case LX_W_FP8: return w->scale != nullptr;
    case LX_W_NVFP4: return w->scale != nullptr && K % 16 == 0;
    default: return false;
  }
}

void set_weight(Product& p, const lx_weight& w) {
  p.w = WeightRef{w.type, w.data, w.scale};
  p.ep.row_factor = (w.type == LX_W_INT8 || w.type == LX_W_FP8) ? static_cast<const float*>(w.scale) : nullptr;
  p.ep.row_factor_scalar = w.type == LX_W_NVFP4 ? w.global : 1.0f;
}

// The decode GEMV of every format: weight-only, f32 math, x read as it is. BF16X weights (lx_gemv_bf16x
// only) run through the same product as BF16, so the two give the same bits.
int gemv_any(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight& w, const void* bias, lx_dtype bias_dtype, void* y,
             lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta) {
  if (M == 0 || N == 0) return kOk;
  if (M < 0 || M > 8 || K <= 0 || ld_x < K || ld_y < N) return fail(kInvalid);
  if (!valid_dtype(x_dtype) || !valid_dtype(y_dtype) || !valid_dtype(bias_dtype)) return fail(kInvalid);
  const bool bf16x = w.type == LX_W_BF16X && w.data && K < bf16x::kMaxK;
  if (!x || !y || !(bf16x || valid_weight(&w, K))) return fail(kInvalid);
  Product p;
  p.path = kPathF32;
  p.M = M;
  p.N = N;
  p.K = K;
  float* tmp = x_dtype == LX_F32 ? nullptr : static_cast<float*>(scratch(sizeof(float) * M * K, 1));
  p.xf = acts_f32(x, x_dtype, ld_x, M, K, tmp, &p.ld_x);
  set_weight(p, w);
  p.ep.bias = bias;
  p.ep.bias_dtype = bias_dtype;
  p.ep.y = y;
  p.ep.y_dtype = y_dtype;
  p.ep.ld_y = ld_y;
  p.ep.alpha = alpha;
  p.ep.beta = beta;
  return product(p);
}

}  // namespace

// The GEMM of any format and math (lx_gemm_w's contract), with the activations prepared in `ws` (at least
// gemm_ws_bytes(M, K, math) bytes) or in internal scratch when ws is null.
int gemm_any(const void* x, int x_dtype, int64_t ld_x, const lx_weight& w, int math, const void* bias, int bias_dtype,
             void* y, int y_dtype, int64_t ld_y, int64_t M, int64_t N, int64_t K, float alpha, float beta, void* ws) {
  if (M == 0 || N == 0) return kOk;
  if (M < 0 || N < 0 || K <= 0 || ld_x < K || ld_y < N) return fail(kInvalid);
  if (!valid_dtype(x_dtype) || !valid_dtype(y_dtype) || (bias && !valid_dtype(bias_dtype))) return fail(kInvalid);
  if (!x || !y || !valid_weight(&w, K)) return fail(kInvalid);
  switch (math) {
    case LX_MATH_BF16: break;
    case LX_MATH_INT8:
      if (w.type != LX_W_INT8) return fail(kNotSupported);
      break;
    case LX_MATH_FP8:
      if (w.type != LX_W_FP8 && w.type != LX_W_NVFP4) return fail(kNotSupported);
      break;
    case LX_MATH_FP4:
      if (w.type != LX_W_NVFP4 || K % 16 != 0) return fail(kNotSupported);
      break;
    default: return fail(kInvalid);
  }
  if (!ws) ws = scratch(static_cast<size_t>(gemm_ws_bytes(M, K, math)), 1);
  const Kern& k = kern();
  Product p;
  p.M = M;
  p.N = N;
  p.K = K;
  set_weight(p, w);
  p.ep.bias = bias;
  p.ep.bias_dtype = bias_dtype;
  p.ep.y = y;
  p.ep.y_dtype = y_dtype;
  p.ep.ld_y = ld_y;
  p.ep.alpha = alpha;
  p.ep.beta = beta;
  char* wsb = static_cast<char*>(ws);
  float* scales = reinterpret_cast<float*>(wsb);  // [M], first, so the rest stays 64-byte aligned
  char* rows = wsb + (M * 4 + 63) / 64 * 64;
  // The FP8 and FP4 maths multiply quantized values, so their dots are summed exactly (f32 rows, double
  // sums); BF16 runs on the bf16 dot instructions where the level has them.
  const bool quantized = math == LX_MATH_FP8 || math == LX_MATH_FP4;
  const bool bf16_dot = k.dot_bf16 != nullptr && M > 8 && !quantized;

  if (math == LX_MATH_INT8) {
    int8_t* q = reinterpret_cast<int8_t*>(rows);
    parallel_for(M, grain_for(M, 4), [&](int64_t m0, int64_t m1) {
      for (int64_t m = m0; m < m1; ++m)
        quantize_act_row(x, x_dtype, m, ld_x, LX_MATH_INT8, static_cast<int>(K), reinterpret_cast<uint8_t*>(q + m * K),
                         scales + m, nullptr);
    });
    p.path = kPathI8;
    p.xi = q;
    p.ld_x = K;
    p.ep.x_scale = scales;
    return product(p);
  }

  // BF16, FP8 and FP4: the activations' exact values as bf16 or f32 rows, the per-token scale apart.
  float* xf = reinterpret_cast<float*>(rows);
  uint16_t* xb = reinterpret_cast<uint16_t*>(rows);
  parallel_for(M, grain_for(M, 4), [&](int64_t m0, int64_t m1) {
    uint8_t* qrow = static_cast<uint8_t*>(scratch(K + K / 16 + 64, 0));
    for (int64_t m = m0; m < m1; ++m) {
      float* fr = xf + m * K;
      uint16_t* br = xb + m * K;
      if (math == LX_MATH_BF16) {
        for (int64_t kk = 0; kk < K; ++kk) {
          const uint16_t b = x_dtype == LX_BF16 ? static_cast<const uint16_t*>(x)[m * ld_x + kk]
                                                : f32_to_bf16(static_cast<const float*>(x)[m * ld_x + kk]);
          if (bf16_dot) br[kk] = b;
          else fr[kk] = bf16_to_f32(b);
        }
        continue;
      }
      float s = 1.0f;
      uint8_t* bs = qrow + K;
      quantize_act_row(x, x_dtype, m, ld_x, math, static_cast<int>(K), qrow, &s, bs);
      scales[m] = s;
      for (int64_t kk = 0; kk < K; ++kk) {
        float v;
        if (math == LX_MATH_FP8) {
          v = e4m3_to_f32(qrow[kk]);
        } else {
          const uint8_t b = qrow[kk >> 1];
          v = e2m1_to_f32((kk & 1) ? (b >> 4) : (b & 15u)) * e4m3_to_f32(bs[kk >> 4]);
        }
        if (bf16_dot) br[kk] = f32_to_bf16(v);
        else fr[kk] = v;
      }
    }
  });
  if (math != LX_MATH_BF16) p.ep.x_scale = scales;
  p.ld_x = K;
  if (bf16_dot) {
    p.path = kPathBF16;
    p.xb = xb;
  } else {
    p.path = kPathF32;
    p.xf = xf;
    p.exact = quantized;
  }
  return product(p);
}

int64_t gemm_ws_bytes(int64_t M, int64_t K, int math) {
  if (M <= 0 || K <= 0) return 0;
  const int64_t elem = math == LX_MATH_INT8 ? 1 : 4;
  const int64_t bytes = (M * 4 + 63) / 64 * 64 + M * K * elem;
  return (bytes + 255) / 256 * 256;
}

}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_gemm_bf16(const void* x, int ld_x, const void* w, const void* bias, void* y, lx_dtype y_dtype, int ld_y, int M,
                 int N, int K, float alpha, float beta, void* stream) {
  LX_PROFILE("gemm_bf16");
  (void)stream;
  if (y_dtype != LX_F32 && y_dtype != LX_BF16) return fail(kInvalid);
  if (M < 0 || N < 0 || K <= 0 || ld_x < K || ld_y < N) return fail(kInvalid);
  if (M == 0 || N == 0) return kOk;
  if (!x || !w || !y) return fail(kInvalid);
  const lx_weight wt{LX_W_BF16, w, nullptr, 1.0f, nullptr};
  return gemm_any(x, LX_BF16, ld_x, wt, LX_MATH_BF16, bias, LX_BF16, y, y_dtype, ld_y, M, N, K, alpha, beta, nullptr);
}

int lx_gemm_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N, int K,
                void* stream) {
  LX_PROFILE("gemm_f32");
  (void)stream;
  if (M < 0 || N < 0 || K <= 0 || ld_x < K || ld_y < N) return fail(kInvalid);
  if (M == 0 || N == 0) return kOk;
  if (!x || !w || !y) return fail(kInvalid);
  Product p;
  p.path = kPathF32;
  p.M = M;
  p.N = N;
  p.K = K;
  p.xf = x;
  p.ld_x = ld_x;
  p.w = WeightRef{kW_F32, w, nullptr};
  p.ep.bias = bias;
  p.ep.bias_dtype = LX_F32;
  p.ep.y = y;
  p.ep.y_dtype = LX_F32;
  p.ep.ld_y = ld_y;
  return product(p);
}

int lx_gemv(const void* x, lx_dtype x_dtype, int ld_x, const void* w, lx_wtype w_type, const float* w_scale, const void* bias,
            lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
            void* stream) {
  LX_PROFILE("gemv");
  (void)stream;
  if (w_type != LX_W_BF16 && w_type != LX_W_INT8) return fail(kInvalid);
  if (w_type == LX_W_INT8 && !w_scale && M > 0 && N > 0) return fail(kInvalid);
  const lx_weight wt{w_type, w, w_scale, 1.0f, nullptr};
  return gemv_any(x, x_dtype, ld_x, wt, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta);
}

int lx_gemv_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias, lx_dtype bias_dtype,
              void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta, void* stream) {
  LX_PROFILE("gemv_w");
  (void)stream;
  if (!w || w->type == LX_W_BF16X) return fail(kInvalid);  // BF16X has its own entry points
  return gemv_any(x, x_dtype, ld_x, *w, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta);
}

// Lossless BF16: the bf16 GEMV's product on the encoded rows (Row<kW_BF16X> in kern/isa.cpp hands the dot
// the same bf16 values in the same lanes), so y is lx_gemv's to the bit while the weights stream at about
// 12.2 bits a value instead of 16.
int lx_gemv_bf16x(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias, lx_dtype bias_dtype,
                  void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta, void* stream) {
  LX_PROFILE("gemv_bf16x");
  (void)stream;
  if (M != 0 && N != 0 && (reinterpret_cast<uintptr_t>(w) & 7) != 0) return fail(kInvalid);
  const lx_weight wt{LX_W_BF16X, w, nullptr, 1.0f, nullptr};
  return gemv_any(x, x_dtype, ld_x, wt, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta);
}

// The GPU's like-for-like baseline for BF16X; on the CPU it is lx_gemv itself.
int lx_gemv_decode_bf16(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias, lx_dtype bias_dtype,
                        void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
                        void* stream) {
  LX_PROFILE("gemv_decode_bf16");
  (void)stream;
  const lx_weight wt{LX_W_BF16, w, nullptr, 1.0f, nullptr};
  return gemv_any(x, x_dtype, ld_x, wt, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta);
}

int lx_gemm_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, lx_math math, const void* bias,
              lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
              void* workspace, void* stream) {
  LX_PROFILE("gemm_w");
  (void)stream;
  if (!w) return fail(kInvalid);
  return gemm_any(x, x_dtype, ld_x, *w, math, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, workspace);
}

int64_t lx_gemm_workspace_bytes(int M, int K, lx_math math) { return gemm_ws_bytes(M, K, math); }

}  // extern "C"
