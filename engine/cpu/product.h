// The dense product driver shared by the GEMV, GEMM and expert ops (gemm.cpp).
#pragma once

#include "cpu/cpu.h"

namespace lx::cpu {

enum : int { kPathF32 = 0, kPathI8 = 1, kPathBF16 = 2 };

// y[m, n] = alpha * (row_factor[n] * x_scale[m] * sums[m, n] + bias[n]) + beta * y[m, n]; y is read
// only when beta != 0. row_factor null means row_factor_scalar for every row.
struct Epilogue {
  const float* row_factor = nullptr;
  float row_factor_scalar = 1.0f;
  const float* x_scale = nullptr;
  const void* bias = nullptr;
  int bias_dtype = LX_F32;
  void* y = nullptr;
  int y_dtype = LX_F32;
  int64_t ld_y = 0;
  float alpha = 1.0f, beta = 0.0f;
  void apply(const float* sums, int64_t ld_s, int64_t m0, int64_t m1, int64_t n0, int64_t n1) const;
};

struct Product {
  int path = kPathF32;
  const float* xf = nullptr;  // activations for kPathF32
  const int8_t* xi = nullptr;  // for kPathI8 (int8 weights only)
  const uint16_t* xb = nullptr;  // for kPathBF16 (levels with bf16 dot products)
  int64_t ld_x = 0;
  int64_t M = 0, N = 0, K = 0;
  WeightRef w{};
  Epilogue ep;
};

int product(const Product& p);

// lx_gemm_w's contract for any format and math; ws holds the prepared activations (gemm_ws_bytes), or
// is null for internal scratch.
int gemm_any(const void* x, int x_dtype, int64_t ld_x, const lx_weight& w, int math, const void* bias, int bias_dtype,
             void* y, int y_dtype, int64_t ld_y, int64_t M, int64_t N, int64_t K, float alpha, float beta, void* ws);
int64_t gemm_ws_bytes(int64_t M, int64_t K, int math);

}  // namespace lx::cpu
