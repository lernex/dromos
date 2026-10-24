// Decode GEMV on every weight format (kernels/gemv.metal), and lx_gemm_f32 on up to 8 rows.
#include <algorithm>
#include <cstdio>

#include "cuda/tunable.h"
#include "metal/backend.h"
#include "metal/gemm.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kRows = 4;    // rows per simdgroup (R in the kernel)
constexpr int kGroups = 4;  // simdgroups per threadgroup
constexpr int kMaxM = 8;

// Variant index: format, x bf16, x vector loads, M bucket (1, 4, 8), scalar.
int variant(int f, bool xb, bool xv, int mm, bool scalar) {
  const int mb = mm == 1 ? 0 : mm == 4 ? 1 : 2;
  return ((((f * 2 + xb) * 2 + (scalar ? 0 : xv)) * 3 + mb) * 2) + scalar;
}
std::string gemv_name(int i) {
  const int scalar = i % 2, mb = (i / 2) % 3, xv = (i / 6) % 2, xb = (i / 12) % 2, f = i / 24;
  const int mm = mb == 0 ? 1 : mb == 1 ? 4 : 8;
  char name[64];
  std::snprintf(name, sizeof(name), "gemv_%d_%d%d_%d_%d_%d", f, xb, xv, mm, kRows, scalar);
  return name;
}
KernelSet<120> g_gemv(gemv_name);  // the four formats, then f32 weights (format 4)

const lx::TunableRegistration reg_gemv("gemv", {"s1", "s2", "s4"});

// One launch over format f (0 .. 3 the lx_wtype, 4 f32 weights) once the arguments are checked.
int launch(int f, const void* data, const void* scale, float global, const void* x, lx_dtype x_dtype, int ld_x,
           const void* bias, lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
           float beta, void* stream) {
  const bool xb = x_dtype == LX_BF16;
  const size_t xs = dtype_size(x_dtype);
  const int values = f == LX_W_BF16 ? 8 : f == LX_W_NVFP4 ? 32 : f == kWeightF32 ? 4 : 16;
  const bool scalar = K % values != 0 || (reinterpret_cast<uintptr_t>(data) & 15) != 0;
  const bool xv = (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (size_t(ld_x) * xs) % 16 == 0;
  const int mm = M == 1 ? 1 : M <= 4 ? 4 : 8;

  // Simdgroups per row group: one while the row groups alone fill the GPU, more for short outputs.
  int s = lx_tune_get("gemv", M, N, K);
  if (s < 0) s = N >= 2048 ? 0 : N >= 768 ? 1 : 2;
  const int S = 1 << s;

  const size_t data_bytes = f == kWeightF32 ? size_t(N) * K * 4 : weight_data_bytes(lx_wtype(f), N, K);
  const size_t scale_bytes = f == kWeightF32 ? 0 : weight_scale_bytes(lx_wtype(f), N, K);
  GemvArgs a{};
  a.w = WeightRef{addr(stream, data, data_bytes), addr(stream, scale, scale_bytes), 0, global, f};
  a.x = addr(stream, x, span(M, K, ld_x, xs));
  a.bias = addr(stream, bias, bias ? size_t(N) * dtype_size(bias_dtype) : 0);
  a.y = addr_w(stream, y, span(M, N, ld_y, dtype_size(y_dtype)));
  a.ld_x = ld_x;
  a.ld_y = ld_y;
  a.M = M;
  a.N = N;
  a.K = K;
  a.alpha = alpha;
  a.beta = beta;
  a.x_bf16 = xb;
  a.bias_bf16 = bias_dtype == LX_BF16;
  a.y_bf16 = y_dtype == LX_BF16;
  a.split = S;
  const int rows_per_group = kRows * kGroups / S;
  const NSUInteger red = S > 1 ? NSUInteger(kGroups * kRows * mm * 4) : 0;
  return g_gemv.launch(variant(f, xb, xv, mm, scalar), stream, a, size1(ceil_div(N, rows_per_group)),
                       size1(32 * kGroups), red);
}

}  // namespace

namespace lx::metal {

int gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N, int K,
             void* stream) {
  return launch(kWeightF32, w, nullptr, 1.0f, x, LX_F32, ld_x, bias, LX_F32, y, LX_F32, ld_y, M, N, K, 1.0f, 0.0f,
                stream);
}

}  // namespace lx::metal

extern "C" {

int lx_gemv_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias, lx_dtype bias_dtype,
              void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta, void* stream) {
  if (!w || M < 1 || M > kMaxM || N <= 0 || K <= 0 || ld_x < K || ld_y < N) return kInvalidValue;
  if (!valid_dtype(x_dtype) || !valid_dtype(y_dtype) || (bias && !valid_dtype(bias_dtype))) return kInvalidValue;
  if (w->type < LX_W_BF16 || w->type > LX_W_NVFP4 || !w->data) return kInvalidValue;
  if (w->type != LX_W_BF16 && !w->scale) return kInvalidValue;
  if (w->type == LX_W_NVFP4 && K % 16 != 0) return kInvalidValue;
  return launch(w->type, w->data, w->scale, w->type == LX_W_NVFP4 ? w->global : 1.0f, x, x_dtype, ld_x, bias,
                bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
}

int lx_gemv(const void* x, lx_dtype x_dtype, int ld_x, const void* w, lx_wtype w_type, const float* w_scale,
            const void* bias, lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K,
            float alpha, float beta, void* stream) {
  if (w_type != LX_W_BF16 && w_type != LX_W_INT8) return kInvalidValue;
  const lx_weight lw{w_type, w, w_scale, 1.0f, nullptr};
  return lx_gemv_w(x, x_dtype, ld_x, &lw, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
}

}  // extern "C"
