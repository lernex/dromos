// The tile GEMM (kernels/gemm.metal): lx_gemm_w on every weight format and math, lx_gemm_bf16 and
// lx_gemm_f32.
#include <algorithm>
#include <cstdio>

#include "cuda/tunable.h"
#include "metal/backend.h"
#include "metal/gemm.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace lx::metal {

namespace {

std::string gemm_name(int i) {
  char name[32];
  std::snprintf(name, sizeof(name), "gemm_%d%d_%d", i / 15, (i / 3) % 5, i % 3);
  return name;
}
std::string gemm_nax_name(int i) {
  char name[32];
  std::snprintf(name, sizeof(name), "gemm_nax_%d%d_%d", i / 15, (i / 3) % 5, i % 3);
  return name;
}
KernelSet<90> g_gemm(gemm_name);
KernelSet<90> g_gemm_nax(gemm_nax_name, true);
std::string gemm_nax_dev_name(int i) { return "gemm_nax_dev_" + std::to_string(i); }
KernelSet<2> g_gemm_nax_dev(gemm_nax_dev_name, true);  // BF16, INT8 weights from bf16 activations

}  // namespace

int launch_gemm(int af, int wf, int mode, void* stream, const GemmArgs& args, MTLSize groups) {
  const int v = gemm_variant(af, wf, mode);
  if (Device::get().nax()) return g_gemm_nax.launch(v, stream, args, groups, size1(kGemmThreads), kGemmSmemNax);
  return g_gemm.launch(v, stream, args, groups, size1(kGemmThreads), kGemmSmemBf16);
}

int64_t round256(int64_t b) { return (b + 255) / 256 * 256; }

QuantizedRows quantized_rows(int rows, int K, lx_math math) {
  QuantizedRows q{};
  if (math == LX_MATH_BF16 || rows <= 0 || K <= 0) return q;
  const bool fp4 = math == LX_MATH_FP4;
  q.ld = int((fp4 ? K / 2 : K) + 15) / 16 * 16;
  q.ld_blocks = fp4 ? (K / 16 + 15) / 16 * 16 : 0;
  q.scale_at = round256(int64_t(rows) * q.ld);
  q.blocks_at = q.scale_at + round256(int64_t(rows) * 4);
  q.bytes = q.blocks_at + round256(int64_t(rows) * q.ld_blocks);
  return q;
}

int activation_format(lx_math math, lx_dtype x_dtype) {
  switch (math) {
    case LX_MATH_INT8: return kActInt8;
    case LX_MATH_FP8: return kActE4m3;
    case LX_MATH_FP4: return kActFp4;
    default: return x_dtype == LX_BF16 ? kActBf16 : kActF32;
  }
}

}  // namespace lx::metal

extern "C" {

int64_t lx_gemm_workspace_bytes(int M, int K, lx_math math) { return quantized_rows(M, K, math).bytes; }

int lx_gemm_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, lx_math math, const void* bias,
              lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
              void* workspace, void* stream) {
  if (!w || !w->data || w->type < LX_W_BF16 || w->type > LX_W_NVFP4) return kInvalidValue;
  if (M <= 0 || N <= 0 || K <= 0) return M == 0 || N == 0 ? kOk : kInvalidValue;
  if (ld_x < K || ld_y < N || !valid_dtype(x_dtype) || !valid_dtype(y_dtype) || (bias && !valid_dtype(bias_dtype)))
    return kInvalidValue;
  if (w->type != LX_W_BF16 && !w->scale) return kInvalidValue;
  if (math < LX_MATH_BF16 || math > LX_MATH_FP4) return kInvalidValue;
  if (!math_supported(w->type, math)) return kNotSupported;
  if ((w->type == LX_W_NVFP4 || math == LX_MATH_FP4) && K % 16 != 0) return kInvalidValue;

  GemmArgs a{};
  a.w = WeightRef{addr(stream, w->data, weight_data_bytes(w->type, N, K)),
                  addr(stream, w->scale, weight_scale_bytes(w->type, N, K)), 0,
                  w->type == LX_W_NVFP4 ? w->global : 1.0f, w->type};
  const int af = activation_format(math, x_dtype);
  if (math == LX_MATH_BF16) {
    const size_t xs = dtype_size(x_dtype);
    a.x = addr(stream, x, span(M, K, ld_x, xs));
    a.ld_x = ld_x;
    a.x_vec = (reinterpret_cast<uintptr_t>(x) & 15) == 0 && (size_t(ld_x) * xs) % 16 == 0;
  } else {
    // Quantize the activations per token into the workspace first.
    const QuantizedRows q = quantized_rows(M, K, math);
    if (!workspace) return kInvalidValue;
    uint8_t* ws = static_cast<uint8_t*>(workspace);
    if (int e = lx_quantize_act(x, x_dtype, ld_x, math, ws, q.ld, reinterpret_cast<float*>(ws + q.scale_at),
                                ws + q.blocks_at, q.ld_blocks, M, K, stream))
      return e;
    a.x = addr(stream, ws, size_t(q.scale_at));
    a.x_scale = addr(stream, ws + q.scale_at, size_t(M) * 4);
    a.x_block = math == LX_MATH_FP4 ? addr(stream, ws + q.blocks_at, size_t(M) * q.ld_blocks) : 0;
    a.ld_x = q.ld;
    a.ld_xb = q.ld_blocks;
    a.x_vec = (reinterpret_cast<uintptr_t>(ws) & 15) == 0;
  }
  a.bias = addr(stream, bias, bias ? size_t(N) * dtype_size(bias_dtype) : 0);
  a.y = addr_w(stream, y, span(M, N, ld_y, dtype_size(y_dtype)));
  a.ld_y = ld_y;
  a.M = M;
  a.N = N;
  a.K = K;
  a.alpha = alpha;
  a.beta = beta;
  a.bias_bf16 = bias_dtype == LX_BF16;
  a.y_bf16 = y_dtype == LX_BF16;
  const MTLSize groups = size2(ceil_div(N, kTileN), ceil_div(M, kTileM));
  // bf16 activations on BF16 or INT8 weights: the accelerators read both from device memory.
  if (af == kActBf16 && (w->type == LX_W_BF16 || w->type == LX_W_INT8) && a.x_vec && K % 16 == 0 &&
      (reinterpret_cast<uintptr_t>(w->data) & 15) == 0 && Device::get().nax())
    return g_gemm_nax_dev.launch(w->type, stream, a, groups, size1(kGemmThreads), kGemmSmemNax);
  return launch_gemm(af, w->type, 0, stream, a, groups);
}

int lx_gemm_bf16(const void* x, int ld_x, const void* w, const void* bias, void* y, lx_dtype y_dtype, int ld_y, int M,
                 int N, int K, float alpha, float beta, void* stream) {
  const lx_weight lw{LX_W_BF16, w, nullptr, 1.0f, nullptr};
  return lx_gemm_w(x, LX_BF16, ld_x, &lw, LX_MATH_BF16, bias, LX_BF16, y, y_dtype, ld_y, M, N, K, alpha, beta, nullptr,
                   stream);
}

int lx_gemm_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N, int K,
                void* stream) {
  if (M <= 0 || N <= 0) return kOk;
  if (K <= 0 || ld_x < K || ld_y < N || !x || !w || !y) return kInvalidValue;
  if (M <= 8) return gemv_f32(x, ld_x, w, bias, y, ld_y, M, N, K, stream);  // a few rows: one pass over w
  GemmArgs a{};
  a.w = WeightRef{addr(stream, w, size_t(N) * K * 4), 0, 0, 1.0f, kWeightF32};
  a.x = addr(stream, x, span(M, K, ld_x, 4));
  a.ld_x = ld_x;
  a.x_vec = (reinterpret_cast<uintptr_t>(x) & 15) == 0 && ld_x % 4 == 0;
  a.bias = addr(stream, bias, bias ? size_t(N) * 4 : 0);
  a.y = addr_w(stream, y, span(M, N, ld_y, 4));
  a.ld_y = ld_y;
  a.M = M;
  a.N = N;
  a.K = K;
  a.alpha = 1.0f;
  a.beta = 0.0f;
  // kernels/gemm.metal gemm_f32x: 32 x 32 tiles, two staged 32 x 36 f32 blocks.
  static Kernel f32x("gemm_f32x");
  return f32x.launch(stream, a, size2(ceil_div(N, 32), ceil_div(M, 32)), size1(kGemmThreads), 2 * 32 * 36 * 4);
}

}  // extern "C"
