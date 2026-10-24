// Weight and activation quantization (kernels/quant.metal).
#include <algorithm>
#include <cstring>

#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

NSUInteger row_threads(int64_t n) { return NSUInteger(std::min<int64_t>(256, std::max<int64_t>(32, (n + 31) / 32 * 32))); }

// A small zeroed word per stream for the NVFP4 absmax. Device memory, read back after a sync.
uint32_t* amax_word(void* stream) {
  static std::mutex mu;
  static std::unordered_map<Stream*, void*> words;
  std::lock_guard<std::mutex> g(mu);
  void*& p = words[stream_of(stream)];
  if (!p && Device::get().alloc(&p, 256) != kOk) return nullptr;
  return static_cast<uint32_t*>(p);
}

}  // namespace

extern "C" {

int lx_quantize_weight(const void* w, lx_weight* out, int64_t rows, int cols, void* stream) {
  static Kernel int8_k("quantize_rows_int8"), fp8_k("quantize_rows_fp8"), amax_k("nvfp4_amax"),
      nvfp4_k("nvfp4_quant");
  if (!out) return kInvalidValue;
  if (rows <= 0 || cols <= 0) return kOk;
  if (!w || !out->data || (out->type != LX_W_BF16 && !out->scale)) return kInvalidValue;
  QuantArgs a{};
  a.w = addr(stream, w, size_t(rows) * cols * 2);
  a.q = addr_w(stream, out->data, weight_data_bytes(out->type, rows, cols));
  a.scale = addr_w(stream, out->scale, weight_scale_bytes(out->type, rows, cols));
  a.rows = rows;
  a.cols = cols;
  switch (out->type) {
    case LX_W_BF16:
      return lx_rt_copy(const_cast<void*>(out->data), w, rows * cols * 2, LX_COPY_D2D, stream);
    case LX_W_INT8:
      out->global = 1.0f;
      return int8_k.launch(stream, a, size1(NSUInteger(rows)), size1(row_threads(cols)));
    case LX_W_FP8:
      out->global = 1.0f;
      return fp8_k.launch(stream, a, size1(NSUInteger(rows)), size1(row_threads(cols)));
    case LX_W_NVFP4: {
      if (cols % 16 != 0) return kInvalidValue;
      uint32_t* word = amax_word(stream);
      if (!word) return kOutOfMemory;
      if (int e = lx_rt_memset(word, 0, 4, stream)) return e;
      a.amax = addr_w(stream, word, 4);
      const int64_t n = rows * cols;
      const NSUInteger groups = NSUInteger(std::min<int64_t>(1024, std::max<int64_t>(1, n / (256 * 16))));
      if (int e = amax_k.launch(stream, a, size1(groups), size1(256))) return e;
      // The global is the caller's (a host float in the lx_weight), so the absmax comes back first.
      if (int e = lx_rt_stream_sync(stream)) return e;
      float amax = 0.0f;
      std::memcpy(&amax, word, 4);
      a.global = amax / 2688.0f;  // as kernels.h states it, also for a zero matrix (global 0)
      out->global = a.global;
      // The sync retired the transient wraps of host memory, so the addresses are taken again.
      a.w = addr(stream, w, size_t(rows) * cols * 2);
      a.q = addr_w(stream, out->data, weight_data_bytes(out->type, rows, cols));
      a.scale = addr_w(stream, out->scale, weight_scale_bytes(out->type, rows, cols));
      a.amax = 0;
      return nvfp4_k.launch(stream, a, size1(ceil_div(rows * (cols / 16), 256)), size1(256));
    }
  }
  return kInvalidValue;
}

int lx_quantize_w8(const void* w, int8_t* q, float* scale, int64_t rows, int cols, void* stream) {
  lx_weight out{LX_W_INT8, q, scale, 1.0f, nullptr};
  return lx_quantize_weight(w, &out, rows, cols, stream);
}

int lx_dequantize_weight(const lx_weight* w, void* out, int64_t rows, int cols, void* stream) {
  static Kernel k0("dequantize_0"), k1("dequantize_1"), k2("dequantize_2"), k3("dequantize_3");
  if (!w || !out) return kInvalidValue;
  if (rows <= 0 || cols <= 0) return kOk;
  if (w->type < LX_W_BF16 || w->type > LX_W_NVFP4 || !w->data) return kInvalidValue;
  if (w->type != LX_W_BF16 && !w->scale) return kInvalidValue;
  if (w->type == LX_W_NVFP4 && cols % 16 != 0) return kInvalidValue;
  DequantArgs a{};
  a.w = WeightRef{addr(stream, w->data, weight_data_bytes(w->type, rows, cols)),
                  addr(stream, w->scale, weight_scale_bytes(w->type, rows, cols)), 0,
                  w->type == LX_W_NVFP4 ? w->global : 1.0f, w->type};
  a.out = addr_w(stream, out, size_t(rows) * cols * 2);
  a.rows = rows;
  a.cols = cols;
  Kernel* ks[4] = {&k0, &k1, &k2, &k3};
  return ks[w->type]->launch(stream, a, size2(ceil_div(cols, 64), NSUInteger(rows)), size2(64, 1));
}

int lx_quantize_act(const void* x, lx_dtype x_dtype, int ld_x, lx_math math, void* q, int ld_q, float* scale,
                    void* block_scale, int ld_bs, int M, int K, void* stream) {
  static Kernel k1("quantize_act_1"), k2("quantize_act_2"), k3("quantize_act_3");
  if (M <= 0 || K <= 0) return kOk;
  if (!valid_dtype(x_dtype) || ld_x < K || !x || !q || !scale) return kInvalidValue;
  if (math != LX_MATH_INT8 && math != LX_MATH_FP8 && math != LX_MATH_FP4) return kInvalidValue;
  const bool fp4 = math == LX_MATH_FP4;
  if (fp4 && (K % 16 != 0 || !block_scale || ld_bs < K / 16 || ld_q < K / 2)) return kInvalidValue;
  if (!fp4 && ld_q < K) return kInvalidValue;
  QuantActArgs a{};
  a.x = addr(stream, x, span(M, K, ld_x, dtype_size(x_dtype)));
  a.q = addr_w(stream, q, span(M, fp4 ? K / 2 : K, ld_q, 1));
  a.scale = addr_w(stream, scale, size_t(M) * 4);
  a.block_scale = fp4 ? addr_w(stream, block_scale, span(M, K / 16, ld_bs, 1)) : 0;
  a.ld_x = ld_x;
  a.ld_q = ld_q;
  a.ld_bs = ld_bs;
  a.M = M;
  a.K = K;
  a.x_bf16 = x_dtype == LX_BF16;
  Kernel& k = math == LX_MATH_INT8 ? k1 : math == LX_MATH_FP8 ? k2 : k3;
  return k.launch(stream, a, size1(NSUInteger(M)), size1(row_threads(fp4 ? K / 16 : K)));
}

}  // extern "C"
