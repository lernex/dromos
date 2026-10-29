// Norms, elementwise kernels and the model glue (kernels/elementwise.metal).
#include <algorithm>

#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

// Threads for one row of `dim` values: a simdgroup multiple, at most 1024.
NSUInteger row_threads(int dim, int cap = 1024) {
  const int t = std::min(cap, std::max(32, (dim + 31) / 32 * 32));
  return static_cast<NSUInteger>(t);
}

// Grid for a [rows, cols] elementwise launch, 256 threads over the columns; rows go over y (and z
// past 65535) and the kernels stride over any remainder.
constexpr int kThreads = 256;

}  // namespace

extern "C" {

int lx_rmsnorm(const void* x, lx_dtype x_dtype, const void* w, void* y, lx_dtype y_dtype, int rows, int dim, float eps,
               void* stream) {
  static Kernel k00("rmsnorm_00"), k01("rmsnorm_01"), k10("rmsnorm_10"), k11("rmsnorm_11");
  if (rows <= 0) return kOk;
  if (dim <= 0 || !valid_dtype(x_dtype) || !valid_dtype(y_dtype)) return kInvalidValue;
  Kernel& k = x_dtype == LX_F32 ? (y_dtype == LX_F32 ? k00 : k01) : (y_dtype == LX_F32 ? k10 : k11);
  RmsnormArgs a{addr(stream, x, span(rows, dim, dim, dtype_size(x_dtype))), addr(stream, w, size_t(dim) * 2),
                addr_w(stream, y, span(rows, dim, dim, dtype_size(y_dtype))), dim, eps};
  return k.launch(stream, a, size1(rows), size1(row_threads(dim)));
}

int lx_copy2d(const void* src, lx_dtype src_dtype, int ld_src, void* dst, lx_dtype dst_dtype, int ld_dst, int rows,
              int cols, void* stream) {
  static Kernel k00("copy2d_00"), k01("copy2d_01"), k10("copy2d_10"), k11("copy2d_11");
  if (rows <= 0 || cols <= 0) return kOk;
  if (rows > 1 && (ld_src < cols || ld_dst < cols)) return kInvalidValue;
  if (!valid_dtype(src_dtype) || !valid_dtype(dst_dtype)) return kInvalidValue;
  Kernel& k = src_dtype == LX_F32 ? (dst_dtype == LX_F32 ? k00 : k01) : (dst_dtype == LX_F32 ? k10 : k11);
  Copy2dArgs a{addr(stream, src, span(rows, cols, ld_src, dtype_size(src_dtype))),
               addr_w(stream, dst, span(rows, cols, ld_dst, dtype_size(dst_dtype))), ld_src, ld_dst, rows, cols};
  const NSUInteger gy = static_cast<NSUInteger>(std::min(rows, 65535));
  const NSUInteger gz = std::min<NSUInteger>(ceil_div(rows, 65535), 64);
  return k.launch(stream, a, size3(ceil_div(cols, kThreads), gy, gz), size1(kThreads));
}

int lx_activate_f32(float* y, int64_t n, int act, void* stream) {
  static Kernel k0("activate_0"), k1("activate_1"), k2("activate_2");
  if (n <= 0) return kOk;
  if (act < 0 || act > 2) return kInvalidValue;
  Kernel& k = act == 0 ? k0 : act == 1 ? k1 : k2;
  ActivateArgs a{addr_w(stream, y, size_t(n) * 4), n};
  const NSUInteger groups = std::min<NSUInteger>(ceil_div(n, kThreads), 4096);
  return k.launch(stream, a, size1(groups), size1(kThreads));
}

int lx_silu_mul(const float* x, int ld_x, void* out, lx_dtype out_dtype, int rows, int cols, void* stream) {
  static Kernel k0("silu_mul_0"), k1("silu_mul_1");
  if (rows <= 0 || cols <= 0) return kOk;
  if (ld_x < 2 * cols || !valid_dtype(out_dtype)) return kInvalidValue;
  SiluMulArgs a{addr(stream, x, span(rows, 2 * cols, ld_x, 4)),
                addr_w(stream, out, span(rows, cols, cols, dtype_size(out_dtype))), ld_x, rows, cols};
  return (out_dtype == LX_F32 ? k0 : k1)
      .launch(stream, a, size2(ceil_div(cols, kThreads), std::min(rows, 65535)), size1(kThreads));
}

int lx_add_row_bf16(float* y, const void* row, int rows, int cols, void* stream) {
  static Kernel k("add_row");
  if (rows <= 0 || cols <= 0) return kOk;
  AddRowArgs a{addr_w(stream, y, span(rows, cols, cols, 4)), addr(stream, row, size_t(cols) * 2), rows, cols};
  return k.launch(stream, a, size2(ceil_div(cols, kThreads), std::min(rows, 65535)), size1(kThreads));
}

// The two adds as two launches here (the CUDA build fuses them): the same bits.
int lx_add_two_rows_bf16(float* y, const void* row_a, const void* row_b, int rows, int cols, void* stream) {
  if (int e = lx_add_row_bf16(y, row_a, rows, cols, stream)) return e;
  return lx_add_row_bf16(y, row_b, rows, cols, stream);
}

int lx_axpby_f32(float* y, const float* x, float alpha, float beta, int64_t n, void* stream) {
  static Kernel k("axpby");
  if (n <= 0) return kOk;
  AxpbyArgs a{addr_w(stream, y, size_t(n) * 4), addr(stream, x, size_t(n) * 4), n, alpha, beta};
  return k.launch(stream, a, size1(std::min<NSUInteger>(ceil_div(n, kThreads), 4096)), size1(kThreads));
}

int lx_sub_f32(const float* a_, const float* b_, float* out, int64_t n, void* stream) {
  static Kernel k("sub_f32");
  if (n <= 0) return kOk;
  SubArgs a{addr(stream, a_, size_t(n) * 4), addr(stream, b_, size_t(n) * 4), addr_w(stream, out, size_t(n) * 4), n};
  return k.launch(stream, a, size1(std::min<NSUInteger>(ceil_div(n, kThreads), 4096)), size1(kThreads));
}

int lx_meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                    float inv_log_experts, void* meta, int ld, int col, int T, void* stream) {
  static Kernel k("meta_scalars");
  if (T <= 0) return kOk;
  if (ld < col + 4) return kInvalidValue;
  MetaScalarsArgs a{addr(stream, entropy, size_t(T) * 4), addr(stream, confidence, size_t(T) * 4),
                    addr(stream, continuation, size_t(T) * 4), addr_w(stream, meta, span(T, col + 4, ld, 2)),
                    mean_k, inv_log_experts, ld, col, T};
  return k.launch(stream, a, size1(ceil_div(T, kThreads)), size1(kThreads));
}

int lx_gather_last(const float* src, const int32_t* cu, float* out, int n_seq, int D, void* stream) {
  static Kernel k("gather_last");
  if (n_seq <= 0 || D <= 0) return kOk;
  // The rows read are on the device (cu), so the extent of src is unknown here.
  GatherLastArgs a{addr(stream, src, kUnknownExtent), addr(stream, cu, size_t(n_seq + 1) * 4),
                   addr_w(stream, out, span(n_seq, D, D, 4)), D};
  return k.launch(stream, a, size1(n_seq), size1(row_threads(D, 256)));
}

// The four copies (the CUDA build writes the f32 row in one launch): the same bits.
int lx_concat4_f32(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R, int T,
                   void* stream) {
  const int W = 3 * D + R;
  if (T <= 0 || D <= 0 || R <= 0) return kOk;
  char* o = static_cast<char*>(out);
  if (int e = lx_copy2d(a, LX_F32, D, o, LX_F32, W, T, D, stream)) return e;
  if (int e = lx_copy2d(b, LX_F32, D, o + 4 * D, LX_F32, W, T, D, stream)) return e;
  if (int e = lx_copy2d(c, LX_F32, D, o + 4 * 2 * D, LX_F32, W, T, D, stream)) return e;
  return lx_copy2d(d, LX_F32, R, o + 4 * 3 * D, LX_F32, W, T, R, stream);
}

int lx_concat4_bf16(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R, int T,
                    void* stream) {
  static Kernel k("concat4_bf16");
  const int W = 3 * D + R;
  if (T <= 0 || W <= 0) return kOk;
  Concat4Args g{addr(stream, a, span(T, D, D, 4)), addr(stream, b, span(T, D, D, 4)),
                addr(stream, c, span(T, D, D, 4)), addr(stream, d, span(T, R, R, 4)),
                addr_w(stream, out, span(T, W, W, 2)), D, R, T};
  return k.launch(stream, g, size3(ceil_div(W, kThreads), T, 1), size1(kThreads));
}

int lx_pack_router(float* router_in, void* latent, const float* features, int Lat, int R, int T, void* stream) {
  static Kernel k("pack_router");
  const int W = Lat + R;
  if (T <= 0 || W <= 0) return kOk;
  PackRouterArgs g{addr_w(stream, router_in, span(T, W, W, 4)), addr_w(stream, latent, span(T, Lat, Lat, 2)),
                   addr(stream, features, span(T, R, R, 4)), Lat, R, T};
  return k.launch(stream, g, size3(ceil_div(W, kThreads), T, 1), size1(kThreads));
}

int lx_meta_entry(const float* summary, const float* entropy, const float* confidence, const float* continuation,
                  float mean_k, float inv_log_experts, void* meta, int R, int T, void* stream) {
  static Kernel k("meta_entry");
  const int ld = R + 4;
  if (T <= 0 || R <= 0) return kOk;
  MetaEntryArgs g{addr(stream, summary, span(T, R, R, 4)), addr(stream, entropy, size_t(T) * 4),
                  addr(stream, confidence, size_t(T) * 4), addr(stream, continuation, size_t(T) * 4),
                  addr_w(stream, meta, span(T, ld, ld, 2)), mean_k, inv_log_experts, R, T};
  return k.launch(stream, g, size3(ceil_div(ld, kThreads), T, 1), size1(kThreads));
}

}  // extern "C"
