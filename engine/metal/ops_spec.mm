// Speculative decoding on Metal (kernels/spec.metal): the kernels.h section of that name.
#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kGemvRows = 8;

}  // namespace

extern "C" {

int lx_spec_gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N,
                     int K, void* stream) {
  static Kernel k("spec_gemv_f32");
  if (M == 0 || N == 0) return kOk;
  if (M < 0 || M > kGemvRows || K <= 0 || ld_x < K || ld_y < N || !x || !w || !y) return kInvalidValue;
  SpecGemvArgs a{addr(stream, x, span(M, K, ld_x, 4)), addr(stream, w, size_t(N) * K * 4),
                 addr(stream, bias, size_t(N) * 4), addr_w(stream, y, span(M, N, ld_y, 4)), ld_x, ld_y, M, N, K};
  return k.launch(stream, a, size1(ceil_div(N, 4)), size1(128));
}

}  // extern "C"
