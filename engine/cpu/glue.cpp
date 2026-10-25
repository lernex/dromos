// The model glue (kernels.h, "Model glue"): the small ops the Metis forward pass needs between the library
// kernels. A decode step's sizes run on the calling thread; a prefill's split over the pool.
#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int64_t kGrainElems = 64 * 1024;

// fn(i0, i1) over [0, n), on one thread when n is small.
template <class Fn>
void over(int64_t n, Fn&& fn) {
  if (n <= kGrainElems) {
    fn(int64_t{0}, n);
    return;
  }
  parallel_for(n, grain_for(n, kGrainElems), fn);
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_axpby_f32(float* y, const float* x, float a, float b, int64_t n, void* stream) {
  LX_PROFILE("axpby_f32");
  (void)stream;
  if (n <= 0) return kOk;
  if (!y || !x) return fail(kInvalid);
  over(n, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) y[i] = a * y[i] + b * x[i];
  });
  return kOk;
}

int lx_sub_f32(const float* a, const float* b, float* out, int64_t n, void* stream) {
  LX_PROFILE("sub_f32");
  (void)stream;
  if (n <= 0) return kOk;
  if (!a || !b || !out) return fail(kInvalid);
  over(n, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) out[i] = a[i] - b[i];
  });
  return kOk;
}

// The four scalar columns of a memory entry's metadata, after the expert summary.
int lx_meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                    float inv_log_experts, void* meta, int ld, int col, int T, void* stream) {
  LX_PROFILE("meta_scalars");
  (void)stream;
  if (T <= 0) return kOk;
  if (!entropy || !confidence || !continuation || !meta || col < 0 || (T > 1 && ld < col + 4)) return fail(kInvalid);
  uint16_t* m = static_cast<uint16_t*>(meta);
  const uint16_t k = f32_to_bf16(mean_k);
  for (int t = 0; t < T; ++t) {
    uint16_t* row = m + static_cast<int64_t>(t) * ld + col;
    row[0] = k;
    row[1] = f32_to_bf16(entropy[t] * inv_log_experts);
    row[2] = f32_to_bf16(confidence[t]);
    row[3] = f32_to_bf16(continuation[t]);
  }
  return kOk;
}

// Each sequence's last token: out[i, :] = src[cu[i + 1] - 1, :].
int lx_gather_last(const float* src, const int32_t* cu, float* out, int n_seq, int D, void* stream) {
  LX_PROFILE("gather_last");
  (void)stream;
  if (n_seq <= 0 || D <= 0) return kOk;
  if (!src || !cu || !out) return fail(kInvalid);
  for (int i = 0; i < n_seq; ++i) {
    if (cu[i + 1] < 1) return fail(kInvalid);
    memcpy(out + static_cast<int64_t>(i) * D, src + (static_cast<int64_t>(cu[i + 1]) - 1) * D, sizeof(float) * D);
  }
  return kOk;
}

}  // extern "C"
