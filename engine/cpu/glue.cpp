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

// [a | b | c | d] per token, bf16: a, b, c are D wide, d is R wide.
int lx_concat4_bf16(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R, int T,
                    void* stream) {
  LX_PROFILE("concat4_bf16");
  (void)stream;
  if (T <= 0 || D <= 0 || R <= 0) return kOk;
  if (!a || !b || !c || !d || !out) return fail(kInvalid);
  uint16_t* o = static_cast<uint16_t*>(out);
  const int W = 3 * D + R;
  for (int t = 0; t < T; ++t) {
    uint16_t* row = o + static_cast<int64_t>(t) * W;
    const float* segs[4] = {a + static_cast<int64_t>(t) * D, b + static_cast<int64_t>(t) * D,
                            c + static_cast<int64_t>(t) * D, d + static_cast<int64_t>(t) * R};
    for (int j = 0; j < D; ++j) row[j] = f32_to_bf16(segs[0][j]);
    for (int j = 0; j < D; ++j) row[D + j] = f32_to_bf16(segs[1][j]);
    for (int j = 0; j < D; ++j) row[2 * D + j] = f32_to_bf16(segs[2][j]);
    for (int j = 0; j < R; ++j) row[3 * D + j] = f32_to_bf16(segs[3][j]);
  }
  return kOk;
}

// [a | b | c | d] per token, f32: lx_concat4_bf16's layout, copied exactly.
int lx_concat4_f32(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R, int T,
                   void* stream) {
  LX_PROFILE("concat4_f32");
  (void)stream;
  if (T <= 0 || D <= 0 || R <= 0) return kOk;
  if (!a || !b || !c || !d || !out) return fail(kInvalid);
  float* o = static_cast<float*>(out);
  const int W = 3 * D + R;
  for (int t = 0; t < T; ++t) {
    float* row = o + static_cast<int64_t>(t) * W;
    const float* sa = a + static_cast<int64_t>(t) * D;
    const float* sb = b + static_cast<int64_t>(t) * D;
    const float* sc = c + static_cast<int64_t>(t) * D;
    const float* sd = d + static_cast<int64_t>(t) * R;
    for (int j = 0; j < D; ++j) row[j] = sa[j];
    for (int j = 0; j < D; ++j) row[D + j] = sb[j];
    for (int j = 0; j < D; ++j) row[2 * D + j] = sc[j];
    for (int j = 0; j < R; ++j) row[3 * D + j] = sd[j];
  }
  return kOk;
}

// The stream mean (into mean, as lx_stream_mean writes it), then the routing concat over it: the same bits.
int lx_stream_mean(const float* streams, float* mean, int T, int S, int D, void* stream);
int lx_stream_concat4_bf16(const float* streams, float* mean, int S, const float* b, const float* c, const float* d,
                           void* out, int D, int R, int T, void* stream) {
  if (int e = lx_stream_mean(streams, mean, T, S, D, stream)) return e;
  return lx_concat4_bf16(mean, b, c, d, out, D, R, T, stream);
}

// The router's input once the features are ready: the latent's bf16 copy, and the features into the tail.
int lx_pack_router(float* router_in, void* latent, const float* features, int Lat, int R, int T, void* stream) {
  LX_PROFILE("pack_router");
  (void)stream;
  if (T <= 0 || Lat <= 0 || R <= 0) return kOk;
  if (!router_in || !latent || !features) return fail(kInvalid);
  uint16_t* l = static_cast<uint16_t*>(latent);
  const int W = Lat + R;
  for (int t = 0; t < T; ++t) {
    float* row = router_in + static_cast<int64_t>(t) * W;
    for (int j = 0; j < Lat; ++j) l[static_cast<int64_t>(t) * Lat + j] = f32_to_bf16(row[j]);
    memcpy(row + Lat, features + static_cast<int64_t>(t) * R, sizeof(float) * R);
  }
  return kOk;
}

// A memory entry's metadata row: the summary as bf16, then the four scalars (as lx_meta_scalars writes them).
int lx_meta_entry(const float* summary, const float* entropy, const float* confidence, const float* continuation,
                  float mean_k, float inv_log_experts, void* meta, int R, int T, void* stream) {
  LX_PROFILE("meta_entry");
  (void)stream;
  if (T <= 0) return kOk;
  if (!summary || !entropy || !confidence || !continuation || !meta || R <= 0) return fail(kInvalid);
  uint16_t* m = static_cast<uint16_t*>(meta);
  const int ld = R + 4;
  const uint16_t k = f32_to_bf16(mean_k);
  for (int t = 0; t < T; ++t) {
    uint16_t* row = m + static_cast<int64_t>(t) * ld;
    for (int j = 0; j < R; ++j) row[j] = f32_to_bf16(summary[static_cast<int64_t>(t) * R + j]);
    row[R] = k;
    row[R + 1] = f32_to_bf16(entropy[t] * inv_log_experts);
    row[R + 2] = f32_to_bf16(confidence[t]);
    row[R + 3] = f32_to_bf16(continuation[t]);
  }
  return kOk;
}

}  // extern "C"
