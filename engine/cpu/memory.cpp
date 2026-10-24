// Depth memory (kernels.h, "Depth memory"): the per-token attention over its bank, the gated fuse into
// the streams, and the depth head's last step. Tokens are independent, so the work splits by token.
#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int kMaxEntries = 64;

inline float dot(const float* a, const float* b, int n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int i = 0;
  for (; i + 8 <= n; i += 8)
    for (int j = 0; j < 8; ++j) acc[j] += a[i + j] * b[i + j];
  float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
  for (; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline int64_t token_grain(int64_t T, int64_t per_token) {
  const int64_t per = (32 * 1024 + per_token - 1) / (per_token > 0 ? per_token : 1);
  return grain_for(T, per < 1 ? 1 : per);
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_memory_attend(const float* q, const float* bank_k, const float* bank_v, int n_entries, int max_entries, void* out,
                     int T, int S, int M, void* stream) {
  (void)stream;
  if (n_entries < 1 || n_entries > max_entries || n_entries > kMaxEntries) return fail(kInvalid);
  if (T <= 0 || S <= 0 || M <= 0) return kOk;
  if (!q || !bank_k || !bank_v || !out) return fail(kInvalid);
  uint16_t* o = static_cast<uint16_t*>(out);
  const float scale = 1.0f / std::sqrt(static_cast<float>(M));
  parallel_for(T, token_grain(T, static_cast<int64_t>(S) * n_entries * M), [&](int64_t t0, int64_t t1) {
    float* acc = static_cast<float*>(scratch(sizeof(float) * M));
    for (int64_t t = t0; t < t1; ++t) {
      const float* kb = bank_k + t * max_entries * M;
      const float* vb = bank_v + t * max_entries * M;
      for (int s = 0; s < S; ++s) {
        const int64_t row = t * S + s;
        const float* qr = q + row * M;
        // Entries past n_entries are never read, so a bank may hold stale or NaN slots there.
        float score[kMaxEntries];
        float top = -INFINITY;
        for (int e = 0; e < n_entries; ++e) {
          score[e] = dot(qr, kb + e * M, M) * scale;
          top = score[e] > top ? score[e] : top;
        }
        float total = 0.0f;
        for (int e = 0; e < n_entries; ++e) {
          score[e] = std::exp(score[e] - top);
          total += score[e];
        }
        const float norm = 1.0f / total;
        for (int m = 0; m < M; ++m) acc[m] = score[0] * vb[m];
        for (int e = 1; e < n_entries; ++e) {
          const float w = score[e];
          const float* ve = vb + e * M;
          for (int m = 0; m < M; ++m) acc[m] += w * ve[m];
        }
        uint16_t* orow = o + row * M;
        for (int m = 0; m < M; ++m) orow[m] = f32_to_bf16(acc[m] * norm);
      }
    }
  });
  return kOk;
}

int lx_memory_fuse(float* streams, const float* projected, const void* gate, const void* gate_bias, float* summary,
                   int T, int S, int D, void* stream) {
  (void)stream;
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (!streams || !projected || !gate || !gate_bias || !summary) return fail(kInvalid);
  const uint16_t* gw = static_cast<const uint16_t*>(gate);
  const uint16_t* gb = static_cast<const uint16_t*>(gate_bias);
  const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(D));
  parallel_for(T, token_grain(T, static_cast<int64_t>(S) * D * 2), [&](int64_t t0, int64_t t1) {
    float* g32 = static_cast<float*>(scratch(sizeof(float) * D));
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t base = t * S * D;
      for (int s = 0; s < S; ++s) {
        float* row = streams + base + static_cast<int64_t>(s) * D;
        const float* proj = projected + base + static_cast<int64_t>(s) * D;
        for (int d = 0; d < D; ++d) g32[d] = bf16_to_f32(gw[s * D + d]);
        // The gate reads the row before it is updated.
        const float g = sigmoid(dot(row, g32, D) * inv_sqrt_d + bf16_to_f32(gb[s]));
        for (int d = 0; d < D; ++d) row[d] += g * proj[d];
      }
      float* sum = summary + t * D;
      const float* p0 = projected + base;
      for (int d = 0; d < D; ++d) sum[d] = p0[d];
      for (int s = 1; s < S; ++s)
        for (int d = 0; d < D; ++d) sum[d] += p0[static_cast<int64_t>(s) * D + d];
      for (int d = 0; d < D; ++d) sum[d] /= S;
    }
  });
  return kOk;
}

int lx_continuation_finish(const float* logit, const float* state, const float* difference, float limit, float* out,
                           int T, int D, void* stream) {
  (void)stream;
  if (T <= 0 || D <= 0) return kOk;
  if (!logit || !state || !difference || !out) return fail(kInvalid);
  parallel_for(T, token_grain(T, 2 * static_cast<int64_t>(D)), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const float* st = state + t * D;
      const float* df = difference + t * D;
      const float ss = dot(st, st, D), dd = dot(df, df, D);
      // The 1e-6 floor keeps the ratio finite for an all-zero state.
      float scale = std::sqrt(ss / D);
      scale = scale > 1e-6f ? scale : 1e-6f;
      const float innovation = std::sqrt(dd / D);
      float z = logit[t] + std::tanh(innovation / scale) - 0.5f;
      z = z < -limit ? -limit : (z > limit ? limit : z);
      out[t] = sigmoid(z);
    }
  });
  return kOk;
}

}  // extern "C"
