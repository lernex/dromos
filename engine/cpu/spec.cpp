// Speculative decoding's dense parts and the block drafter's (kernels.h, "Speculative decoding" and "Block
// drafter"): the batch-invariant f32 GEMV, the per-head norm and RoPE, and block attention. The Mamba-2
// verify kernels are in mamba.cpp, beside the plain ones they must match; the sampling ones in sampler.cpp.
#include <algorithm>
#include <cmath>

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int kSpecGemvRows = 8;

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

// lx_gemm_f32 limited to decode-sized steps. The CPU's products are in dot form, so a row's sums never
// depend on the other rows: this is lx_gemm_f32 itself.
int lx_spec_gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N,
                     int K, void* stream) {
  if (M == 0 || N == 0) return kOk;
  if (M < 0 || M > kSpecGemvRows || K <= 0 || ld_x < K || ld_y < N || !x || !w || !y) return fail(kInvalid);
  return lx_gemm_f32(x, ld_x, w, bias, y, ld_y, M, N, K, stream);
}

// Per (row, head), in place: the RMS norm (when norm_w is given), then the rotation of the pairs (i,
// i + Dh / 2) by (pos0 + r) * theta^(-2i / Dh). Positions run past 2^17, where a float angle is off by more
// than its own rounding, so the angle, its sine and its cosine are computed in double.
int lx_spec_head_norm_rope(float* x, int ld_x, int rows, int heads, int head_dim, const void* norm_w, float eps,
                           int64_t pos0, double theta, void* stream) {
  LX_PROFILE("spec_head_norm_rope");
  (void)stream;
  if (rows <= 0 || heads <= 0) return kOk;
  if (!x || head_dim < 2 || head_dim % 2 || ld_x < heads * head_dim || pos0 < 0 || !(theta > 1.0))
    return fail(kInvalid);
  const int half = head_dim / 2;
  double* inv = static_cast<double*>(scratch(sizeof(double) * half, 1));
  for (int i = 0; i < half; ++i) inv[i] = std::pow(theta, -2.0 * i / head_dim);
  const uint16_t* wb = static_cast<const uint16_t*>(norm_w);
  const int64_t items = static_cast<int64_t>(rows) * heads;
  parallel_for(items, grain_for(items, std::max<int64_t>(1, 4096 / head_dim)), [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      const int64_t r = item / heads, h = item % heads;
      float* v = x + r * ld_x + h * head_dim;
      float scale = 1.0f;
      if (wb) {
        float ss = 0.0f;
        for (int i = 0; i < head_dim; ++i) ss += v[i] * v[i];
        scale = 1.0f / std::sqrt(ss / static_cast<float>(head_dim) + eps);
      }
      const double pos = static_cast<double>(pos0 + r);
      for (int i = 0; i < half; ++i) {
        float x1 = v[i], x2 = v[i + half];
        if (wb) {
          x1 = x1 * scale * bf16_to_f32(wb[i]);
          x2 = x2 * scale * bf16_to_f32(wb[i + half]);
        }
        const double angle = pos * inv[i];
        const float c = static_cast<float>(std::cos(angle)), s = static_cast<float>(std::sin(angle));
        v[i] = x1 * c - x2 * s;
        v[i + half] = x2 * c + x1 * s;
      }
    }
  });
  return kOk;
}

// Per (query row b, head h): softmax over the n_ctx context keys and the B block keys, all visible to every
// query, of q . key * scale, then the weighted sum of the values. f32 throughout, the exponentials from the
// level's vector exp.
int lx_spec_block_attention(const float* q, int ld_q, const float* k_ctx, const float* v_ctx, int n_ctx,
                            const float* k_blk, const float* v_blk, int ld_blk, float* out, int ld_out, int B, int H,
                            int KVH, int head_dim, float scale, void* stream) {
  LX_PROFILE("spec_block_attention");
  (void)stream;
  if (B <= 0) return kOk;
  if (!q || !k_blk || !v_blk || !out || n_ctx < 0 || (n_ctx > 0 && (!k_ctx || !v_ctx)) || H <= 0 || KVH <= 0 ||
      H % KVH || head_dim <= 0 || ld_q < H * head_dim || ld_out < H * head_dim || ld_blk < KVH * head_dim)
    return fail(kInvalid);
  const int n = n_ctx + B, Dh = head_dim;
  const Kern& k = kern();
  const int64_t items = static_cast<int64_t>(B) * H;
  parallel_for(items, 1, [&](int64_t i0, int64_t i1) {
    float* sc = static_cast<float*>(scratch(sizeof(float) * (n + Dh), 0));
    float* acc = sc + n;
    for (int64_t item = i0; item < i1; ++item) {
      const int b = static_cast<int>(item / H), h = static_cast<int>(item % H), kv = h / (H / KVH);
      const float* qr = q + static_cast<int64_t>(b) * ld_q + static_cast<int64_t>(h) * Dh;
      auto key = [&](int j) {
        return j < n_ctx ? k_ctx + (static_cast<int64_t>(j) * KVH + kv) * Dh
                         : k_blk + static_cast<int64_t>(j - n_ctx) * ld_blk + static_cast<int64_t>(kv) * Dh;
      };
      auto value = [&](int j) {
        return j < n_ctx ? v_ctx + (static_cast<int64_t>(j) * KVH + kv) * Dh
                         : v_blk + static_cast<int64_t>(j - n_ctx) * ld_blk + static_cast<int64_t>(kv) * Dh;
      };
      float m = -INFINITY;
      for (int j = 0; j < n; ++j) {
        const float* kr = key(j);
        float s = 0.0f;
        for (int d = 0; d < Dh; ++d) s += qr[d] * kr[d];
        sc[j] = s * scale;
        m = std::max(m, sc[j]);
      }
      for (int j = 0; j < n; ++j) sc[j] -= m;
      k.exp_f32(sc, sc, n);
      float z = 0.0f;
      for (int j = 0; j < n; ++j) z += sc[j];
      for (int d = 0; d < Dh; ++d) acc[d] = 0.0f;
      for (int j = 0; j < n; ++j) {
        const float* vr = value(j);
        const float p = sc[j];
        for (int d = 0; d < Dh; ++d) acc[d] += p * vr[d];
      }
      float* o = out + static_cast<int64_t>(b) * ld_out + static_cast<int64_t>(h) * Dh;
      for (int d = 0; d < Dh; ++d) o[d] = acc[d] / z;
    }
  });
  return kOk;
}

}  // extern "C"
