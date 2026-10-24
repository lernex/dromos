// Mamba-2 (kernels.h, "Mamba-2"): the causal conv carrying its state, the selective scan (per (sequence,
// head), kern/isa.cpp), and the gated RMSNorm. The SiLUs go through the level's vector exp.
#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int kConvBlock = 256;  // channels per conv work item

struct SeqInfo {
  int start, len, slot, has;
};

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_conv1d_varlen(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                     const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* out,
                     int C, int W, void* stream) {
  (void)stream;
  if (n_seq <= 0) return kOk;
  if (C <= 0 || W < 1 || ld_x < C || !x || !w || !bias || !conv_state || !slot_ids || !has_state || !cu_seqlens || !out)
    return fail(kInvalid);
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  const uint16_t* bb = static_cast<const uint16_t*>(bias);
  const int blocks = (C + kConvBlock - 1) / kConvBlock;
  const int64_t items = static_cast<int64_t>(n_seq) * blocks;
  const int64_t T = cu_seqlens[n_seq];
  const Kern& k = kern();
  // A work item is one (sequence, channel block): its outputs read the old state, then it writes the
  // block's new state. Sequences own distinct slots, so items never share state.
  const int64_t grain = T * C < 64 * 1024 ? items : 1;
  parallel_for(items, grain, [&](int64_t i0, int64_t i1) {
    float* acc = static_cast<float*>(scratch(sizeof(float) * 2 * kConvBlock));
    float* e = acc + kConvBlock;
    for (int64_t item = i0; item < i1; ++item) {
      const int s = static_cast<int>(item / blocks);
      const int c0 = static_cast<int>(item % blocks) * kConvBlock;
      const int nc = C - c0 < kConvBlock ? C - c0 : kConvBlock;
      const int start = cu_seqlens[s], len = cu_seqlens[s + 1] - start;
      const bool has = has_state[s] != 0;
      float* st = conv_state + static_cast<int64_t>(slot_ids[s]) * (W - 1) * C;
      for (int t = 0; t < len; ++t) {
        for (int c = 0; c < nc; ++c) acc[c] = bf16_to_f32(bb[c0 + c]);
        for (int kk = 0; kk < W; ++kk) {
          // Position relative to the sequence start; negative positions are the carried state.
          const int p = t - (W - 1) + kk;
          if (p >= 0) {
            const float* xr = x + static_cast<int64_t>(start + p) * ld_x + c0;
            for (int c = 0; c < nc; ++c) acc[c] += bf16_to_f32(wb[(c0 + c) * W + kk]) * xr[c];
          } else if (has) {
            const float* sr = st + static_cast<int64_t>(p + W - 1) * C + c0;
            for (int c = 0; c < nc; ++c) acc[c] += bf16_to_f32(wb[(c0 + c) * W + kk]) * sr[c];
          }
        }
        for (int c = 0; c < nc; ++c) e[c] = -acc[c];
        k.exp_f32(e, e, nc);
        float* o = out + static_cast<int64_t>(start + t) * C + c0;
        for (int c = 0; c < nc; ++c) o[c] = acc[c] / (1.0f + e[c]);
      }
      // The new state is the last W - 1 inputs of [old state rows, x rows]. Row j takes combined input
      // len + j; an old row m is read before it is written, since m > j whenever len > 0.
      for (int j = 0; j < W - 1; ++j) {
        const int m = len + j;
        float* dst = st + static_cast<int64_t>(j) * C + c0;
        if (m >= W - 1) {
          const float* xr = x + static_cast<int64_t>(start + m - (W - 1)) * ld_x + c0;
          for (int c = 0; c < nc; ++c) dst[c] = xr[c];
        } else if (has) {
          const float* sr = st + static_cast<int64_t>(m) * C + c0;
          for (int c = 0; c < nc; ++c) dst[c] = sr[c];
        } else {
          for (int c = 0; c < nc; ++c) dst[c] = 0.0f;
        }
      }
    }
  });
  return kOk;
}

int lx_ssm_scan_varlen(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B, const float* C,
                       int ld_bc, const float* A_log, const float* D, const float* dt_bias, float* ssm_state,
                       const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* y,
                       int H, int P, int G, int N, void* stream) {
  (void)stream;
  if (n_seq <= 0) return kOk;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || N < 1) return fail(kInvalid);
  if (ld_x < H * P || ld_dt < H || ld_bc < G * N) return fail(kInvalid);
  if (!x || !dt_raw || !B || !C || !A_log || !D || !dt_bias || !ssm_state || !slot_ids || !has_state || !cu_seqlens || !y)
    return fail(kInvalid);
  const Kern& k = kern();
  const int64_t items = static_cast<int64_t>(n_seq) * H;
  parallel_for(items, 1, [&](int64_t i0, int64_t i1) {
    for (int64_t item = i0; item < i1; ++item) {
      const int s = static_cast<int>(item / H), h = static_cast<int>(item % H);
      const int start = cu_seqlens[s];
      ScanItem a;
      a.x = x + static_cast<int64_t>(start) * ld_x;
      a.ld_x = ld_x;
      a.dt = dt_raw + static_cast<int64_t>(start) * ld_dt;
      a.ld_dt = ld_dt;
      a.B = B + static_cast<int64_t>(start) * ld_bc;
      a.C = C + static_cast<int64_t>(start) * ld_bc;
      a.ld_bc = ld_bc;
      a.state = ssm_state + (static_cast<int64_t>(slot_ids[s]) * H + h) * P * N;
      a.y = y + static_cast<int64_t>(start) * H * P;
      a.ld_y = static_cast<int64_t>(H) * P;
      a.len = cu_seqlens[s + 1] - start;
      a.h = h;
      a.g = h / (H / G);
      a.P = P;
      a.N = N;
      a.negA = -std::exp(A_log[h]);
      a.D = D[h];
      a.dt_bias = dt_bias[h];
      a.has_state = has_state[s];
      k.scan(a);
    }
  });
  return kOk;
}

int lx_gated_rmsnorm(const float* y, const float* z, int ld_z, const void* w, void* out, int T, int I, int groups,
                     float eps, void* stream) {
  (void)stream;
  if (T <= 0) return kOk;
  if (I <= 0 || groups <= 0 || I % groups != 0 || ld_z < I || !y || !z || !w || !out) return fail(kInvalid);
  const int K = I / groups;
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  uint16_t* ob = static_cast<uint16_t*>(out);
  const Kern& k = kern();
  const int64_t per = (32 * 1024 + I - 1) / I;
  parallel_for(T, grain_for(T, per < 1 ? 1 : per), [&](int64_t t0, int64_t t1) {
    float* v = static_cast<float*>(scratch(sizeof(float) * I));
    for (int64_t t = t0; t < t1; ++t) {
      const float* yr = y + t * I;
      const float* zr = z + t * ld_z;
      for (int i = 0; i < I; ++i) v[i] = -zr[i];
      k.exp_f32(v, v, I);
      for (int i = 0; i < I; ++i) v[i] = yr[i] * (zr[i] / (1.0f + v[i]));
      for (int g = 0; g < groups; ++g) {
        const float* vg = v + g * K;
        float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        int i = 0;
        for (; i + 8 <= K; i += 8)
          for (int j = 0; j < 8; ++j) acc[j] += vg[i + j] * vg[i + j];
        float sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
        for (; i < K; ++i) sum += vg[i] * vg[i];
        const float scale = 1.0f / std::sqrt(sum / K + eps);
        uint16_t* o = ob + t * I + g * K;
        for (i = 0; i < K; ++i) o[i] = f32_to_bf16(vg[i] * scale * bf16_to_f32(wb[g * K + i]));
      }
    }
  });
  return kOk;
}

}  // extern "C"
