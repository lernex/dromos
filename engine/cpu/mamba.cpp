// Mamba-2 (kernels.h, "Mamba-2"): the causal conv carrying its state, the selective scan (per (sequence,
// head), kern/isa.cpp), and the gated RMSNorm. The SiLUs go through the level's vector exp.
//
// The speculative forms (kernels.h, "Speculative decoding": lx_spec_conv1d, lx_spec_ssm_scan) live here
// too, because their contract is bit-identity with these: a verify step's outputs, and the base state
// after a fold, must equal feeding the accepted tokens through the plain kernels. So they run the same
// code. The conv's taps go through one function for both, and the scan replays pending positions through
// the level's own scan kernel, one call per run of positions, which is exactly a plain call's arithmetic.
#include <algorithm>

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int kConvBlock = 256;  // channels per conv work item
constexpr int kMaxTaps = 64;
constexpr int kMaxPending = LX_SPEC_MAX_PENDING;

// One conv output row for channels [c0, c0 + nc): silu(bias + sum over taps kk, in order, of w[c, kk] *
// taps[kk][c]), skipping absent taps (null: positions before a sequence with no state).
void conv_token(const uint16_t* wb, const uint16_t* bb, int c0, int nc, int W, const float* const* taps,
                const Kern& k, float* acc, float* e, float* o) {
  for (int c = 0; c < nc; ++c) acc[c] = bf16_to_f32(bb[c0 + c]);
  for (int kk = 0; kk < W; ++kk) {
    const float* r = taps[kk];
    if (!r) continue;
    for (int c = 0; c < nc; ++c) acc[c] += bf16_to_f32(wb[(c0 + c) * W + kk]) * r[c];
  }
  for (int c = 0; c < nc; ++c) e[c] = -acc[c];
  k.exp_f32(e, e, nc);
  for (int c = 0; c < nc; ++c) o[c] = acc[c] / (1.0f + e[c]);
}

// The scan's pending positions: per slot, kMaxPending positions of hist_row(H, P, N) floats, each holding
// every head's x [H, P], B [H, N] (each head's group's row, so a head reads its own) and dt_raw [H].
int64_t hist_row(int64_t H, int64_t P, int64_t N) { return H * (P + N + 1); }

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_conv1d_varlen(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                     const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* out,
                     int C, int W, void* stream) {
  LX_PROFILE("conv1d_varlen");
  (void)stream;
  if (n_seq <= 0) return kOk;
  if (C <= 0 || W < 1 || ld_x < C || !x || !w || !bias || !conv_state || !slot_ids || !has_state || !cu_seqlens || !out)
    return fail(kInvalid);
  if (W > kMaxTaps) return fail(kInvalid);
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  const uint16_t* bb = static_cast<const uint16_t*>(bias);
  const int blocks = (C + kConvBlock - 1) / kConvBlock;
  const int64_t items = static_cast<int64_t>(n_seq) * blocks;
  const int64_t T = cu_seqlens[n_seq];
  const Kern& k = kern();
  // A work item is one (sequence, channel block): its outputs read the old state, then it writes the
  // block's new state. Sequences own distinct slots, so items never share state. A decode step's
  // thousands of channels are worth spreading too (the taps are strided, so a channel is slow alone).
  const int64_t grain = T * C < 4096 ? items : 1;
  parallel_for(items, grain, [&](int64_t i0, int64_t i1) {
    float* acc = static_cast<float*>(scratch(sizeof(float) * 2 * kConvBlock));
    float* e = acc + kConvBlock;
    const float* taps[kMaxTaps];
    for (int64_t item = i0; item < i1; ++item) {
      const int s = static_cast<int>(item / blocks);
      const int c0 = static_cast<int>(item % blocks) * kConvBlock;
      const int nc = C - c0 < kConvBlock ? C - c0 : kConvBlock;
      const int start = cu_seqlens[s], len = cu_seqlens[s + 1] - start;
      const bool has = has_state[s] != 0;
      float* st = conv_state + static_cast<int64_t>(slot_ids[s]) * (W - 1) * C;
      for (int t = 0; t < len; ++t) {
        for (int kk = 0; kk < W; ++kk) {
          // Position relative to the sequence start; negative positions are the carried state.
          const int p = t - (W - 1) + kk;
          taps[kk] = p >= 0 ? x + static_cast<int64_t>(start + p) * ld_x + c0
                     : has  ? st + static_cast<int64_t>(p + W - 1) * C + c0
                            : nullptr;
        }
        conv_token(wb, bb, c0, nc, W, taps, k, acc, e, out + static_cast<int64_t>(start + t) * C + c0);
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
  LX_PROFILE("ssm_scan_varlen");
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

int64_t lx_spec_ssm_hist_floats(int H, int P, int N) {
  if (H <= 0 || P <= 0 || N <= 0) return 0;
  return kMaxPending * hist_row(H, P, N);
}

// A verify step's conv (kernels.h, lx_spec_conv1d). Sequence s's inputs, oldest first, are: the base
// state's W - 1 rows (absent when it holds nothing), its folded positions fold[s] (hist rows), the pending
// positions it keeps (hist rows after those), then its new tokens (x rows). Each new token's taps point at
// the right rows of that list, so its output is conv_token's on the same rows a plain call over the
// accepted tokens would read. Then the base state takes the fold (the last W - 1 rows up to the folded
// positions, as the plain conv's state update leaves them) and hist becomes [kept rows, new rows].
int lx_spec_conv1d(const float* x, int ld_x, const void* w, const void* bias, float* conv_state, float* conv_hist,
                   const int32_t* slot_ids, const int32_t* has_state, const int32_t* fold, const int32_t* keep,
                   const int32_t* cu_seqlens, int n_seq, float* out, int C, int W, void* stream) {
  LX_PROFILE("spec_conv1d");
  (void)stream;
  if (n_seq <= 0) return kOk;
  if (C <= 0 || W < 1 || W > kMaxTaps || ld_x < C || !x || !w || !bias || !conv_state || !conv_hist || !slot_ids ||
      !has_state || !fold || !keep || !cu_seqlens || !out)
    return fail(kInvalid);
  for (int s = 0; s < n_seq; ++s) {
    const int len = cu_seqlens[s + 1] - cu_seqlens[s];
    if (fold[s] < 0 || keep[s] < 0 || len < 0 || fold[s] + keep[s] > kMaxPending || keep[s] + len > kMaxPending)
      return fail(kInvalid);
  }
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  const uint16_t* bb = static_cast<const uint16_t*>(bias);
  const int blocks = (C + kConvBlock - 1) / kConvBlock;
  const int64_t items = static_cast<int64_t>(n_seq) * blocks;
  const Kern& k = kern();
  parallel_for(items, 1, [&](int64_t i0, int64_t i1) {
    float* acc = static_cast<float*>(scratch(sizeof(float) * 2 * kConvBlock));
    float* e = acc + kConvBlock;
    const float* taps[kMaxTaps];
    for (int64_t item = i0; item < i1; ++item) {
      const int s = static_cast<int>(item / blocks);
      const int c0 = static_cast<int>(item % blocks) * kConvBlock;
      const int nc = C - c0 < kConvBlock ? C - c0 : kConvBlock;
      const int start = cu_seqlens[s], len = cu_seqlens[s + 1] - start;
      const int f = fold[s], kp = keep[s];
      const bool has = has_state[s] != 0;
      float* st = conv_state + static_cast<int64_t>(slot_ids[s]) * (W - 1) * C;
      float* hist = conv_hist + static_cast<int64_t>(slot_ids[s]) * kMaxPending * C;
      // Input i of the list after the base state (i < 0 reaches into it), for channels from c0.
      auto input = [&](int i) -> const float* {
        if (i >= f + kp) return x + static_cast<int64_t>(start + i - f - kp) * ld_x + c0;
        if (i >= 0) return hist + static_cast<int64_t>(i) * C + c0;
        return has ? st + static_cast<int64_t>(i + W - 1) * C + c0 : nullptr;
      };
      for (int t = 0; t < len; ++t) {
        const int q = f + kp + t;  // the token's index in the list
        for (int kk = 0; kk < W; ++kk) taps[kk] = input(q - (W - 1) + kk);
        conv_token(wb, bb, c0, nc, W, taps, k, acc, e, out + static_cast<int64_t>(start + t) * C + c0);
      }
      // The fold: row j of the new base state is input f - (W - 1) + j, read before it can be overwritten
      // (rows move toward the front). Absent rows become zeros, as in the plain conv's state.
      if (f > 0) {
        for (int j = 0; j < W - 1; ++j) {
          const float* src = input(f - (W - 1) + j);
          float* dst = st + static_cast<int64_t>(j) * C + c0;
          if (src) memmove(dst, src, sizeof(float) * nc);
          else memset(dst, 0, sizeof(float) * nc);
        }
      }
      // hist: the kept rows to the front, then the new tokens' inputs.
      for (int i = 0; i < kp; ++i)
        memmove(hist + static_cast<int64_t>(i) * C + c0, hist + static_cast<int64_t>(f + i) * C + c0, sizeof(float) * nc);
      for (int t = 0; t < len; ++t)
        memcpy(hist + static_cast<int64_t>(kp + t) * C + c0, x + static_cast<int64_t>(start + t) * ld_x + c0,
               sizeof(float) * nc);
    }
  });
  return kOk;
}

// A verify step's scan (kernels.h, lx_spec_ssm_scan), per (sequence, head). The fold replays its
// positions from hist into the base state with the level's scan kernel; the kept positions replay the
// same way into a copy; the new tokens run from the copy, writing y; then hist becomes [kept rows, new
// rows]. Each call is the kernel's own token-by-token arithmetic, so the base state after a fold and
// every y equal a plain scan over the accepted tokens bit for bit.
int lx_spec_ssm_scan(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B, const float* C,
                     int ld_bc, const float* A_log, const float* D, const float* dt_bias, float* ssm_state,
                     float* ssm_hist, const int32_t* slot_ids, const int32_t* has_state, const int32_t* fold,
                     const int32_t* keep, const int32_t* cu_seqlens, int n_seq, float* y, int H, int P, int G, int N,
                     void* stream) {
  LX_PROFILE("spec_ssm_scan");
  (void)stream;
  if (n_seq <= 0) return kOk;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || N < 1) return fail(kInvalid);
  if (ld_x < H * P || ld_dt < H || ld_bc < G * N) return fail(kInvalid);
  if (!x || !dt_raw || !B || !C || !A_log || !D || !dt_bias || !ssm_state || !ssm_hist || !slot_ids || !has_state ||
      !fold || !keep || !cu_seqlens || !y)
    return fail(kInvalid);
  for (int s = 0; s < n_seq; ++s) {
    const int len = cu_seqlens[s + 1] - cu_seqlens[s];
    if (fold[s] < 0 || keep[s] < 0 || len < 0 || fold[s] + keep[s] > kMaxPending || keep[s] + len > kMaxPending)
      return fail(kInvalid);
  }
  const Kern& k = kern();
  const int64_t row = hist_row(H, P, N), PN = static_cast<int64_t>(P) * N;
  const int64_t items = static_cast<int64_t>(n_seq) * H;
  parallel_for(items, 1, [&](int64_t i0, int64_t i1) {
    float* tmp = static_cast<float*>(scratch(sizeof(float) * (PN + kMaxPending * P)));
    float* ybuf = tmp + PN;  // the replays' outputs, unused
    for (int64_t item = i0; item < i1; ++item) {
      const int s = static_cast<int>(item / H), h = static_cast<int>(item % H), g = h / (H / G);
      const int start = cu_seqlens[s], len = cu_seqlens[s + 1] - start;
      const int f = fold[s], kp = keep[s];
      float* base = ssm_state + (static_cast<int64_t>(slot_ids[s]) * H + h) * PN;
      float* hist = ssm_hist + static_cast<int64_t>(slot_ids[s]) * kMaxPending * row;
      ScanItem a;
      a.P = P;
      a.N = N;
      a.negA = -std::exp(A_log[h]);
      a.D = D[h];
      a.dt_bias = dt_bias[h];
      // Positions [p0, p0 + n) of hist through the kernel, on `state`.
      auto replay = [&](int p0, int n, float* state, int has) {
        const float* r = hist + static_cast<int64_t>(p0) * row;
        a.x = r + static_cast<int64_t>(h) * P;
        a.ld_x = row;
        a.B = a.C = r + static_cast<int64_t>(H) * P + static_cast<int64_t>(h) * N;
        a.ld_bc = row;
        a.dt = r + static_cast<int64_t>(H) * (P + N) + h;
        a.ld_dt = row;
        a.h = a.g = 0;  // the pointers are the head's already
        a.y = ybuf;
        a.ld_y = P;
        a.state = state;
        a.len = n;
        a.has_state = has;
        k.scan(a);
      };
      if (f > 0) replay(0, f, base, has_state[s]);
      const bool valid = has_state[s] != 0 || f > 0;
      if (valid && kp + len > 0) memcpy(tmp, base, sizeof(float) * PN);
      if (kp > 0) replay(f, kp, tmp, valid);
      if (len > 0) {
        a.x = x + static_cast<int64_t>(start) * ld_x;
        a.ld_x = ld_x;
        a.dt = dt_raw + static_cast<int64_t>(start) * ld_dt;
        a.ld_dt = ld_dt;
        a.B = B + static_cast<int64_t>(start) * ld_bc;
        a.C = C + static_cast<int64_t>(start) * ld_bc;
        a.ld_bc = ld_bc;
        a.h = h;
        a.g = g;
        a.y = y + static_cast<int64_t>(start) * H * P;
        a.ld_y = static_cast<int64_t>(H) * P;
        a.state = tmp;
        a.len = len;
        a.has_state = valid || kp > 0;
        k.scan(a);
      }
      // hist, this head's part of each position: the kept ones to the front, then the new tokens'.
      for (int i = 0; i < kp; ++i) {
        float* d = hist + static_cast<int64_t>(i) * row;
        const float* r = hist + static_cast<int64_t>(f + i) * row;
        memmove(d + static_cast<int64_t>(h) * P, r + static_cast<int64_t>(h) * P, sizeof(float) * P);
        memmove(d + static_cast<int64_t>(H) * P + static_cast<int64_t>(h) * N,
                r + static_cast<int64_t>(H) * P + static_cast<int64_t>(h) * N, sizeof(float) * N);
        d[static_cast<int64_t>(H) * (P + N) + h] = r[static_cast<int64_t>(H) * (P + N) + h];
      }
      for (int t = 0; t < len; ++t) {
        float* d = hist + static_cast<int64_t>(kp + t) * row;
        const int64_t tok = start + t;
        memcpy(d + static_cast<int64_t>(h) * P, x + tok * ld_x + static_cast<int64_t>(h) * P, sizeof(float) * P);
        memcpy(d + static_cast<int64_t>(H) * P + static_cast<int64_t>(h) * N, B + tok * ld_bc + static_cast<int64_t>(g) * N,
               sizeof(float) * N);
        d[static_cast<int64_t>(H) * (P + N) + h] = dt_raw[tok * ld_dt + h];
      }
    }
  });
  return kOk;
}

int lx_gated_rmsnorm(const float* y, const float* z, int ld_z, const void* w, void* out, int T, int I, int groups,
                     float eps, void* stream) {
  LX_PROFILE("gated_rmsnorm");
  (void)stream;
  if (T <= 0) return kOk;
  if (I <= 0 || groups <= 0 || I % groups != 0 || ld_z < I || !y || !z || !w || !out) return fail(kInvalid);
  const int K = I / groups;
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  uint16_t* ob = static_cast<uint16_t*>(out);
  const Kern& k = kern();
  // Groups normalize independently, so a work item is one (token, group): a decode step's one token
  // still spreads over the pool.
  const int64_t items = static_cast<int64_t>(T) * groups;
  const int64_t per = (16 * 1024 + K - 1) / K;
  parallel_for(items, grain_for(items, per < 1 ? 1 : per), [&](int64_t i0, int64_t i1) {
    float* v = static_cast<float*>(scratch(sizeof(float) * K));
    for (int64_t item = i0; item < i1; ++item) {
      const int64_t t = item / groups;
      const int g = static_cast<int>(item % groups);
      const float* yr = y + t * I + g * K;
      const float* zr = z + t * ld_z + g * K;
      for (int i = 0; i < K; ++i) v[i] = -zr[i];
      k.exp_f32(v, v, K);
      for (int i = 0; i < K; ++i) v[i] = yr[i] * (zr[i] / (1.0f + v[i]));
      float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      int i = 0;
      for (; i + 8 <= K; i += 8)
        for (int j = 0; j < 8; ++j) acc[j] += v[i + j] * v[i + j];
      float sum = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
      for (; i < K; ++i) sum += v[i] * v[i];
      const float scale = 1.0f / std::sqrt(sum / K + eps);
      uint16_t* o = ob + t * I + g * K;
      for (i = 0; i < K; ++i) o[i] = f32_to_bf16(v[i] * scale * bf16_to_f32(wb[g * K + i]));
    }
  });
  return kOk;
}

// The decode step (kernels.h, lx_mamba_decode_step) is the three kernels above run in turn, one token per
// sequence, so its bits are theirs by construction. The conv's output and the identity cu_seqlens share one
// scratch block (slot 7, which no kernel above uses), so a warm step allocates nothing.
int lx_mamba_decode_step(const float* zxbcdt, int ld, const void* conv_w, const void* conv_b, float* conv_state,
                         float* ssm_state, const float* A_log, const float* D, const float* dt_bias,
                         const int32_t* slot_ids, const int32_t* has_state, const void* norm_w, float eps, float* y,
                         void* out, int32_t* sync, int n_seq, int H, int P, int G, int N, int W, void* stream) {
  LX_PROFILE("mamba_decode_step");
  (void)stream;
  (void)sync;  // the CUDA kernel's group counters: the three kernels need none
  if (n_seq <= 0) return kOk;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || N < 1 || W < 1 || W > 8) return fail(kInvalid);
  const int I = H * P, GN = G * N, conv_dim = I + 2 * GN;
  if (ld < 2 * I + 2 * GN + H) return fail(kInvalid);
  if (!zxbcdt || !conv_w || !conv_b || !conv_state || !ssm_state || !A_log || !D || !dt_bias || !slot_ids ||
      !has_state || !norm_w || !y || !out)
    return fail(kInvalid);
  const size_t xbc_bytes = sizeof(float) * static_cast<size_t>(n_seq) * conv_dim;
  char* block = static_cast<char*>(scratch(xbc_bytes + 64 + sizeof(int32_t) * (n_seq + 1), 7));
  float* xbc = reinterpret_cast<float*>(block);
  int32_t* cu = reinterpret_cast<int32_t*>(block + ((xbc_bytes + 63) & ~static_cast<size_t>(63)));
  for (int s = 0; s <= n_seq; ++s) cu[s] = s;
  int status = lx_conv1d_varlen(zxbcdt + I, ld, conv_w, conv_b, conv_state, slot_ids, has_state, cu, n_seq, xbc,
                                conv_dim, W, nullptr);
  if (status) return status;
  status = lx_ssm_scan_varlen(xbc, conv_dim, zxbcdt + I + conv_dim, ld, xbc + I, xbc + I + GN, conv_dim, A_log, D,
                              dt_bias, ssm_state, slot_ids, has_state, cu, n_seq, y, H, P, G, N, nullptr);
  if (status) return status;
  return lx_gated_rmsnorm(y, zxbcdt, ld, norm_w, out, n_seq, I, G, eps, nullptr);
}

}  // extern "C"
