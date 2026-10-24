// Mamba-2 mixer kernels: the causal conv that carries its state between steps, the selective scan with
// f32 state per slot, and the gated RMSNorm after it.

// ---- causal conv ------------------------------------------------------------------------------
// Each output depends on the W inputs ending at its token, never on another output. Threadgroups are
// 32 channels by 8 token lanes; grid (channel blocks, sequences).
kernel void conv1d_out(constant ConvArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                       uint2 tid [[thread_position_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
  const int s = tg.y;
  const int c = tg.x * tpg.x + tid.x;
  if (c >= a.C) return;
  const int W = a.W, C = a.C;
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bool has = a.has_state[s] != 0;
  const device float* st = a.state + long(a.slot_ids[s]) * (W - 1) * C;
  float wk[8];
  for (int k = 0; k < 8; ++k) wk[k] = k < W ? bf(a.w[c * W + k]) : 0.0f;
  const float b = bf(a.bias[c]);
  for (int t = tid.y; t < len; t += tpg.y) {
    float acc = b;
    for (int k = 0; k < 8; ++k) {
      if (k < W) {
        const int p = t - (W - 1) + k;  // position in the sequence; negative is the carried state
        float v = 0.0f;
        if (p >= 0) v = a.x[long(start + p) * a.ld_x + c];
        else if (has) v = st[(p + W - 1) * C + c];
        acc = fma(wk[k], v, acc);
      }
    }
    a.out[long(start + t) * C + c] = silu(acc);
  }
}

// The new state: the last W - 1 inputs of [old state, x]. One thread per (channel, sequence) writes its
// rows in order; an old row is read before it is overwritten.
kernel void conv1d_state(constant ConvArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int s = gid.y;
  const int c = gid.x;
  if (c >= a.C) return;
  const int W = a.W, C = a.C;
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bool has = a.has_state[s] != 0;
  device float* st = a.state + long(a.slot_ids[s]) * (W - 1) * C;
  for (int j = 0; j < W - 1; ++j) {
    const int m = len + j;
    float v = 0.0f;
    if (m >= W - 1) v = a.x[long(start + m - (W - 1)) * a.ld_x + c];
    else if (has) v = st[m * C + c];
    st[j * C + c] = v;
  }
}

// ---- selective scan --------------------------------------------------------------------------
// One threadgroup per (head, sequence). Four lanes own each state row p; lane q holds the columns
// 4 * (q + 4 i) + e (i < NI, e < 4) in registers for the whole sequence. Tokens are staged through
// threadgroup memory in chunks: x[P] | B[N] | C[N] | dt per token, with dt and the decay formed once
// per token. A row's y is summed over its four lanes with two shuffles.
constant constexpr int kScanChunk = 8;

template <int NI>
kernel void ssm_scan(constant ScanArgs& a [[buffer(0)]], threadgroup float* stage [[threadgroup(0)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                     uint2 tpg [[threads_per_threadgroup]]) {
  const uint tid = tid2.x, threads = tpg.x;
  threadgroup float dts[kScanChunk], decays[kScanChunk];
  constexpr int BW = 16 * NI;  // B and C row width, padded with zeros past N
  const int P = a.P, N = a.N;
  const int BO = (P + 3) & ~3, CO = BO + BW, DO = CO + BW;
  const int SW = DO + 4;  // floats per staged token
  const int q = tid & 3, p = tid >> 2;
  const bool valid = p < P;
  const int h = tg.x, s = tg.y;
  const int g = h / (a.H / a.G);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  device float* st = a.state + (long(a.slot_ids[s]) * a.H + h) * P * N;
  const bool has = a.has_state[s] != 0;
  const float negA = -exp(a.A_log[h]);
  const float Dh = a.D[h], dt_bias = a.dt_bias[h];

  // Zero the staging once: the padding columns stay zero, so their state stays zero.
  for (int i = tid; i < kScanChunk * SW; i += threads) stage[i] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float S[NI][4];
  for (int i = 0; i < NI; ++i)
    for (int e = 0; e < 4; ++e) {
      const int n = 4 * (q + 4 * i) + e;
      S[i][e] = (has && valid && n < N) ? st[p * N + n] : 0.0f;
    }

  for (int c0 = 0; c0 < len; c0 += kScanChunk) {
    const int count = min(kScanChunk, len - c0);
    for (int tok = 0; tok < count; ++tok) {
      const long t = start + c0 + tok;
      threadgroup float* d = stage + tok * SW;
      const device float* xr = a.x + t * a.ld_x + h * P;
      const device float* br = a.B + t * a.ld_bc + g * N;
      const device float* cr = a.C + t * a.ld_bc + g * N;
      for (int j = tid; j < P; j += threads) d[j] = xr[j];
      for (int j = tid; j < N; j += threads) {
        d[BO + j] = br[j];
        d[CO + j] = cr[j];
      }
      if (tid == 0) d[DO] = a.dt_raw[t * a.ld_dt + h];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < count) {
      const float dt = softplus(stage[tid * SW + DO] + dt_bias);
      dts[tid] = dt;
      decays[tid] = exp(dt * negA);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int tok = 0; tok < count; ++tok) {
      const threadgroup float* row = stage + tok * SW;
      const float xp = valid ? row[p] : 0.0f;
      const float dtx = dts[tok] * xp;
      const float decay = decays[tok];
      float acc = 0.0f;
      for (int i = 0; i < NI; ++i) {
        const int c = 4 * (q + 4 * i);
        const float4 bb = *(const threadgroup float4*)(row + BO + c);
        const float4 cc = *(const threadgroup float4*)(row + CO + c);
        S[i][0] = fma(decay, S[i][0], dtx * bb.x);
        S[i][1] = fma(decay, S[i][1], dtx * bb.y);
        S[i][2] = fma(decay, S[i][2], dtx * bb.z);
        S[i][3] = fma(decay, S[i][3], dtx * bb.w);
        acc = fma(S[i][0], cc.x, acc);
        acc = fma(S[i][1], cc.y, acc);
        acc = fma(S[i][2], cc.z, acc);
        acc = fma(S[i][3], cc.w, acc);
      }
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      if (valid && q == 0) a.y[(long(start + c0 + tok) * a.H + h) * P + p] = acc + Dh * xp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  if (valid) {
    for (int i = 0; i < NI; ++i)
      for (int e = 0; e < 4; ++e) {
        const int n = 4 * (q + 4 * i) + e;
        if (n < N) st[p * N + n] = S[i][e];
      }
  }
}
template [[host_name("ssm_scan_1")]] kernel void ssm_scan<1>(constant ScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("ssm_scan_2")]] kernel void ssm_scan<2>(constant ScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("ssm_scan_4")]] kernel void ssm_scan<4>(constant ScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("ssm_scan_8")]] kernel void ssm_scan<8>(constant ScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("ssm_scan_16")]] kernel void ssm_scan<16>(constant ScanArgs&, threadgroup float*, uint2, uint2, uint2);

// ---- gated RMSNorm ---------------------------------------------------------------------------
// One threadgroup per (token, group): v = y * silu(z), normalized over the group's K channels.
kernel void gated_rmsnorm(constant GatedNormArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int K = a.I / a.groups;
  const int t = g / a.groups, grp = g % a.groups;
  const int base = grp * K;
  const device float* yr = a.y + long(t) * a.I + base;
  const device float* zr = a.z + long(t) * a.ld_z + base;
  float sum = 0.0f;
  for (int i = tid; i < K; i += threads) {
    const float v = yr[i] * silu(zr[i]);
    sum += v * v;
  }
  const float scale = rsqrt(group_sum(sum, scratch, tid, threads) / K + a.eps);
  for (int i = tid; i < K; i += threads) {
    const float v = yr[i] * silu(zr[i]);
    a.out[long(t) * a.I + base + i] = to_bf(v * scale * bf(a.w[base + i]));
  }
}
