// Depth memory: the per-token attention over its bank, the gated fuse into the streams, and the depth
// head's last step.

constant constexpr int kMaxEntries = 8;

// One simdgroup per (token, stream) row. Lanes split M for the dots and the output; each score is a
// simdgroup sum, so every lane holds the whole softmax. Entries past n_entries are never read.
kernel void memory_attend(constant MemoryAttendArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                          uint sgs [[simdgroups_per_threadgroup]]) {
  const int row = g * sgs + sg;
  if (row >= a.rows) return;
  const int M = a.M;
  const int t = row / a.S;
  const device float* qr = a.q + long(row) * M;
  const device float* kb = a.bank_k + long(t) * a.max_entries * M;
  const device float* vb = a.bank_v + long(t) * a.max_entries * M;
  const float scale = rsqrt(float(M));

  float score[kMaxEntries];
  for (int e = 0; e < kMaxEntries; ++e) {
    score[e] = -INFINITY;
    if (e < a.n_entries) {
      float dot = 0.0f;
      for (int m = lane; m < M; m += 32) dot += qr[m] * kb[e * M + m];
      score[e] = simd_sum(dot) * scale;
    }
  }
  float top = score[0];
  for (int e = 1; e < kMaxEntries; ++e)
    if (e < a.n_entries) top = max(top, score[e]);
  float total = 0.0f;
  for (int e = 0; e < kMaxEntries; ++e) {
    if (e < a.n_entries) {
      score[e] = exp(score[e] - top);
      total += score[e];
    }
  }
  const float norm = 1.0f / total;
  for (int m = lane; m < M; m += 32) {
    float acc = 0.0f;
    for (int e = 0; e < kMaxEntries; ++e)
      if (e < a.n_entries) acc += score[e] * vb[e * M + m];
    a.out[long(row) * M + m] = to_bf(acc * norm);
  }
}

// Gated fuse, threadgroup per (token, stream) when split = S, or per token (split = 1). The gate of a
// stream depends only on its own row, which only that threadgroup writes, after the dot: in place is
// safe. The summary (the mean of projected over streams) is written by the stream-0 threadgroup.
kernel void memory_fuse(constant MemoryFuseArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int S = a.S, D = a.D;
  const int per = a.split == 1 ? S : 1;  // streams per threadgroup
  const int t = a.split == 1 ? int(g) : int(g) / S;
  const int s0 = a.split == 1 ? 0 : int(g) % S;
  const float inv_sqrt_d = rsqrt(float(D));
  const long base = long(t) * S * D;
  for (int s = s0; s < s0 + per; ++s) {
    device float* row = a.streams + base + long(s) * D;
    const device float* proj = a.projected + base + long(s) * D;
    float dot = 0.0f;
    for (int d = tid; d < D; d += threads) dot += row[d] * bf(a.gate[s * D + d]);
    const float gte = sigmoid(group_sum(dot, scratch, tid, threads) * inv_sqrt_d + bf(a.gate_bias[s]));
    for (int d = tid; d < D; d += threads) row[d] += gte * proj[d];
  }
  if (s0 == 0) {
    for (int d = tid; d < D; d += threads) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) acc += a.projected[base + long(s) * D + d];
      a.summary[long(t) * D + d] = acc / S;
    }
  }
}

// memory_fuse on float4 (D a multiple of 4, rows 16-byte aligned, the gate 8-byte aligned).
kernel void memory_fuse_vec(constant MemoryFuseArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                            uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int S = a.S, D = a.D, D4 = D / 4;
  const int per = a.split == 1 ? S : 1;
  const int t = a.split == 1 ? int(g) : int(g) / S;
  const int s0 = a.split == 1 ? 0 : int(g) % S;
  const float inv_sqrt_d = rsqrt(float(D));
  const long base = long(t) * S * D4;
  const device uint2* gate = (const device uint2*)a.gate;
  for (int s = s0; s < s0 + per; ++s) {
    device float4* row = (device float4*)a.streams + base + long(s) * D4;
    const device float4* proj = (const device float4*)a.projected + base + long(s) * D4;
    float dt = 0.0f;
    for (int c = tid; c < D4; c += threads) {
      const uint2 w = gate[s * D4 + c];
      dt += dot(row[c], float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y)));
    }
    const float gte = sigmoid(group_sum(dt, scratch, tid, threads) * inv_sqrt_d + bf(a.gate_bias[s]));
    for (int c = tid; c < D4; c += threads) row[c] += gte * proj[c];
  }
  if (s0 == 0) {
    const device float4* proj = (const device float4*)a.projected + base;
    for (int c = tid; c < D4; c += threads) {
      float4 acc = float4(0.0f);
      for (int s = 0; s < S; ++s) acc += proj[long(s) * D4 + c];
      ((device float4*)a.summary)[long(t) * D4 + c] = acc / S;
    }
  }
}

// out[t] = sigmoid(clip(logit + tanh(innovation / scale) - 0.5, +-limit)), one threadgroup per token.
kernel void continuation_finish(constant ContinuationArgs& a [[buffer(0)]], uint t [[threadgroup_position_in_grid]],
                                uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const device float* st = a.state + long(t) * a.D;
  const device float* df = a.difference + long(t) * a.D;
  float ss = 0.0f, dd = 0.0f;
  for (int d = tid; d < a.D; d += threads) {
    ss += st[d] * st[d];
    dd += df[d] * df[d];
  }
  const float state_ms = group_sum(ss, scratch, tid, threads);
  const float diff_ms = group_sum(dd, scratch, tid, threads);
  const float scale = max(sqrt(state_ms / a.D), 1e-6f);
  const float innovation = sqrt(diff_ms / a.D);
  const float z = clamp(a.logit[t] + precise::tanh(innovation / scale) - 0.5f, -a.limit, a.limit);
  if (tid == 0) a.out[t] = sigmoid(z);
}
