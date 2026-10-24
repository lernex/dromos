// Embedding and the residual streams. Token-major kernels run one threadgroup per token. At decode a
// token's row on one threadgroup runs on one GPU core while the rest idle, so the decode read splits
// each row over `split` threadgroups (columns slices); each recomputes the row's sum of squares, which
// it reads from cache (the row is a few KB), instead of synchronizing with its peers.

constant constexpr int kMaxStreams = 8;

kernel void embed_streams(constant EmbedArgs& a [[buffer(0)]], uint t [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  const device ushort* row = a.table + long(a.ids[t]) * a.D;
  device float* out = a.streams + long(t) * a.S * a.D;
  for (int d = tid; d < a.D; d += threads) {
    const float v = bf(row[d]);
    for (int s = 0; s < a.S; ++s) out[s * a.D + d] = v + bf(a.stream_emb[s * a.D + d]);
  }
}

kernel void stream_mean(constant StreamMeanArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]],
                        uint2 grid [[threads_per_grid]]) {
  const int d = gid.x;
  if (d >= a.D) return;
  for (long t = gid.y; t < a.T; t += grid.y) {
    const device float* in = a.streams + t * a.S * a.D + d;
    float acc = 0.0f;
    for (int s = 0; s < a.S; ++s) acc += in[s * a.D];
    a.mean[t * a.D + d] = acc / a.S;
  }
}

// Hyper-connection read: mixed = matrix @ streams, source = read . streams, h = rmsnorm(source).
//   mhc_read_row    one threadgroup per token: one pass writes mixed and keeps the source row in
//                   threadgroup memory until the sum of squares is known.
//   mhc_read_split  `split` threadgroups per token (decode): part p writes the columns
//                   [p * W, p * W + W); each part takes the sum of squares over the whole row, in the
//                   same order, so every part normalizes alike.
kernel void mhc_read_row(constant MhcReadArgs& a [[buffer(0)]], threadgroup float* source [[threadgroup(0)]],
                         uint t [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int S = a.S, D = a.D;
  const device float* in = a.streams + long(t) * S * D;
  device float* mix = a.mixed + long(t) * S * D;
  float rd[kMaxStreams], mat[kMaxStreams * kMaxStreams];
  for (int i = 0; i < kMaxStreams; ++i) rd[i] = i < S ? a.read[i] : 0.0f;
  for (int i = 0; i < kMaxStreams * kMaxStreams; ++i) mat[i] = i < S * S ? a.matrix[i] : 0.0f;

  float sq = 0.0f;
  for (int d = tid; d < D; d += threads) {
    float x[kMaxStreams];
    for (int i = 0; i < kMaxStreams; ++i) x[i] = i < S ? in[i * D + d] : 0.0f;
    float src = 0.0f;
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += rd[i] * x[i];
    for (int o = 0; o < S; ++o) {
      float acc = 0.0f;
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) acc += mat[o * S + i] * x[i];
      mix[o * D + d] = acc;
    }
    source[d] = src;
    sq += src * src;
  }
  const float scale = rsqrt(group_sum(sq, scratch, tid, threads) / D + a.eps);
  for (int d = tid; d < D; d += threads) a.h[long(t) * D + d] = to_bf(source[d] * scale * bf(a.norm_w[d]));
}

kernel void mhc_read_split(constant MhcReadArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                           uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int S = a.S, D = a.D;
  const int t = g / a.split, part = g % a.split;
  const int W = (D + a.split - 1) / a.split;
  const int c0 = part * W, c1 = min(D, c0 + W);
  const device float* in = a.streams + long(t) * S * D;
  device float* mix = a.mixed + long(t) * S * D;

  float rd[kMaxStreams], mat[kMaxStreams * kMaxStreams];
  for (int i = 0; i < kMaxStreams; ++i) rd[i] = i < S ? a.read[i] : 0.0f;
  for (int i = 0; i < kMaxStreams * kMaxStreams; ++i) mat[i] = i < S * S ? a.matrix[i] : 0.0f;

  float sq = 0.0f;
  for (int d = tid; d < D; d += threads) {
    float src = 0.0f;
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += rd[i] * in[i * D + d];
    sq += src * src;
  }
  const float scale = rsqrt(group_sum(sq, scratch, tid, threads) / D + a.eps);

  for (int d = c0 + tid; d < c1; d += threads) {
    float x[kMaxStreams];
    for (int i = 0; i < kMaxStreams; ++i) x[i] = i < S ? in[i * D + d] : 0.0f;
    float src = 0.0f;
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += rd[i] * x[i];
    for (int o = 0; o < S; ++o) {
      float acc = 0.0f;
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) acc += mat[o * S + i] * x[i];
      mix[o * D + d] = acc;
    }
    a.h[long(t) * D + d] = to_bf(src * scale * bf(a.norm_w[d]));
  }
}

// streams = mixed + write[s] * update; mixed and streams may alias (each element is read and written
// by one thread).
kernel void mhc_write(constant MhcWriteArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]],
                      uint2 grid [[threads_per_grid]]) {
  const int d = gid.x;
  if (d >= a.D) return;
  for (long t = gid.y; t < a.T; t += grid.y) {
    const float u = a.update[t * a.D + d];
    const long base = t * a.S * a.D + d;
    for (int s = 0; s < a.S; ++s) a.streams[base + s * a.D] = a.mixed[base + s * a.D] + a.write[s] * u;
  }
}

// ---- the same on float4 ---------------------------------------------------------------------------
// Rows of a multiple of 4 floats, 16-byte aligned (bf16 rows 8-byte aligned): the kernels above with
// each thread taking four columns per step, the same arithmetic per column. At prefill the rows do not
// fit the cache and these run at the memory's pace; the scalar forms above serve every other shape.
inline float4 bf4(uint2 w) { return float4(bf_lo(w.x), bf_hi(w.x), bf_lo(w.y), bf_hi(w.y)); }
inline uint2 to_bf4(float4 v) {
  return uint2(uint(to_bf(v.x)) | (uint(to_bf(v.y)) << 16), uint(to_bf(v.z)) | (uint(to_bf(v.w)) << 16));
}

constant constexpr int kReadKeep = 4;  // source chunks a thread keeps in registers (one per token form)

// mhc_read on float4. split == 1: one threadgroup per token, the source chunks kept in registers until
// the sum of squares is known (the host sizes the threadgroup so they fit). split > 1: the decode form,
// each part going over the whole row for the sum of squares and then writing its own columns.
kernel void mhc_read_vec(constant MhcReadArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int S = a.S, D = a.D, D4 = D / 4;
  const int t = g / a.split, part = g % a.split;
  const device float4* in = (const device float4*)(a.streams + long(t) * S * D);
  device float4* mix = (device float4*)(a.mixed + long(t) * S * D);
  device uint2* h = (device uint2*)(a.h + long(t) * D);
  const device uint2* nw = (const device uint2*)a.norm_w;
  float rd[kMaxStreams], mat[kMaxStreams * kMaxStreams];
  for (int i = 0; i < kMaxStreams; ++i) rd[i] = i < S ? a.read[i] : 0.0f;
  for (int i = 0; i < kMaxStreams * kMaxStreams; ++i) mat[i] = i < S * S ? a.matrix[i] : 0.0f;

  if (a.split == 1) {
    float4 keep[kReadKeep];
    float sq = 0.0f;
    for (int j = 0; j < kReadKeep; ++j) {
      const int c = int(tid) + j * int(threads);
      keep[j] = float4(0.0f);
      if (c >= D4) continue;
      float4 x[kMaxStreams];
      for (int i = 0; i < kMaxStreams; ++i) x[i] = i < S ? in[i * D4 + c] : float4(0.0f);
      float4 src = float4(0.0f);
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) src += rd[i] * x[i];
      for (int o = 0; o < S; ++o) {
        float4 acc = float4(0.0f);
        for (int i = 0; i < kMaxStreams; ++i)
          if (i < S) acc += mat[o * S + i] * x[i];
        mix[o * D4 + c] = acc;
      }
      keep[j] = src;
      sq += dot(src, src);
    }
    const float scale = rsqrt(group_sum(sq, scratch, tid, threads) / D + a.eps);
    for (int j = 0; j < kReadKeep; ++j) {
      const int c = int(tid) + j * int(threads);
      if (c < D4) h[c] = to_bf4(keep[j] * scale * bf4(nw[c]));
    }
    return;
  }

  const int W = (D4 + a.split - 1) / a.split;
  const int c0 = part * W, c1 = min(D4, c0 + W);
  float sq = 0.0f;
  for (int c = tid; c < D4; c += threads) {
    float4 src = float4(0.0f);
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += rd[i] * in[i * D4 + c];
    sq += dot(src, src);
  }
  const float scale = rsqrt(group_sum(sq, scratch, tid, threads) / D + a.eps);
  for (int c = c0 + tid; c < c1; c += threads) {
    float4 x[kMaxStreams];
    for (int i = 0; i < kMaxStreams; ++i) x[i] = i < S ? in[i * D4 + c] : float4(0.0f);
    float4 src = float4(0.0f);
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += rd[i] * x[i];
    for (int o = 0; o < S; ++o) {
      float4 acc = float4(0.0f);
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) acc += mat[o * S + i] * x[i];
      mix[o * D4 + c] = acc;
    }
    h[c] = to_bf4(src * scale * bf4(nw[c]));
  }
}

// mhc_write on float4: one thread per (token, four columns).
kernel void mhc_write_vec(constant MhcWriteArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const int D4 = a.D / 4;
  if (long(i) >= long(a.T) * D4) return;
  const long t = i / D4, c = i % D4;
  const float4 u = ((const device float4*)a.update)[t * D4 + c];
  const long base = t * a.S * D4 + c;
  for (int s = 0; s < a.S; ++s)
    ((device float4*)a.streams)[base + s * D4] = ((const device float4*)a.mixed)[base + s * D4] + a.write[s] * u;
}

// stream_mean on float4.
kernel void stream_mean_vec(constant StreamMeanArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const int D4 = a.D / 4;
  if (long(i) >= long(a.T) * D4) return;
  const long t = i / D4, c = i % D4;
  const device float4* in = (const device float4*)a.streams + t * a.S * D4 + c;
  float4 acc = float4(0.0f);
  for (int s = 0; s < a.S; ++s) acc += in[s * D4];
  ((device float4*)a.mean)[t * D4 + c] = acc / a.S;
}
