// Decode GEMV on every weight format (kernels.h lx_gemv_w), and on f32 weights for lx_gemm_f32 on a few
// rows: memory-bound, one pass over the weight.
//
// A simdgroup owns R consecutive output rows and walks K in 16-byte chunks of each row, lane l taking
// chunks l, l + 32, ... (or, when S simdgroups share the rows, every S * 32-th chunk from its part's
// start). The R chunks of one step are loaded before any is used, so each lane keeps R loads in flight.
// A row's sum runs in an order that S alone sets (R and the number of x rows only change how the loads
// are issued), so with S fixed per shape a row's bits do not depend on the batch.
// x is read from the cache: for each of the M <= 8 rows, the chunk's values once per step, shared by
// the R weight rows. The math is f32; the per-row factor is applied once, after the reduction.
//
// The dot core is shared with the few-token expert kernels (moe.metal), which feed it gathered rows.

typedef const device uchar* dptr;

// x values [k0, k0 + V) of one row, f32 or bf16, with 16-byte loads when the row allows (XV).
template <int V, bool XB, bool XV>
inline void load_x(const device uchar* row, int k0, thread float* v) {
  if (XB) {
    const device ushort* p = (const device ushort*)row + k0;
    if (XV) {
      for (int i = 0; i < V; i += 8) {
        const uint4 w = *(const device uint4*)(p + i);
        v[i] = bf_lo(w.x); v[i + 1] = bf_hi(w.x); v[i + 2] = bf_lo(w.y); v[i + 3] = bf_hi(w.y);
        v[i + 4] = bf_lo(w.z); v[i + 5] = bf_hi(w.z); v[i + 6] = bf_lo(w.w); v[i + 7] = bf_hi(w.w);
      }
    } else {
      for (int i = 0; i < V; ++i) v[i] = bf(p[i]);
    }
  } else {
    const device float* p = (const device float*)row + k0;
    if (XV) {
      for (int i = 0; i < V; i += 4) {
        const float4 f = *(const device float4*)(p + i);
        v[i] = f.x; v[i + 1] = f.y; v[i + 2] = f.z; v[i + 3] = f.w;
      }
    } else {
      for (int i = 0; i < V; ++i) v[i] = p[i];
    }
  }
}

// acc + (one 16-byte chunk of weight codes) . x[0 .. V), without the row factor. The 8-bit and 4-bit
// floats are widened through half, two at a time: an e4m3 byte placed in a half's exponent and mantissa
// is its value / 256, an e2m1 nibble its value / 16384 (both exact, subnormals included), and those
// powers of two are applied once per chunk (FP8) or folded into the block scale (NVFP4, s0 and s1).
template <int F> inline float dot_chunk(uint4 w, float s0, float s1, const thread float* x, float acc);
template <> inline float dot_chunk<W_BF16>(uint4 w, float, float, const thread float* x, float acc) {
  acc = fma(bf_lo(w.x), x[0], acc); acc = fma(bf_hi(w.x), x[1], acc);
  acc = fma(bf_lo(w.y), x[2], acc); acc = fma(bf_hi(w.y), x[3], acc);
  acc = fma(bf_lo(w.z), x[4], acc); acc = fma(bf_hi(w.z), x[5], acc);
  acc = fma(bf_lo(w.w), x[6], acc); acc = fma(bf_hi(w.w), x[7], acc);
  return acc;
}
template <> inline float dot_chunk<W_INT8>(uint4 w, float, float, const thread float* x, float acc) {
  const uint words[4] = {w.x, w.y, w.z, w.w};
  for (int i = 0; i < 4; ++i) {
    const float4 v = float4(as_type<char4>(words[i]));
    acc = fma(v.x, x[4 * i], acc);
    acc = fma(v.y, x[4 * i + 1], acc);
    acc = fma(v.z, x[4 * i + 2], acc);
    acc = fma(v.w, x[4 * i + 3], acc);
  }
  return acc;
}
template <> inline float dot_chunk<W_F32>(uint4 w, float, float, const thread float* x, float acc) {
  acc = fma(as_type<float>(w.x), x[0], acc);
  acc = fma(as_type<float>(w.y), x[1], acc);
  acc = fma(as_type<float>(w.z), x[2], acc);
  acc = fma(as_type<float>(w.w), x[3], acc);
  return acc;
}
inline half2 e4m3_pair(uint t) {  // bytes at bits 0 and 16 of t
  return as_type<half2>(((t & 0x00800080u) << 8) | ((t & 0x007f007fu) << 7));
}
template <> inline float dot_chunk<W_FP8>(uint4 w, float, float, const thread float* x, float acc) {
  const uint words[4] = {w.x, w.y, w.z, w.w};
  float sum = 0.0f;
  for (int i = 0; i < 4; ++i) {
    const float2 a = float2(e4m3_pair(words[i]));       // k = 4i, 4i + 2
    const float2 b = float2(e4m3_pair(words[i] >> 8));  // k = 4i + 1, 4i + 3
    sum = fma(a.x, x[4 * i], sum);
    sum = fma(b.x, x[4 * i + 1], sum);
    sum = fma(a.y, x[4 * i + 2], sum);
    sum = fma(b.y, x[4 * i + 3], sum);
  }
  return fma(sum, 256.0f, acc);
}
inline half2 e2m1_pair(uint t) {  // nibbles at bits 0 and 16 of t
  return as_type<half2>(((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12));
}
template <> inline float dot_chunk<W_NVFP4>(uint4 w, float s0, float s1, const thread float* x, float acc) {
  const uint words[4] = {w.x, w.y, w.z, w.w};
  float sum[2] = {0.0f, 0.0f};
  for (int i = 0; i < 4; ++i) {
    // Word i holds k = 8i .. 8i + 7; byte j has k = 8i + 2j (low nibble) and 8i + 2j + 1.
    const uint v = words[i];
    const float2 a = float2(e2m1_pair(v));        // k + 0, k + 4
    const float2 b = float2(e2m1_pair(v >> 4));   // k + 1, k + 5
    const float2 c = float2(e2m1_pair(v >> 8));   // k + 2, k + 6
    const float2 d = float2(e2m1_pair(v >> 12));  // k + 3, k + 7
    const thread float* xk = x + 8 * i;
    float t = sum[i >> 1];
    t = fma(a.x, xk[0], t);
    t = fma(b.x, xk[1], t);
    t = fma(c.x, xk[2], t);
    t = fma(d.x, xk[3], t);
    t = fma(a.y, xk[4], t);
    t = fma(b.y, xk[5], t);
    t = fma(c.y, xk[6], t);
    t = fma(d.y, xk[7], t);
    sum[i >> 1] = t;
  }
  return fma(sum[1], s1, fma(sum[0], s0, acc));
}

// Lane-partial dot products of NR weight rows with up to MM x rows (M of them live), over the
// chunks c = first, first + stride, ... below K / V. acc[r * MM + m] accumulates row r with x row m.
// Rows hold K values of format F from rows[r]; NVFP4 block scales from scales[r].
template <int F, bool XB, bool XV, int NR, int MM>
inline void dot_core(const thread dptr* rows, const thread dptr* scales,
                     const thread dptr* xrows, int M, int K, int first, int stride,
                     thread float* acc) {
  constexpr int V = chunk_values<F>();
  constexpr int SHIFT = F == W_BF16 ? 1 : F == W_NVFP4 ? -1 : F == W_F32 ? 2 : 0;  // bytes = k << SHIFT
  const int chunks = K / V;
  for (int c = first; c < chunks; c += stride) {
    const int k0 = c * V;
    const long off = SHIFT >= 0 ? (long(k0) << SHIFT) : (long(k0) >> 1);
    uint4 w[NR];
    float s0[NR], s1[NR];
    for (int r = 0; r < NR; ++r) {
      w[r] = *(const device uint4*)(rows[r] + off);
      if (F == W_NVFP4) {
        // The chunk's two block scales are adjacent: one 2-byte load (rows of K % 32 == 0 values from an
        // even scale address, which the host checks for this path).
        const uint sc = *(const device ushort*)(scales[r] + k0 / 16);
        s0[r] = e4m3(sc & 0xffu) * 16384.0f;
        s1[r] = e4m3(sc >> 8) * 16384.0f;
      } else {
        s0[r] = 1.0f;
        s1[r] = 1.0f;
      }
    }
    if (MM == 1) {
      float xv[V];
      load_x<V, XB, XV>(xrows[0], k0, xv);
      for (int r = 0; r < NR; ++r) acc[r] = dot_chunk<F>(w[r], s0[r], s1[r], xv, acc[r]);
      continue;
    }
    if (F == W_FP8 || F == W_NVFP4) {
      // These formats' single-row dot sums a chunk (FP8) or a scale block (NVFP4) on its own before adding
      // it in. Several rows do exactly the same per row, so a row's bits do not depend on M.
      for (int m = 0; m < MM; ++m) {
        if (m >= M) break;
        float xv[V];
        load_x<V, XB, XV>(xrows[m], k0, xv);
        for (int r = 0; r < NR; ++r) acc[r * MM + m] = dot_chunk<F>(w[r], s0[r], s1[r], xv, acc[r * MM + m]);
      }
      continue;
    }
    // Several x rows: decode each weight chunk once, then one dot per row, in the single-row dot's order.
    float wv[NR][V];
    for (int r = 0; r < NR; ++r) decode_chunk<F>(w[r], s0[r] * (F == W_NVFP4 ? 1.0f / 16384.0f : 1.0f),
                                                 s1[r] * (F == W_NVFP4 ? 1.0f / 16384.0f : 1.0f), wv[r]);
    for (int m = 0; m < MM; ++m) {
      if (m >= M) break;
      float xv[V];
      load_x<V, XB, XV>(xrows[m], k0, xv);
      for (int r = 0; r < NR; ++r) {
        float t = acc[r * MM + m];
        for (int i = 0; i < V; ++i) t = fma(wv[r][i], xv[i], t);
        acc[r * MM + m] = t;
      }
    }
  }
}

// The same one element at a time, for rows that are not whole 16-byte chunks or not aligned.
template <int F, bool XB, int NR, int MM>
inline void dot_scalar(const thread dptr* rows, const thread dptr* scales,
                       const thread dptr* xrows, int M, int K, int first, int stride,
                       thread float* acc) {
  for (int k = first; k < K; k += stride) {
    float wv[NR];
    for (int r = 0; r < NR; ++r) wv[r] = element<F>(rows[r], scales[r], 0, k, K);
    for (int m = 0; m < MM; ++m) {
      if (MM > 1 && m >= M) break;
      const float xv = XB ? bf(((const device ushort*)xrows[m])[k]) : ((const device float*)xrows[m])[k];
      for (int r = 0; r < NR; ++r) acc[r * MM + m] = fma(wv[r], xv, acc[r * MM + m]);
    }
  }
}

// Sums the lane partials of a simdgroup, then the S parts of a row group in threadgroup memory in
// part order. Returns true on the threads that hold the totals (every lane of part 0).
template <int N>
inline bool reduce_parts(thread float* acc, threadgroup float* red, int S, int part, int sg, uint lane) {
  for (int i = 0; i < N; ++i) acc[i] = simd_sum(acc[i]);
  if (S == 1) return true;
  if (lane == 0)
    for (int i = 0; i < N; ++i) red[sg * N + i] = acc[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (part != 0) return false;
  for (int i = 0; i < N; ++i) {
    float t = 0.0f;
    for (int p = 0; p < S; ++p) t += red[(sg + p) * N + i];
    acc[i] = t;
  }
  return true;
}

constant constexpr int kGemvGroups = 4;  // simdgroups per threadgroup

// One threadgroup: kGemvGroups simdgroups in row groups of S (a.split); each group owns R rows.
template <int F, bool XB, bool XV, int MM, int R, bool SCALAR>
kernel void gemv(constant GemvArgs& a [[buffer(0)]], threadgroup float* red [[threadgroup(0)]],
                 uint tg [[threadgroup_position_in_grid]], uint sgu [[simdgroup_index_in_threadgroup]],
                 uint lane [[thread_index_in_simdgroup]]) {
  const int S = a.split, sg = int(sgu);
  const int part = sg % S, group = sg / S;
  const int n0 = (int(tg) * (kGemvGroups / S) + group) * R;
  const int K = a.K, N = a.N, M = a.M;
  // A row group past the end still takes part in the reduction's barrier; it only skips the stores.

  dptr rows[R];
  dptr scales[R];
  for (int r = 0; r < R; ++r) {
    const int n = min(n0 + r, N - 1);
    rows[r] = a.w.data + row_offset<F>(n, K);
    scales[r] = F == W_NVFP4 ? a.w.scale + long(n) * (K / 16) : a.w.scale;
  }
  dptr xrows[MM];
  for (int m = 0; m < MM; ++m)
    xrows[m] = (const device uchar*)a.x + long(min(m, M - 1)) * a.ld_x * (XB ? 2 : 4);

  float acc[R * MM];
  for (int i = 0; i < R * MM; ++i) acc[i] = 0.0f;
  if (SCALAR) dot_scalar<F, XB, R, MM>(rows, scales, xrows, M, K, part * 32 + int(lane), S * 32, acc);
  else dot_core<F, XB, XV, R, MM>(rows, scales, xrows, M, K, part * 32 + int(lane), S * 32, acc);

  if (!reduce_parts<R * MM>(acc, red, S, part, sg, lane)) return;

  // y = alpha * (acc * factor + bias) + beta * y, one output per lane.
  for (int i = 0; i < R * MM; ++i) {
    if (int(lane) != (i & 31)) continue;
    const int r = i / MM, m = i % MM;
    const int n = n0 + r;
    if (n >= N || m >= M) continue;
    float v = acc[i] * row_factor<F>(a.w.scale, n, a.w.global);
    if (a.bias) v += load_val(a.bias, n, a.bias_bf16);
    v *= a.alpha;
    const long at = long(m) * a.ld_y + n;
    if (a.beta != 0.0f) v += a.beta * load_val(a.y, at, a.y_bf16);
    store_val(a.y, at, v, a.y_bf16);
  }
}

#define LX_GEMV(F, XB, XV, MM, R, SC)                                                                   \
  template [[host_name("gemv_" #F "_" #XB #XV "_" #MM "_" #R "_" #SC)]] kernel void                    \
  gemv<F, bool(XB), bool(XV), MM, R, bool(SC)>(constant GemvArgs&, threadgroup float*, uint, uint, uint);
#define LX_GEMV_X(F, MM, R)  \
  LX_GEMV(F, 0, 0, MM, R, 0) \
  LX_GEMV(F, 0, 1, MM, R, 0) \
  LX_GEMV(F, 1, 0, MM, R, 0) \
  LX_GEMV(F, 1, 1, MM, R, 0) \
  LX_GEMV(F, 0, 0, MM, R, 1) \
  LX_GEMV(F, 1, 0, MM, R, 1)
#define LX_GEMV_F(F) LX_GEMV_X(F, 1, 1) LX_GEMV_X(F, 4, 2) LX_GEMV_X(F, 8, 2)
LX_GEMV_F(0)
LX_GEMV_F(1)
LX_GEMV_F(2)
LX_GEMV_F(3)
LX_GEMV_F(4)  // f32 weights: lx_gemm_f32 on a few rows
