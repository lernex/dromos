// The latent mixture of experts: routing, the expert summary, and the routed experts in two forms.
//
// Few tokens (lx_moe_experts_decode, and the bf16 / int8 entry points at every T): two GEMV-shaped
// stages on the dot core of gemv.metal. Stage 1 gives each pair its bf16 SwiGLU intermediate; pairs on
// the same expert are batched (up to 8 per batch) so a weight chunk is read once for all of them.
// Stage 2 gives each token's output row by row, summing its top_k experts in order. Every pair's
// arithmetic is the same whichever batch it lands in, so a token's output does not depend on the other
// tokens of the call. The grouped form for prefill is at the end of this file.

// ---- routing ----------------------------------------------------------------------------------
// One simdgroup per token: softmax statistics over all E logits, then top_k rounds that each pick the
// best expert not yet picked by (logit + bias, lower index), then the weights, a softmax over the
// picks' unbiased logits. Lane j ends holding pick j.
constant constexpr int kRouteGroups = 4;
constant constexpr int kMaxTopK = 32;

kernel void moe_route(constant RouteArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int picked[kRouteGroups][kMaxTopK];
  const int t = int(tg) * kRouteGroups + int(sg);
  if (t >= a.T) return;  // the whole simdgroup leaves together; no threadgroup barrier follows
  const int E = a.E;
  const device float* row = a.logits + long(t) * E;

  float m = -INFINITY;
  for (int i = lane; i < E; i += 32) m = max(m, row[i]);
  m = simd_max(m);
  float z = 0.0f;
  for (int i = lane; i < E; i += 32) z += exp(row[i] - m);
  z = simd_sum(z);
  float h = 0.0f;
  for (int i = lane; i < E; i += 32) {
    const float p = exp(row[i] - m) / z;
    h -= p * log(max(p, 1e-9f));
  }
  h = simd_sum(h);
  if (lane == 0) {
    a.entropy[t] = h;
    a.confidence[t] = 1.0f / z;
  }

  threadgroup int* pick = picked[sg];
  for (int r = 0; r < a.top_k; ++r) {
    float best = -INFINITY;
    int best_i = E;
    for (int i = lane; i < E; i += 32) {
      bool taken = false;
      for (int j = 0; j < r; ++j) taken |= pick[j] == i;
      if (taken) continue;
      const float v = row[i] + (a.bias ? a.bias[i] : 0.0f);
      if (v > best || (v == best && i < best_i)) {
        best = v;
        best_i = i;
      }
    }
    for (ushort o = 16; o > 0; o >>= 1) {
      const float ov = simd_shuffle_xor(best, o);
      const int oi = simd_shuffle_xor(best_i, o);
      if (ov > best || (ov == best && oi < best_i)) {
        best = ov;
        best_i = oi;
      }
    }
    if (lane == 0) pick[r] = best_i;
    simdgroup_barrier(mem_flags::mem_threadgroup);
  }

  const bool live = int(lane) < a.top_k;
  const int id = live ? pick[lane] : 0;
  const float l = live ? row[id] : -INFINITY;
  const float lmax = simd_max(l);
  const float e = live ? exp(l - lmax) : 0.0f;
  const float esum = simd_sum(e);
  if (live) {
    a.ids[long(t) * a.top_k + lane] = id;
    a.weights[long(t) * a.top_k + lane] = e / esum;
  }
}

// summary[t, r] = sum_k weights[t, k] * emb[ids[t, k], r], k in order.
kernel void moe_summary(constant SummaryArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int r = gid.x, t = gid.y;
  if (r >= a.R || t >= a.T) return;
  float acc = 0.0f;
  for (int k = 0; k < a.top_k; ++k) {
    const long p = long(t) * a.top_k + k;
    acc += a.weights[p] * bf(a.emb[long(a.ids[p]) * a.R + r]);
  }
  a.summary[long(t) * a.R + r] = acc;
}

// ---- few-token experts --------------------------------------------------------------------------
constant constexpr int kBatch = 8;       // pairs per stage-1 batch
constant constexpr int kExpertRows = 4;  // intermediate columns (stage 1) or output rows (stage 2) per simdgroup
constant constexpr int kExpertGroups = 4;

kernel void expert_links(constant LinkArgs& a [[buffer(0)]], uint p [[thread_position_in_grid]]) {
  if (int(p) >= a.P) return;
  const int e = a.ids[p];
  int rank = 0, next = -1;
  for (int q = 0; q < a.P; ++q) {
    if (a.ids[q] != e) continue;
    if (q < int(p)) ++rank;
    else if (q > int(p) && next < 0) next = q;
  }
  a.links[p] = rank;
  a.links[a.P + p] = next;
}

// The dot core over a batch of `count` pairs, with the one-pair form taken when that is all there is.
template <int F, bool XV, bool SCALAR, int NR, int MM>
inline void expert_dots(const thread dptr* rows, const thread dptr* scales, const thread dptr* xrows, int count,
                        int K, uint lane, thread float* acc) {
  for (int i = 0; i < NR * MM; ++i) acc[i] = 0.0f;
  if (SCALAR) dot_scalar<F, true, NR, MM>(rows, scales, xrows, count, K, int(lane), 32, acc);
  else dot_core<F, true, XV, NR, MM>(rows, scales, xrows, count, K, int(lane), 32, acc);
  for (int i = 0; i < NR * MM; ++i) acc[i] = simd_sum(acc[i]);
}

// Stage 1. Grid (column blocks of I, pairs). The threadgroup of a batch leader computes its batch.
template <int F, bool XV, bool SCALAR>
kernel void expert_up(constant ExpertUpArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                      uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int R = kExpertRows;
  const int p = tg.y, P = a.P, I = a.I, L = a.L;
  if (a.links[p] % kBatch != 0) return;
  const int i0 = (int(tg.x) * kExpertGroups + int(sg)) * R;
  if (i0 >= I) return;  // no barriers in this kernel

  int pairs[kBatch];
  int count = 0;
  for (int q = p; q >= 0 && count < kBatch; q = a.links[P + q]) pairs[count++] = q;

  const device uchar* data;
  const device uchar* scale;
  float global;
  expert_slice<F>(a.w, a.ids[p], 2 * I, L, data, scale, global);
  dptr rows[2 * R], scales[2 * R];
  for (int r = 0; r < R; ++r) {
    const int i = min(i0 + r, I - 1);
    rows[r] = data + row_offset<F>(i, L);
    rows[R + r] = data + row_offset<F>(I + i, L);
    scales[r] = F == W_NVFP4 ? scale + long(i) * (L / 16) : scale;
    scales[R + r] = F == W_NVFP4 ? scale + long(I + i) * (L / 16) : scale;
  }
  dptr xrows[kBatch];
  for (int j = 0; j < kBatch; ++j)
    xrows[j] = (const device uchar*)(a.latent + long(pairs[min(j, count - 1)] / a.top_k) * L);

  if (count == 1) {
    float acc[2 * R];
    expert_dots<F, XV, SCALAR, 2 * R, 1>(rows, scales, xrows, 1, L, lane, acc);
    for (int r = 0; r < R; ++r) {
      const int i = i0 + r;
      if (int(lane) != r || i >= I) continue;
      const float g = acc[r] * row_factor<F>(scale, i, global);
      const float u = acc[R + r] * row_factor<F>(scale, I + i, global);
      a.inter[long(p) * I + i] = to_bf(silu(g) * u);
    }
  } else {
    float acc[2 * R * kBatch];
    expert_dots<F, XV, SCALAR, 2 * R, kBatch>(rows, scales, xrows, count, L, lane, acc);
    for (int r = 0; r < R; ++r) {
      const int i = i0 + r;
      if (i >= I) continue;
      const float fg = row_factor<F>(scale, i, global), fu = row_factor<F>(scale, I + i, global);
      for (int j = 0; j < kBatch; ++j) {
        if (int(lane) != ((r * kBatch + j) & 31) || j >= count) continue;
        const float g = acc[r * kBatch + j] * fg;
        const float u = acc[(R + r) * kBatch + j] * fu;
        a.inter[long(pairs[j]) * I + i] = to_bf(silu(g) * u);
      }
    }
  }
}

// Stage 2. Grid (row blocks of L, tokens). Each lane weights its partial dot of each expert by the
// pair's weight and the row's factor, then the lanes are summed once.
template <int F, bool XV, bool SCALAR>
kernel void expert_down(constant ExpertDownArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  constexpr int R = kExpertRows;
  const int t = tg.y, L = a.L, I = a.I;
  const int l0 = (int(tg.x) * kExpertGroups + int(sg)) * R;
  if (l0 >= L) return;
  float tot[R];
  for (int r = 0; r < R; ++r) tot[r] = 0.0f;
  for (int k = 0; k < a.top_k; ++k) {
    const long p = long(t) * a.top_k + k;
    const device uchar* data;
    const device uchar* scale;
    float global;
    expert_slice<F>(a.w, a.ids[p], L, I, data, scale, global);
    dptr rows[R], scales[R];
    for (int r = 0; r < R; ++r) {
      const int l = min(l0 + r, L - 1);
      rows[r] = data + row_offset<F>(l, I);
      scales[r] = F == W_NVFP4 ? scale + long(l) * (I / 16) : scale;
    }
    dptr xrows[1] = {(const device uchar*)(a.inter + p * I)};
    float acc[R];
    for (int r = 0; r < R; ++r) acc[r] = 0.0f;
    if (SCALAR) dot_scalar<F, true, R, 1>(rows, scales, xrows, 1, I, int(lane), 32, acc);
    else dot_core<F, true, XV, R, 1>(rows, scales, xrows, 1, I, int(lane), 32, acc);
    const float wk = a.weights[p];
    for (int r = 0; r < R; ++r) tot[r] = fma(acc[r], wk * row_factor<F>(scale, min(l0 + r, L - 1), global), tot[r]);
  }
  for (int r = 0; r < R; ++r) {
    const float v = simd_sum(tot[r]);
    if (lane == 0 && l0 + r < L) a.out[long(t) * L + l0 + r] = v;
  }
}

#define LX_EXPERT(F, XV, SC)                                                                            \
  template [[host_name("expert_up_" #F "_" #XV #SC)]] kernel void expert_up<F, bool(XV), bool(SC)>(     \
      constant ExpertUpArgs&, uint2, uint, uint);                                                       \
  template [[host_name("expert_down_" #F "_" #XV #SC)]] kernel void expert_down<F, bool(XV), bool(SC)>( \
      constant ExpertDownArgs&, uint2, uint, uint);
#define LX_EXPERT_F(F) LX_EXPERT(F, 0, 0) LX_EXPERT(F, 1, 0) LX_EXPERT(F, 0, 1)
LX_EXPERT_F(0)
LX_EXPERT_F(1)
LX_EXPERT_F(2)
LX_EXPERT_F(3)

// ---- grouped experts ----------------------------------------------------------------------------
// Pairs are bucketed by expert (moe_group), each expert's run is cut into tiles of up to 64 pairs, and
// the two stages run as tile GEMMs (gemm.metal, modes 1 and 2) over the tile list. Stage 2 writes each
// pair's weighted row; moe_combine sums a token's top_k rows in order. Within an expert the pairs keep
// their order, so the tiling, and with it every bit of the result, is the same on every run.

// One threadgroup. Threadgroup memory: counts [E], next free slot [E], one chunk of expert ids.
kernel void moe_group(constant GroupArgs& a [[buffer(0)]], threadgroup int* sh [[threadgroup(0)]],
                      uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  const int E = a.E, P = a.P;
  threadgroup atomic_int* counts = (threadgroup atomic_int*)sh;
  threadgroup int* slot = sh + E;
  threadgroup int* chunk = sh + 2 * E;
  for (int e = tid; e < E; e += threads) atomic_store_explicit(&counts[e], 0, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int p = tid; p < P; p += threads) atomic_fetch_add_explicit(&counts[a.ids[p]], 1, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (tid == 0) {
    int at = 0, tiles = 0;
    for (int e = 0; e < E; ++e) {
      const int c = atomic_load_explicit(&counts[e], memory_order_relaxed);
      a.counts[e] = at;
      slot[e] = at;
      for (int r = 0; r < c; r += a.tile_rows) {
        a.tiles[1 + 3 * tiles] = e;
        a.tiles[2 + 3 * tiles] = at + r;
        a.tiles[3 + 3 * tiles] = min(a.tile_rows, c - r);
        ++tiles;
      }
      at += c;
    }
    a.counts[E] = at;
    a.tiles[0] = tiles;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  // Stable placement, one chunk of pairs at a time: a pair's place is its expert's next free slot plus
  // the number of earlier pairs of the chunk on the same expert.
  for (int c0 = 0; c0 < P; c0 += threads) {
    const int p = c0 + int(tid);
    const int e = p < P ? a.ids[p] : -1;
    chunk[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= 0) {
      int rank = 0;
      for (int j = 0; j < int(tid); ++j) rank += chunk[j] == e ? 1 : 0;
      a.order[slot[e] + rank] = p;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (e >= 0) atomic_fetch_add_explicit((threadgroup atomic_int*)&slot[e], 1, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// out[t, l] = sum_k rows[t * top_k + k, l], k in order.
kernel void moe_combine(constant CombineArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int l = gid.x, t = gid.y;
  if (l >= a.L || t >= a.T) return;
  float acc = 0.0f;
  for (int k = 0; k < a.top_k; ++k) acc += a.rows[(long(t) * a.top_k + k) * a.L + l];
  a.out[long(t) * a.L + l] = acc;
}
