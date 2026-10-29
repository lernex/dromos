// Causal grouped-query attention with no positional encoding, over varlen sequences.
//
// attn_write_kv rounds this step's k and v to bf16 into the cache first, so prefill and decode read the
// same values. attention takes one (token, kv head) item per threadgroup and serves its G query heads
// from one pass over the cache: four simdgroups, one query head each (more heads take more passes).
// The cache goes through threadgroup memory in tiles of 32 positions; in a tile, lane j scores key j
// (rows padded so the lanes hit distinct banks), the simdgroup keeps an online softmax, and lane l
// accumulates output dims l, l + 32, ...
//
// Every item reduces in `splits` parts, part r taking tiles r, r + splits, ..., and the parts are merged in
// attn_combine's order. Few items (decode): the parts run in separate threadgroups, which write their softmax
// state (max, sum, accumulator) to a scratch buffer, and attn_combine merges them. Many items (prefill): one
// threadgroup runs the parts in turn and merges them in place, with the same arithmetic, so a token's bits do
// not depend on how many items run beside it. T is device data (the
// ABI passes cu_seqlens, not a count), so both kernels read T = cu[n_seq] and grid-stride over the
// work, and every threadgroup makes the same choice. On GPUs with neural accelerators, prefill items
// go to the flash kernel in nax/attention.metal instead (attn_flash_takes).

constant constexpr int kAttnTile = 32;
constant constexpr int kAttnHeads = 4;  // query heads per pass: one per simdgroup
constant constexpr int kSplitBelow = 64;
constant constexpr int kAttnSplitMax = 16;  // the most parts an item takes (the host's kAttnSplits)

// Whether the whole items go to the flash kernel on the neural accelerators (nax/attention.metal, which
// makes the same choice): the host allows it, the items are not split, and the sequences carry enough
// tokens each to fill its query blocks.
inline bool attn_flash_takes(constant AttnArgs& a, int T) {
  return a.flash != 0 && !(a.splits > 1 && T * a.KVH < kSplitBelow) && T >= 4 * a.n_seq;
}

inline int seq_of_token(const device int* cu, int n_seq, int t) {
  int lo = 1, hi = n_seq;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (cu[mid] > t) hi = mid;
    else lo = mid + 1;
  }
  return lo - 1;
}

kernel void attn_write_kv(constant AttnArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                          uint groups [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                          uint threads [[threads_per_threadgroup]]) {
  const int T = a.cu[a.n_seq];
  const int width = a.KVH * a.Dh;
  for (int t = g; t < T; t += groups) {
    const int s = seq_of_token(a.cu, a.n_seq, t);
    const long pos = a.start_pos[s] + (t - a.cu[s]);
    const long row = (long(a.slot_ids[s]) * a.capacity + pos) * width;
    const device float* k = a.qkv + long(t) * a.ld_qkv + a.H * a.Dh;
    const device float* v = k + width;
    for (int i = tid; i < width; i += threads) {
      a.k_cache[row + i] = to_bf(k[i]);
      a.v_cache[row + i] = to_bf(v[i]);
    }
  }
}

// Threadgroup memory: K and V tiles of kAttnTile rows of Dh + 2 bf16 (the pad spreads the rows over
// the banks), then the pass's scaled queries, kAttnHeads rows of Dh floats.
kernel void attention(constant AttnArgs& a [[buffer(0)]], threadgroup uchar* smem [[threadgroup(0)]],
                      uint g [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint sg [[simdgroup_index_in_threadgroup]],
                      uint lane [[thread_index_in_simdgroup]]) {
  const int Dh = a.Dh, H = a.H, KVH = a.KVH;
  const int G = H / KVH;
  const int RS = Dh + 2;  // tile row stride in bf16
  threadgroup ushort* sK = (threadgroup ushort*)smem;
  threadgroup ushort* sV = sK + kAttnTile * RS;
  threadgroup float* sQ = (threadgroup float*)(sV + kAttnTile * RS);
  const int T = a.cu[a.n_seq];
  if (attn_flash_takes(a, T)) return;
  // Few items run their parts in parallel; otherwise a threadgroup runs its item's parts in turn. The parts
  // are the architecture's (S = splits), so the reduction is the same either way.
  const bool split = a.splits > 1 && T * KVH < kSplitBelow;
  const int S = a.splits;
  const int part = split ? int(g) % S : 0;
  const int first = split ? int(g) / S : int(g);
  const int stride = split ? int(groups) / S : int(groups);
  const int NCH = Dh / 8;  // 16-byte chunks per row

  for (int item = first; item < T * KVH; item += stride) {
    const int t = item / KVH, kvh = item % KVH;
    const int s = seq_of_token(a.cu, a.n_seq, t);
    const long slot = a.slot_ids[s];
    const int L = a.start_pos[s] + (t - a.cu[s]) + 1;  // visible positions 0 .. L - 1
    const int n_tiles = (L + kAttnTile - 1) / kAttnTile;
    const device float* q_row = a.qkv + long(t) * a.ld_qkv;

    for (int h0 = 0; h0 < G; h0 += kAttnHeads) {
      const int hq = kvh * G + h0 + int(sg);  // this simdgroup's query head
      const bool live = h0 + int(sg) < G;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (int i = tid; i < kAttnHeads * Dh; i += 128) {
        const int hh = i / Dh, d = i % Dh;
        sQ[i] = (h0 + hh < G) ? q_row[(kvh * G + h0 + hh) * Dh + d] * a.scale : 0.0f;
      }
      // The item's parts, each reduced from a softmax state of its own: one part here in split mode, all S in turn
      // otherwise. The merge at the end is attn_combine's, in the same order.
      const int pr_lo = split ? part : 0;
      const int pr_hi = split ? part + 1 : S;
      float pm[kAttnSplitMax], pl[kAttnSplitMax];
      float pacc[kAttnSplitMax][4];
      for (int pr = pr_lo; pr < pr_hi; ++pr) {
        float m = -INFINITY, l = 0.0f;
        float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};  // dims lane + 32 j, Dh <= 128

        for (int tile = pr; tile < n_tiles; tile += S) {
          threadgroup_barrier(mem_flags::mem_threadgroup);
          for (int c = tid; c < kAttnTile * NCH; c += 128) {
            const int r = c / NCH, ch = c % NCH;
            const int p = tile * kAttnTile + r;
            uint4 kv = uint4(0), vv = uint4(0);
            if (p < L) {
              const long src = ((slot * a.capacity + p) * KVH + kvh) * Dh + ch * 8;
              kv = *(const device uint4*)(a.k_cache + src);
              vv = *(const device uint4*)(a.v_cache + src);
            }
            threadgroup uint* dk = (threadgroup uint*)(sK + r * RS + ch * 8);
            threadgroup uint* dv = (threadgroup uint*)(sV + r * RS + ch * 8);
            dk[0] = kv.x; dk[1] = kv.y; dk[2] = kv.z; dk[3] = kv.w;
            dv[0] = vv.x; dv[1] = vv.y; dv[2] = vv.z; dv[3] = vv.w;
          }
          threadgroup_barrier(mem_flags::mem_threadgroup);
          if (!live) continue;
          const int p = tile * kAttnTile + int(lane);
          float score = -INFINITY;
          if (p < L) {
            const threadgroup ushort* kr = sK + lane * RS;
            const threadgroup float* qr = sQ + sg * Dh;
            float dot = 0.0f;
            for (int d = 0; d < Dh; d += 2) {
              const uint w = *(const threadgroup uint*)(kr + d);
              dot = fma(qr[d], bf_lo(w), dot);
              dot = fma(qr[d + 1], bf_hi(w), dot);
            }
            score = dot;
          }
          const float mn = max(m, simd_max(score));
          if (mn == -INFINITY) continue;  // no visible key in this tile (only possible for a tail split)
          const float corr = fast::exp(m - mn);
          const float pj = p < L ? fast::exp(score - mn) : 0.0f;
          l = l * corr + simd_sum(pj);
          for (int j = 0; j < 4; ++j) acc[j] *= corr;
          for (int j = 0; j < kAttnTile; ++j) {
            const float w = simd_shuffle(pj, ushort(j));
            const threadgroup ushort* vr = sV + j * RS;
            for (int k = 0; k < 4; ++k) {
              const int d = int(lane) + 32 * k;
              if (d < Dh) acc[k] = fma(w, bf(vr[d]), acc[k]);
            }
          }
          m = mn;
        }
        pm[pr] = m;
        pl[pr] = l;
        for (int k = 0; k < 4; ++k) pacc[pr][k] = acc[k];
      }

      if (live) {
        if (split) {
          // Partial state: [item][part][head][Dh + 2] = accumulator, max, sum.
          device float* ps = a.partial + ((long(item) * S + part) * G + (h0 + sg)) * (Dh + 2);
          for (int k = 0; k < 4; ++k) {
            const int d = int(lane) + 32 * k;
            if (d < Dh) ps[d] = pacc[part][k];
          }
          if (lane == 0) {
            ps[Dh] = pm[part];
            ps[Dh + 1] = pl[part];
          }
        } else {
          device ushort* o = a.out + long(t) * H * Dh + long(hq) * Dh;
          if (S == 1) {
            const float inv = 1.0f / pl[0];
            for (int k = 0; k < 4; ++k) {
              const int d = int(lane) + 32 * k;
              if (d < Dh) o[d] = to_bf(pacc[0][k] * inv);
            }
          } else {
            // attn_combine's merge of the parts: the same order and arithmetic, so the bits match a split item.
            float M = -INFINITY;
            for (int r = 0; r < S; ++r) M = max(M, pm[r]);
            float Z = 0.0f;
            float A[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            for (int r = 0; r < S; ++r) {
              if (pm[r] == -INFINITY) continue;
              const float w = fast::exp(pm[r] - M);
              Z += pl[r] * w;
              for (int k = 0; k < 4; ++k) A[k] += pacc[r][k] * w;
            }
            for (int k = 0; k < 4; ++k) {
              const int d = int(lane) + 32 * k;
              if (d < Dh) o[d] = to_bf(A[k] / Z);
            }
          }
        }
      }
    }
  }
}

// Merges the split parts of each item (only in split mode; a whole item is merged in the attention kernel).
kernel void attn_combine(constant AttnArgs& a [[buffer(0)]], uint g [[threadgroup_position_in_grid]],
                         uint groups [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]],
                         uint threads [[threads_per_threadgroup]]) {
  const int T = a.cu[a.n_seq];
  const int KVH = a.KVH, Dh = a.Dh, G = a.H / KVH, S = a.splits;
  if (!(S > 1 && T * KVH < kSplitBelow)) return;
  for (int item = g; item < T * KVH; item += groups) {
    const int t = item / KVH, kvh = item % KVH;
    for (int i = tid; i < G * Dh; i += threads) {
      const int hh = i / Dh, d = i % Dh;
      float M = -INFINITY;
      for (int r = 0; r < S; ++r) M = max(M, a.partial[((long(item) * S + r) * G + hh) * (Dh + 2) + Dh]);
      float Z = 0.0f, A = 0.0f;
      for (int r = 0; r < S; ++r) {
        const device float* ps = a.partial + ((long(item) * S + r) * G + hh) * (Dh + 2);
        const float mr = ps[Dh];
        if (mr == -INFINITY) continue;
        const float w = fast::exp(mr - M);
        Z += ps[Dh + 1] * w;
        A += ps[d] * w;
      }
      a.out[long(t) * a.H * Dh + long(kvh * G + hh) * Dh + d] = to_bf(A / Z);
    }
  }
}
