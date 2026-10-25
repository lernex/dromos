// Prefill attention on the neural accelerators: flash attention over the cache, for the items the
// per-token kernel (attention.metal) would otherwise take whole. That kernel reads the visible cache once
// per token, which grows with the square of the prompt; here a threadgroup takes a block of
// kFlashRows / G tokens of one sequence with all G query heads of one kv head (kFlashRows query rows),
// and reads each cache tile once for all of them.
//
// Per tile of kFlashKeys positions: S = Q K^T on the accelerators (q in f32 against the bf16 cache, f32
// scores) into threadgroup memory; an online softmax over each row (two threads per row) writes P as bf16 and the row's
// rescale factor; then O = O * corr + P V, O a cooperative f32 tensor that stays in registers.
// The next tile's K and V are loaded into registers while the current one is multiplied. Positions past
// the block's last visible one are never read (they are zero in the staging), and each row masks the
// positions past its own token. Head size 64 only (the host checks); the result matches the per-token
// kernel to bf16 rounding of the probabilities, as tensor-core flash attention does.
//
// Threadgroup memory is exactly Metal's 32 KB: Q in f32, a K and a V tile, and the scores, whose
// space P, the rescale factors and the row sums reuse once every thread holds its scores in registers.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

constant constexpr int kFlashRows = 64;   // query rows per threadgroup: kFlashRows / G tokens x G heads
constant constexpr int kFlashKeys = 32;   // cache positions per tile
constant constexpr int kFlashDh = 64;     // head size
constant constexpr int kFlashBelow = 64;  // kSplitBelow of attention.metal

// The choice attention.metal's attn_flash_takes makes, on the same device data.
inline bool flash_takes(constant AttnArgs& a, int T) {
  return a.flash != 0 && !(a.splits > 1 && T * a.KVH < kFlashBelow) && T >= 4 * a.n_seq;
}

kernel void attention_flash(constant AttnArgs& a [[buffer(0)]], threadgroup uchar* smem [[threadgroup(0)]],
                            uint g [[threadgroup_position_in_grid]], uint groups [[threadgroups_per_grid]],
                            uint tid [[thread_index_in_threadgroup]]) {
  constexpr int R = kFlashRows, BK = kFlashKeys, D = kFlashDh;
  constexpr int CH = D / 8;  // 16-byte chunks per cache row
  const int T = a.cu[a.n_seq];
  if (!flash_takes(a, T)) return;
  const int KVH = a.KVH, H = a.H, G = H / KVH, BQ = R / G;
  threadgroup float* sQ = (threadgroup float*)smem;             // [R][D]
  threadgroup bfloat* sK = (threadgroup bfloat*)(sQ + R * D);   // [BK][D]
  threadgroup bfloat* sV = sK + BK * D;                         // [BK][D]
  threadgroup float* sS = (threadgroup float*)(sV + BK * D);    // [R][BK]
  threadgroup bfloat* sP = (threadgroup bfloat*)sS;             // [R][BK], over the first half of sS
  threadgroup float* sCorr = sS + R * BK / 2;                   // [R], in the second half
  threadgroup float* sL = sCorr + R;                            // [R]

  using TQ = tensor<threadgroup float, extents<int32_t, D, R>, tensor_inline>;
  using TK = tensor<threadgroup bfloat, extents<int32_t, D, BK>, tensor_inline>;
  using TS = tensor<threadgroup float, extents<int32_t, BK, R>, tensor_inline>;
  using TP = tensor<threadgroup bfloat, extents<int32_t, BK, R>, tensor_inline>;
  using TV = tensor<threadgroup bfloat, extents<int32_t, D, BK>, tensor_inline>;
  TQ tq(sQ, extents<int32_t, D, R>());
  TK tk(sK, extents<int32_t, D, BK>());
  TS ts(sS, extents<int32_t, BK, R>());
  TP tp(sP, extents<int32_t, BK, R>());
  TV tv(sV, extents<int32_t, D, BK>());
  constexpr auto sdesc = matmul2d_descriptor(R, BK, D, false, true, false, matmul2d_descriptor::mode::multiply);
  constexpr auto odesc =
      matmul2d_descriptor(R, D, BK, false, false, false, matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<sdesc, execution_simdgroups<4>> sop;
  matmul2d<odesc, execution_simdgroups<4>> oop;

  // Items: (sequence, block of BQ tokens, kv head), the sequences in order.
  int total = 0;
  for (int s = 0; s < a.n_seq; ++s) total += (a.cu[s + 1] - a.cu[s] + BQ - 1) / BQ;
  total *= KVH;
  const int r = int(tid) >> 1, side = int(tid) & 1;  // the softmax's row and half of it
  for (int item = int(g); item < total; item += int(groups)) {
    const int kvh = item % KVH;
    int blk = item / KVH, s = 0;
    for (;; ++s) {
      const int nb = (a.cu[s + 1] - a.cu[s] + BQ - 1) / BQ;
      if (blk < nb) break;
      blk -= nb;
    }
    const int t0 = a.cu[s] + blk * BQ;               // the block's first token
    const int ntok = min(BQ, a.cu[s + 1] - t0);      // its tokens
    const int p0 = a.start_pos[s] + (t0 - a.cu[s]);  // the first token's cache position
    const int last = p0 + ntok - 1;                  // the last position any row sees
    const long slot = a.slot_ids[s];
    const int limit = p0 + min(r / G, ntok - 1);     // the last position row r sees

    // Q rows r = token * G + head, scaled.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int i = int(tid); i < R * D; i += 128) {
      const int row = i / D, d = i % D, tok = row / G;
      float v = 0.0f;
      if (tok < ntok) v = a.qkv[long(t0 + tok) * a.ld_qkv + (kvh * G + row % G) * D + d] * a.scale;
      sQ[i] = v;
    }

    auto o = oop.template get_destination_cooperative_tensor<TP, TV, float>();
    for (uint16_t i = 0; i < o.get_capacity(); ++i)
      if (o.is_valid_element(i)) o[i] = 0.0f;
    float m = -INFINITY, l = 0.0f;

    // This thread's two 16-byte chunks of a K tile and of a V tile.
    uint4 pk[2], pv[2];
    const int n_tiles = last / BK + 1;
    for (int c = 0; c < 2; ++c) {
      const int idx = int(tid) + c * 128, kr = idx / CH, ch = idx % CH;
      const long src = ((slot * a.capacity + kr) * KVH + kvh) * D + ch * 8;
      pk[c] = kr <= last ? *(const device uint4*)(a.k_cache + src) : uint4(0);
      pv[c] = kr <= last ? *(const device uint4*)(a.v_cache + src) : uint4(0);
    }
    for (int tile = 0; tile < n_tiles; ++tile) {
      threadgroup_barrier(mem_flags::mem_threadgroup);  // the last tile's reads of K, V and P are done
      for (int c = 0; c < 2; ++c) {
        const int idx = int(tid) + c * 128;
        ((threadgroup uint4*)sK)[idx] = pk[c];
        ((threadgroup uint4*)sV)[idx] = pv[c];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tile + 1 < n_tiles) {
        for (int c = 0; c < 2; ++c) {
          const int idx = int(tid) + c * 128, pos = (tile + 1) * BK + idx / CH, ch = idx % CH;
          const long src = ((slot * a.capacity + pos) * KVH + kvh) * D + ch * 8;
          pk[c] = pos <= last ? *(const device uint4*)(a.k_cache + src) : uint4(0);
          pv[c] = pos <= last ? *(const device uint4*)(a.v_cache + src) : uint4(0);
        }
      }

      auto sc = sop.template get_destination_cooperative_tensor<TQ, TK, float>();
      sop.run(tq, tk, sc);
      sc.store(ts);
      threadgroup_barrier(mem_flags::mem_threadgroup);

      // Online softmax over row r, this thread's 16 columns.
      float sv[16];
      float mx = -INFINITY;
      for (int j = 0; j < 16; ++j) {
        const int c = side * 16 + j;
        sv[j] = tile * BK + c <= limit ? sS[r * BK + c] : -INFINITY;
        mx = max(mx, sv[j]);
      }
      mx = max(mx, simd_shuffle_xor(mx, 1));
      threadgroup_barrier(mem_flags::mem_threadgroup);  // every row's scores are read: P may reuse them
      const float mn = max(m, mx);
      const float corr = mn == -INFINITY ? 1.0f : exp(m - mn);
      float sum = 0.0f;
      for (int j = 0; j < 16; ++j) {
        const bfloat pj = bfloat(sv[j] == -INFINITY ? 0.0f : exp(sv[j] - mn));
        sum += float(pj);  // the rounded probabilities, as P V sees them
        sP[r * BK + side * 16 + j] = pj;
      }
      sum += simd_shuffle_xor(sum, 1);
      l = l * corr + sum;
      m = mn;
      if (side == 0) sCorr[r] = corr;
      threadgroup_barrier(mem_flags::mem_threadgroup);

      for (uint16_t i = 0; i < o.get_capacity(); ++i)
        if (o.is_valid_element(i)) o[i] *= sCorr[o.get_multidimensional_index(i)[1]];
      oop.run(tp, tv, o);
    }

    if (side == 0) sL[r] = l;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint16_t i = 0; i < o.get_capacity(); ++i) {
      if (!o.is_valid_element(i)) continue;
      const auto at = o.get_multidimensional_index(i);
      const int row = at[1], d = at[0], tok = row / G;
      if (tok < ntok)
        a.out[long(t0 + tok) * H * D + long(kvh * G + row % G) * D + d] = to_bf(o[i] / sL[row]);
    }
  }
}
