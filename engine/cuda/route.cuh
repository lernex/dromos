// The router's arithmetic, shared by lx_moe_route (moe.cu) and the router's fused epilogue (lx_router_route, moe.cu).
//
// Each function is one warp's work on one row of router logits, and the float operations and their order are the
// ones lx_moe_route has always made, so a row gives the same bits whichever kernel routes it. The GEMV partials are
// the chains lx_spec_gemv_f32 sums: thread t owns k = t, t + 256, ... in order, with one fused multiply-add each.
#pragma once

#include <cstdint>

#include "cuda/common.cuh"

namespace lx::route {

constexpr int kMaxTopK = 32;  // the picks of a row: one per lane of its warp

// Softmax statistics of one row: the max, the normaliser z and the entropy. Lane 0 writes the entropy and the
// confidence (the largest probability, 1 / z). The full warp runs it.
__device__ __forceinline__ void row_stats(const float* row, int E, int lane, float* entropy, float* confidence) {
  float m = -INFINITY;
  for (int i = lane; i < E; i += 32) m = fmaxf(m, row[i]);
  m = lx::warp_max(m);
  float z = 0.0f;
  for (int i = lane; i < E; i += 32) z += expf(row[i] - m);
  z = lx::warp_sum(z);
  float h = 0.0f;
  for (int i = lane; i < E; i += 32) {
    const float p = expf(row[i] - m) / z;
    h -= p * logf(fmaxf(p, 1e-9f));
  }
  h = lx::warp_sum(h);
  if (lane == 0) {
    *entropy = h;
    *confidence = 1.0f / z;
  }
}

// The top_k picks of one row by logits + bias: each round, every lane scans its strided slice for the best expert not
// yet picked, then the warp reduces to one winner. The (value, lower index) order makes ties deterministic. Writes
// pick[0 .. top_k) in pick order. The full warp runs it, and the picks are visible to the warp on return.
__device__ __forceinline__ void row_picks(const float* row, const float* bias, int E, int lane, int top_k, int* pick) {
  for (int r = 0; r < top_k; ++r) {
    float best = -INFINITY;
    int best_i = E;
    for (int i = lane; i < E; i += 32) {
      bool taken = false;
      for (int j = 0; j < r; ++j) taken |= pick[j] == i;
      if (taken) continue;
      const float v = row[i] + (bias ? bias[i] : 0.0f);
      if (v > best || (v == best && i < best_i)) {
        best = v;
        best_i = i;
      }
    }
    for (int o = 16; o > 0; o >>= 1) {
      const float ov = __shfl_xor_sync(0xffffffffu, best, o);
      const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
      if (ov > best || (ov == best && oi < best_i)) {
        best = ov;
        best_i = oi;
      }
    }
    if (lane == 0) pick[r] = best_i;
    __syncwarp();
  }
}

// The weights over one row's picks: a softmax over the picks of their unbiased logits. Lane j holds pick j, and writes
// its expert id and weight. The full warp runs it, after row_picks.
__device__ __forceinline__ void row_weights(const float* row, const int* pick, int top_k, int lane, int32_t* ids,
                                            float* weights) {
  const bool live = lane < top_k;
  const int id = live ? pick[lane] : 0;
  const float l = live ? row[id] : -INFINITY;
  const float lmax = lx::warp_max(l);
  const float e = live ? expf(l - lmax) : 0.0f;
  const float esum = lx::warp_sum(e);
  if (live) {
    ids[lane] = id;
    weights[lane] = e / esum;
  }
}

// The router's weights for one output column: thread t (of 256) takes k = t + 256 j, j = 0 .. J - 1, zero past K.
// They are static, so the caller loads them before the predecessor's wait.
template <int J>
__device__ __forceinline__ void load_w(const float* __restrict__ wr, int K, int t, float (&wv)[J]) {
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int k = t + j * 256;
    wv[j] = k < K ? wr[k] : 0.0f;
  }
}

// The router's GEMV partials for one output column, for the rows m < M: thread t (of 256) returns in acc[m] its
// chain over k = t + 256 j, j = 0 .. J - 1, in that order, with wv from load_w. The activation loads are issued before
// the first product, so the J * M loads of a thread are in flight together. The caller reduces acc across the warp
// exactly as lx_spec_gemv_f32 does.
template <int J, int Rows>
__device__ __forceinline__ void gemv_partials(const float* __restrict__ x, int ld_x, const float (&wv)[J], int M,
                                              int K, int t, float (&acc)[Rows]) {
  float xv[J][Rows];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int k = t + j * 256;
    const bool in = k < K;
#pragma unroll
    for (int m = 0; m < Rows; ++m) xv[j][m] = (in && m < M) ? x[m * ld_x + k] : 0.0f;
  }
#pragma unroll
  for (int m = 0; m < Rows; ++m) acc[m] = 0.0f;
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const bool in = t + j * 256 < K;
#pragma unroll
    for (int m = 0; m < Rows; ++m)
      if (in && m < M) acc[m] = fmaf(wv[j], xv[j][m], acc[m]);
  }
}

// The picks of a row in one pass, for the fused launch. Each lane keeps the best TK of its own experts; a butterfly then
// merges the lists. The best TK of two sorted lists is the elementwise better of one list and the other reversed, which
// is bitonic, and a bitonic sort (half-cleaners) puts it in order. Every lane ends with the row's best TK in row_picks'
// order: value descending, ties to the lower index. A NaN bias is never picked, as in row_picks. row_picks runs top_k
// rounds of scan, shuffle reduction and shared-memory write; this gives the same picks in one pass.
template <int TK>
struct TopList {
  float v[TK];
  int i[TK];
};

__device__ __forceinline__ bool beats(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// Insert (v, i) into a list kept best first.
template <int TK>
__device__ __forceinline__ void top_insert(TopList<TK>& t, float v, int i) {
  if (!beats(v, i, t.v[TK - 1], t.i[TK - 1])) return;
  t.v[TK - 1] = v;
  t.i[TK - 1] = i;
#pragma unroll
  for (int k = TK - 1; k > 0; --k) {
    if (beats(t.v[k], t.i[k], t.v[k - 1], t.i[k - 1])) {
      const float tv = t.v[k - 1];
      const int ti = t.i[k - 1];
      t.v[k - 1] = t.v[k];
      t.i[k - 1] = t.i[k];
      t.v[k] = tv;
      t.i[k] = ti;
    }
  }
}

// The best TK of two best-first lists (TK a power of two), kept best first.
template <int TK>
__device__ __forceinline__ void top_merge(TopList<TK>& a, const TopList<TK>& b) {
  TopList<TK> c;
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    const int j = TK - 1 - k;
    const bool take_a = beats(a.v[k], a.i[k], b.v[j], b.i[j]);
    c.v[k] = take_a ? a.v[k] : b.v[j];
    c.i[k] = take_a ? a.i[k] : b.i[j];
  }
#pragma unroll
  for (int d = TK / 2; d >= 1; d /= 2) {
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      if ((k & d) == 0) {
        const int j = k + d;
        if (beats(c.v[j], c.i[j], c.v[k], c.i[k])) {
          const float tv = c.v[k];
          const int ti = c.i[k];
          c.v[k] = c.v[j];
          c.i[k] = c.i[j];
          c.v[j] = tv;
          c.i[j] = ti;
        }
      }
    }
  }
  a = c;
}

// Every lane of the warp ends holding the warp's best TK.
template <int TK>
__device__ __forceinline__ void top_warp(TopList<TK>& t) {
#pragma unroll
  for (int o = 16; o > 0; o >>= 1) {
    TopList<TK> p;
#pragma unroll
    for (int k = 0; k < TK; ++k) {
      p.v[k] = __shfl_xor_sync(0xffffffffu, t.v[k], o);
      p.i[k] = __shfl_xor_sync(0xffffffffu, t.i[k], o);
    }
    top_merge(t, p);
  }
}

// The picks of one row, the same as row_picks writes them for top_k <= TK. Lane j scans experts j, j + 32, ... The full
// warp runs it, and the picks are visible to the warp on return.
template <int TK>
__device__ __forceinline__ void top_picks(const float* row, const float* bias, int E, int lane, int top_k, int* pick) {
  TopList<TK> t;
#pragma unroll
  for (int k = 0; k < TK; ++k) {
    t.v[k] = -INFINITY;
    t.i[k] = E;
  }
  for (int i = lane; i < E; i += 32) top_insert(t, row[i] + (bias ? bias[i] : 0.0f), i);
  top_warp(t);
#pragma unroll
  for (int k = 0; k < TK; ++k)
    if (lane == 0 && k < top_k) pick[k] = t.i[k];
  __syncwarp();
}

}  // namespace lx::route
