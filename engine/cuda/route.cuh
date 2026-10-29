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

// The router's GEMV partials for one output column, for the rows m < M: thread t (of 256) returns in acc[m] its
// chain over k = t + 256 j, j = 0 .. J - 1, in that order. Every load is issued before the first product, so the
// J * (1 + M) loads of a thread are in flight together rather than one DRAM round trip after another. The caller
// reduces acc across the warp exactly as lx_spec_gemv_f32 does.
template <int J, int Rows>
__device__ __forceinline__ void gemv_partials(const float* __restrict__ x, int ld_x, const float* __restrict__ wr,
                                              int M, int K, int t, float (&acc)[Rows]) {
  float wv[J];
  float xv[J][Rows];
#pragma unroll
  for (int j = 0; j < J; ++j) {
    const int k = t + j * 256;
    const bool in = k < K;
    wv[j] = in ? wr[k] : 0.0f;
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

}  // namespace lx::route
