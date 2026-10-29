// Latent mixture of experts (kernels.h, "Latent mixture of experts").
//
// Routing is one warp per token. The top-k picks are k warp-wide argmax rounds over the biased
// logits, so the order is deterministic: descending biased logit, ties to the lower index.
//
// The experts run as two kernels with no atomics, so every output is written once:
//   1. intermediate: one warp per intermediate row i of a (token, k) pair computes the gate and up
//      dot products with the token's latent and writes bf16(silu(gate) * up) to a [T * top_k, I]
//      scratch buffer.
//   2. down: one warp per output row l of a token. The token's top_k intermediates are staged in
//      shared memory once per block, and the weighted sum over k stays in registers.
// The scratch comes from cudaMallocAsync on the caller's stream: kernels.h has no workspace
// argument, and stream-ordered allocation keeps the buffer valid under graph capture.
#include "cuda/common.cuh"
#include "cuda/route.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;                   // expert kernels: 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kRouteWarps = 4;                  // route kernel: one token per warp, 4 per block
constexpr int kMaxTopK = lx::route::kMaxTopK;   // route keeps one pick per lane of its warp
constexpr size_t kMaxSmem = 48 * 1024;          // default dynamic shared memory limit

// Eight bf16 values in one 16-byte load.
struct alignas(16) Bf16x8 {
  __nv_bfloat162 v[4];
};

// Dot product of a bf16 row with a bf16 vector x of length n, summed over the warp: every lane
// gets the total. When n is a multiple of 8 a warp reads 256 elements per step as 16-byte chunks,
// one contiguous 512-byte read; otherwise it falls back to scalar loads.
__device__ __forceinline__ float warp_dot(const bf16* row, const bf16* x, int n, int lane) {
  float acc = 0.0f;
  if (n % 8 == 0) {
    for (int c = lane * 8; c < n; c += 256) {
      const Bf16x8 a = *reinterpret_cast<const Bf16x8*>(row + c);
      const Bf16x8 b = *reinterpret_cast<const Bf16x8*>(x + c);
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const float2 fa = __bfloat1622float2(a.v[j]);
        const float2 fb = __bfloat1622float2(b.v[j]);
        acc += fa.x * fb.x + fa.y * fb.y;
      }
    }
  } else {
    for (int c = lane; c < n; c += 32) acc += lx::to_f32(row[c]) * lx::to_f32(x[c]);
  }
  return lx::warp_sum(acc);
}

// Routing: softmax statistics, then the top-k picks, then the weights over the picks. One warp per token; the per-row
// arithmetic lives in lx::route (route.cuh), shared with the router's fused epilogue.
__global__ void moe_route_kernel(const float* __restrict__ logits, const float* __restrict__ bias,
                                 int32_t* __restrict__ ids, float* __restrict__ weights,
                                 float* __restrict__ entropy, float* __restrict__ confidence, int T, int E,
                                 int top_k) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ int picked[kRouteWarps][kMaxTopK];
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int t = blockIdx.x * kRouteWarps + warp;
  if (t >= T) return;  // whole warp returns together, so the shuffles below stay full-warp
  const float* row = logits + static_cast<int64_t>(t) * E;
  lx::route::row_stats(row, E, lane, entropy + t, confidence + t);
  lx::route::row_picks(row, bias, E, lane, top_k, picked[warp]);
  lx::route::row_weights(row, picked[warp], top_k, lane, ids + static_cast<int64_t>(t) * top_k,
                         weights + static_cast<int64_t>(t) * top_k);
}

// The router's GEMV and its routing in one launch (decode-sized steps: M <= 8 rows). Block n computes logit n of every
// row, with the GEMV's partials and warp reduction as lx_spec_gemv_f32 has them, and writes it to logits [M, E]. The
// last block to finish (a counter, as the thread-fence reductions elsewhere in the tree) stages the logits in shared
// memory and routes them with moe_route's arithmetic. For M <= 4 each row has two warps: warp 2r makes the statistics
// (entropy, confidence) and warp 2r + 1 the picks and weights. Above that one warp routes a row whole.
constexpr int kRouterThreads = 256;  // the GEMV's block: 8 warps
constexpr int kRouterRows = 8;       // decode-sized steps, as lx_spec_gemv_f32
constexpr int kRouterMaxE = 1024;    // the epilogue stages M * E logits in shared memory

template <int J>
__global__ void __launch_bounds__(kRouterThreads) router_route_kernel(
    const float* __restrict__ x, int ld_x, const float* __restrict__ w, const float* __restrict__ bias,
    float* __restrict__ logits, const float* __restrict__ selection_bias, int32_t* __restrict__ ids,
    float* __restrict__ weights, float* __restrict__ entropy, float* __restrict__ confidence,
    int32_t* __restrict__ ticket, int M, int K, int top_k) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ float part[kRouterThreads / 32][kRouterRows];
  __shared__ float rows[kRouterRows][kRouterMaxE];
  __shared__ float bsm[kRouterMaxE];  // the selection bias, staged once per block for the picks
  __shared__ int picked[kRouterRows][kMaxTopK];
  __shared__ int last;
  const int E = gridDim.x;
  const int n = blockIdx.x;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;

  if (selection_bias)
    for (int i = threadIdx.x; i < E; i += kRouterThreads) bsm[i] = selection_bias[i];
  float acc[kRouterRows];
  lx::route::gemv_partials<J, kRouterRows>(x, ld_x, w + static_cast<int64_t>(n) * K, M, K, threadIdx.x, acc);
#pragma unroll
  for (int m = 0; m < kRouterRows; ++m)
    if (m < M) acc[m] = lx::warp_sum(acc[m]);  // rows past M are never read
  if (lane == 0) {
#pragma unroll
    for (int m = 0; m < kRouterRows; ++m) part[warp][m] = acc[m];
  }
  __syncthreads();
  if (threadIdx.x < M) {
    float sum = 0.0f;
#pragma unroll
    for (int i = 0; i < kRouterThreads / 32; ++i) sum += part[i][threadIdx.x];
    if (bias) sum += bias[n];
    logits[threadIdx.x * E + n] = sum;
  }
  // Publish this block's logits, then count it. The block that counts the last one routes every row.
  __threadfence();
  __syncthreads();
  if (threadIdx.x == 0) last = atomicAdd(ticket, 1) == E - 1;
  __syncthreads();
  if (!last) return;

  __threadfence();
  if (threadIdx.x == 0) *ticket = 0;  // the counter is back at zero for the next router
  for (int i = threadIdx.x; i < M * E; i += kRouterThreads) rows[i / E][i % E] = __ldcg(logits + i);
  __syncthreads();
  const bool paired = M <= kRouterRows / 2;
  const int r = paired ? warp >> 1 : warp;
  if (r < M) {
    const float* row = rows[r];
    if (!paired || (warp & 1) == 0) lx::route::row_stats(row, E, lane, entropy + r, confidence + r);
    if (!paired || (warp & 1) == 1) {
      if (top_k == 4) lx::route::top_picks<4>(row, selection_bias ? bsm : nullptr, E, lane, top_k, picked[r]);
      else lx::route::row_picks(row, selection_bias, E, lane, top_k, picked[r]);
      lx::route::row_weights(row, picked[r], top_k, lane, ids + static_cast<int64_t>(r) * top_k,
                             weights + static_cast<int64_t>(r) * top_k);
    }
  }
}

// Kernel 1. Block b covers rows 8b..8b+7 of pair p (grid = row_blocks per pair); warp w takes row
// i = 8b + w. The pair's expert is ids[p] and its token's latent is staged in shared memory once
// per block, since every warp reads all of it.
__global__ void moe_intermediate_kernel(const bf16* __restrict__ latent, const int32_t* __restrict__ ids,
                                        const bf16* __restrict__ gate_up, bf16* __restrict__ inter, int L,
                                        int I, int top_k, int row_blocks) {
  lx::pdl_trigger();
  lx::pdl_wait();
  extern __shared__ __align__(16) bf16 x[];
  const int64_t p = static_cast<int64_t>(blockIdx.x) / row_blocks;
  const int i = static_cast<int>(blockIdx.x % row_blocks) * kWarps + (threadIdx.x >> 5);
  const int64_t t = p / top_k;
  for (int c = threadIdx.x; c < L; c += kThreads) x[c] = latent[t * L + c];
  __syncthreads();
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  // gate_up[e] is [2I, L]: rows 0..I-1 are the gate, rows I..2I-1 the up projection.
  const bf16* w = gate_up + static_cast<int64_t>(ids[p]) * 2 * I * L;
  const float gate = warp_dot(w + static_cast<int64_t>(i) * L, x, L, lane);
  const float up = warp_dot(w + static_cast<int64_t>(I + i) * L, x, L, lane);
  // The intermediate is rounded to bf16 before the down projection, as the reference does.
  if (lane == 0) inter[p * I + i] = lx::from_f32<bf16>(lx::silu(gate) * up);
}

// Kernel 2. Block b covers output rows 8b..8b+7 of token t (grid = row_blocks per token); warp w
// takes row l = 8b + w. The token's top_k intermediates are staged in shared memory, and the
// weighted sum over k is accumulated in registers, so out[t, l] is written exactly once.
__global__ void moe_down_kernel(const bf16* __restrict__ inter, const int32_t* __restrict__ ids,
                                const float* __restrict__ weights, const bf16* __restrict__ down,
                                float* __restrict__ out, int L, int I, int top_k, int row_blocks) {
  lx::pdl_trigger();
  lx::pdl_wait();
  extern __shared__ __align__(16) bf16 h[];
  const int64_t t = static_cast<int64_t>(blockIdx.x) / row_blocks;
  const int l = static_cast<int>(blockIdx.x % row_blocks) * kWarps + (threadIdx.x >> 5);
  const bf16* src = inter + t * top_k * I;
  for (int c = threadIdx.x; c < top_k * I; c += kThreads) h[c] = src[c];
  __syncthreads();
  if (l >= L) return;
  const int lane = threadIdx.x & 31;
  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int64_t p = t * top_k + k;
    // down[e] is [L, I]: row l is the projection of the intermediate back to latent coordinate l.
    const bf16* row = down + (static_cast<int64_t>(ids[p]) * L + l) * I;
    acc += weights[p] * warp_dot(row, h + k * I, I, lane);
  }
  if (lane == 0) out[t * L + l] = acc;
}

// summary[t, r] = sum_k weights[t, k] * expert_emb[ids[t, k], r]. One thread per output; the
// token's ids and weights are the same for all threads of the token, so they broadcast.
__global__ void moe_summary_kernel(const int32_t* __restrict__ ids, const float* __restrict__ weights,
                                   const bf16* __restrict__ emb, float* __restrict__ summary, int64_t n, int R,
                                   int top_k) {
  lx::pdl_trigger();
  lx::pdl_wait();
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const int64_t t = idx / R;
  const int r = static_cast<int>(idx % R);
  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int64_t p = t * top_k + k;
    acc += weights[p] * lx::to_f32(emb[static_cast<int64_t>(ids[p]) * R + r]);
  }
  summary[idx] = acc;
}

// moe_summary_kernel, then history = a * history + b * summary on the same element: the history update reads
// nothing but its own summary, so it rides in the same launch (the separate axpby after it did the same arithmetic).
__global__ void moe_summary_history_kernel(const int32_t* __restrict__ ids, const float* __restrict__ weights,
                                          const bf16* __restrict__ emb, float* __restrict__ summary,
                                          float* __restrict__ history, float a, float b, int64_t n, int R,
                                          int top_k) {
  const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= n) return;
  const int64_t t = idx / R;
  const int r = static_cast<int>(idx % R);
  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int64_t p = t * top_k + k;
    acc += weights[p] * lx::to_f32(emb[static_cast<int64_t>(ids[p]) * R + r]);
  }
  summary[idx] = acc;
  history[idx] = a * history[idx] + b * acc;
}

}  // namespace

extern "C" int lx_moe_route(const float* logits, const float* selection_bias, int32_t* ids, float* weights,
                            float* entropy, float* confidence, int T, int E, int top_k, void* stream) {
  if (T <= 0) return 0;
  if (top_k < 1 || top_k > kMaxTopK || top_k > E) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  lx::launch_pdl(moe_route_kernel, (T + kRouteWarps - 1) / kRouteWarps, kRouteWarps * 32, 0, s, logits, selection_bias, ids, weights, entropy, confidence, T, E, top_k);
  return lx::launch_status();
}

namespace {
template <int J, typename... A>
void launch_router_route(dim3 grid, cudaStream_t s, A... args) {
  lx::launch_pdl(router_route_kernel<J>, grid, dim3(kRouterThreads), 0, s, args...);
}
}  // namespace

extern "C" int lx_router_route(const float* x, int ld_x, const float* w, const float* bias, float* logits,
                               const float* selection_bias, int32_t* ids, float* weights, float* entropy,
                               float* confidence, int32_t* ticket, int M, int E, int K, int top_k, void* stream) {
  if (M <= 0) return 0;
  if (M > kRouterRows || E <= 0 || E > kRouterMaxE || K <= 0 || ld_x < K || top_k < 1 || top_k > kMaxTopK ||
      top_k > E || !x || !w || !logits || !ids || !weights || !entropy || !confidence || !ticket)
    return static_cast<int>(cudaErrorInvalidValue);
  const dim3 grid(E);
  auto s = static_cast<cudaStream_t>(stream);
  switch ((K + kRouterThreads - 1) / kRouterThreads) {
    case 1: launch_router_route<1>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 2: launch_router_route<2>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 3: launch_router_route<3>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 4: launch_router_route<4>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 5: launch_router_route<5>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 6: launch_router_route<6>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 7: launch_router_route<7>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    case 8: launch_router_route<8>(grid, s, x, ld_x, w, bias, logits, selection_bias, ids, weights, entropy, confidence, ticket, M, K, top_k); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return lx::launch_status();
}

extern "C" int lx_moe_experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                              const void* down, float* out, int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return 0;
  const size_t smem_latent = sizeof(bf16) * static_cast<size_t>(L);
  const size_t smem_inter = sizeof(bf16) * static_cast<size_t>(top_k) * static_cast<size_t>(I);
  if (L <= 0 || I <= 0 || top_k < 1 || smem_latent > kMaxSmem || smem_inter > kMaxSmem)
    return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);

  const int64_t pairs = static_cast<int64_t>(T) * top_k;
  bf16* inter = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&inter), pairs * I * sizeof(bf16), s);
  if (err != cudaSuccess) return static_cast<int>(err);

  const int row_blocks_a = (I + kWarps - 1) / kWarps;
  lx::launch_pdl(moe_intermediate_kernel, static_cast<unsigned int>(row_blocks_a * pairs), kThreads, smem_latent, s, static_cast<const bf16*>(latent), ids, static_cast<const bf16*>(gate_up), inter, L, I, top_k, row_blocks_a);

  const int row_blocks_b = (L + kWarps - 1) / kWarps;
  lx::launch_pdl(moe_down_kernel, static_cast<unsigned int>(T * row_blocks_b), kThreads, smem_inter, s, inter, ids, weights, static_cast<const bf16*>(down), out, L, I, top_k, row_blocks_b);

  cudaFreeAsync(inter, s);
  return lx::launch_status();
}

extern "C" int lx_moe_summary(const int32_t* ids, const float* weights, const void* expert_emb, float* summary,
                              int T, int R, int top_k, void* stream) {
  if (T <= 0 || R <= 0) return 0;
  const int64_t n = static_cast<int64_t>(T) * R;
  auto s = static_cast<cudaStream_t>(stream);
  lx::launch_pdl(moe_summary_kernel, static_cast<unsigned int>((n + kThreads - 1) / kThreads), kThreads, 0, s, ids, weights, static_cast<const bf16*>(expert_emb), summary, n, R, top_k);
  return lx::launch_status();
}

extern "C" int lx_moe_summary_history(const int32_t* ids, const float* weights, const void* expert_emb, float* summary,
                                      float* history, float a, float b, int T, int R, int top_k, void* stream) {
  if (T <= 0 || R <= 0) return 0;
  const int64_t n = static_cast<int64_t>(T) * R;
  auto s = static_cast<cudaStream_t>(stream);
  moe_summary_history_kernel<<<static_cast<unsigned int>((n + kThreads - 1) / kThreads), kThreads, 0, s>>>(
      ids, weights, static_cast<const bf16*>(expert_emb), summary, history, a, b, n, R, top_k);
  return lx::launch_status();
}
