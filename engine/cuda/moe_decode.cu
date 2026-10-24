// Few-token latent mixture of experts on any weight format (kernels.h, lx_moe_experts_decode).
//
// The contract of lx_moe_experts, for T <= 8 tokens, with the expert weights read in their stored format
// (wformat.cuh): decode is bound by the expert weights, so each pass streams the chosen experts' rows
// once, one warp per row, 8 bytes per lane per step. Two stages, no atomics, every output written once:
//   1. gate/up: one warp per intermediate row i of a (token, k) pair. The warp reads gate row i and up
//      row I + i of the pair's expert and writes bf16(silu(gate) * up) to the pair's row of the workspace.
//   2. down: one warp per output row l of a token, summed over the token's top_k experts in registers.
// The intermediates are bf16, as the contract says; the per-row factors (int8 / fp8 row scales, nvfp4
// globals, per expert for a stack that has them) apply once per row. Each stage reads its own stack's
// format, so the gate/up and down stacks may differ.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/wformat.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;  // 8 warps per block, one row per warp
constexpr int kWarps = kThreads / 32;
constexpr int kMaxT = 8;       // the decode contract: a few tokens
constexpr int kMaxTopK = 8;

template <int F>
constexpr int kVal = lx::WordFormat<F>::kValues;  // values per lane word

// A lane's word of one row and step, with the block scale of its values (NVFP4; 1 otherwise).
struct Lane {
  uint2 word;
  float scale;
};

// Row n's word at element k0 (a multiple of the format's values per word): zero past K.
template <int F>
__device__ __forceinline__ Lane load_word(const lx_weight& w, int64_t n, int64_t k0, int64_t K) {
  Lane out{make_uint2(0, 0), 1.0f};
  if (k0 >= K) return out;
  out.word = __ldg(static_cast<const uint2*>(w.data) + (n * K + k0) / kVal<F>);
  if constexpr (F == LX_W_NVFP4) out.scale = lx::e4m3_to_f32(__ldg(static_cast<const uint8_t*>(w.scale) + n * (K / 16) + k0 / 16));
  return out;
}

// kV bf16 activations at p as f32 (kV is 4, 8 or 16: whole 8-byte loads).
template <int kV>
__device__ __forceinline__ void load_x(const bf16* p, float* f) {
  const uint2* q = reinterpret_cast<const uint2*>(p);
#pragma unroll
  for (int i = 0; i < kV / 4; ++i) {
    const uint2 v = __ldg(q + i);
    f[4 * i] = __uint_as_float(v.x << 16);
    f[4 * i + 1] = __uint_as_float(v.x & 0xffff0000u);
    f[4 * i + 2] = __uint_as_float(v.y << 16);
    f[4 * i + 3] = __uint_as_float(v.y & 0xffff0000u);
  }
}

// The dot product of one lane word with the activations xf (kV values).
template <int F>
__device__ __forceinline__ float dot_word(const Lane& a, const float* xf) {
  float wf[kVal<F>];
  lx::WordFormat<F>::decode(a.word, a.scale, wf);
  float acc = 0.0f;
#pragma unroll
  for (int t = 0; t < kVal<F>; ++t) acc = fmaf(wf[t], xf[t], acc);
  return acc;
}

// Stage 1. Grid: pairs * rows_per_pair blocks; warp w of block b is intermediate row
// (b % rows_per_pair) * kWarps + w of pair b / rows_per_pair.
template <int F>
__global__ void __launch_bounds__(kThreads) moe_gate_up_kernel(const bf16* __restrict__ latent,
                                                              const int32_t* __restrict__ ids,
                                                              const lx_weight gate_up, bf16* __restrict__ inter,
                                                              int L, int I, int top_k, int rows_per_pair) {
  constexpr int kV = kVal<F>;
  const int p = blockIdx.x / rows_per_pair;
  const int i = (blockIdx.x % rows_per_pair) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];
  // gate_up[e] is [2I, L]: row i the gate and row I + i the up projection.
  const lx_weight w = lx::expert_slice(gate_up, e, 2 * I, L);
  const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;

  float g = 0.0f, u = 0.0f;
  for (int64_t k0 = static_cast<int64_t>(lane) * kV; k0 < L; k0 += 32 * kV) {
    float xf[kV];
    load_x<kV>(x + k0, xf);
    g += dot_word<F>(load_word<F>(w, i, k0, L), xf);
    u += dot_word<F>(load_word<F>(w, static_cast<int64_t>(I) + i, k0, L), xf);
  }
  g = lx::warp_sum(g);
  u = lx::warp_sum(u);
  if (lane == 0) {
    const float sg = lx::WordFormat<F>::row_factor(w, i);
    const float su = lx::WordFormat<F>::row_factor(w, static_cast<int64_t>(I) + i);
    inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// Stage 2. Grid: tokens * ceil(L / kWarps) blocks, with the token index fastest. Warp w of block b is
// output row (b / T) * kWarps + w of token b % T.
template <int F>
__global__ void __launch_bounds__(kThreads) moe_down_kernel(const bf16* __restrict__ inter,
                                                           const int32_t* __restrict__ ids,
                                                           const float* __restrict__ weights, const lx_weight down,
                                                           float* __restrict__ out, int T, int L, int I, int top_k) {
  constexpr int kV = kVal<F>;
  const int t = blockIdx.x % T;
  const int l = (blockIdx.x / T) * kWarps + (threadIdx.x >> 5);
  if (l >= L) return;  // whole warps: the row index is per warp
  const int lane = threadIdx.x & 31;

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int p = t * top_k + k;
    const int e = ids[p];
    const lx_weight w = lx::expert_slice(down, e, L, I);
    const float ck = weights[p] * lx::WordFormat<F>::row_factor(w, l);
    const bf16* h = inter + static_cast<int64_t>(p) * I;
    float dot = 0.0f;
    for (int64_t k0 = static_cast<int64_t>(lane) * kV; k0 < I; k0 += 32 * kV) {
      float xf[kV];
      load_x<kV>(h + k0, xf);
      dot += dot_word<F>(load_word<F>(w, l, k0, I), xf);
    }
    acc = fmaf(ck, dot, acc);
  }
  acc = lx::warp_sum(acc);
  if (lane == 0) out[static_cast<int64_t>(t) * L + l] = acc;
}

// The stage 1 kernel for a format, and the stage 2 kernel for a format, as launches.
template <int F>
cudaError_t launch_gate_up(const bf16* latent, const int32_t* ids, const lx_weight& w, bf16* inter, int L, int I,
                           int top_k, int P, cudaStream_t s) {
  const int rows_per_pair = (I + kWarps - 1) / kWarps;
  moe_gate_up_kernel<F><<<static_cast<unsigned int>(P * rows_per_pair), kThreads, 0, s>>>(latent, ids, w, inter, L,
                                                                                         I, top_k, rows_per_pair);
  return cudaGetLastError();
}

template <int F>
cudaError_t launch_down(const bf16* inter, const int32_t* ids, const float* weights, const lx_weight& w, float* out,
                        int T, int L, int I, int top_k, cudaStream_t s) {
  const int blocks_per_token = (L + kWarps - 1) / kWarps;
  moe_down_kernel<F><<<static_cast<unsigned int>(T * blocks_per_token), kThreads, 0, s>>>(inter, ids, weights, w,
                                                                                         out, T, L, I, top_k);
  return cudaGetLastError();
}

// Whether a format's word covers the row length: every row is whole words.
bool whole_words(const lx_weight& w, int64_t K) {
  switch (w.type) {
    case LX_W_BF16: return K % lx::WordFormat<LX_W_BF16>::kValues == 0;
    case LX_W_INT8: return K % lx::WordFormat<LX_W_INT8>::kValues == 0;
    case LX_W_FP8: return K % lx::WordFormat<LX_W_FP8>::kValues == 0;
    case LX_W_NVFP4: return K % lx::WordFormat<LX_W_NVFP4>::kValues == 0;
    default: return false;
  }
}

}  // namespace

extern "C" int lx_moe_experts_decode(const void* latent, const int32_t* ids, const float* weights,
                                     const lx_weight* gate_up, const lx_weight* down, float* out, void* workspace,
                                     int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return 0;
  if (T > kMaxT || top_k < 1 || top_k > kMaxTopK || L <= 0 || I <= 0) return static_cast<int>(cudaErrorInvalidValue);
  if (!latent || !ids || !weights || !gate_up || !down || !out || !workspace) return static_cast<int>(cudaErrorInvalidValue);
  if (!gate_up->data || !down->data) return static_cast<int>(cudaErrorInvalidValue);
  if (gate_up->type == LX_W_BF16X || down->type == LX_W_BF16X) {
    // Lossless stacks (bf16x.cu) run their own kernels; both stacks must be encoded, a mixed pair has none.
    if (gate_up->type != LX_W_BF16X || down->type != LX_W_BF16X) return static_cast<int>(cudaErrorNotSupported);
    return lx_moe_experts_bf16x(latent, ids, weights, gate_up->data, down->data, out, workspace, T, L, I, top_k,
                                stream);
  }
  if (!whole_words(*gate_up, L) || !whole_words(*down, I)) return static_cast<int>(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(workspace) & 15) != 0) return static_cast<int>(cudaErrorInvalidValue);

  auto s = static_cast<cudaStream_t>(stream);
  const int P = T * top_k;
  auto* inter = static_cast<bf16*>(workspace);
  const auto* lat = static_cast<const bf16*>(latent);
  cudaError_t err = cudaSuccess;
  switch (gate_up->type) {
    case LX_W_BF16: err = launch_gate_up<LX_W_BF16>(lat, ids, *gate_up, inter, L, I, top_k, P, s); break;
    case LX_W_INT8: err = launch_gate_up<LX_W_INT8>(lat, ids, *gate_up, inter, L, I, top_k, P, s); break;
    case LX_W_FP8: err = launch_gate_up<LX_W_FP8>(lat, ids, *gate_up, inter, L, I, top_k, P, s); break;
    case LX_W_NVFP4: err = launch_gate_up<LX_W_NVFP4>(lat, ids, *gate_up, inter, L, I, top_k, P, s); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  if (err != cudaSuccess) return static_cast<int>(err);
  switch (down->type) {
    case LX_W_BF16: err = launch_down<LX_W_BF16>(inter, ids, weights, *down, out, T, L, I, top_k, s); break;
    case LX_W_INT8: err = launch_down<LX_W_INT8>(inter, ids, weights, *down, out, T, L, I, top_k, s); break;
    case LX_W_FP8: err = launch_down<LX_W_FP8>(inter, ids, weights, *down, out, T, L, I, top_k, s); break;
    case LX_W_NVFP4: err = launch_down<LX_W_NVFP4>(inter, ids, weights, *down, out, T, L, I, top_k, s); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return static_cast<int>(err);
}

// The workspace: the bf16 intermediate of every (token, k) pair, rounded up to 256 bytes.
extern "C" int64_t lx_moe_decode_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t bytes = 2 * static_cast<int64_t>(T) * top_k * I;
  const int64_t lossless = lx_moe_bf16x_workspace_bytes(T, I, top_k);  // the same caller buffer serves both
  return std::max((bytes + 255) / 256 * 256, lossless);
}
