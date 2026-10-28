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
//
// Pipelining. A warp's row is a short run of steps (a lane's word per step), and the old loops issued each
// step's loads only after the previous step's math, so a warp had one or two loads in flight and the
// memory pipe idled between them. Each loop now keeps a ring of kDepth steps in flight: the loads of step
// j + kDepth go out as soon as step j is consumed, so the warp always has kDepth steps requested. The
// arithmetic is unchanged: the same products, in the same order, into the same accumulators, so every
// output keeps its bits (the batch invariance and the spec check depend on it).
//
// The down stage is one flattened sequence of (expert k, step s) items per warp. Its weights and row
// factors come from the routing's ids and weights, which the gate/up stage does not change, so the stage
// is launched programmatically: its prologue issues the first ring of weight loads while the gate/up
// stage still streams, and only the activations (the gate/up stage's output) wait for that stage to finish.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <type_traits>

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

// Steps in flight per warp. A step is one 8-byte word per row the loop reads, per lane. FP8 and INT8 keep four,
// which their memory pipe needs with a few warps per SM. NVFP4 and BF16 keep one: a ring of two cost NVFP4 8 to
// 17% at 2 to 8 streams, and a ring of four cost BF16 5% at one stream. Their words are already wide enough, and
// the ring raised their registers and cut the blocks per SM.
template <int F>
constexpr int kDepth = (F == LX_W_FP8 || F == LX_W_INT8) ? 4 : 1;

// Compile-time static loop: f(integral_constant<int, i>) for i = 0 .. N - 1, so i indexes the ring's slots
// statically (the slots live in registers).
template <int N, int I = 0, typename Fn>
__device__ __forceinline__ void static_for(Fn&& f) {
  if constexpr (I < N) {
    f(std::integral_constant<int, I>{});
    static_for<N, I + 1>(f);
  }
}

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

// kV bf16 activations at p, as they sit in memory (kV is 4, 8 or 16: whole 8-byte loads).
template <int kV>
__device__ __forceinline__ void load_x(const bf16* p, uint2 (&r)[kV / 4]) {
  const uint2* q = reinterpret_cast<const uint2*>(p);
#pragma unroll
  for (int i = 0; i < kV / 4; ++i) r[i] = __ldg(q + i);
}

// The loaded activations as f32, the same values load_x's words always held.
template <int kV>
__device__ __forceinline__ void unpack_x(const uint2 (&r)[kV / 4], float (&f)[kV]) {
#pragma unroll
  for (int i = 0; i < kV / 4; ++i) {
    f[4 * i] = __uint_as_float(r[i].x << 16);
    f[4 * i + 1] = __uint_as_float(r[i].x & 0xffff0000u);
    f[4 * i + 2] = __uint_as_float(r[i].y << 16);
    f[4 * i + 3] = __uint_as_float(r[i].y & 0xffff0000u);
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

// ---- stage 1: gate and up ----------------------------------------------------------------------------

// One step of a gate/up row pair: the gate word and the up word of this lane, and the activations.
template <int F>
struct GateUpStep {
  Lane g, u;
  uint2 x[kVal<F> / 4];
};

template <int F>
__device__ __forceinline__ void issue_gate_up(const lx_weight& w, int64_t i, int I, int L, const bf16* x, int64_t k0,
                                              GateUpStep<F>& s) {
  s.g = load_word<F>(w, i, k0, L);
  s.u = load_word<F>(w, static_cast<int64_t>(I) + i, k0, L);
  load_x<kVal<F>>(x + k0, s.x);
}

// Stage 1. Grid: pairs * rows_per_pair blocks; warp w of block b is intermediate row
// (b % rows_per_pair) * kWarps + w of pair b / rows_per_pair.
template <int F>
__global__ void __launch_bounds__(kThreads) moe_gate_up_kernel(const bf16* __restrict__ latent,
                                                              const int32_t* __restrict__ ids,
                                                              const lx_weight gate_up, bf16* __restrict__ inter,
                                                              int L, int I, int top_k, int rows_per_pair) {
  constexpr int kV = kVal<F>;
  constexpr int D = kDepth<F>;
  // The down stage may start its prologue now: it reads only ids and weights before it waits for this stage.
  lx::pdl_trigger();
  const int p = blockIdx.x / rows_per_pair;
  const int i = (blockIdx.x % rows_per_pair) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;  // whole warps: the row index is per warp
  const int lane = threadIdx.x & 31;
  const int e = ids[p];
  // gate_up[e] is [2I, L]: row i the gate and row I + i the up projection.
  const lx_weight w = lx::expert_slice(gate_up, e, 2 * I, L);
  const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;

  // This lane's steps: k0 = lane_k + s * stride, for every s with k0 < L.
  const int64_t lane_k = static_cast<int64_t>(lane) * kV, stride = 32 * kV;
  const int count = lane_k < L ? static_cast<int>((L - lane_k + stride - 1) / stride) : 0;

  GateUpStep<F> ring[D];
  static_for<D>([&](auto d) {
    if (d < count) issue_gate_up<F>(w, i, I, L, x, lane_k + d * stride, ring[d]);
  });
  float g = 0.0f, u = 0.0f;
  for (int j0 = 0; j0 < count; j0 += D) {
    static_for<D>([&](auto d) {
      if (j0 + d < count) {
        float xf[kV];
        unpack_x<kV>(ring[d].x, xf);
        g += dot_word<F>(ring[d].g, xf);
        u += dot_word<F>(ring[d].u, xf);
        if (j0 + d + D < count) issue_gate_up<F>(w, i, I, L, x, lane_k + (j0 + d + D) * stride, ring[d]);
      }
    });
  }
  g = lx::warp_sum(g);
  u = lx::warp_sum(u);
  if (lane == 0) {
    const float sg = lx::WordFormat<F>::row_factor(w, i);
    const float su = lx::WordFormat<F>::row_factor(w, static_cast<int64_t>(I) + i);
    inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// ---- stage 2: down -----------------------------------------------------------------------------------

// One item of a warp's down sequence, the weight half: the word of the down row, and the product's factor
// (the pair's route weight times the row factor of the expert's row). They come from ids, weights and the
// weights alone, so they can be loaded before the gate/up stage's output is ready.
template <int F>
struct DownWeight {
  Lane w;
  float ck;
};

// The item's activations: the intermediate of pair p, the same step's values (after the gate/up stage).
template <int F>
using DownAct = uint2[kVal<F> / 4];

// Item j = k * steps + s of a warp's sequence: expert k of the token's top_k, step s of its row.
// `steps` is the step count of the down row (the same for every lane); this lane has words in step s only
// when lane_k + s * stride < I, and the words past the row are zero.
template <int F>
__device__ __forceinline__ void issue_down_weight(const lx_weight& down, const int32_t* ids, const float* weights,
                                                  int64_t p_base, int64_t l, int L, int I, int64_t lane_k, int steps,
                                                  int j, DownWeight<F>& out) {
  constexpr int kV = kVal<F>;
  const int k = j / steps, s = j - k * steps;
  const int64_t p = p_base + k;
  const int e = ids[p];
  const lx_weight w = lx::expert_slice(down, e, L, I);
  out.ck = weights[p] * lx::WordFormat<F>::row_factor(w, l);
  out.w = load_word<F>(w, l, lane_k + static_cast<int64_t>(s) * 32 * kV, I);
}

template <int F>
__device__ __forceinline__ void issue_down_act(const bf16* inter, int64_t p_base, int I, int64_t lane_k, int steps,
                                               int j, uint2 (&x)[kVal<F> / 4]) {
  constexpr int kV = kVal<F>;
  const int k = j / steps, s = j - k * steps;
  const int64_t k0 = lane_k + static_cast<int64_t>(s) * 32 * kV;
  if (k0 < I) {
    load_x<kV>(inter + (p_base + k) * I + k0, x);
  } else {
    for (int i = 0; i < kV / 4; ++i) x[i] = make_uint2(0, 0);
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
  constexpr int D = kDepth<F>;
  const int t = blockIdx.x % T;
  const int l = (blockIdx.x / T) * kWarps + (threadIdx.x >> 5);
  if (l >= L) return;  // whole warps: the row index is per warp
  const int lane = threadIdx.x & 31;

  const int64_t lane_k = static_cast<int64_t>(lane) * kV, stride = 32 * kV;
  const int steps = (I + stride - 1) / stride;  // steps of a down row: the same for every lane
  const int n = top_k * steps;                   // items of the warp's sequence
  const int64_t p_base = static_cast<int64_t>(t) * top_k;

  // The weight halves of the first ring go out before the wait: they depend on the routing only.
  DownWeight<F> wring[D];
  DownAct<F> xring[D];
  static_for<D>([&](auto d) {
    if (d < n) issue_down_weight<F>(down, ids, weights, p_base, l, L, I, lane_k, steps, d, wring[d]);
  });
  // The gate/up stage's intermediates are read from here on.
  lx::pdl_wait();
  static_for<D>([&](auto d) {
    if (d < n) issue_down_act<F>(inter, p_base, I, lane_k, steps, d, xring[d]);
  });

  float acc = 0.0f;
  float dot = 0.0f;
  for (int j0 = 0; j0 < n; j0 += D) {
    static_for<D>([&](auto d) {
      if (j0 + d < n) {
        const int s = (j0 + d) % steps;
        if (lane_k + static_cast<int64_t>(s) * stride < I) {  // this lane has a word in this step
          float xf[kV];
          unpack_x<kV>(xring[d], xf);
          dot += dot_word<F>(wring[d].w, xf);
        }
        // The last step of expert k: its dot product joins the pair's sum, in k order.
        if (s == steps - 1) {
          acc = fmaf(wring[d].ck, dot, acc);
          dot = 0.0f;
        }
        if (j0 + d + D < n) {
          issue_down_weight<F>(down, ids, weights, p_base, l, L, I, lane_k, steps, j0 + d + D, wring[d]);
          issue_down_act<F>(inter, p_base, I, lane_k, steps, j0 + d + D, xring[d]);
        }
      }
    });
  }
  acc = lx::warp_sum(acc);
  if (lane == 0) out[static_cast<int64_t>(t) * L + l] = acc;
}

// ---- launches ----------------------------------------------------------------------------------------

// The down stage, launched programmatically on the gate/up stage (below sm_90 an ordinary launch, in stream
// order): it may start once that stage has triggered, and its own wait holds back the intermediates' reads.
template <int F>
cudaError_t launch_down(const bf16* inter, const int32_t* ids, const float* weights, const lx_weight& w, float* out,
                        int T, int L, int I, int top_k, cudaStream_t s) {
  const int blocks_per_token = (L + kWarps - 1) / kWarps;
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(static_cast<unsigned int>(T * blocks_per_token));
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cfg.attrs = &attr;
  cfg.numAttrs = lx::pdl_available() ? 1 : 0;
  cudaLaunchKernelEx(&cfg, moe_down_kernel<F>, inter, ids, weights, w, out, T, L, I, top_k);
  return cudaGetLastError();
}

// The stage 1 kernel for a format, as a launch.
template <int F>
cudaError_t launch_gate_up(const bf16* latent, const int32_t* ids, const lx_weight& w, bf16* inter, int L, int I,
                           int top_k, int P, cudaStream_t s) {
  const int rows_per_pair = (I + kWarps - 1) / kWarps;
  moe_gate_up_kernel<F><<<static_cast<unsigned int>(P * rows_per_pair), kThreads, 0, s>>>(latent, ids, w, inter, L,
                                                                                         I, top_k, rows_per_pair);
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
