// Latent mixture of experts with int8 expert weights (kernels.h, lx_moe_experts_w8).
//
// The same math as moe.cu, with each expert matrix stored as int8 plus one f32 scale per row.
// Decode is bound by the expert weights, so the two stages are built around the DRAM stream:
//   1. gate/up: one warp per intermediate row i of a (pair, row block) block. The warp reads gate
//      row i and up row i of the pair's expert and writes bf16(silu(gate) * up) to the pair's row.
//   2. down: one warp per output row l of a token, summed over the token's top_k experts in
//      registers. Its routing and down rows do not depend on stage 1, so they are fetched before
//      griddepcontrol.wait (programmatic dependent launch: the kernel starts behind stage 1 and
//      copies its rows into shared memory with cp.async); the wait then covers only the intermediates.
//
// Stage 1 has two forms. Decode-sized calls (up to kSharePairs pairs) give every pair its own warps;
// the few pairs that share an expert read its rows from L2. Larger calls (prefill) share each expert's
// rows: a link pass orders the pairs of every expert, and the first pair of an expert (its lead)
// computes the expert's intermediates for every pair in the chain. Every intermediate and every
// output is written by exactly one warp, in a fixed order: no atomics.
//
// The scale applies after the int8 dot product (sum of q * s is s * sum of q), so the inner loop is
// an int8 to float conversion and an FMA. Rows of a multiple of 16 bytes take 16-byte loads; other
// shapes take scalar paths, which the odd-shape tests exercise.
#include "cuda/common.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;             // 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kMaxTopK = 8;               // stage 2 holds routing weights in registers up to this
constexpr int kMaxVecChunks = 4;          // vector paths cover rows of up to 4 * 512 bytes
constexpr size_t kMaxSmem = 48 * 1024;    // default dynamic shared memory limit
constexpr int kSharePairs = 32;           // above this many pairs, stage 1 shares expert rows
// Register caps (blocks per SM): 64 registers for stage 1, which the decode form fits without spills,
// and 40 for stage 2. Stage 2 holds no registers across the wait, and a tighter stage 1 cap measured
// slower: the spills cost more than the overlap they would allow.
constexpr int kMinBlocksA = 4;
constexpr int kMinBlocksB = 6;

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint4 load16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }

// Dot product of 16 int8 weights (four packed words) with 16 bf16 activations (two uint4). Weight
// element n pairs with activation element n: activation n sits in word n / 2, low half when n is even.
__device__ __forceinline__ float dot16(const uint4& w, const uint4& xa, const uint4& xb) {
  const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
  const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int b = 0; b < 4; ++b) {
      const uint32_t xw = xv[2 * j + (b >> 1)];
      const float x = (b & 1) ? bf16_hi(xw) : bf16_lo(xw);
      acc = fmaf(static_cast<float>(static_cast<int8_t>(wv[j] >> (8 * b))), x, acc);
    }
  }
  return acc;
}

// Byte offset of the link arrays in the workspace: the intermediates, rounded to 16 bytes.
__host__ __device__ inline int64_t chain_offset(int P, int I) {
  return (2 * static_cast<int64_t>(P) * I + 15) / 16 * 16;
}

// Asynchronous 16-byte global to shared copy, used by stage 2 to stage weights before the wait.
__device__ __forceinline__ void cp_async16(void* smem, const void* gmem) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(dst), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Stage 1, decode form, vector path: one pair per block row, NC chunks of 512 bytes per row. Each lane
// owns 16 bytes per chunk, so a warp reads 512 contiguous bytes per load. Requires L % 16 == 0.
template <int NC>
__global__ void __launch_bounds__(kThreads, kMinBlocksA) w8_gate_up_pair_vec(const bf16* __restrict__ latent,
                                                                             const int32_t* __restrict__ ids,
                                                                             const int8_t* __restrict__ gate_up,
                                                                             const float* __restrict__ gate_up_scale,
                                                                             bf16* __restrict__ inter, int L,
                                                                             int I, int top_k, int rows) {
  lx::pdl_trigger();  // stage 2 may start its prologue now
  const int p = blockIdx.x / rows;
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];

  // gate_up[e] is [2I, L]: row i the gate and row I + i the up projection.
  const int64_t gate_row = (static_cast<int64_t>(e) * 2 * I + i) * L;
  uint4 wg[NC], wu[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int col = 512 * c + 16 * lane;
    wg[c] = make_uint4(0, 0, 0, 0);
    wu[c] = make_uint4(0, 0, 0, 0);
    if (col < L) {
      wg[c] = load16(gate_up + gate_row + col);
      wu[c] = load16(gate_up + gate_row + static_cast<int64_t>(I) * L + col);
    }
  }

  const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;
  float g = 0.0f, u = 0.0f;
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int col = 512 * c + 16 * lane;
    if (col < L) {
      const uint4 xa = load16(x + col);
      const uint4 xb = load16(x + col + 8);
      g += dot16(wg[c], xa, xb);
      u += dot16(wu[c], xa, xb);
    }
  }
  g = lx::warp_sum(g);
  u = lx::warp_sum(u);
  if (lane == 0) {
    const float sg = gate_up_scale[static_cast<int64_t>(e) * 2 * I + i];
    const float su = gate_up_scale[static_cast<int64_t>(e) * 2 * I + I + i];
    inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// Stage 1, decode form, scalar path: any row length.
__global__ void __launch_bounds__(kThreads) w8_gate_up_pair_scalar(const bf16* __restrict__ latent,
                                                                   const int32_t* __restrict__ ids,
                                                                   const int8_t* __restrict__ gate_up,
                                                                   const float* __restrict__ gate_up_scale,
                                                                   bf16* __restrict__ inter, int L, int I,
                                                                   int top_k, int rows) {
  lx::pdl_trigger();
  const int p = blockIdx.x / rows;
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];
  const int8_t* gate = gate_up + (static_cast<int64_t>(e) * 2 * I + i) * L;
  const int8_t* up = gate + static_cast<int64_t>(I) * L;
  const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;
  float g = 0.0f, u = 0.0f;
  for (int c = lane; c < L; c += 32) {
    const float xv = lx::to_f32(x[c]);
    g = fmaf(static_cast<float>(gate[c]), xv, g);
    u = fmaf(static_cast<float>(up[c]), xv, u);
  }
  g = lx::warp_sum(g);
  u = lx::warp_sum(u);
  if (lane == 0) {
    const float sg = gate_up_scale[static_cast<int64_t>(e) * 2 * I + i];
    const float su = gate_up_scale[static_cast<int64_t>(e) * 2 * I + I + i];
    inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// Prefill links: for every pair p, the next pair after p that routed to the same expert (next, -1
// when none), and whether p is the first pair of its expert (lead). One thread per pair scans ids,
// which is P int32s, so the pass costs a few microseconds and the stage 1 blocks never scan.
__global__ void __launch_bounds__(kThreads) w8_pair_links(const int32_t* __restrict__ ids, int P,
                                                          int32_t* __restrict__ next, int32_t* __restrict__ lead) {
  const int p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= P) return;
  const int e = ids[p];
  int32_t is_lead = 1, following = -1;
  for (int q = 0; q < P; ++q) {
    if (q == p || ids[q] != e) continue;
    if (q < p) {
      is_lead = 0;
    } else {
      following = q;
      break;
    }
  }
  next[p] = following;
  lead[p] = is_lead;
}

// Stage 1, sharing form, vector path: the lead pair of each expert walks the chain of pairs that
// routed to it, so the expert's gate and up rows are read once for all of them. Non-lead blocks exit.
template <int NC>
__global__ void __launch_bounds__(kThreads, kMinBlocksA) w8_gate_up_chain_vec(const bf16* __restrict__ latent,
                                                                              const int32_t* __restrict__ ids,
                                                                              const int8_t* __restrict__ gate_up,
                                                                              const float* __restrict__ gate_up_scale,
                                                                              bf16* __restrict__ inter,
                                                                              const int32_t* __restrict__ next,
                                                                              const int32_t* __restrict__ lead,
                                                                              int L, int I, int top_k, int rows) {
  lx::pdl_trigger();
  const int p = blockIdx.x / rows;
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (!lead[p] || i >= I) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];

  const int64_t gate_row = (static_cast<int64_t>(e) * 2 * I + i) * L;
  uint4 wg[NC], wu[NC];
#pragma unroll
  for (int c = 0; c < NC; ++c) {
    const int col = 512 * c + 16 * lane;
    wg[c] = make_uint4(0, 0, 0, 0);
    wu[c] = make_uint4(0, 0, 0, 0);
    if (col < L) {
      wg[c] = load16(gate_up + gate_row + col);
      wu[c] = load16(gate_up + gate_row + static_cast<int64_t>(I) * L + col);
    }
  }
  const float sg = gate_up_scale[static_cast<int64_t>(e) * 2 * I + i];
  const float su = gate_up_scale[static_cast<int64_t>(e) * 2 * I + I + i];

  for (int q = p; q >= 0; q = next[q]) {
    const bf16* x = latent + static_cast<int64_t>(q / top_k) * L;
    float g = 0.0f, u = 0.0f;
#pragma unroll
    for (int c = 0; c < NC; ++c) {
      const int col = 512 * c + 16 * lane;
      if (col < L) {
        const uint4 xa = load16(x + col);
        const uint4 xb = load16(x + col + 8);
        g += dot16(wg[c], xa, xb);
        u += dot16(wu[c], xa, xb);
      }
    }
    g = lx::warp_sum(g);
    u = lx::warp_sum(u);
    if (lane == 0) inter[static_cast<int64_t>(q) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// Stage 1, sharing form, scalar path: any row length.
__global__ void __launch_bounds__(kThreads) w8_gate_up_chain_scalar(const bf16* __restrict__ latent,
                                                                    const int32_t* __restrict__ ids,
                                                                    const int8_t* __restrict__ gate_up,
                                                                    const float* __restrict__ gate_up_scale,
                                                                    bf16* __restrict__ inter,
                                                                    const int32_t* __restrict__ next,
                                                                    const int32_t* __restrict__ lead, int L, int I,
                                                                    int top_k, int rows) {
  lx::pdl_trigger();
  const int p = blockIdx.x / rows;
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (!lead[p] || i >= I) return;
  const int lane = threadIdx.x & 31;
  const int e = ids[p];
  const int8_t* gate = gate_up + (static_cast<int64_t>(e) * 2 * I + i) * L;
  const int8_t* up = gate + static_cast<int64_t>(I) * L;
  const float sg = gate_up_scale[static_cast<int64_t>(e) * 2 * I + i];
  const float su = gate_up_scale[static_cast<int64_t>(e) * 2 * I + I + i];

  for (int q = p; q >= 0; q = next[q]) {
    const bf16* x = latent + static_cast<int64_t>(q / top_k) * L;
    float g = 0.0f, u = 0.0f;
    for (int c = lane; c < L; c += 32) {
      const float xv = lx::to_f32(x[c]);
      g = fmaf(static_cast<float>(gate[c]), xv, g);
      u = fmaf(static_cast<float>(up[c]), xv, u);
    }
    g = lx::warp_sum(g);
    u = lx::warp_sum(u);
    if (lane == 0) inter[static_cast<int64_t>(q) * I + i] = lx::from_f32<bf16>(lx::silu(sg * g) * (su * u));
  }
}

// Stage 2, vector path (I % 16 == 0). Block b covers output rows 8b..8b+7 of token t: grid =
// row_blocks * T, with t the fastest index so consecutive blocks reuse the same down rows in L2.
// Each warp copies its top_k down rows into shared memory with asynchronous copies before
// griddepcontrol.wait, so they arrive while stage 1 streams; after the wait only the intermediates
// are read. Shared memory is (2 + kWarps) * top_k * I bytes.
__global__ void __launch_bounds__(kThreads, kMinBlocksB) w8_down_vec(const bf16* __restrict__ inter,
                                                                     const int32_t* __restrict__ ids,
                                                                     const float* __restrict__ weights,
                                                                     const int8_t* __restrict__ down,
                                                                     const float* __restrict__ down_scale,
                                                                     float* __restrict__ out, int T, int L,
                                                                     int I, int top_k) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* h = reinterpret_cast<bf16*>(smem_raw);  // this token's top_k intermediates
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int8_t* rows = reinterpret_cast<int8_t*>(h + top_k * I) + warp * top_k * I;  // this warp's rows
  const int t = blockIdx.x % T;
  const int l = (blockIdx.x / T) * kWarps + warp;
  const bool live = l < L;

  // c[k] folds the routing weight into the row scale: out[l] = sum_k c[k] * dot(q[e_k, l], h[k]).
  float c[kMaxTopK];
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k) {
    c[k] = 0.0f;
    if (k < top_k && live) {
      const int p = t * top_k + k;
      const int64_t row = static_cast<int64_t>(ids[p]) * L + l;
      c[k] = weights[p] * down_scale[row];
      for (int v = lane; v < I / 16; v += 32) cp_async16(rows + k * I + 16 * v, down + row * I + 16 * v);
    }
  }

  lx::pdl_wait();  // stage 1 has written the intermediates
  cp_async_wait_all();              // this thread's rows have landed
  const uint4* src = reinterpret_cast<const uint4*>(inter + static_cast<int64_t>(t) * top_k * I);
  uint4* dst = reinterpret_cast<uint4*>(h);
  for (int v = threadIdx.x; v < top_k * I / 8; v += kThreads) dst[v] = src[v];
  __syncthreads();
  if (!live) return;

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < kMaxTopK; ++k) {
    if (k < top_k) {
      for (int col = 16 * lane; col < I; col += 512) {
        const uint4 xa = *reinterpret_cast<const uint4*>(h + k * I + col);
        const uint4 xb = *reinterpret_cast<const uint4*>(h + k * I + col + 8);
        const uint4 w = *reinterpret_cast<const uint4*>(rows + k * I + col);
        acc += c[k] * dot16(w, xa, xb);
      }
    }
  }
  acc = lx::warp_sum(acc);
  if (lane == 0) out[static_cast<int64_t>(t) * L + l] = acc;
}

// Stage 2, scalar path: any I. Same reduction order as the vector path up to float rounding.
__global__ void __launch_bounds__(kThreads) w8_down_scalar(const bf16* __restrict__ inter,
                                                           const int32_t* __restrict__ ids,
                                                           const float* __restrict__ weights,
                                                           const int8_t* __restrict__ down,
                                                           const float* __restrict__ down_scale,
                                                           float* __restrict__ out, int T, int L, int I,
                                                           int top_k) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  bf16* h = reinterpret_cast<bf16*>(smem_raw);
  const int t = blockIdx.x % T;
  const int l = (blockIdx.x / T) * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool live = l < L;

  lx::pdl_wait();
  for (int v = threadIdx.x; v < top_k * I; v += kThreads) h[v] = inter[static_cast<int64_t>(t) * top_k * I + v];
  __syncthreads();
  if (!live) return;

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int p = t * top_k + k;
    const int64_t row = static_cast<int64_t>(ids[p]) * L + l;
    const float ck = weights[p] * down_scale[row];
    float dot = 0.0f;
    for (int col = lane; col < I; col += 32) {
      dot = fmaf(static_cast<float>(down[row * I + col]), lx::to_f32(h[k * I + col]), dot);
    }
    acc = fmaf(ck, dot, acc);
  }
  acc = lx::warp_sum(acc);
  if (lane == 0) out[static_cast<int64_t>(t) * L + l] = acc;
}

// Launches `kernel` on `s` through cudaLaunchKernelEx. With `dependent`, the launch is programmatic:
// the kernel may start once the previous kernel in the stream has issued its trigger, and it must
// call cudaGridDependencySynchronize before reading anything that kernel writes.
template <typename... Params, typename... Args>
cudaError_t launch_kernel(void (*kernel)(Params...), unsigned int grid, size_t smem, cudaStream_t s,
                          bool dependent, Args... args) {
  cudaLaunchAttribute attr{};
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(grid);
  cfg.blockDim = dim3(kThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = s;
  cfg.attrs = &attr;
  cfg.numAttrs = dependent && lx::pdl_available() ? 1 : 0;  // below sm_90 an ordinary launch: stream order holds
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

}  // namespace

extern "C" int lx_moe_experts_w8(const void* latent, const int32_t* ids, const float* weights,
                                 const int8_t* gate_up, const float* gate_up_scale, const int8_t* down,
                                 const float* down_scale, float* out, void* workspace, int T, int L, int I,
                                 int top_k, void* stream) {
  if (T <= 0) return 0;
  const int64_t pairs = static_cast<int64_t>(T) * top_k;
  const int64_t rows_a = (I + kWarps - 1) / kWarps;  // stage 1 blocks per pair
  const int64_t rows_b = (L + kWarps - 1) / kWarps;  // stage 2 blocks per token
  const size_t smem = sizeof(bf16) * static_cast<size_t>(top_k) * static_cast<size_t>(I > 0 ? I : 0);
  const size_t smem_vec = static_cast<size_t>(2 + kWarps) * static_cast<size_t>(top_k) * static_cast<size_t>(I > 0 ? I : 0);
  // The intermediates are read back as 16-byte vectors, so the workspace must be 16-byte aligned.
  if (L <= 0 || I <= 0 || top_k < 1 || top_k > kMaxTopK || smem > kMaxSmem || workspace == nullptr ||
      (reinterpret_cast<uintptr_t>(workspace) & 15) != 0 || pairs * rows_a > INT32_MAX ||
      static_cast<int64_t>(T) * rows_b > INT32_MAX)
    return static_cast<int>(cudaErrorInvalidValue);

  auto s = static_cast<cudaStream_t>(stream);
  auto* inter = static_cast<bf16*>(workspace);
  const auto* lat = static_cast<const bf16*>(latent);
  const int P = static_cast<int>(pairs);
  const int ra = static_cast<int>(rows_a);
  const auto grid_a = static_cast<unsigned int>(pairs * rows_a);
  const int nca = (L + 511) / 512;
  const int vec_a = L % 16 == 0 && nca <= kMaxVecChunks ? nca : 0;

  // Stage 1 is a plain launch: it reads the routing, which the stream has already finished.
  cudaError_t err = cudaSuccess;
  if (P <= kSharePairs) {
    switch (vec_a) {
      case 1: err = launch_kernel(w8_gate_up_pair_vec<1>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, L, I, top_k, ra); break;
      case 2: err = launch_kernel(w8_gate_up_pair_vec<2>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, L, I, top_k, ra); break;
      case 3: err = launch_kernel(w8_gate_up_pair_vec<3>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, L, I, top_k, ra); break;
      case 4: err = launch_kernel(w8_gate_up_pair_vec<4>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, L, I, top_k, ra); break;
      default: err = launch_kernel(w8_gate_up_pair_scalar, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, L, I, top_k, ra); break;
    }
  } else {
    // The links live in the workspace after the intermediates (see lx_moe_workspace_bytes).
    auto* next = reinterpret_cast<int32_t*>(static_cast<uint8_t*>(workspace) + chain_offset(P, I));
    int32_t* lead = next + P;
    const auto grid_links = static_cast<unsigned int>((P + kThreads - 1) / kThreads);
    err = launch_kernel(w8_pair_links, grid_links, 0, s, false, ids, P, next, lead);
    if (err != cudaSuccess) return static_cast<int>(err);
    switch (vec_a) {
      case 1: err = launch_kernel(w8_gate_up_chain_vec<1>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, next, lead, L, I, top_k, ra); break;
      case 2: err = launch_kernel(w8_gate_up_chain_vec<2>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, next, lead, L, I, top_k, ra); break;
      case 3: err = launch_kernel(w8_gate_up_chain_vec<3>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, next, lead, L, I, top_k, ra); break;
      case 4: err = launch_kernel(w8_gate_up_chain_vec<4>, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, next, lead, L, I, top_k, ra); break;
      default: err = launch_kernel(w8_gate_up_chain_scalar, grid_a, 0, s, false, lat, ids, gate_up, gate_up_scale, inter, next, lead, L, I, top_k, ra); break;
    }
  }
  if (err != cudaSuccess) return static_cast<int>(err);

  // Stage 2 launches programmatically behind stage 1, so its prologue overlaps stage 1's stream.
  const auto grid_b = static_cast<unsigned int>(static_cast<int64_t>(T) * rows_b);
  if (I % 16 == 0 && smem_vec <= kMaxSmem) {
    err = launch_kernel(w8_down_vec, grid_b, smem_vec, s, true, inter, ids, weights, down, down_scale, out, T, L, I, top_k);
  } else {
    err = launch_kernel(w8_down_scalar, grid_b, smem, s, true, inter, ids, weights, down, down_scale, out, T, L, I, top_k);
  }
  return static_cast<int>(err);
}

// Workspace: the bf16 intermediate of every (token, k) pair, then the two int32 link arrays of the
// prefill form, rounded up to 256 bytes. The kernels take all their scratch from here and allocate
// nothing, so the call can sit in a CUDA graph.
extern "C" int64_t lx_moe_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t pairs = static_cast<int64_t>(T) * top_k;
  const int64_t bytes = static_cast<int64_t>(chain_offset(static_cast<int>(pairs), I)) + 8 * pairs;
  return (bytes + 255) / 256 * 256;
}
