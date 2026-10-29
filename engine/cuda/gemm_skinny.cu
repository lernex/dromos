// Skinny decode GEMM: the contract and the reduction order are in kernels.h ("Skinny decode GEMM").
//
// The weight streaming is the routed-expert kernels' (gemm_tc.cuh): a weight row's 64-element k-tile is 128 bytes, read
// by 16-byte cp.async copies into a swizzled shared-memory stage, several stages in flight, and the fragments come out
// of shared memory by ldmatrix into mma.sync m16n8k16.
//
// Split-K across a cluster. A tile of BN output columns can be streamed by S CTAs of one cluster, each taking a
// contiguous slice of the K-tiles. The slices are fixed by (K, S): slice r holds tiles [r * KT / S, (r + 1) * KT / S).
// Each CTA computes its slice's partial sums, writes them to its own shared memory, and after a cluster barrier every
// CTA sums the S partials of its share of the outputs, reading the peers' shared memory through the cluster (DSMEM), in
// ascending slice order. No workspace, no atomics, one launch. S and BN depend only on (K, N) and the card's SM count, so
// they do not change with M, and the bits of a row do not depend on M.
//
// Warps. Each CTA has four warps. With BN = 32 the warps split the four n8 tiles two ways and the row tiles two ways, so
// no warp reads the whole activation tile; with BN = 16 or 8 the same rule gives one n8 tile per warp group.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/gemm_tc.cuh"
#include "cuda/kernels.h"
#include "cuda/mma.cuh"

#include "cuda/common.cuh"

namespace {

constexpr int kKC = 64;    // k per tile: 64 bf16 = 128 bytes per row, the segment the card streams at full rate
constexpr int kRB = 128;   // bytes per staged row
constexpr int kRowsPerCall = 64;
constexpr int kMaxSplit = 8;  // the largest cluster the card's portable launches allow

// The cluster barrier, as PTX states it. Arrive publishes this thread's shared-memory writes (release) and wait takes the
// peers' writes (acquire). The memory clobber keeps the compiler from moving the partial stores or the peer reads across
// the barrier; cooperative groups' sync() does not promise that for the partial exchange.
// Below sm_90 there are no clusters and the host never launches S > 1, so both are empty there.
#if LX_SM90_DEVICE
__device__ __forceinline__ void cluster_arrive() { asm volatile("barrier.cluster.arrive.release.aligned;" ::: "memory"); }
__device__ __forceinline__ void cluster_wait() { asm volatile("barrier.cluster.wait.acquire.aligned;" ::: "memory"); }
#else
__device__ __forceinline__ void cluster_arrive() {}
__device__ __forceinline__ void cluster_wait() {}
#endif


__device__ __forceinline__ void epilogue_store(float v, int row, int col, const void* bias, bool bias_f32, void* y,
                                               bool y_bf16, int ld_y, float alpha, float beta) {
  if (bias != nullptr)
    v = v + (bias_f32 ? static_cast<const float*>(bias)[col]
                      : __bfloat162float(static_cast<const __nv_bfloat16*>(bias)[col]));
  v = alpha * v;
  if (y_bf16) {
    __nv_bfloat16* out = static_cast<__nv_bfloat16*>(y) + static_cast<size_t>(row) * ld_y + col;
    if (beta != 0.f) v = v + beta * __bfloat162float(*out);
    *out = __float2bfloat16_rn(v);
  } else {
    float* out = static_cast<float*>(y) + static_cast<size_t>(row) * ld_y + col;
    if (beta != 0.f) v = v + beta * (*out);
    *out = v;
  }
}

// One CTA: BN output columns of a cluster of S CTAs, each streaming its slice of K through ST cp.async stages.
template <int BM, int BN, int ST, int S>
__global__ void __launch_bounds__(128) skinny_bf16_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                          const __nv_bfloat16* __restrict__ w, const void* bias,
                                                          bool bias_f32, void* y, bool y_bf16, int ld_y, int M, int K,
                                                          float alpha, float beta) {
  lx::pdl_trigger();
  lx::pdl_wait();
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int kMT = BM / 16;                       // row tiles
  constexpr int kNTG = BN / 8 < 2 ? BN / 8 : 2;      // warp groups splitting the n8 tiles
  constexpr int kNTw = BN / 8 / kNTG;                // n8 tiles per warp
  constexpr int kMG = 4 / kNTG;                      // warp groups splitting the row tiles
  constexpr int kMTw = (kMT + kMG - 1) / kMG;        // row tiles per warp
  constexpr int kAStage = BM * kRB, kStage = (BM + BN) * kRB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int ng = warp % kNTG, mg = warp / kNTG;
  const int KT = K / kKC;
  const int rank = S > 1 ? static_cast<int>(lx::cluster_block_rank()) : 0;
  const int n0 = (S > 1 ? static_cast<int>(blockIdx.x) / S : static_cast<int>(blockIdx.x)) * BN;
  const int kt0 = rank * KT / S;                     // this CTA's K slice: tiles kt0 .. kt0 + KTr - 1
  const int KTr = (rank + 1) * KT / S - kt0;
  const uint32_t s0 = lx::mma::smem_addr(smem);

  // Loader: 16-byte chunks. A holds BM rows of x (rows past M are zero-filled, not read); B holds BN rows of w.
  auto load_tile = [&](int stage, int kl) {
    const uint32_t base = s0 + stage * kStage;
    const int kb = (kt0 + kl) * kKC;
#pragma unroll
    for (int c = tid; c < BM * 8; c += 128) {
      const int r = c >> 3, ch = c & 7;
      const bool ok = r < M;
      lx::mma::cp_async16(base + lx::tc::swz<kRB, 0>(r, ch), x + static_cast<size_t>(ok ? r : 0) * ld_x + kb + ch * 8, ok);
    }
#pragma unroll
    for (int c = tid; c < BN * 8; c += 128) {
      const int r = c >> 3, ch = c & 7;
      lx::mma::cp_async16(base + kAStage + lx::tc::swz<kRB, 0>(r, ch),
                          w + static_cast<size_t>(n0 + r) * K + kb + ch * 8, true);
    }
  };

  float acc[kNTw][kMTw][4];
#pragma unroll
  for (int j = 0; j < kNTw; ++j)
#pragma unroll
    for (int jm = 0; jm < kMTw; ++jm)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[j][jm][e] = 0.f;

  // Prologue: ST - 1 tiles in flight. Each iteration waits for its tile, refills the stage the previous iteration used
  // (the barrier retires its reads first), then multiplies the tile's four k16 steps in ascending order.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < KTr) load_tile(s, s);
    lx::mma::cp_async_commit();
  }
  for (int kl = 0; kl < KTr; ++kl) {
    lx::mma::cp_async_wait<ST - 2>();
    __syncthreads();
    if (kl + ST - 1 < KTr) load_tile((kl + ST - 1) % ST, kl + ST - 1);
    lx::mma::cp_async_commit();
    const uint32_t sA = s0 + (kl % ST) * kStage, sB = sA + kAStage;
#pragma unroll
    for (int ks = 0; ks < kKC / 16; ++ks) {
      // B: lanes 0-7 and 8-15 read the two 8-element k halves of an n8 tile (the x4 duplicates its upper matrices)
      uint32_t bf[kNTw][2];
#pragma unroll
      for (int j = 0; j < kNTw; ++j) {
        uint32_t rb[4];
        lx::mma::ldmatrix_x4(rb, sB + lx::tc::swz<kRB, 0>((ng * kNTw + j) * 8 + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
        bf[j][0] = rb[0];
        bf[j][1] = rb[1];
      }
#pragma unroll
      for (int jm = 0; jm < kMTw; ++jm) {
        const int mt = mg + jm * kMG;
        if (mt >= kMT) continue;  // a warp group with fewer row tiles than groups (BM = 16)
        uint32_t a[4];
        const int a_r = (lane & 7) + ((lane >> 3) & 1) * 8, a_c = lane >> 4;
        lx::mma::ldmatrix_x4(a, sA + lx::tc::swz<kRB, 0>(mt * 16 + a_r, 2 * ks + a_c));
#pragma unroll
        for (int j = 0; j < kNTw; ++j) lx::mma::mma_bf16(acc[j][jm], a, bf[j][0], bf[j][1]);
      }
    }
  }
  lx::mma::cp_async_wait<0>();

  // The C fragment: element e is row g (+8 for e >= 2), column 2t (+1 for odd e) of an n8 tile, within the tile's BN.
  if constexpr (S == 1) {
#pragma unroll
    for (int j = 0; j < kNTw; ++j)
#pragma unroll
      for (int jm = 0; jm < kMTw; ++jm)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int mt = mg + jm * kMG;
          const int row = mt * 16 + g + (e >> 1) * 8;
          if (mt >= kMT || row >= M) continue;
          const int col = (ng * kNTw + j) * 8 + 2 * t + (e & 1);
          epilogue_store(acc[j][jm][e], row, n0 + col, bias, bias_f32, y, y_bf16, ld_y, alpha, beta);
        }
  } else {
    // Partial sums to this CTA's shared memory (the stages are done: the wait above and a barrier retire them), then
    // the cluster barrier, then each CTA sums the partials of its share of the outputs in ascending slice order.
    __syncthreads();
    float* part = reinterpret_cast<float*>(smem);  // [BM][BN]
#pragma unroll
    for (int j = 0; j < kNTw; ++j)
#pragma unroll
      for (int jm = 0; jm < kMTw; ++jm)
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          const int mt = mg + jm * kMG;
          if (mt >= kMT) continue;
          const int row = mt * 16 + g + (e >> 1) * 8;
          const int col = (ng * kNTw + j) * 8 + 2 * t + (e & 1);
          part[row * BN + col] = acc[j][jm][e];
        }
    // Phase 1: every partial is in its owner's shared memory. Phase 2: each CTA sums the peers' partials for the outputs it
    // owns (i = rank + S * m, so the CTAs of a cluster split the tile by residue) into registers. A second barrier
    // then ends every peer read, so the CTA may write its outputs and exit. No global access sits between the DSMEM
    // reads and the barrier.
    cluster_arrive();
    cluster_wait();
    constexpr int kOwn = (BM * BN + S * 128 - 1) / (S * 128);  // outputs per thread, rounded up
    float sums[kOwn];
#pragma unroll
    for (int k = 0; k < kOwn; ++k) {
      const int i = rank + S * (static_cast<int>(tid) + 128 * k);
      float sum = 0.f;
      if (i < BM * BN) {
#pragma unroll
        for (int q = 0; q < S; ++q) {
          const float v = lx::cluster_peer(part, static_cast<unsigned>(q))[i];
          sum = q == 0 ? v : sum + v;
        }
      }
      sums[k] = sum;
    }
    cluster_arrive();
    cluster_wait();  // every peer read of this CTA's partials has finished
#pragma unroll
    for (int k = 0; k < kOwn; ++k) {
      const int i = rank + S * (static_cast<int>(tid) + 128 * k);
      if (i < BM * BN) {
        const int row = i / BN, col = i % BN;
        if (row < M) epilogue_store(sums[k], row, n0 + col, bias, bias_f32, y, y_bf16, ld_y, alpha, beta);
      }
    }
  }
}

template <int BM, int BN, int ST, int S>
void launch(const __nv_bfloat16* x, int ld_x, const __nv_bfloat16* w, const void* bias, bool bias_f32, void* y,
            bool y_bf16, int ld_y, int M, int N, int K, float alpha, float beta, cudaStream_t s) {
  constexpr int bytes = ST * (BM + BN) * kRB;
  cudaFuncSetAttribute(skinny_bf16_kernel<BM, BN, ST, S>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(static_cast<unsigned>(N / BN * S));
  cfg.blockDim = dim3(128);
  cfg.dynamicSmemBytes = bytes;
  cfg.stream = s;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeClusterDimension;
  attr[0].val.clusterDim.x = S;
  attr[0].val.clusterDim.y = 1;
  attr[0].val.clusterDim.z = 1;
  cfg.attrs = attr;
  cfg.numAttrs = S > 1 ? 1 : 0;
  cudaLaunchKernelEx(&cfg, skinny_bf16_kernel<BM, BN, ST, S>, x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, K, alpha,
                     beta);
}

// The split for a (K, N) on a card with `sms` SMs: BN = 32 and S = ceil(2 sms / (N / 32)) when that S is at most
// kMaxSplit; else BN = 16 and S = ceil(2 sms / (N / 16)). S never exceeds the K-tiles. Two CTAs per SM is the target.
int choose_split(int N, int K, int sms, int* bn) {
  const int ktiles = K / kKC;
  if (N % 32 == 0) {
    const int tiles = N / 32, s = std::max(1, (2 * sms + tiles - 1) / tiles);
    if (s <= kMaxSplit && s <= ktiles) {
      *bn = 32;
      return s;
    }
  }
  const int tiles = N / 16, s = std::max(1, (2 * sms + tiles - 1) / tiles);
  *bn = 16;
  return std::min({s, kMaxSplit, ktiles});
}

template <int BM>
void launch_bm(const __nv_bfloat16* x, int ld_x, const __nv_bfloat16* w, const void* bias, bool bias_f32, void* y,
               bool y_bf16, int ld_y, int M, int N, int K, float alpha, float beta, cudaStream_t s, int bn, int S) {
  if (bn == 32) {
    switch (S) {
      case 1: launch<BM, 32, 4, 1>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 2: launch<BM, 32, 4, 2>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 3: launch<BM, 32, 4, 3>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 4: launch<BM, 32, 4, 4>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 5: launch<BM, 32, 4, 5>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 6: launch<BM, 32, 4, 6>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 7: launch<BM, 32, 4, 7>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      default: launch<BM, 32, 4, 8>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
    }
  } else {
    switch (S) {
      case 1: launch<BM, 16, 6, 1>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 2: launch<BM, 16, 6, 2>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 3: launch<BM, 16, 6, 3>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 4: launch<BM, 16, 6, 4>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 5: launch<BM, 16, 6, 5>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 6: launch<BM, 16, 6, 6>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      case 7: launch<BM, 16, 6, 7>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
      default: launch<BM, 16, 6, 8>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s); break;
    }
  }
}

}  // namespace

LX_API int lx_gemm_skinny(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                          lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                          float beta, void* stream) {
  if (M < 0 || N <= 0 || K <= 0 || ld_x < K || ld_y < N) return static_cast<int>(cudaErrorInvalidValue);
  if (M == 0) return 0;
  if (!w || !w->data || !x || !y) return static_cast<int>(cudaErrorInvalidValue);
  if (w->type != LX_W_BF16 || x_dtype != LX_BF16 || N % 16 != 0 || K % kKC != 0) return LX_ERR_NOT_SUPPORTED;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  int dev = 0, sms = 1;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  int bn = 16;
  const int S = choose_split(N, K, sms, &bn);
  const bool bias_f32 = bias_dtype == LX_F32;
  const bool y_bf16 = y_dtype == LX_BF16;
  const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x);
  const __nv_bfloat16* wp = static_cast<const __nv_bfloat16*>(w->data);
  for (int r0 = 0; r0 < M; r0 += kRowsPerCall) {
    const int rows = std::min(kRowsPerCall, M - r0);
    const __nv_bfloat16* xr = xp + static_cast<size_t>(r0) * ld_x;
    void* yr = static_cast<char*>(y) + static_cast<size_t>(r0) * ld_y * (y_bf16 ? 2 : 4);
    if (rows <= 16)
      launch_bm<16>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, bn, S);
    else if (rows <= 32)
      launch_bm<32>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, bn, S);
    else
      launch_bm<64>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, bn, S);
  }
  return static_cast<int>(cudaGetLastError());
}
