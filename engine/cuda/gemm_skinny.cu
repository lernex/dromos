// Skinny decode GEMM: the contract and the reduction order are in kernels.h ("Skinny decode GEMM").
//
// The weight streaming is the routed-expert kernels' (gemm_tc.cuh): a weight row's 64-element k-tile is 128 bytes, read
// by 16-byte cp.async copies into a swizzled shared-memory stage, several stages in flight, and the fragments come out
// of shared memory by ldmatrix into mma.sync m16n8k16. The difference is that every expert is the same matrix here: one
// CTA owns BN consecutive output columns for all K, and the activations (up to 64 rows of one K-tile at a time) are
// staged beside the weights, so each K-tile of x is loaded once per CTA.
//
// Tiles. Every CTA has four warps. Warp w owns n8 tile w % (BN / 8) and the row tiles of group w / (BN / 8), so BN = 32
// gives each warp one n8 tile over all the row tiles, and BN = 16 gives two n8 tiles, each taken by two warps that split
// the row tiles. BN is 32, or 16 when N is narrow (N / 32 < 2 * SMs), so the grid has enough CTAs to fill the card. BM is
// the rows in the call (16, 32 or 64). The chain for each output element is the same whatever BM, BN and the warp split
// are, so none of these choices changes any bits.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/gemm_tc.cuh"
#include "cuda/kernels.h"
#include "cuda/mma.cuh"

namespace {

constexpr int kKC = 64;    // k per tile: 64 bf16 = 128 bytes per row, the segment the card streams at full rate
constexpr int kRB = 128;   // bytes per staged row
constexpr int kRowsPerCall = 64;

// One CTA: BN output columns and BM activation rows, K in tiles of kKC through ST cp.async stages.
template <int BM, int BN, int ST>
__global__ void __launch_bounds__(128) skinny_bf16_kernel(const __nv_bfloat16* __restrict__ x, int ld_x,
                                                              const __nv_bfloat16* __restrict__ w, const void* bias,
                                                              bool bias_f32, void* y, bool y_bf16, int ld_y, int M,
                                                              int K, float alpha, float beta) {
  extern __shared__ __align__(16) uint8_t smem[];
  constexpr int kThreads = 128;
  constexpr int kMT = BM / 16;
  constexpr int kNT = BN / 8;               // n8 tiles per CTA
  constexpr int kNG = 4 / kNT;              // warp groups splitting the row tiles
  constexpr int kMTw = (kMT + kNG - 1) / kNG;  // row tiles one warp holds
  constexpr int kAStage = BM * kRB, kStage = (BM + BN) * kRB;
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int nt = warp % kNT, mg = warp / kNT;  // this warp's n8 tile, and its group of row tiles
  const int n0 = blockIdx.x * BN;
  const int KT = K / kKC;
  const uint32_t s0 = lx::mma::smem_addr(smem);

  // Loader: 16-byte chunks. A holds BM rows of x (rows past M are zero-filled, not read); B holds BN rows of w.
  auto load_tile = [&](int stage, int kt) {
    const uint32_t base = s0 + stage * kStage;
    const int kb = kt * kKC;
#pragma unroll
    for (int c = tid; c < BM * 8; c += kThreads) {
      const int r = c >> 3, ch = c & 7;
      const bool ok = r < M;
      lx::mma::cp_async16(base + lx::tc::swz<kRB, 0>(r, ch), x + static_cast<size_t>(ok ? r : 0) * ld_x + kb + ch * 8, ok);
    }
#pragma unroll
    for (int c = tid; c < BN * 8; c += kThreads) {
      const int r = c >> 3, ch = c & 7;
      lx::mma::cp_async16(base + kAStage + lx::tc::swz<kRB, 0>(r, ch),
                          w + static_cast<size_t>(n0 + r) * K + kb + ch * 8, true);
    }
  };

  float acc[kMTw][4];
#pragma unroll
  for (int j = 0; j < kMTw; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[j][e] = 0.f;

  // Prologue: ST - 1 tiles in flight. Each iteration waits for its tile, refills the stage the previous iteration used
  // (the barrier retires its reads first), then multiplies the tile's four k16 steps in ascending order.
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < KT) load_tile(s, s);
    lx::mma::cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    lx::mma::cp_async_wait<ST - 2>();
    __syncthreads();
    if (kt + ST - 1 < KT) load_tile((kt + ST - 1) % ST, kt + ST - 1);
    lx::mma::cp_async_commit();
    const uint32_t sA = s0 + (kt % ST) * kStage, sB = sA + kAStage;
#pragma unroll
    for (int ks = 0; ks < kKC / 16; ++ks) {
      // B: lanes 0-7 and 8-15 read the two 8-element k halves of this warp's n8 rows (the ldmatrix x4 duplicates its
      // upper matrices, which are unused)
      uint32_t rb[4];
      lx::mma::ldmatrix_x4(rb, sB + lx::tc::swz<kRB, 0>(nt * 8 + (lane & 7), 2 * ks + ((lane >> 3) & 1)));
      const uint32_t b0 = rb[0], b1 = rb[1];
#pragma unroll
      for (int j = 0; j < kMTw; ++j) {
        const int mt = mg + j * kNG;
        if (mt >= kMT) continue;  // a warp group with fewer row tiles than groups (BM = 16)
        uint32_t a[4];
        const int a_r = (lane & 7) + ((lane >> 3) & 1) * 8, a_c = lane >> 4;
        lx::mma::ldmatrix_x4(a, sA + lx::tc::swz<kRB, 0>(mt * 16 + a_r, 2 * ks + a_c));
        lx::mma::mma_bf16(acc[j], a, b0, b1);
      }
    }
  }
  lx::mma::cp_async_wait<0>();

  // The C fragment: element e is row g (+8 for e >= 2), column 2t (+1 for odd e) of this warp's 16x8 tile.
  const int col0 = n0 + nt * 8 + 2 * t;
#pragma unroll
  for (int j = 0; j < kMTw; ++j)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int mt = mg + j * kNG;
      const int row = mt * 16 + g + (e >> 1) * 8;
      if (mt >= kMT || row >= M) continue;
      const int col = col0 + (e & 1);
      float v = acc[j][e];
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
}

template <int BM, int BN, int ST>
void launch(const __nv_bfloat16* x, int ld_x, const __nv_bfloat16* w, const void* bias, bool bias_f32, void* y,
            bool y_bf16, int ld_y, int M, int N, int K, float alpha, float beta, cudaStream_t s) {
  constexpr int bytes = ST * (BM + BN) * kRB;
  cudaFuncSetAttribute(skinny_bf16_kernel<BM, BN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes);
  skinny_bf16_kernel<BM, BN, ST><<<N / BN, 128, bytes, s>>>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, K, alpha,
                                                              beta);
}

// BN = 32 streams four stages; BN = 16 (narrow N, so more CTAs) needs more stages in flight per CTA to keep the bytes
// coming: six 2-KB B stages. BN = 8 was measured and is not used: it doubles the x re-reads per weight byte, and at M = 32
// it loses 20 to 40 percent on the 2048-wide shapes.
template <int BM>
void launch_bn(const __nv_bfloat16* x, int ld_x, const __nv_bfloat16* w, const void* bias, bool bias_f32, void* y,
               bool y_bf16, int ld_y, int M, int N, int K, float alpha, float beta, cudaStream_t s, bool wide) {
  if (wide) launch<BM, 32, 4>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s);
  else launch<BM, 16, 6>(x, ld_x, w, bias, bias_f32, y, y_bf16, ld_y, M, N, K, alpha, beta, s);
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
  const bool wide = N % 32 == 0 && N / 32 >= 2 * sms;
  const bool bias_f32 = bias_dtype == LX_F32;
  const bool y_bf16 = y_dtype == LX_BF16;
  const __nv_bfloat16* xp = static_cast<const __nv_bfloat16*>(x);
  const __nv_bfloat16* wp = static_cast<const __nv_bfloat16*>(w->data);
  for (int r0 = 0; r0 < M; r0 += kRowsPerCall) {
    const int rows = std::min(kRowsPerCall, M - r0);
    const __nv_bfloat16* xr = xp + static_cast<size_t>(r0) * ld_x;
    void* yr = static_cast<char*>(y) + static_cast<size_t>(r0) * ld_y * (y_bf16 ? 2 : 4);
    if (rows <= 16) launch_bn<16>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, wide);
    else if (rows <= 32) launch_bn<32>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, wide);
    else launch_bn<64>(xr, ld_x, wp, bias, bias_f32, yr, y_bf16, ld_y, rows, N, K, alpha, beta, s, wide);
  }
  return static_cast<int>(cudaGetLastError());
}
