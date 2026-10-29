// The router's f32 GEMM at decode-sized row counts (9 to 64 rows), bit for bit as lx_gemm_f32 computes them.
//
// lx_gemm_f32 runs cuBLASLt's pinned algorithm, a simt sgemm with no split-K: each output adds its K terms in
// increasing k into one accumulator, with fmaf, and adds the bias last. This kernel keeps that chain exactly and
// changes how the operands reach it. A block owns a 16 x 4 tile of outputs, one thread each, and streams K through a
// four-stage ring of shared-memory chunks filled by cp.async, so the next chunks load while the current one runs its
// chain. The tiles are small enough to spread over the GPU, and no chain depends on another, so the bits do not depend
// on the tile shape or the ring depth. The M <= 8 path (lx_spec_gemv_f32) sums in another order and is not touched.

#include <cuda_pipeline.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/common.cuh"
#include "cuda/kernels.h"

namespace {

constexpr int kTileN = 16;         // output columns per block (one thread each, with kTileM rows)
constexpr int kTileM = 4;          // output rows per block
constexpr int kChunk = 128;        // k terms per staged chunk
constexpr int kStages = 4;         // chunks in flight: the ring fills kStages - 1 chunks ahead of use
constexpr int kStride = kChunk + 4;  // floats per staged row: a 16-byte multiple, conflict-free for float4 reads
constexpr int kThreads = kTileN * kTileM;
constexpr int kMaxRows = 64;

constexpr size_t kSmemBytes = static_cast<size_t>(kStages) * (kTileN + kTileM) * kStride * sizeof(float);

__global__ void __launch_bounds__(kThreads) seq_gemm_f32_kernel(const float* __restrict__ x, int ld_x,
                                                               const float* __restrict__ w,
                                                               const float* __restrict__ bias, float* __restrict__ y,
                                                               int ld_y, int M, int N, int K) {
  extern __shared__ __align__(16) float smem[];
  float* sw = smem;                                  // [kStages][kTileN][kStride]
  float* sx = smem + kStages * kTileN * kStride;     // [kStages][kTileM][kStride]
  const int tn = threadIdx.x % kTileN, tm = threadIdx.x / kTileN;
  const int n0 = blockIdx.x * kTileN, m0 = blockIdx.y * kTileM;
  const int m = m0 + tm, n = n0 + tn;
  const int nchunks = (K + kChunk - 1) / kChunk;
  constexpr int kPieces = kChunk / 4;  // 16-byte pieces per staged row

  // Stage chunk c into ring slot c % kStages. Rows past N or M are clamped to a valid row; their results are never stored.
  auto issue = [&](int c) {
    if (c >= nchunks) return;
    const int buf = c % kStages;
    const int k0 = c * kChunk;
    const int kc = min(kChunk, K - k0);
    for (int i = threadIdx.x; i < kTileN * kPieces; i += kThreads) {
      const int r = i / kPieces, p = i - (i / kPieces) * kPieces;
      if (p * 4 < kc)
        __pipeline_memcpy_async(sw + (buf * kTileN + r) * kStride + p * 4,
                                w + static_cast<int64_t>(min(n0 + r, N - 1)) * K + k0 + p * 4, 16);
    }
    for (int i = threadIdx.x; i < kTileM * kPieces; i += kThreads) {
      const int r = i / kPieces, p = i - (i / kPieces) * kPieces;
      if (p * 4 < kc)
        __pipeline_memcpy_async(sx + (buf * kTileM + r) * kStride + p * 4,
                                x + static_cast<int64_t>(min(m0 + r, M - 1)) * ld_x + k0 + p * 4, 16);
    }
  };

  float acc = 0.0f;
#pragma unroll
  for (int c = 0; c < kStages - 1; ++c) {
    issue(c);
    __pipeline_commit();
  }
  for (int c = 0; c < nchunks; ++c) {
    issue(c + kStages - 1);
    __pipeline_commit();
    __pipeline_wait_prior(kStages - 1);
    __syncthreads();
    const int buf = c % kStages;
    const float4* wr = reinterpret_cast<const float4*>(sw + (buf * kTileN + tn) * kStride);
    const float4* xr = reinterpret_cast<const float4*>(sx + (buf * kTileM + tm) * kStride);
    const int kc = min(kChunk, K - c * kChunk);
    if (kc == kChunk) {
#pragma unroll
      for (int j = 0; j < kChunk / 4; ++j) {
        const float4 a = xr[j], b = wr[j];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
      }
    } else {
      for (int j = 0; j < kc / 4; ++j) {
        const float4 a = xr[j], b = wr[j];
        acc = fmaf(a.x, b.x, acc);
        acc = fmaf(a.y, b.y, acc);
        acc = fmaf(a.z, b.z, acc);
        acc = fmaf(a.w, b.w, acc);
      }
    }
    __syncthreads();
  }
  if (m < M && n < N) y[static_cast<int64_t>(m) * ld_y + n] = bias ? acc + bias[n] : acc;
}

}  // namespace

extern "C" int lx_gemm_f32_seq(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M,
                               int N, int K, void* stream) {
  if (M <= 0 || N <= 0) return 0;
  if (M > kMaxRows || K <= 0 || K % 4 != 0 || ld_x < K || ld_x % 4 != 0 || ld_y < N || !x || !w || !y)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16 != 0)
    return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  if (cudaError_t e = cudaFuncSetAttribute(seq_gemm_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(kSmemBytes));
      e != cudaSuccess)
    return static_cast<int>(e);
  seq_gemm_f32_kernel<<<dim3((N + kTileN - 1) / kTileN, (M + kTileM - 1) / kTileM), kThreads, kSmemBytes, s>>>(
      x, ld_x, w, bias, y, ld_y, M, N, K);
  return lx::launch_status();
}
