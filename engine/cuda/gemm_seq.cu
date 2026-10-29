// The router's f32 GEMM at decode-sized row counts (9 to 64 rows), bit for bit as lx_gemm_f32 computes them.
//
// lx_gemm_f32 runs cuBLASLt's pinned algorithm, a simt sgemm with no split-K: each output adds its K terms in
// increasing k into one accumulator, with fmaf, and the bias last. This kernel keeps that chain exactly. What it
// changes is the layout: a block stages 8 rows of x and 8 rows of w into shared memory with vector loads, and each
// of its 64 outputs runs its chain from there. The cuBLASLt algorithm runs the whole output in one block at these
// shapes (grid 1), which is why it takes 78 us at 32 rows and this takes about 12.
//
// The chain order is the only thing that fixes the bits, so the tile sizes are free to change. The M <= 8 path
// (lx_spec_gemv_f32) sums in a different order and is not touched.

#include <cuda_runtime.h>

#include <cstdint>

#include "cuda/common.cuh"
#include "cuda/kernels.h"

namespace {

constexpr int kSeqTile = 8;        // x rows and w rows per block: an 8 x 8 tile of outputs
constexpr int kSeqThreads = 256;   // all of them stage; the first 64 each run one output
constexpr int kSeqMaxRows = 64;
constexpr int kSeqMaxK = 1536;     // 16 staged rows of K + 1 floats stay under the 99 KB opt-in shared memory

size_t seq_smem_bytes(int K) { return static_cast<size_t>(2 * kSeqTile) * static_cast<size_t>(K + 1) * sizeof(float); }

__global__ void __launch_bounds__(kSeqThreads) seq_gemm_f32_kernel(const float* __restrict__ x, int ld_x,
                                                                  const float* __restrict__ w,
                                                                  const float* __restrict__ bias, float* __restrict__ y,
                                                                  int ld_y, int M, int N, int K) {
  extern __shared__ float smem[];
  const int ldk = K + 1;
  float* sx = smem;
  float* sw = smem + kSeqTile * ldk;
  const int n0 = blockIdx.x * kSeqTile, m0 = blockIdx.y * kSeqTile;
  const int K4 = K >> 2;
  for (int r = 0; r < kSeqTile; ++r) {
    float* dst = sx + r * ldk;
    if (m0 + r < M) {
      const float4* src = reinterpret_cast<const float4*>(x + static_cast<int64_t>(m0 + r) * ld_x);
      for (int i = threadIdx.x; i < K4; i += kSeqThreads) {
        const float4 v = src[i];
        dst[4 * i] = v.x, dst[4 * i + 1] = v.y, dst[4 * i + 2] = v.z, dst[4 * i + 3] = v.w;
      }
    } else {
      for (int i = threadIdx.x; i < K; i += kSeqThreads) dst[i] = 0.0f;
    }
  }
  for (int r = 0; r < kSeqTile; ++r) {
    float* dst = sw + r * ldk;
    if (n0 + r < N) {
      const float4* src = reinterpret_cast<const float4*>(w + static_cast<int64_t>(n0 + r) * K);
      for (int i = threadIdx.x; i < K4; i += kSeqThreads) {
        const float4 v = src[i];
        dst[4 * i] = v.x, dst[4 * i + 1] = v.y, dst[4 * i + 2] = v.z, dst[4 * i + 3] = v.w;
      }
    } else {
      for (int i = threadIdx.x; i < K; i += kSeqThreads) dst[i] = 0.0f;
    }
  }
  __syncthreads();
  if (threadIdx.x < kSeqTile * kSeqTile) {
    const int tx = threadIdx.x % kSeqTile, ty = threadIdx.x / kSeqTile;
    const int n = n0 + tx, m = m0 + ty;
    if (m < M && n < N) {
      const float* xr = sx + ty * ldk;
      const float* wr = sw + tx * ldk;
      float acc = 0.0f;
#pragma unroll 8
      for (int c = 0; c < K; ++c) acc = fmaf(xr[c], wr[c], acc);
      y[static_cast<int64_t>(m) * ld_y + n] = bias ? acc + bias[n] : acc;
    }
  }
}

}  // namespace

extern "C" int lx_gemm_f32_seq(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M,
                               int N, int K, void* stream) {
  if (M <= 0 || N <= 0) return 0;
  if (M > kSeqMaxRows || K <= 0 || K > kSeqMaxK || K % 4 != 0 || ld_x < K || ld_x % 4 != 0 || ld_y < N || !x || !w || !y)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((reinterpret_cast<uintptr_t>(x) | reinterpret_cast<uintptr_t>(w)) % 16 != 0)
    return static_cast<int>(cudaErrorInvalidValue);
  const size_t smem = seq_smem_bytes(K);
  auto s = static_cast<cudaStream_t>(stream);
  if (cudaError_t e = cudaFuncSetAttribute(seq_gemm_f32_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                           static_cast<int>(smem));
      e != cudaSuccess)
    return static_cast<int>(e);
  seq_gemm_f32_kernel<<<dim3((N + kSeqTile - 1) / kSeqTile, (M + kSeqTile - 1) / kSeqTile), kSeqThreads, smem, s>>>(
      x, ld_x, w, bias, y, ld_y, M, N, K);
  return lx::launch_status();
}
