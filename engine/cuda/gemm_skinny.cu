// Skinny decode GEMM: the contract and the reduction order are in kernels.h ("Skinny decode GEMM").
//
// The output tile is 64 rows by 32 columns per CTA: four warps, warp w owning columns [8w, 8w + 8) of the CTA's 32.
// The CTA walks K in 64-wide stages. Each stage stages the x rows and the weight rows in shared memory (a weight row's
// 64 bf16 are 128 bytes), then four k-tiles of 16 go through mma.sync m16n8k16: the warp's B fragment comes from the
// weight rows, and each row tile's A fragment from x. Every output element keeps its own accumulator and sees its
// k-tiles in ascending order, which is the reduction order the header states.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>

#include "cuda/kernels.h"

namespace {

constexpr int kRows = 64;   // rows per launch; M above this is chunked by the host
constexpr int kNT = 32;     // output columns per CTA
constexpr int kKC = 64;     // k per staged tile: 128 bytes of bf16 per weight row
constexpr int kThreads = 128;
constexpr int kMT = kRows / 16;

__device__ __forceinline__ float load_f(float v) { return v; }
__device__ __forceinline__ float load_f(__nv_bfloat16 v) { return __bfloat162float(v); }

template <typename T>
__device__ __forceinline__ T store_f(float v);
template <>
__device__ __forceinline__ float store_f<float>(float v) {
  return v;
}
template <>
__device__ __forceinline__ __nv_bfloat16 store_f<__nv_bfloat16>(float v) {
  return __float2bfloat16_rn(v);
}

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
  return static_cast<uint32_t>(__bfloat16_as_ushort(lo)) | (static_cast<uint32_t>(__bfloat16_as_ushort(hi)) << 16);
}

// d += A * B for one 16x8 output tile: a0..a3 are the four A registers of the thread, b0 and b1 its two B registers.
__device__ __forceinline__ void mma_bf16(float (&d)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
      "{%0, %1, %2, %3};\n"
      : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
      : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

template <typename XT, typename YT, typename BT>
__global__ void __launch_bounds__(kThreads)
    skinny_bf16_kernel(const XT* __restrict__ x, int ld_x, const __nv_bfloat16* __restrict__ w, const BT* __restrict__ bias,
                       YT* __restrict__ y, int ld_y, int M, int K, float alpha, float beta) {
  __shared__ __nv_bfloat16 xs[kRows][kKC];
  __shared__ __nv_bfloat16 ws[kNT][kKC];
  const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
  const int g = lane >> 2, t = lane & 3;
  const int n0 = blockIdx.x * kNT;
  const int nb = warp * 8;
  const int mt_count = (M + 15) / 16;

  float acc[kMT][4];
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
    for (int e = 0; e < 4; ++e) acc[mt][e] = 0.f;

  for (int k0 = 0; k0 < K; k0 += kKC) {
    for (int i = tid; i < kRows * kKC; i += kThreads) {
      const int r = i / kKC, c = i % kKC;
      const float v = r < M ? load_f(x[static_cast<size_t>(r) * ld_x + k0 + c]) : 0.f;
      xs[r][c] = store_f<__nv_bfloat16>(v);
    }
    for (int i = tid; i < kNT * kKC; i += kThreads) {
      const int n = i / kKC, c = i % kKC;
      ws[n][c] = w[static_cast<size_t>(n0 + n) * K + k0 + c];
    }
    __syncthreads();
#pragma unroll
    for (int kk = 0; kk < kKC; kk += 16) {
      const uint32_t b0 = pack2(ws[nb + g][kk + 2 * t], ws[nb + g][kk + 2 * t + 1]);
      const uint32_t b1 = pack2(ws[nb + g][kk + 2 * t + 8], ws[nb + g][kk + 2 * t + 9]);
      for (int mt = 0; mt < mt_count; ++mt) {
        const int r = mt * 16 + g;
        const uint32_t a0 = pack2(xs[r][kk + 2 * t], xs[r][kk + 2 * t + 1]);
        const uint32_t a1 = pack2(xs[r + 8][kk + 2 * t], xs[r + 8][kk + 2 * t + 1]);
        const uint32_t a2 = pack2(xs[r][kk + 2 * t + 8], xs[r][kk + 2 * t + 9]);
        const uint32_t a3 = pack2(xs[r + 8][kk + 2 * t + 8], xs[r + 8][kk + 2 * t + 9]);
        mma_bf16(acc[mt], a0, a1, a2, a3, b0, b1);
      }
    }
    __syncthreads();
  }

  // The C fragment: element e of a thread is row g (+8 for e >= 2), column 2t (+1 for odd e) of its 16x8 tile.
  const int col0 = n0 + nb + 2 * t;
  for (int mt = 0; mt < mt_count; ++mt)
    for (int e = 0; e < 4; ++e) {
      const int row = mt * 16 + g + (e >> 1) * 8;
      if (row >= M) continue;
      const int col = col0 + (e & 1);
      float v = acc[mt][e];
      if (bias != nullptr) v = v + load_f(bias[col]);
      v = alpha * v;
      YT* out = y + static_cast<size_t>(row) * ld_y + col;
      if (beta != 0.f) v = v + beta * load_f(*out);
      *out = store_f<YT>(v);
    }
}

template <typename XT, typename YT, typename BT>
void launch(const void* x, int ld_x, const void* w, const void* bias, void* y, int ld_y, int M, int N, int K, float alpha,
            float beta, cudaStream_t s) {
  skinny_bf16_kernel<XT, YT, BT><<<N / kNT, kThreads, 0, s>>>(
      static_cast<const XT*>(x), ld_x, static_cast<const __nv_bfloat16*>(w), static_cast<const BT*>(bias),
      static_cast<YT*>(y), ld_y, M, K, alpha, beta);
}

template <typename XT, typename YT>
void dispatch_bias(const void* x, int ld_x, const void* w, const void* bias, lx_dtype bias_dtype, void* y, int ld_y,
                   int M, int N, int K, float alpha, float beta, cudaStream_t s) {
  if (bias == nullptr) launch<XT, YT, float>(x, ld_x, w, nullptr, y, ld_y, M, N, K, alpha, beta, s);
  else if (bias_dtype == LX_F32) launch<XT, YT, float>(x, ld_x, w, bias, y, ld_y, M, N, K, alpha, beta, s);
  else launch<XT, YT, __nv_bfloat16>(x, ld_x, w, bias, y, ld_y, M, N, K, alpha, beta, s);
}

template <typename XT>
void dispatch_y(const void* x, int ld_x, const void* w, const void* bias, lx_dtype bias_dtype, void* y, lx_dtype y_dtype,
                int ld_y, int M, int N, int K, float alpha, float beta, cudaStream_t s) {
  if (y_dtype == LX_F32)
    dispatch_bias<XT, float>(x, ld_x, w, bias, bias_dtype, y, ld_y, M, N, K, alpha, beta, s);
  else
    dispatch_bias<XT, __nv_bfloat16>(x, ld_x, w, bias, bias_dtype, y, ld_y, M, N, K, alpha, beta, s);
}

}  // namespace

LX_API int lx_gemm_skinny(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                          lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                          float beta, void* stream) {
  if (M < 0 || N <= 0 || K <= 0 || ld_x < K || ld_y < N) return static_cast<int>(cudaErrorInvalidValue);
  if (M == 0) return 0;
  if (!w || !w->data || !x || !y) return static_cast<int>(cudaErrorInvalidValue);
  if (w->type != LX_W_BF16 || N % kNT != 0 || K % kKC != 0) return LX_ERR_NOT_SUPPORTED;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const size_t xs = x_dtype == LX_F32 ? 4 : 2, ys = y_dtype == LX_F32 ? 4 : 2;
  for (int r0 = 0; r0 < M; r0 += kRows) {
    const int rows = std::min(kRows, M - r0);
    const void* xr = static_cast<const char*>(x) + xs * static_cast<size_t>(r0) * ld_x;
    void* yr = static_cast<char*>(y) + ys * static_cast<size_t>(r0) * ld_y;
    if (x_dtype == LX_F32)
      dispatch_y<float>(xr, ld_x, w->data, bias, bias_dtype, yr, y_dtype, ld_y, rows, N, K, alpha, beta, s);
    else
      dispatch_y<__nv_bfloat16>(xr, ld_x, w->data, bias, bias_dtype, yr, y_dtype, ld_y, rows, N, K, alpha, beta, s);
  }
  return static_cast<int>(cudaGetLastError());
}
