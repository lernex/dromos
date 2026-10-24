// Elementwise kernels: the strided dtype-casting copy, the activations, the packed gate|up product
// and the row bias add. One thread per element; rows go over blockIdx.y with a stride loop, so
// any row count fits the grid.
#include <algorithm>

#include "cuda/common.cuh"

namespace {

constexpr int kThreads = 256;

// Grid for a [rows, cols] launch: blockIdx.x covers the columns, blockIdx.y the rows.
inline dim3 grid2d(int rows, int cols) {
  return dim3(static_cast<unsigned>((cols + kThreads - 1) / kThreads),
              static_cast<unsigned>(std::min(rows, 65535)));
}

__device__ __forceinline__ float sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

// dst[r, c] = cast(src[r, c]) for c < cols.
template <typename TSrc, typename TDst>
__global__ void copy2d_kernel(const TSrc* src, int ld_src, TDst* dst, int ld_dst, int rows, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y)
    dst[static_cast<int64_t>(r) * ld_dst + c] =
        lx::from_f32<TDst>(lx::to_f32(src[static_cast<int64_t>(r) * ld_src + c]));
}

// In place y = act(y). The activation is a template argument, so each launch is one branch-free loop.
template <int kAct>
__global__ void activate_kernel(float* y, int64_t n) {
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n; i += stride) {
    const float v = y[i];
    if constexpr (kAct == 0) {
      y[i] = tanhf(v);
    } else if constexpr (kAct == 1) {
      y[i] = v * sigmoid(v);
    } else {
      y[i] = sigmoid(v);
    }
  }
}

// out[r, c] = silu(x[r, c]) * x[r, cols + c]: the gate and up halves of one packed row.
template <typename TOut>
__global__ void silu_mul_kernel(const float* x, int ld_x, TOut* out, int rows, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  for (int r = blockIdx.y; r < rows; r += gridDim.y) {
    const float* xr = x + static_cast<int64_t>(r) * ld_x;
    const float gate = xr[c];
    out[static_cast<int64_t>(r) * cols + c] = lx::from_f32<TOut>(gate * sigmoid(gate) * xr[cols + c]);
  }
}

// y[r, c] += row[c]. Each column thread loads its bf16 bias once and reuses it down the rows.
__global__ void add_row_kernel(float* y, const __nv_bfloat16* row, int rows, int cols) {
  const int c = blockIdx.x * blockDim.x + threadIdx.x;
  if (c >= cols) return;
  const float add = lx::to_f32(row[c]);
  for (int r = blockIdx.y; r < rows; r += gridDim.y) y[static_cast<int64_t>(r) * cols + c] += add;
}

}  // namespace

extern "C" int lx_copy2d(const void* src, lx_dtype src_dtype, int ld_src, void* dst, lx_dtype dst_dtype,
                         int ld_dst, int rows, int cols, void* stream) {
  if (rows <= 0 || cols <= 0) return 0;
  // A window read out of a wider buffer needs strides that clear it. One row needs no stride.
  if (rows > 1 && (ld_src < cols || ld_dst < cols)) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  LX_DISPATCH_DTYPE(src_dtype, TSrc, LX_DISPATCH_DTYPE(dst_dtype, TDst,
    copy2d_kernel<TSrc, TDst><<<grid2d(rows, cols), kThreads, 0, s>>>(
        static_cast<const TSrc*>(src), ld_src, static_cast<TDst*>(dst), ld_dst, rows, cols)))
  return lx::launch_status();
}

extern "C" int lx_activate_f32(float* y, int64_t n, int act, void* stream) {
  if (n <= 0) return 0;
  // Enough blocks to fill the GPU several times over; the kernel's stride loop covers the rest.
  const int blocks = static_cast<int>(std::min<int64_t>((n + kThreads - 1) / kThreads, 1 << 16));
  auto s = static_cast<cudaStream_t>(stream);
  switch (act) {
    case 0: activate_kernel<0><<<blocks, kThreads, 0, s>>>(y, n); break;
    case 1: activate_kernel<1><<<blocks, kThreads, 0, s>>>(y, n); break;
    case 2: activate_kernel<2><<<blocks, kThreads, 0, s>>>(y, n); break;
    default: return static_cast<int>(cudaErrorInvalidValue);
  }
  return lx::launch_status();
}

extern "C" int lx_silu_mul(const float* x, int ld_x, void* out, lx_dtype out_dtype, int rows, int cols,
                           void* stream) {
  if (rows <= 0 || cols <= 0) return 0;
  if (ld_x < 2 * cols) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  LX_DISPATCH_DTYPE(out_dtype, TOut,
    silu_mul_kernel<TOut><<<grid2d(rows, cols), kThreads, 0, s>>>(
        x, ld_x, static_cast<TOut*>(out), rows, cols))
  return lx::launch_status();
}

extern "C" int lx_add_row_bf16(float* y, const void* row, int rows, int cols, void* stream) {
  if (rows <= 0 || cols <= 0) return 0;
  add_row_kernel<<<grid2d(rows, cols), kThreads, 0, static_cast<cudaStream_t>(stream)>>>(
      y, static_cast<const __nv_bfloat16*>(row), rows, cols);
  return lx::launch_status();
}
