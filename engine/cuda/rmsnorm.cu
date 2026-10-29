#include "cuda/common.cuh"

namespace {

// One block per row; float32 throughout.
template <typename TIn, typename TOut>
__global__ void rmsnorm_kernel(const TIn* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                               TOut* __restrict__ y, int dim, float eps) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ float scratch[32];
  const TIn* row = x + static_cast<int64_t>(blockIdx.x) * dim;
  TOut* out = y + static_cast<int64_t>(blockIdx.x) * dim;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < dim; i += blockDim.x) {
    const float v = lx::to_f32(row[i]);
    sum += v * v;
  }
  const float scale = rsqrtf(lx::block_sum(sum, scratch) / dim + eps);
  for (int i = threadIdx.x; i < dim; i += blockDim.x)
    out[i] = lx::from_f32<TOut>(lx::to_f32(row[i]) * scale * __bfloat162float(w[i]));
}

}  // namespace

extern "C" int lx_rmsnorm(const void* x, lx_dtype x_dtype, const void* w, void* y, lx_dtype y_dtype,
                          int rows, int dim, float eps, void* stream) {
  if (rows <= 0) return 0;
  const int threads = dim >= 1024 ? 1024 : ((dim + 31) / 32) * 32;
  auto s = static_cast<cudaStream_t>(stream);
  LX_DISPATCH_DTYPE(x_dtype, TIn, LX_DISPATCH_DTYPE(y_dtype, TOut,
    lx::launch_pdl(rmsnorm_kernel<TIn, TOut>, rows, threads, 0, s, static_cast<const TIn*>(x), static_cast<const __nv_bfloat16*>(w), static_cast<TOut*>(y), dim, eps)))
  return lx::launch_status();
}
