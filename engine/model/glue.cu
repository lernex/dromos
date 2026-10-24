// Small kernels the Metis forward pass needs between the library kernels. They are specific to
// how the model wires its pieces together, so they live with the model rather than in the ABI.
#include "model/glue.h"

#include "cuda/common.cuh"

namespace {

__global__ void axpby_kernel(float* y, const float* x, float a, float b, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) y[i] = a * y[i] + b * x[i];
}

__global__ void sub_kernel(const float* a, const float* b, float* out, int64_t n) {
  const int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x;
  if (i < n) out[i] = a[i] - b[i];
}

// The four scalar columns of a memory entry's metadata, after the expert summary.
__global__ void meta_scalars_kernel(const float* entropy, const float* confidence, const float* continuation,
                                    float mean_k, float inv_log_experts, __nv_bfloat16* meta, int ld, int col,
                                    int T) {
  const int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= T) return;
  __nv_bfloat16* row = meta + static_cast<int64_t>(t) * ld + col;
  row[0] = __float2bfloat16_rn(mean_k);
  row[1] = __float2bfloat16_rn(entropy[t] * inv_log_experts);
  row[2] = __float2bfloat16_rn(confidence[t]);
  row[3] = __float2bfloat16_rn(continuation[t]);
}

int blocks(int64_t n) { return static_cast<int>((n + 255) / 256); }

}  // namespace

namespace lx::glue {

int axpby(float* y, const float* x, float a, float b, int64_t n, cudaStream_t s) {
  if (n > 0) axpby_kernel<<<blocks(n), 256, 0, s>>>(y, x, a, b, n);
  return launch_status();
}

int sub(const float* a, const float* b, float* out, int64_t n, cudaStream_t s) {
  if (n > 0) sub_kernel<<<blocks(n), 256, 0, s>>>(a, b, out, n);
  return launch_status();
}

int meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                 float inv_log_experts, void* meta, int ld, int col, int T, cudaStream_t s) {
  if (T > 0)
    meta_scalars_kernel<<<blocks(T), 256, 0, s>>>(entropy, confidence, continuation, mean_k, inv_log_experts,
                                                  static_cast<__nv_bfloat16*>(meta), ld, col, T);
  return launch_status();
}

}  // namespace lx::glue
