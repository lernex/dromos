// Small kernels the Metis forward pass needs between the library kernels (kernels.h, "Model glue"). They
// are specific to how the model wires its pieces together; every backend provides them as plain ops.
#include "cuda/common.cuh"
#include "cuda/kernels.h"

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

// out[i, :] = src[cu[i + 1] - 1, :]: each sequence's last token, read on the device so a captured
// graph stays valid for any sequence lengths with the same counts.
__global__ void gather_last_kernel(const float* src, const int32_t* cu, float* out, int D) {
  const int64_t row = cu[blockIdx.x + 1] - 1;
  for (int d = threadIdx.x; d < D; d += blockDim.x) out[static_cast<int64_t>(blockIdx.x) * D + d] = src[row * D + d];
}

int blocks(int64_t n) { return static_cast<int>((n + 255) / 256); }

}  // namespace

extern "C" {

int lx_axpby_f32(float* y, const float* x, float a, float b, int64_t n, void* stream) {
  if (n > 0) axpby_kernel<<<blocks(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(y, x, a, b, n);
  return lx::launch_status();
}

int lx_sub_f32(const float* a, const float* b, float* out, int64_t n, void* stream) {
  if (n > 0) sub_kernel<<<blocks(n), 256, 0, static_cast<cudaStream_t>(stream)>>>(a, b, out, n);
  return lx::launch_status();
}

int lx_meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                    float inv_log_experts, void* meta, int ld, int col, int T, void* stream) {
  if (T > 0)
    meta_scalars_kernel<<<blocks(T), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        entropy, confidence, continuation, mean_k, inv_log_experts, static_cast<__nv_bfloat16*>(meta), ld, col, T);
  return lx::launch_status();
}

int lx_gather_last(const float* src, const int32_t* cu, float* out, int n_seq, int D, void* stream) {
  if (n_seq > 0) gather_last_kernel<<<n_seq, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, cu, out, D);
  return lx::launch_status();
}

}  // extern "C"
