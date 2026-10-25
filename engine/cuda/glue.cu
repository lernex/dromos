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

// The routing features' input row per token: [a | b | c | d] in bf16. Columns run over the grid's x, tokens over y.
__global__ void concat4_kernel(const float* a, const float* b, const float* c, const float* d, __nv_bfloat16* out,
                               int D, int R, int T) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  const int W = 3 * D + R;
  if (j >= W || t >= T) return;
  float v;
  if (j < D) v = a[static_cast<int64_t>(t) * D + j];
  else if (j < 2 * D) v = b[static_cast<int64_t>(t) * D + (j - D)];
  else if (j < 3 * D) v = c[static_cast<int64_t>(t) * D + (j - 2 * D)];
  else v = d[static_cast<int64_t>(t) * R + (j - 3 * D)];
  out[static_cast<int64_t>(t) * W + j] = __float2bfloat16_rn(v);
}

// The router's input after the features join: the latent's bf16 copy, and the features' f32 copy into the tail.
__global__ void pack_router_kernel(float* router_in, __nv_bfloat16* latent, const float* features, int Lat, int R,
                                   int T) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  const int W = Lat + R;
  if (j >= W || t >= T) return;
  if (j < Lat) latent[static_cast<int64_t>(t) * Lat + j] = __float2bfloat16_rn(router_in[static_cast<int64_t>(t) * W + j]);
  else router_in[static_cast<int64_t>(t) * W + j] = features[static_cast<int64_t>(t) * R + (j - Lat)];
}

// A memory entry's metadata row: the summary as bf16, then the four scalars (as meta_scalars writes them).
__global__ void meta_entry_kernel(const float* summary, const float* entropy, const float* confidence,
                                  const float* continuation, float mean_k, float inv_log_experts, __nv_bfloat16* meta,
                                  int R, int T) {
  const int j = blockIdx.x * blockDim.x + threadIdx.x;
  const int t = blockIdx.y;
  const int ld = R + 4;
  if (j >= ld || t >= T) return;
  __nv_bfloat16* row = meta + static_cast<int64_t>(t) * ld;
  if (j < R) row[j] = __float2bfloat16_rn(summary[static_cast<int64_t>(t) * R + j]);
  else if (j == R) row[j] = __float2bfloat16_rn(mean_k);
  else if (j == R + 1) row[j] = __float2bfloat16_rn(entropy[t] * inv_log_experts);
  else if (j == R + 2) row[j] = __float2bfloat16_rn(confidence[t]);
  else row[j] = __float2bfloat16_rn(continuation[t]);
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

int lx_concat4_bf16(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R, int T,
                    void* stream) {
  const int W = 3 * D + R;
  if (T > 0 && W > 0)
    concat4_kernel<<<dim3(blocks(W), T), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        a, b, c, d, static_cast<__nv_bfloat16*>(out), D, R, T);
  return lx::launch_status();
}

int lx_pack_router(float* router_in, void* latent, const float* features, int Lat, int R, int T, void* stream) {
  const int W = Lat + R;
  if (T > 0 && W > 0)
    pack_router_kernel<<<dim3(blocks(W), T), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        router_in, static_cast<__nv_bfloat16*>(latent), features, Lat, R, T);
  return lx::launch_status();
}

int lx_meta_entry(const float* summary, const float* entropy, const float* confidence, const float* continuation,
                  float mean_k, float inv_log_experts, void* meta, int R, int T, void* stream) {
  if (T > 0)
    meta_entry_kernel<<<dim3(blocks(R + 4), T), 256, 0, static_cast<cudaStream_t>(stream)>>>(
        summary, entropy, confidence, continuation, mean_k, inv_log_experts, static_cast<__nv_bfloat16*>(meta), R, T);
  return lx::launch_status();
}

}  // extern "C"
