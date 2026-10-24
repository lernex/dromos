#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace lx::glue {

// y = a * y + b * x over n f32 elements.
int axpby(float* y, const float* x, float a, float b, int64_t n, cudaStream_t s);
// out = a - b over n f32 elements.
int sub(const float* a, const float* b, float* out, int64_t n, cudaStream_t s);
// meta[t, col .. col + 3] = mean_k, entropy[t] * inv_log_experts, confidence[t], continuation[t] (bf16).
int meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                 float inv_log_experts, void* meta, int ld, int col, int T, cudaStream_t s);

}  // namespace lx::glue
