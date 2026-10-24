// The Metis forward pass's glue ops: the kernels.h model-glue ops under the names the model uses.
#pragma once

#include <stdint.h>

#include "cuda/kernels.h"

namespace lx::glue {

// y = a * y + b * x over n f32 elements.
inline int axpby(float* y, const float* x, float a, float b, int64_t n, void* s) {
  return lx_axpby_f32(y, x, a, b, n, s);
}
// out = a - b over n f32 elements.
inline int sub(const float* a, const float* b, float* out, int64_t n, void* s) {
  return lx_sub_f32(a, b, out, n, s);
}
// meta[t, col .. col + 3] = mean_k, entropy[t] * inv_log_experts, confidence[t], continuation[t] (bf16).
inline int meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                        float inv_log_experts, void* meta, int ld, int col, int T, void* s) {
  return lx_meta_scalars(entropy, confidence, continuation, mean_k, inv_log_experts, meta, ld, col, T, s);
}

// out[i, :] = src[cu[i + 1] - 1, :] for each of n_seq sequences (cu on the device).
inline int gather_last(const float* src, const int32_t* cu, float* out, int n_seq, int D, void* s) {
  return lx_gather_last(src, cu, out, n_seq, D, s);
}

}  // namespace lx::glue
