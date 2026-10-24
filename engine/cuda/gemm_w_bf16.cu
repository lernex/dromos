// lx_gemm_w tile variants: bf16 x bf16 (for an f32 bias, which cuBLASLt cannot take as is).
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kBF16, 80, "gemm_w.bf16.sm80", LX_V(kBF16, 80, 32, 128, 1, 4, 4, 128), LX_V(kBF16, 80, 64, 128, 2, 4, 4, 64), LX_V(kBF16, 80, 128, 128, 2, 4, 3, 64), LX_V(kBF16, 80, 128, 128, 2, 2, 3, 64), LX_V(kBF16, 80, 128, 128, 2, 4, 4, 64), LX_V(kBF16, 80, 128, 256, 2, 4, 4, 64), LX_V(kBF16, 80, 256, 128, 4, 2, 4, 64))

}  // namespace lx::gemmw
