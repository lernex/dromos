// lx_gemm_w tile variants: W8A16: INT8 and FP8 weights expanded to bf16.
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kW8I8, 80, "gemm_w.w8i8.sm80", LX_V(kW8I8, 80, 32, 128, 1, 4, 4, 128), LX_V(kW8I8, 80, 64, 128, 2, 4, 4, 128), LX_V(kW8I8, 80, 128, 128, 2, 4, 3, 128), LX_V(kW8I8, 80, 128, 64, 2, 2, 3, 128), LX_V(kW8I8, 80, 128, 128, 2, 2, 3, 128), LX_VS(kW8I8, 80, 32, 128, 1, 4, 4, 128, 2), LX_VS(kW8I8, 80, 32, 128, 1, 4, 4, 128, 4))
LX_FAMILY(kW8F8, 80, "gemm_w.w8f8.sm80", LX_V(kW8F8, 80, 32, 128, 1, 4, 4, 128), LX_V(kW8F8, 80, 64, 128, 2, 4, 4, 128), LX_V(kW8F8, 80, 128, 128, 2, 4, 3, 128), LX_V(kW8F8, 80, 128, 64, 2, 2, 3, 128), LX_V(kW8F8, 80, 128, 128, 2, 2, 3, 128), LX_VS(kW8F8, 80, 32, 128, 1, 4, 4, 128, 2), LX_VS(kW8F8, 80, 32, 128, 1, 4, 4, 128, 4))
LX_FAMILY(kW8F8, 120, "gemm_w.w8f8.sm120", LX_V(kW8F8, 120, 32, 128, 1, 4, 4, 128), LX_V(kW8F8, 120, 64, 128, 2, 4, 4, 128), LX_V(kW8F8, 120, 128, 128, 2, 4, 3, 128), LX_V(kW8F8, 120, 128, 64, 2, 2, 3, 128), LX_V(kW8F8, 120, 128, 128, 2, 2, 3, 128), LX_VS(kW8F8, 120, 32, 128, 1, 4, 4, 128, 2), LX_VS(kW8F8, 120, 32, 128, 1, 4, 4, 128, 4))

}  // namespace lx::gemmw
