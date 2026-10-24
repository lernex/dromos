// lx_gemm_w tile variants: NVFP4 weights under BF16 (W4A16) and FP8 (W4A8) math.
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kW4A16, 80, "gemm_w.w4a16.sm80", LX_V(kW4A16, 80, 32, 128, 1, 4, 4, 128), LX_V(kW4A16, 80, 64, 128, 2, 4, 4, 128), LX_V(kW4A16, 80, 128, 128, 2, 4, 3, 128), LX_V(kW4A16, 80, 128, 64, 2, 2, 3, 128), LX_V(kW4A16, 80, 128, 128, 2, 2, 3, 128), LX_VS(kW4A16, 80, 32, 128, 1, 4, 4, 128, 2), LX_VS(kW4A16, 80, 32, 128, 1, 4, 4, 128, 4))
LX_FAMILY(kW4A8, 120, "gemm_w.w4a8.sm120", LX_V(kW4A8, 120, 32, 128, 1, 4, 4, 128), LX_V(kW4A8, 120, 64, 128, 2, 4, 4, 128), LX_V(kW4A8, 120, 128, 128, 2, 4, 3, 128), LX_V(kW4A8, 120, 128, 64, 2, 2, 3, 128), LX_V(kW4A8, 120, 128, 128, 2, 2, 3, 128), LX_VS(kW4A8, 120, 32, 128, 1, 4, 4, 128, 2), LX_VS(kW4A8, 120, 32, 128, 1, 4, 4, 128, 4))

}  // namespace lx::gemmw
