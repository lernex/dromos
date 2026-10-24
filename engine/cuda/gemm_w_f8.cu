// lx_gemm_w tile variants: FP8 math: the sm_89 family (legacy e4m3 MMA) and the sm_120 family (block-scaled kind).
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kF8, 89, "gemm_w.f8.sm89", LX_V(kF8, 89, 32, 128, 1, 4, 4, 128), LX_V(kF8, 89, 64, 128, 2, 4, 4, 64), LX_V(kF8, 89, 128, 128, 2, 4, 3, 64), LX_V(kF8, 89, 128, 128, 2, 2, 3, 64), LX_V(kF8, 89, 128, 128, 2, 4, 4, 64), LX_V(kF8, 89, 128, 256, 2, 4, 4, 64), LX_V(kF8, 89, 256, 128, 4, 2, 4, 64), LX_VS(kF8, 89, 32, 128, 1, 4, 4, 128, 2), LX_VS(kF8, 89, 32, 128, 1, 4, 4, 128, 4))
LX_FAMILY(kF8, 120, "gemm_w.f8.sm120", LX_V(kF8, 120, 32, 128, 1, 4, 4, 128), LX_V(kF8, 120, 64, 128, 2, 4, 4, 64), LX_V(kF8, 120, 128, 128, 2, 4, 3, 64), LX_V(kF8, 120, 128, 128, 2, 2, 3, 64), LX_V(kF8, 120, 128, 128, 2, 4, 4, 64), LX_V(kF8, 120, 128, 256, 2, 4, 4, 64), LX_V(kF8, 120, 256, 128, 4, 2, 4, 64), LX_VS(kF8, 120, 32, 128, 1, 4, 4, 128, 2), LX_VS(kF8, 120, 32, 128, 1, 4, 4, 128, 4))

}  // namespace lx::gemmw
