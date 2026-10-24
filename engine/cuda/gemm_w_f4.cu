// lx_gemm_w tile variants: FP4 math (block-scaled NVFP4 MMA, sm_120).
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kF4, 120, "gemm_w.f4.sm120", LX_V(kF4, 120, 32, 128, 1, 4, 4, 64), LX_V(kF4, 120, 64, 128, 2, 4, 3, 64), LX_V(kF4, 120, 128, 128, 2, 4, 2, 64), LX_V(kF4, 120, 128, 128, 2, 2, 2, 64), LX_V(kF4, 120, 128, 128, 2, 4, 3, 64), LX_V(kF4, 120, 128, 256, 2, 4, 3, 64), LX_V(kF4, 120, 256, 128, 4, 2, 3, 64), LX_VS(kF4, 120, 32, 128, 1, 4, 4, 64, 2), LX_VS(kF4, 120, 32, 128, 1, 4, 4, 64, 4))

}  // namespace lx::gemmw
