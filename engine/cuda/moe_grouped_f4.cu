// lx_moe_experts_grouped tile variants: FP4 math (block-scaled NVFP4 x NVFP4).
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kF4, 120, "moe_grouped.f4.sm120", LX_MV(kF4, 120, 32, 128, 1, 4, 4, 64), LX_MV(kF4, 120, 64, 128, 2, 4, 3, 64), LX_MV(kF4, 120, 128, 128, 2, 4, 3, 64), LX_MV(kF4, 120, 256, 128, 4, 2, 3, 64))

}  // namespace lx::moeg
