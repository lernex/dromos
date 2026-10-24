// lx_moe_experts_grouped tile variants: FP8 math: the legacy e4m3 MMA (sm_89) and the block-scaled kind (sm_120).
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kF8, 89, "moe_grouped.f8.sm89", LX_MV(kF8, 89, 32, 128, 1, 4, 4, 128), LX_MV(kF8, 89, 64, 128, 2, 4, 4, 64), LX_MV(kF8, 89, 128, 128, 2, 2, 3, 64), LX_MV(kF8, 89, 256, 128, 4, 2, 4, 64))

LX_MOEG(kF8, 120, "moe_grouped.f8.sm120", LX_MV(kF8, 120, 32, 128, 1, 4, 4, 128), LX_MV(kF8, 120, 64, 128, 2, 4, 4, 64), LX_MV(kF8, 120, 128, 128, 2, 2, 3, 64), LX_MV(kF8, 120, 256, 128, 4, 2, 4, 64))

}  // namespace lx::moeg
