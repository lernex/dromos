// lx_moe_experts_grouped tile variants: NVFP4 weights under BF16 math (W4A16) and FP8 math (W4A8).
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kW4A16, 80, "moe_grouped.w4a16.sm80", LX_MV(kW4A16, 80, 32, 128, 1, 4, 4, 128), LX_MV(kW4A16, 80, 64, 128, 2, 4, 4, 128), LX_MV(kW4A16, 80, 128, 128, 2, 4, 3, 128))

LX_MOEG(kW4A8, 120, "moe_grouped.w4a8.sm120", LX_MV(kW4A8, 120, 32, 128, 1, 4, 4, 128), LX_MV(kW4A8, 120, 64, 128, 2, 4, 4, 128), LX_MV(kW4A8, 120, 128, 128, 2, 4, 3, 128))

}  // namespace lx::moeg
