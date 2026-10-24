// lx_moe_experts_grouped tile variants: BF16 math on INT8 and FP8 weights (W8A16).
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kW8I8, 80, "moe_grouped.w8i8.sm80", LX_MV(kW8I8, 80, 32, 128, 1, 4, 4, 128), LX_MV(kW8I8, 80, 64, 128, 2, 4, 4, 128), LX_MV(kW8I8, 80, 128, 128, 2, 4, 3, 128))

LX_MOEG(kW8F8, 80, "moe_grouped.w8f8.sm80", LX_MV(kW8F8, 80, 32, 128, 1, 4, 4, 128), LX_MV(kW8F8, 80, 64, 128, 2, 4, 4, 128), LX_MV(kW8F8, 80, 128, 128, 2, 4, 3, 128))

LX_MOEG(kW8F8, 120, "moe_grouped.w8f8.sm120", LX_MV(kW8F8, 120, 32, 128, 1, 4, 4, 128), LX_MV(kW8F8, 120, 64, 128, 2, 4, 4, 128), LX_MV(kW8F8, 120, 128, 128, 2, 4, 3, 128))

}  // namespace lx::moeg
