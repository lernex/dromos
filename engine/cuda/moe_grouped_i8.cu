// lx_moe_experts_grouped tile variants: INT8 math (IMMA).
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kI8, 80, "moe_grouped.i8.sm80", LX_MV(kI8, 80, 32, 128, 1, 4, 4, 128), LX_MV(kI8, 80, 64, 128, 2, 4, 4, 64), LX_MV(kI8, 80, 128, 128, 2, 2, 3, 64), LX_MV(kI8, 80, 256, 128, 4, 2, 4, 64))

}  // namespace lx::moeg
