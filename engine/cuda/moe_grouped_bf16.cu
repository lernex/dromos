// lx_moe_experts_grouped tile variants: BF16 math on BF16 weights.
#include "cuda/moe_grouped.cuh"

namespace lx::moeg {

LX_MOEG(kBF16, 80, "moe_grouped.bf16.sm80", LX_MV(kBF16, 80, 32, 128, 1, 4, 4, 128), LX_MV(kBF16, 80, 64, 128, 2, 4, 4, 64), LX_MV(kBF16, 80, 128, 128, 2, 2, 3, 64), LX_MV(kBF16, 80, 256, 128, 4, 2, 4, 64))

}  // namespace lx::moeg
