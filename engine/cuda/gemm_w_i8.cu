// lx_gemm_w tile variants: INT8 math (IMMA).
#include "cuda/gemm_w.cuh"

namespace lx::gemmw {

LX_FAMILY(kI8, 80, "gemm_w.i8.sm80", LX_V(kI8, 80, 32, 128, 1, 4, 4, 128), LX_V(kI8, 80, 64, 128, 2, 4, 4, 64), LX_V(kI8, 80, 128, 128, 2, 4, 3, 64), LX_V(kI8, 80, 128, 128, 2, 2, 3, 64), LX_V(kI8, 80, 128, 128, 2, 4, 4, 64), LX_V(kI8, 80, 128, 256, 2, 4, 4, 64), LX_V(kI8, 80, 256, 128, 4, 2, 4, 64))

}  // namespace lx::gemmw
