// The decode kernel for programs whose GEMV weights are INT8: the body with the INT8 readers, and the f32 routers and
// depth head and the BF16 entry and experts where a program has them (decode_kernel_body.cuh).
#define LX_DECODE_NS variant_i8
#define LX_DECODE_FMTS ((1u << LX_W_INT8) | (1u << LX_W_BF16) | (1u << kF32))
#include "cuda/decode_kernel_body.cuh"
