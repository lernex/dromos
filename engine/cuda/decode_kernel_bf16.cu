// The decode kernel for programs whose GEMV weights are BF16: the body with the BF16 readers, and the f32 routers and
// depth head where a program has them (decode_kernel_body.cuh).
#define LX_DECODE_NS variant_bf16
#define LX_DECODE_FMTS ((1u << LX_W_BF16) | (1u << kF32))
#include "cuda/decode_kernel_body.cuh"
