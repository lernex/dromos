// The decode kernel for the BF16 family (decode_family.h): the BF16 and BF16X readers, and the f32 routers and depth
// head. A program whose GEMV, expert and entry weights are BF16 or BF16X runs this copy. The body is
// decode_kernel_body.cuh.
#include "cuda/decode_family.h"

#define LX_DECODE_NS family_bf16
#define LX_DECODE_FMTS (LX_FAMILY_BF16 | (1u << LX_W_F32))
#include "cuda/decode_kernel_body.cuh"
