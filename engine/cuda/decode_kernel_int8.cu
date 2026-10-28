// The decode kernel for the INT8 family (decode_family.h): the INT8 readers, with the BF16 and BF16X parts a model keeps
// wider, and the f32 routers and depth head. The body is decode_kernel_body.cuh.
#include "cuda/decode_family.h"

#define LX_DECODE_NS family_int8
#define LX_DECODE_FMTS (LX_FAMILY_INT8 | (1u << LX_W_F32))
#include "cuda/decode_kernel_body.cuh"
