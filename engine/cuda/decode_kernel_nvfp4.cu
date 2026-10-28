// The decode kernel for the NVFP4 family (decode_family.h): the NVFP4 expert readers with the FP8 dense projections the
// GPTQ folder keeps, the BF16 and BF16X parts, and the f32 routers and depth head. The body is decode_kernel_body.cuh.
#include "cuda/decode_family.h"

#define LX_DECODE_NS family_nvfp4
#define LX_DECODE_FMTS (LX_FAMILY_NVFP4 | (1u << LX_W_F32))
#include "cuda/decode_kernel_body.cuh"
