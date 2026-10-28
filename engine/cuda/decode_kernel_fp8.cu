// The decode kernel for the FP8 family (decode_family.h): the FP8 readers (the FP8 checkpoints' experts and dense
// projections), with the BF16 and BF16X parts they keep wider, and the f32 routers and depth head. The body is
// decode_kernel_body.cuh.
#include "cuda/decode_family.h"

#define LX_DECODE_NS family_fp8
#define LX_DECODE_FMTS (LX_FAMILY_FP8 | (1u << LX_W_F32))
#include "cuda/decode_kernel_body.cuh"
