// The decode kernel, every weight format: the generic copy, which runs any program. The kernel's body is
// decode_kernel_body.cuh. The variant copies (decode_kernel_i8.cu, decode_kernel_bf16.cu) are the same body with only
// the formats their programs read compiled in, so a program of one format runs a small kernel.
#define LX_DECODE_NS variant_gen
#include "cuda/decode_kernel_body.cuh"
