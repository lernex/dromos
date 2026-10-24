// NVIDIA test shim: the bfloat16 names the HIP port uses, over CUDA's bf16 types (see hip_runtime.h).
#pragma once

#include <cuda_bf16.h>

typedef __nv_bfloat16 __hip_bfloat16;
typedef __nv_bfloat162 __hip_bfloat162;
#define __float2bfloat16 __float2bfloat16_rn
