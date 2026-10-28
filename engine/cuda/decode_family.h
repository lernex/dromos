// The decode kernel's weight-format families. A decode program runs one kernel copy, and each family has its own copy
// (decode_kernel_bf16.cu, decode_kernel_int8.cu, decode_kernel_fp8.cu, decode_kernel_nvfp4.cu), which compiles only the
// readers its family reads. One copy of every format is a single kernel the compiler took about half an hour over, and
// no program needs it.
//
// Every family reads BF16X, the lossless bf16 copies the planner makes of a model's BF16 projections in any model format,
// and BF16, the memory entry and the parts a quantized model keeps wider. Every copy reads the f32 routers and the depth
// head. A program's GEMV, expert and entry formats must sit in one family: the smallest family that reads them all runs
// it. A mix no family reads has no copy, so lx_program_build refuses it (LX_ERR_NOT_SUPPORTED) and the model keeps the
// kernel-per-op path (metis.cpp, the program gate).
#pragma once

#include "cuda/kernels.h"

// The formats each family's copy reads, as bits of lx_wtype. The f32 bit is in every copy, so it is not listed.
#define LX_FAMILY_BF16 ((1u << LX_W_BF16) | (1u << LX_W_BF16X))
#define LX_FAMILY_INT8 (LX_FAMILY_BF16 | (1u << LX_W_INT8))
#define LX_FAMILY_FP8 (LX_FAMILY_BF16 | (1u << LX_W_FP8))
#define LX_FAMILY_NVFP4 (LX_FAMILY_FP8 | (1u << LX_W_NVFP4))

namespace lx::decode {

// The families, tried smallest first: BF16 sits inside every family and FP8 inside NVFP4, so the first family that reads
// all of a program's formats is the one its copy is built for.
enum Family : int { kFamilyBF16 = 0, kFamilyINT8, kFamilyFP8, kFamilyNVFP4, kFamilyCount, kNoFamily = -1 };

// The family whose copy runs a program that reads the formats in `quant` (bits of lx_wtype; the f32 bit is ignored), or
// kNoFamily when no family reads them all.
inline int family_for(unsigned quant) {
  constexpr unsigned kReads[kFamilyCount] = {LX_FAMILY_BF16, LX_FAMILY_INT8, LX_FAMILY_FP8, LX_FAMILY_NVFP4};
  quant &= ~(1u << LX_W_F32);
  for (int f = 0; f < kFamilyCount; ++f)
    if ((quant & ~kReads[f]) == 0) return f;
  return kNoFamily;
}

}  // namespace lx::decode
