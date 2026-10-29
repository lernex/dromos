// What libdromos_cpu.so exports beyond kernels.h: the controls the offload runtime's CPU lane uses
// (engine/runtime/offload.h). Everything else is the kernels.h ABI.
#pragma once

#include "cuda/kernels.h"

#ifdef __cplusplus
extern "C" {
#endif

// The threads the CPU kernels run on: the calling thread plus the pool's workers (LX_THREADS, default
// the machine's physical cores).
LX_API int lx_cpu_threads(void);

// Tests: run the generic (portable C++) kernels even where the CPU has vector instructions. The same as
// lx_set_arch_override(1); 0 restores the best level.
LX_API void lx_cpu_force_scalar(int on);

#ifdef __cplusplus
}
#endif
