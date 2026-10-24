// Which architecture the kernels target, and which variant of a kernel runs for a shape. C ABI so
// the tests can set both through ctypes.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LX_API __attribute__((visibility("default")))

// The architecture kernel dispatchers choose a code path for, as major * 10 + minor (120 for an
// RTX 50). It is the current device's, lowered by LX_ARCH_OVERRIDE or lx_set_arch_override, so the
// sm_80 / sm_89 / sm_90 paths can run and be tested on a newer card. Raising it is refused.
LX_API int lx_arch(void);
LX_API int lx_set_arch_override(int arch);  // 0 clears; returns the arch now in effect

// Kernel variants. A kernel with alternatives (tile shapes, split counts, thread layouts) asks at
// launch for the variant chosen for its shape, and uses its own heuristic on -1. The autotuner
// (engine/runtime/tuner) times each variant and fills the table. M is bucketed to the next power of
// two (so one entry covers a range of batch sizes); N and K are exact.
LX_API int lx_tune_get(const char* kernel, int M, int N, int K);
LX_API void lx_tune_set(const char* kernel, int M, int N, int K, int variant);
LX_API void lx_tune_clear(void);
// What kernels registered: their names and, per kernel, how many variants and what each is.
LX_API int lx_tune_kernels(const char** names, int capacity);  // returns the total count
LX_API int lx_tune_variants(const char* kernel);               // 0 when unknown
LX_API const char* lx_tune_variant_name(const char* kernel, int variant);

#ifdef __cplusplus
}

#include <initializer_list>

namespace lx {
// Registers a kernel's variants at load time:
//   static const lx::TunableRegistration reg("gemv_w", {"rg2_s1", "rg4_s1", "rg4_s2"});
struct TunableRegistration {
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants);
};
}  // namespace lx
#endif
