// Which architecture the kernels target, and which variant of a kernel runs for a shape. C ABI so
// the tests can set both through ctypes.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LX_API __attribute__((visibility("default")))

// lx_tune_set's rejections: the kernel's order-class rule refuses the variant (see lx::TuneAdmit), or the
// kernel has no such variant. (LX_ERR_NOT_SUPPORTED is 801, in kernels.h.)
#ifndef LX_ERR_TUNE_REJECTED
#define LX_ERR_TUNE_REJECTED 802
#endif
#ifndef LX_ERR_TUNE_VARIANT
#define LX_ERR_TUNE_VARIANT 803
#endif

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
// Fixes a variant for a key. A kernel with an order-class rule refuses a variant the rule does not admit at
// that key: the call returns LX_ERR_TUNE_REJECTED and the table is unchanged. Returns 0 when stored.
LX_API int lx_tune_set(const char* kernel, int M, int N, int K, int variant);
// Stores a variant with no admission check. For the offline tuner's class discovery and for tests that run
// every variant; the engine never calls it.
LX_API int lx_tune_force(const char* kernel, int M, int N, int K, int variant);
LX_API void lx_tune_clear(void);
// What kernels registered: their names and, per kernel, how many variants and what each is.
LX_API int lx_tune_kernels(const char** names, int capacity);  // returns the total count
LX_API int lx_tune_variants(const char* kernel);               // 0 when unknown
LX_API const char* lx_tune_variant_name(const char* kernel, int variant);

#ifdef __cplusplus
}

#include <initializer_list>
#include <string>

namespace lx {

// The first architecture with thread-block clusters and programmatic dependent launch (sm_90).
// Kernels take their one-block paths below it, and the host checks this before a cluster launch.
constexpr int kClusterArch = 90;

// One problem a tunable kernel is timed on: a product of M rows with an N x K weight, or with
// `experts` such weights (an expert stack), in the weight format and math the plan runs it in.
// `weight` is an lx_wtype and `math` an lx_math (0 for both when the kernel has no such choice).
struct TuneProblem {
  int M = 1, N = 1, K = 1;
  int experts = 1;
  int weight = 0;
  int math = 0;
};

// Runs variant `variant` of a kernel on a problem. The kernel makes its own inputs (deterministic,
// and cached per problem), launches that variant rather than the table's choice, and writes its
// result as f32 to `out`, M * N elements, so the tuner can check every variant against variant 0.
// Returns a cudaError_t. cudaErrorNotSupported marks a variant or a problem this device or build
// cannot run; the tuner skips it.
using TuneRunner = int (*)(int variant, const TuneProblem& problem, float* out, void* stream);

// Admission: a kernel whose results must not depend on the batch size registers one. It says whether a
// variant may be the table's choice for a key. The rule must depend on the shape (N, K) and the variant
// only: M is passed so a rule can see it, but one that uses M breaks batch invariance. lx_tune_set asks
// it; lx_tune_force does not.
using TuneAdmit = bool (*)(int variant, int M, int N, int K);

// The key a kernel's table entries live under: the kernel's name, with its format, math and expert
// count appended when they are not the defaults, so one kernel keeps a winner per format it runs.
// Kernels call lx_tune_get with the same string.
std::string tune_key(const char* kernel, const TuneProblem& problem);

// The runner registered for a kernel, or null.
TuneRunner tunable_runner(const char* kernel);

// Registers a kernel's variants at load time:
//   static const lx::TunableRegistration reg("gemv_w", {"rg2_s1", "rg4_s1", "rg4_s2"});
// and, for the autotuner, its runner too:
//   static const lx::TunableRegistration reg("mhc_read", {"block", "cluster8"}, &run_mhc_read);
// and, for a kernel whose results must not depend on M, the rule that admits its variants:
//   static const lx::TunableRegistration reg("gemv_w_bf16", {"s1r2d2", "s2r4d3g"}, &admits_bf16);
struct TunableRegistration {
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants);
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants, TuneRunner runner);
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants, TuneAdmit admit);
};

}  // namespace lx
#endif
