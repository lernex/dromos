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

// The autotuner's entry points, so host code times and stores a kernel's variants through the ABI and
// never sees the kernels' C++ types. A problem is one timed case: M rows against an N x K weight, or
// against `experts` such weights, in the weight format `weight` and math `math` (0 for both when the
// kernel has no such choice). M, N and K are as lx::TuneProblem has them.
typedef struct {
  int M, N, K;
  int experts;
  int weight;
  int math;
} lx_tune_problem;
// Runs variant `variant` of a registered kernel on the problem and writes its result as f32 to `out`
// (M * N elements). Returns LX_ERR_NOT_SUPPORTED when the variant or the problem cannot run here.
LX_API int lx_tune_run(const char* kernel, int variant, const lx_tune_problem* problem, float* out, void* stream);
// 1 when the kernel registered a runner, so lx_tune_run can time its variants; else 0.
LX_API int lx_tune_runnable(const char* kernel);
// The table key of a kernel's problem: the kernel name with the format, math and expert count appended
// when they are not the defaults. Writes it NUL-terminated to `out` (at most `capacity` bytes) and
// returns its length.
LX_API int lx_tune_key(const char* kernel, const lx_tune_problem* problem, char* out, int capacity);

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
struct TunableRegistration {
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants);
  TunableRegistration(const char* kernel, std::initializer_list<const char*> variants, TuneRunner runner);
};

}  // namespace lx
#endif
