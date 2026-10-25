// Shared by the CPU backend's op files (compiled once, for the architecture's baseline). The per-ISA
// kernels do not include this header: they see only kern/table.h, so nothing here is ever compiled with
// instructions the CPU may lack.
#pragma once

#include <stdint.h>
#include <string.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <type_traits>

#include "cpu/kern/table.h"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

namespace lx::cpu {

// Status codes: the CUDA runtime's, so callers read every backend's errors the same way.
constexpr int kOk = 0;
constexpr int kInvalid = 1;        // cudaErrorInvalidValue
constexpr int kNoMemory = 2;       // cudaErrorMemoryAllocation
constexpr int kBadDevice = 101;    // cudaErrorInvalidDevice
constexpr int kBadHandle = 400;    // cudaErrorInvalidResourceHandle
constexpr int kNotSupported = LX_ERR_NOT_SUPPORTED;

// Records the status of the last failing call for lx_rt_last_error, and returns it.
int fail(int status);

// ---- Number formats ---------------------------------------------------------------------------

inline float bits_f32(uint32_t u) {
  float f;
  memcpy(&f, &u, 4);
  return f;
}
inline uint32_t f32_bits(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  return u;
}

inline float bf16_to_f32(uint16_t v) { return bits_f32(static_cast<uint32_t>(v) << 16); }

// Round to nearest, ties to even (NaN stays NaN).
inline uint16_t f32_to_bf16(float f) {
  uint32_t u = f32_bits(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
inline float round_bf16(float f) { return bf16_to_f32(f32_to_bf16(f)); }

// e4m3 (the "fn" variant: no infinities, 0x7f and 0xff are NaN) to f32, exact.
float e4m3_to_f32(uint8_t b);
// f32 to e4m3, round to nearest even, saturating at +-448 (NaN stays NaN).
uint8_t f32_to_e4m3(float f);
// e2m1 code (low 4 bits: sign bit 3) to f32, exact.
float e2m1_to_f32(uint32_t code);
// The nearest e2m1 code of x (ties to the even code, saturating at 6; zero has no sign).
uint32_t f32_to_e2m1(float x);

// lx_quantize_act for one row: q (int8, e4m3, or packed e2m1 with block_scale), and its scale.
void quantize_act_row(const void* x, int x_dtype, int64_t row, int64_t ld_x, int math, int K, uint8_t* q,
                      float* scale, uint8_t* block_scale);

inline float silu(float v) { return v / (1.0f + std::exp(-v)); }
inline float sigmoid(float v) { return 1.0f / (1.0f + std::exp(-v)); }
inline float softplus(float v) { return v > 20.0f ? v : std::log1p(std::exp(v)); }

// Reads element i of a buffer of lx_dtype d as f32; writes one.
inline float load_as_f32(const void* p, int d, int64_t i) {
  return d == LX_BF16 ? bf16_to_f32(static_cast<const uint16_t*>(p)[i]) : static_cast<const float*>(p)[i];
}
inline void store_from_f32(void* p, int d, int64_t i, float v) {
  if (d == LX_BF16) static_cast<uint16_t*>(p)[i] = f32_to_bf16(v);
  else static_cast<float*>(p)[i] = v;
}
inline bool valid_dtype(int d) { return d == LX_F32 || d == LX_BF16; }

// ---- ISA ---------------------------------------------------------------------------------------

// The kernel table of the level in effect (lx_arch, after any override).
const Kern& kern();

// The highest level this CPU supports and the library was built with: lx_arch before any override.
int real_level();

// A level's instruction set, in words; the maths the level in effect multiplies with instructions of its own
// (lx_device_props.native_math); and whether the chip has AMX tiles the OS lets programs use.
const char* level_isa(int code);
int native_math();
bool amx_present();

// ---- Threads ------------------------------------------------------------------------------------

// Worker threads plus the caller. LX_THREADS overrides the default (the machine's physical cores).
int threads();

using RangeFn = void (*)(void* ctx, int64_t begin, int64_t end);
// Runs fn over [0, n) in chunks of `grain`, claimed dynamically by the workers and the caller, and
// returns when every chunk is done. Called from inside a chunk it runs inline (no nesting).
void run_parallel(int64_t n, int64_t grain, RangeFn fn, void* ctx);

template <class F>
inline void parallel_for(int64_t n, int64_t grain, F&& f) {
  if (n <= 0) return;
  if (grain < 1) grain = 1;
  if (n <= grain) {
    f(int64_t{0}, n);
    return;
  }
  run_parallel(n, grain, [](void* c, int64_t b, int64_t e) { (*static_cast<std::remove_reference_t<F>*>(c))(b, e); },
               &f);
}

// Chunk size that gives each thread about `per_thread` chunks of n items, at least `min_grain`.
inline int64_t grain_for(int64_t n, int64_t min_grain, int per_thread = 4) {
  const int64_t chunks = static_cast<int64_t>(threads()) * per_thread;
  int64_t g = (n + chunks - 1) / chunks;
  return g < min_grain ? min_grain : g;
}

// ---- Profile ---------------------------------------------------------------------------------------

// LX_CPU_PROFILE=1: every op's calls and wall time, totalled per op and printed to stderr at exit. Off,
// a scope costs one predictable branch.
extern bool g_profile;
struct ProfScope {
  const char* name;
  int64_t t0;
  explicit ProfScope(const char* n) : name(n), t0(g_profile ? now_ns() : 0) {}
  ~ProfScope() {
    if (t0) record(name, now_ns() - t0);
  }
  static int64_t now_ns();
  static void record(const char* name, int64_t ns);
};
#define LX_PROFILE(name) ::lx::cpu::ProfScope lx_profile_scope_(name)

// ---- Scratch -------------------------------------------------------------------------------------

// A per-thread scratch buffer of at least `bytes`, 64-byte aligned, kept between calls (grown, never
// shrunk), so steady-state calls allocate nothing. Slot picks one of a few independent buffers.
void* scratch(size_t bytes, int slot = 0);

}  // namespace lx::cpu
