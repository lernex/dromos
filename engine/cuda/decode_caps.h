// The decode program's size decisions against the x buffer, as plain integer arithmetic: the builder
// (decode_program.cu) takes them, and the unit test (tests/unit/decode_caps_test.cpp) runs them under the
// old and the new x buffer size, so a change to the shared-memory layout cannot change where a product is
// split or whether an expert takes the tensor-core path without the test saying so.
#pragma once

#include <stdint.h>

namespace lx::decode::caps {

// The x buffer starts after the fixed shared-memory slots (decode_ops.h's kXOff, which asserts this value).
// A larger slot set would shrink the x buffer: at 73344 bytes Praxis's depth-memory entry splits at 1024
// instead of 2048 (the test records it), so the layout keeps this size.
constexpr int64_t kXOff = 25920;
// The opt-in shared memory per block on sm_120 (cudaDevAttrMaxSharedMemoryPerBlockOptin).
constexpr int64_t kSmemOptIn = 101376;
constexpr int64_t x_cap_of(int64_t smem_max, int64_t x_off) { return (smem_max - x_off) / 128 * 128; }

constexpr int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }
constexpr int64_t round_up(int64_t a, int64_t b) { return ceil_div(a, b) * b; }

// A product's K split: a chunk's x (kmax_m tokens by K + xpad values of xe bytes) must fit the x buffer.
// Gated rows round the count up to a power of two. Kc is the chunk width, a multiple of kstep.
struct KSplit {
  int chunks;
  int Kc;
};
constexpr KSplit gemv_k_split(int K, int xpad, int xe, int kstep, bool gated, int kmax_m, int64_t x_cap) {
  int64_t chunks = ceil_div(static_cast<int64_t>(kmax_m) * (K + xpad) * xe, x_cap);
  if (gated && chunks > 1) {
    int64_t p = 1;
    while (p < chunks) p <<= 1;
    chunks = p;
  }
  const int64_t Kc = round_up(ceil_div(K, chunks), kstep);
  return {static_cast<int>(ceil_div(K, Kc)), static_cast<int>(Kc)};
}

// A depth-memory read stages n_entries rows of D floats in the x buffer.
constexpr bool memread_fits(int n_entries, int D, int64_t x_cap) {
  return static_cast<int64_t>(n_entries) * D * 4 <= x_cap;
}

// An expert stack's K by I bf16 rows are staged in the x buffer.
constexpr bool experts_fit(int K, int I, int64_t x_cap) { return static_cast<int64_t>(K) * I * 2 <= x_cap; }

// The tensor-core expert paths stage M tokens' rows of L + 32 (up) or I + 32 (down) bf16 values.
constexpr bool expert_up_tc_fits(int M, int L, int64_t x_cap) {
  return static_cast<int64_t>(M) * (L + 32) * 2 <= x_cap;
}
constexpr bool expert_down_tc_fits(int M, int K, int I, int64_t x_cap) {
  return static_cast<int64_t>(M) * K * (I + 32) * 2 <= x_cap;
}

// The depth-memory entry's kc: the widest slice of its D values whose kmax_m tokens of Km + kc floats fit
// the x buffer. -1 when no slice does.
constexpr int entry_kc(int D, int Km, int kmax_m, int64_t x_cap) {
  int kc = D;
  while (kmax_m * (Km + kc) * 4 > x_cap && kc % 16 == 0) kc /= 2;  // whatever M is
  if (kmax_m * (Km + kc) * 4 > x_cap || D % kc) return -1;
  return kc;
}

}  // namespace lx::decode::caps
