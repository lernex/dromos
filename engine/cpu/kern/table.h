// The hot kernels of the CPU backend, as one table of function pointers per ISA level. kern/isa.cpp is
// compiled once per level (engine/cpu/CMakeLists.txt) and fills a table; isa.cpp picks the table of the
// level in effect. Plain structs only: this header is shared by code built with different instruction
// sets, so it must not define anything with external linkage.
#pragma once

#include <stdint.h>

namespace lx::cpu {

// A weight's format as the kernels read it: kernels.h's lx_wtype, plus f32 (the routers, the depth head).
enum : int { kW_BF16 = 0, kW_INT8 = 1, kW_FP8 = 2, kW_NVFP4 = 3, kW_BF16X = 4, kW_F32 = 5 };

// The LX_W_BF16X layout (kernels.h, "Weight formats"; cuda/bf16x.cuh has it in full). Every function is
// static: this header is compiled once per ISA level, and nothing in it may get external linkage.
namespace bf16x {
constexpr int kBlock = 128;        // values per block (one base exponent)
constexpr int kBlockBytes = 192;   // 128 sign|mantissa bytes, then 64 bytes of 4-bit codes
constexpr int kCodes = 128;        // offset of the codes in a block
constexpr uint32_t kNoPatch = 0xffffffffu;
constexpr int kMaxK = 1 << 23;     // patch positions are 23 bits
static constexpr int blocks(int64_t K) { return static_cast<int>((K + kBlock - 1) / kBlock); }
// At least 2 + K / 2048 patch slots, and as many more as fit in the header's padding to 8 bytes.
static constexpr int header_bytes(int64_t K) {
  return static_cast<int>((4 * (2 + K / 2048) + blocks(K) + 7) / 8 * 8);
}
static constexpr int patch_slots(int64_t K) { return (header_bytes(K) - blocks(K)) / 4; }
static constexpr int base_offset(int64_t K) { return 4 * patch_slots(K); }
static constexpr int64_t row_bytes(int64_t K) {
  return header_bytes(K) + static_cast<int64_t>(kBlockBytes) * blocks(K);
}
}  // namespace bf16x

// One [N, K] matrix in its format (the rows of one expert, for a stack). The per-row factor (the INT8 /
// FP8 row scale, the NVFP4 global) is never applied by the dot kernels: their callers apply it once per
// output, so sum_k x * (q * s) = s * sum_k x * q. BF16X rows are bf16x::row_bytes(K) apart.
struct WeightRef {
  int type;
  const void* data;
  const void* scale;  // f32 [N] for INT8 / FP8, e4m3 [N, K / 16] for NVFP4
};

// sums[m, n] = sum_k x[m, k] * w[n, k] (NVFP4 block scales applied) for f32 activations.
// Accumulation is in f32, lane-wise over the vector width, in k order, then one fixed horizontal sum,
// so a row's result never depends on M, on the tile it falls in or on the thread that computes it.
// With out_d, the sums are exact instead, in double, written to out_d (out is unused): for the FP8 and
// FP4 maths, whose operands are quantized values with exact f32 products (FP8 and NVFP4 weights only).
struct DotF32 {
  const float* x;  // [M, K] f32, row stride ld_x
  int64_t ld_x;
  WeightRef w;
  int64_t K;
  float* out;  // the call's tile: out[(m - m0) * ld_out + (n - n0)]
  int64_t ld_out;
  float* panel;  // per-thread scratch for decoded weight rows (panel_floats f32), or null
  int64_t panel_floats;
  double* out_d = nullptr;  // exact sums, laid out as out
};

// sums[m, n] = sum_k x[m, k] * w[n, k] for int8 activations and int8 weights, exact in int32, written
// as f32.
struct DotI8 {
  const int8_t* x;  // [M, K] int8, row stride ld_x (bytes)
  int64_t ld_x;
  const int8_t* w;  // [N, K] int8
  int64_t K;
  float* out;
  int64_t ld_out;
  void* panel;  // per-thread scratch (panel_bytes), or null
  int64_t panel_bytes;
};

// The same product with bf16 activations and weights expanded to bf16 (exact for every format), on the
// ISA's bf16 dot instructions. Null in the table of a level without them.
struct DotBF16 {
  const uint16_t* x;  // [M, K] bf16, row stride ld_x
  int64_t ld_x;
  WeightRef w;
  int64_t K;
  float* out;
  int64_t ld_out;
  void* panel;
  int64_t panel_bytes;
};

// Causal GQA over a bf16 KV cache for one work item: query rows q0..q1 of one (sequence, kv head).
struct AttnItem {
  const float* qkv;  // the sequence's first token's row
  int64_t ld_qkv;
  const uint16_t* k;  // the slot's cache, [capacity, KVH, Dh]
  const uint16_t* v;
  uint16_t* out;  // the sequence's first token's output row, [T, H * Dh]
  int H, KVH, Dh, kvh;
  int start;  // cached positions before the sequence's first new token
  int q0, q1;  // new-token rows of this item, relative to the sequence
  float scale;
};

// The selective scan for one (sequence, head).
struct ScanItem {
  const float* x;  // the sequence's first token's row, head h at x + h * P
  int64_t ld_x;
  const float* dt;  // dt_raw row of the first token, head h at dt[h]
  int64_t ld_dt;
  const float* B;  // row of the first token, group g at B + g * N
  const float* C;
  int64_t ld_bc;
  float* state;  // [P, N] of this (slot, head)
  float* y;  // the sequence's first token's output row [H, P]
  int64_t ld_y;
  int len, h, g, P, N;
  float negA, D, dt_bias;
  int has_state;
};

struct Kern {
  const char* name;
  int code;  // tunable.h's lx_arch value for this level
  int vec_floats;  // f32 lanes of the level's vectors
  void (*dot_f32)(const DotF32& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1);
  void (*dot_i8)(const DotI8& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1);
  void (*dot_bf16)(const DotBF16& a, int64_t m0, int64_t m1, int64_t n0, int64_t n1);  // may be null
  void (*attention)(const AttnItem& a);
  void (*scan)(const ScanItem& a);
  // y[i] = exp(x[i]) over n f32, |error| <= 2 ulp; used by the softmaxes and activations.
  void (*exp_f32)(const float* x, float* y, int64_t n);
};

// One per level; null when the level was not built.
const Kern* kern_generic();
const Kern* kern_avx2();
const Kern* kern_avxvnni();
const Kern* kern_avx512();
const Kern* kern_avx512vnni();
const Kern* kern_avx512bf16();
const Kern* kern_neon();
const Kern* kern_neondot();
const Kern* kern_neoni8mm();

}  // namespace lx::cpu
