// The tile GEMM's host side, shared by lx_gemm_w (ops_gemm.mm) and the grouped experts (ops_moe.mm).
#pragma once

#include <string>

#include "metal/backend.h"
#include "metal/kernels/args.h"

namespace lx::metal {

// The tile GEMMs' shape and threads, and their threadgroup memory: kernels/gemm.metal stages two 64 x 40
// bf16 tiles, kernels/nax/gemm.metal two stages of two 64 x 32 tiles, then the 64 x 64 f32 result.
constexpr int kTileM = 64, kTileN = 64;
constexpr int kGemmThreads = 128;
constexpr NSUInteger kGemmSmemBf16 = 2 * kTileM * 40 * 2;
constexpr NSUInteger kGemmSmemNax = kTileM * kTileN * 4;

// Activation formats (A_* in the kernel) and the f32 weight of lx_gemm_f32.
constexpr int kActBf16 = 0, kActF32 = 1, kActInt8 = 2, kActE4m3 = 3, kActFp4 = 4, kActF32Exact = 5;
constexpr int kWeightF32 = 4;

// gemm_<AF><WF>_<MODE>: MODE 0 plain, 1 grouped gate_up, 2 grouped down.
inline int gemm_variant(int af, int wf, int mode) { return (af * 5 + wf) * 3 + mode; }

// Launches variant gemm_variant(af, wf, mode) of the tile GEMM over `groups` threadgroups: on the
// neural accelerators (kernels/nax/gemm.metal) where the GPU has them, else on simdgroup matrices
// (kernels/gemm.metal).
int launch_gemm(int af, int wf, int mode, void* stream, const GemmArgs& args, MTLSize groups);

// The activation format a math reads (BF16 math reads x as given).
int activation_format(lx_math math, lx_dtype x_dtype);

// Per-token quantized rows in a workspace: codes [rows, ld] bytes, then scales f32 [rows], then FP4
// block scales [rows, ld_blocks]; offsets in bytes, each part 256-aligned.
struct QuantizedRows {
  int ld = 0, ld_blocks = 0;
  int64_t scale_at = 0, blocks_at = 0, bytes = 0;
};
QuantizedRows quantized_rows(int rows, int K, lx_math math);

int64_t round256(int64_t b);

// lx_gemm_f32 on M <= 8 rows: the GEMV over f32 weights (ops_gemv.mm).
int gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N, int K,
             void* stream);

}  // namespace lx::metal
