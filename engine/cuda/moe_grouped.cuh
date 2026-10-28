// The grouped expert problems and epilogues for lx_moe_experts_grouped (moe_grouped.cu) and its
// tile-variant families, instantiated per kind in moe_grouped_<kind>.cu like gemm_w's.
//
// The (token, slot) pairs are sorted by expert, so each expert's rows are one contiguous run of the
// sorted order, and a grouped GEMM is a dense GEMM whose M tiles each belong to one expert. The routing
// step writes a table of those tiles per tile height (kTileBM): entry i is {first sorted row,
// expert << 16 | rows}, rows = 0 past the last tile, and a kernel's CTA (i, j) takes M tile i and N tile j.
// The grid is the table's capacity times the N tiles, fixed by (T, top_k, E) alone, so the launch never
// waits on the routing (no host round trip; capturable in a CUDA graph).
//
//   gate_up  A: the latent rows of the pairs' tokens, gathered by the loader (no permuted copy);
//            B: the expert's gate_up rows, interleaved so every 16 columns of a tile are 8 gate columns
//            and the 8 up columns that match them; the epilogue writes bf16(silu(gate) * up).
//   down     A: the intermediate in sorted order; the epilogue multiplies by the routing weight and adds
//            into the token's output row (atomic), or writes the pair's own row for an ordered sum.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>

#include "cuda/common.cuh"
#include "cuda/gemm_tc.cuh"
#include "cuda/kernels.h"

namespace lx::moeg {

using lx::tc::Tile;

constexpr int kTileBMs = 4;  // tile heights with a table: 32, 64, 128, 256
__host__ __device__ constexpr int bm_slot(int bm) { return bm == 32 ? 0 : bm == 64 ? 1 : bm == 128 ? 2 : 3; }
// Capacity of the table for tile height bm: every expert adds at most one partial tile.
inline int table_capacity(int pairs, int E, int bm) { return (pairs + bm - 1) / bm + std::min(E, pairs); }

struct TileTables {
  const int2* table[kTileBMs];
  int capacity[kTileBMs];
};

// ---- problems ---------------------------------------------------------------------------------------

struct GroupedParams {
  const uint8_t* a;  // A rows, lda bytes apart: latent rows (gate_up) or sorted rows (down)
  int64_t lda;
  const float* a_scale;  // per A row: INT8 / FP8 scale or the FP4 global
  const uint8_t* a_sf;   // FP4 block scales, ld_asf bytes per row
  int64_t ld_asf;
  const int32_t* pairs;  // sorted row -> pair (token * top_k + slot)
  int top_k;
  const uint8_t* b;  // expert 0's weight, ldb bytes per row; experts b_stride bytes apart
  int64_t ldb, b_stride;
  const float* b_scale;  // per weight row (INT8 / FP8), b_scale_stride floats per expert
  int64_t b_scale_stride;
  const uint8_t* b_sf;  // NVFP4 block scales, ld_bsf bytes per row, b_sf_stride per expert
  int64_t ld_bsf, b_sf_stride;
  float b_global;
  const float* b_expert_global;  // NVFP4 stacks: each expert's own global [E], or null when b_global serves all
  const int2* tiles;  // the tile table for this kernel's BM
  int N, K, tiles_n;
  int half_n;  // gate_up: I, where the up rows start
};

template <int BM, int BN>
__device__ __forceinline__ bool grouped_tile(const GroupedParams& p, Tile& t, int cta) {
  const int i = cta / p.tiles_n, j = cta % p.tiles_n;
  const int2 e = p.tiles[i];
  const int rows = e.y & 0xffff;
  if (rows == 0) return false;
  const int ex = e.y >> 16;
  t.m0 = e.x;
  t.rows = rows;
  t.n0 = j * BN;
  t.cols = min(BN, p.N - t.n0);
  t.expert = ex;
  t.b = p.b + ex * p.b_stride;
  t.b_sf = p.b_sf ? p.b_sf + ex * p.b_sf_stride : nullptr;
  t.b_scale = p.b_scale ? p.b_scale + ex * p.b_scale_stride : nullptr;
  return true;
}

// The weight global of a tile's expert: an NVFP4 stack carries one per expert (kernels.h, "Weight formats").
__device__ __forceinline__ float grouped_global(const GroupedParams& p, const Tile& t) {
  return p.b_expert_global ? __ldg(p.b_expert_global + t.expert) : p.b_global;
}

// gate_up: A rows gathered by token, B rows interleaved gate / up in blocks of 8.
struct GateUpProb {
  using Params = GroupedParams;
  template <int BM, int BN>
  static __device__ __forceinline__ bool tile(const Params& p, Tile& t, int cta) {
    return grouped_tile<BM, BN>(p, t, cta);
  }
  static __device__ __forceinline__ int64_t token(const Params& p, const Tile& t, int r) {
    return __ldg(p.pairs + t.m0 + r) / p.top_k;
  }
  static __device__ __forceinline__ const uint8_t* a_row(const Params& p, const Tile& t, int r) {
    return p.a + token(p, t, r) * p.lda;
  }
  static __device__ __forceinline__ const uint8_t* a_sf_row(const Params& p, const Tile& t, int r) {
    return p.a_sf + token(p, t, r) * p.ld_asf;
  }
  static __device__ __forceinline__ float a_scale(const Params& p, const Tile& t, int r) {
    return __ldg(p.a_scale + token(p, t, r));
  }
  // Tile column c -> gate_up row: even blocks of 8 are gate rows, odd blocks the up rows that match.
  static __device__ __forceinline__ int64_t b_row(const Params& p, const Tile& t, int c) {
    const int blk = c >> 3;
    const int i = (t.n0 >> 1) + (blk >> 1) * 8 + (c & 7);
    return (blk & 1) ? p.half_n + i : i;
  }
  static __device__ __forceinline__ float b_global(const Params& p, const Tile& t) { return grouped_global(p, t); }
};

// down: A in sorted order, B rows as stored.
struct DownProb {
  using Params = GroupedParams;
  template <int BM, int BN>
  static __device__ __forceinline__ bool tile(const Params& p, Tile& t, int cta) {
    return grouped_tile<BM, BN>(p, t, cta);
  }
  static __device__ __forceinline__ const uint8_t* a_row(const Params& p, const Tile& t, int r) {
    return p.a + (t.m0 + r) * p.lda;
  }
  static __device__ __forceinline__ const uint8_t* a_sf_row(const Params& p, const Tile& t, int r) {
    return p.a_sf + (t.m0 + r) * p.ld_asf;
  }
  static __device__ __forceinline__ float a_scale(const Params& p, const Tile& t, int r) {
    return __ldg(p.a_scale + t.m0 + r);
  }
  static __device__ __forceinline__ int64_t b_row(const Params&, const Tile& t, int c) { return t.n0 + c; }
  static __device__ __forceinline__ float b_global(const Params& p, const Tile& t) { return grouped_global(p, t); }
};

// ---- epilogues --------------------------------------------------------------------------------------

// The intermediate, bf16 [pairs, ld] in sorted order: bf16(silu(gate) * up), as lx_moe_experts rounds it.
struct GateUpEpi {
  static constexpr bool kGated = true;
  using Row = int64_t;  // the sorted row
  struct Params {
    __nv_bfloat16* inter;
    int64_t ld;  // elements, even
  };
  static __device__ __forceinline__ Row row(const Params&, const Tile& t, int r) { return t.m0 + r; }
  static __device__ __forceinline__ void store_gated(const Params& e, const Tile& t, Row r, int c, float g0, float g1,
                                                     float u0, float u1) {
    const int col = (t.n0 >> 1) + (c >> 4) * 8 + (c & 7);
    *reinterpret_cast<__nv_bfloat162*>(e.inter + r * e.ld + col) =
        __floats2bfloat162_rn(lx::silu(g0) * u0, lx::silu(g1) * u1);
  }
};

// weight * (down @ intermediate): added into out[token] (atomic), or written to row `pair` of an f32
// [pairs, ld] buffer that an ordered pass then sums per token, slot 0 first.
struct DownEpi {
  static constexpr bool kGated = false;
  struct Row {
    float w;
    int64_t dst;  // output row: the token (atomic) or the pair
  };
  struct Params {
    const int32_t* pairs;  // sorted row -> pair
    int top_k;
    const float* weights;  // routing weight per pair
    float* out;
    int64_t ld;  // elements
    bool atomic, pair_ok;
  };
  static __device__ __forceinline__ Row row(const Params& e, const Tile& t, int r) {
    const int p = __ldg(e.pairs + t.m0 + r);
    return Row{__ldg(e.weights + p), e.atomic ? p / e.top_k : p};
  }
  static __device__ __forceinline__ void store2(const Params& e, const Tile& t, const Row& r, int c, float v0,
                                                float v1) {
    float* y = e.out + r.dst * e.ld + t.n0 + c;
    const bool two = c + 1 < t.cols;
    v0 *= r.w;
    v1 *= r.w;
    if (e.atomic) {
#if __CUDA_ARCH__ >= 900
      if (two && e.pair_ok) {
        atomicAdd(reinterpret_cast<float2*>(y), make_float2(v0, v1));
        return;
      }
#endif
      atomicAdd(y, v0);
      if (two) atomicAdd(y + 1, v1);
    } else if (two && e.pair_ok) {
      *reinterpret_cast<float2*>(y) = make_float2(v0, v1);
    } else {
      y[0] = v0;
      if (two) y[1] = v1;
    }
  }
};

// ---- kernels and variants -----------------------------------------------------------------------------

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB, class Prob, class Epi>
__global__ void __launch_bounds__(WM * WN * 32, (lx::tc::Config<kKind, kArch, BM, BN, WM, WN, ST, RB>::kMinBlocks))
    grouped_kernel(const GroupedParams pp, const typename Epi::Params ep) {
  extern __shared__ __align__(128) uint8_t smem[];
  lx::tc::gemm_body<kKind, kArch, BM, BN, WM, WN, ST, RB, Prob, Epi>(pp, ep, smem);
}

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB, class Prob, class Epi>
inline int launch_grouped(GroupedParams pp, const typename Epi::Params& ep, const TileTables& tt, cudaStream_t s) {
  using C = lx::tc::Config<kKind, kArch, BM, BN, WM, WN, ST, RB>;
  auto kernel = grouped_kernel<kKind, kArch, BM, BN, WM, WN, ST, RB, Prob, Epi>;
  static const cudaError_t attr = [&] {
    cudaError_t e = cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, C::kSmem);
    if (e == cudaSuccess)
      e = cudaFuncSetAttribute(kernel, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    return e;
  }();
  if (attr != cudaSuccess) return static_cast<int>(attr);
  pp.tiles = tt.table[bm_slot(BM)];
  pp.tiles_n = (pp.N + BN - 1) / BN;
  const int64_t grid = static_cast<int64_t>(tt.capacity[bm_slot(BM)]) * pp.tiles_n;
  kernel<<<static_cast<unsigned>(grid), C::kThreads, C::kSmem, s>>>(pp, ep);
  return lx::launch_status();
}

using GateUpFn = int (*)(GroupedParams, const GateUpEpi::Params&, const TileTables&, cudaStream_t);
using DownFn = int (*)(GroupedParams, const DownEpi::Params&, const TileTables&, cudaStream_t);

struct Variant {
  const char* name;
  int bm;
  GateUpFn gate_up;
  DownFn down;
  int min_pairs = 0, max_pairs = 0;  // a variant picked for [min_pairs, max_pairs] (token, slot) pairs, see pick()
};

template <int kKind, int kArch>
struct Family {
  static const char* name();
  static const Variant* variants(int& n);  // listed small to large
};

#define LX_MOEG_FAMILY(K, A)                        \
  template <>                                       \
  const char* Family<lx::tc::K, A>::name();         \
  template <>                                       \
  const Variant* Family<lx::tc::K, A>::variants(int& n);
LX_MOEG_FAMILY(kBF16, 80)
LX_MOEG_FAMILY(kI8, 80)
LX_MOEG_FAMILY(kF8, 89)
LX_MOEG_FAMILY(kF8, 120)
LX_MOEG_FAMILY(kF4, 120)
LX_MOEG_FAMILY(kW8I8, 80)
LX_MOEG_FAMILY(kW8F8, 80)
LX_MOEG_FAMILY(kW8F8, 120)
LX_MOEG_FAMILY(kW4A16, 80)
LX_MOEG_FAMILY(kW4A8, 120)
#undef LX_MOEG_FAMILY

// A variant: BMxBN (BM one of the table heights), warps, stages, A bytes per row per k-tile; both stages.
// B rows must be 128 bytes or more per k-tile (gemm_tc.cuh, Config::kBRow): a 64-byte row segment streams at 604 GB/s
// against 853 GB/s for 128 bytes. The W8F8 family's variants are RB = 256 for that reason. Check any new variant's kBRow.
#define LX_MVP(K, A, BM, BN, WM, WN, ST, RB, MINP, MAXP)                                             \
  Variant {                                                                                          \
    #BM "x" #BN "_w" #WM "x" #WN "_s" #ST "_k" #RB, BM,                                              \
        launch_grouped<lx::tc::K, A, BM, BN, WM, WN, ST, RB, GateUpProb, GateUpEpi>,                 \
        launch_grouped<lx::tc::K, A, BM, BN, WM, WN, ST, RB, DownProb, DownEpi>, MINP, MAXP          \
  }
#define LX_MV(K, A, BM, BN, WM, WN, ST, RB) LX_MVP(K, A, BM, BN, WM, WN, ST, RB, 0, 0)
#define LX_MOEG(K, A, NAME, ...)                                    \
  template <>                                                       \
  const char* Family<lx::tc::K, A>::name() {                        \
    return NAME;                                                    \
  }                                                                 \
  template <>                                                       \
  const Variant* Family<lx::tc::K, A>::variants(int& n) {           \
    static const Variant v[] = {__VA_ARGS__};                       \
    n = static_cast<int>(sizeof(v) / sizeof(v[0]));                 \
    return v;                                                       \
  }

}  // namespace lx::moeg
