// The tensor-core GEMM every low-precision product runs on: C[M, N] = A[M, K] . B[N, K]^T with A the
// activations and B a weight in its stored format (kernels.h, "Weight formats"), both K-contiguous.
// One template covers every (weight format, math) pair, the dense GEMM (gemm_w.cu) and the grouped
// expert GEMMs (moe_grouped.cu); a Prob policy says where a tile's rows live and an Epi policy what
// happens to the results.
//
// Mainloop: a CTA computes a BM x BN tile with WM x WN warps; K advances in tiles of 128 bytes of A per
// row (64 bf16, 128 int8 / e4m3, 256 e2m1), streamed through ST shared-memory stages by cp.async
// (zero-filled past M, N and K), so the copy of tile k + ST - 1 overlaps the MMAs on tile k.
//
// Operand kinds:
//   homogeneous (BF16, INT8, FP8, FP4): A and B have the same element type. Fragments come from
//     ldmatrix on a 128-byte swizzled layout (16-byte chunk ^ row % 8) and feed the MMA directly. FP4
//     also stages the e4m3 block scales of both sides and passes them to the block-scaled MMA.
//   bf16 x quantized weight (W8A16 for INT8 / FP8 weights, W4A16 for NVFP4): the bf16 MMA, with B
//     expanded to bf16 in registers. The expansion yields contiguous K runs, so K is permuted inside
//     each 32-wide step: thread q owns K 8q..8q+7, the first MMA takes 8q..8q+3 as its logical
//     columns (2q, 2q+1, 2q+8, 2q+9) and the second 8q+4..8q+7. A is read with the same permutation
//     (one 16-byte load per row), so the product is unchanged.
//   W4A8 (NVFP4 weight x e4m3 activation): the weight's per-16 scales are not representable inside an
//     FP8 operand, so each 32-wide step runs two block-scaled FP8 MMAs, one per 16-block (the other
//     block's B values zeroed), into fresh registers that are scaled and accumulated in f32. The same
//     K permutation puts all of a thread's B values in one block.
// The per-row and per-token scales and the globals are applied once, in the epilogue.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

#include <type_traits>

#include "cuda/common.cuh"
#include "cuda/mma.cuh"

namespace lx::tc {

enum Kind : int {
  kBF16 = 0,  // bf16 x bf16
  kI8,        // int8 per token x int8 per row (IMMA)
  kF8,        // e4m3 per token x e4m3 per row
  kF4,        // NVFP4 x NVFP4 (block scales in the MMA; per-token and weight globals)
  kW8I8,      // bf16 x int8 weight expanded to bf16
  kW8F8,      // bf16 x e4m3 weight expanded to bf16
  kW4A16,     // bf16 x NVFP4 weight expanded to bf16 with its block scale
  kW4A8,      // e4m3 per token x NVFP4 weight, per-block products
};

template <int kKind>
struct Traits;
template <>
struct Traits<kBF16> {
  static constexpr int kABits = 16, kBBits = 16;
  static constexpr bool kLdm = true, kASf = false, kBSf = false, kRowScale = false, kColScale = false, kGlobal = false;
};
template <>
struct Traits<kI8> {
  static constexpr int kABits = 8, kBBits = 8;
  static constexpr bool kLdm = true, kASf = false, kBSf = false, kRowScale = true, kColScale = true, kGlobal = false;
};
template <>
struct Traits<kF8> : Traits<kI8> {};
template <>
struct Traits<kF4> {
  static constexpr int kABits = 4, kBBits = 4;
  static constexpr bool kLdm = true, kASf = true, kBSf = true, kRowScale = true, kColScale = false, kGlobal = true;
};
template <>
struct Traits<kW8I8> {
  static constexpr int kABits = 16, kBBits = 8;
  static constexpr bool kLdm = false, kASf = false, kBSf = false, kRowScale = false, kColScale = true, kGlobal = false;
};
template <>
struct Traits<kW8F8> : Traits<kW8I8> {};
template <>
struct Traits<kW4A16> {
  static constexpr int kABits = 16, kBBits = 4;
  static constexpr bool kLdm = false, kASf = false, kBSf = true, kRowScale = false, kColScale = false, kGlobal = true;
};
template <>
struct Traits<kW4A8> {
  static constexpr int kABits = 8, kBBits = 4;
  static constexpr bool kLdm = false, kASf = false, kBSf = true, kRowScale = true, kColScale = false, kGlobal = true;
};

// Everything the mainloop and the epilogue know about one CTA's tile, filled in by the Prob policy.
struct Tile {
  int rows, cols;    // valid rows (<= BM) and columns (<= BN) of the tile
  int m0, n0;        // the tile's first row and column in the problem's numbering
  int expert;        // grouped problems: the expert; dense: 0
  const uint8_t* b;  // the weight (the expert's slice), row 0
  const uint8_t* b_sf;
  const float* b_scale;
};

// Swizzled byte offset of 16-byte chunk `c` of row `r` in a tile of RB-byte rows. The 128-byte line
// holding the chunk is XORed with a per-line pattern chosen for the access pattern that reads it.
template <int RB, int kPattern>
__device__ __forceinline__ uint32_t swz(int r, int c) {
  const int lin = r * RB + c * 16;
  const int line = lin >> 7;
  int chunk = (lin >> 4) & 7;
  if constexpr (kPattern == 0) chunk ^= line & 7;         // ldmatrix: 8 rows of one chunk
  else if constexpr (kPattern == 1) chunk ^= (line & 1) << 2;  // 16-byte loads, rows in pairs
  else if constexpr (kPattern == 2) chunk ^= (line & 3) << 1;  // 8-byte loads, rows in fours
  else if constexpr (kPattern == 3) chunk ^= (line & 1) << 1;  // 8-byte loads of 64-byte rows
  else if constexpr (kPattern == 4) chunk ^= line & 1;         // 4-byte loads of 32-byte rows
  else chunk ^= line & 3;                                       // 4-byte loads of 64-byte rows
  return static_cast<uint32_t>((line << 7) | (chunk << 4));
}

template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB = 128>
struct Config {
  using T = Traits<kKind>;
  static constexpr int kThreads = WM * WN * 32;
  static constexpr int kWTM = BM / WM, kWTN = BN / WN;
  static constexpr int kMT = kWTM / 16, kNT = kWTN / 8;
  static constexpr int kARow = RB;                         // A bytes per row per k-tile (64 or 128)
  static constexpr int kKTile = kARow * 8 / T::kABits;     // K per k-tile
  static constexpr int kBRow = kKTile * T::kBBits / 8;     // B bytes per row per k-tile
  static constexpr int kSfRow = kKTile / 16;               // scale bytes per row per k-tile (FP4 / NVFP4)
  // Scale rows are padded by one row every 8 so the 32 rows the FP4 scale loads touch fall in 32 banks.
  static constexpr int kASfBytes = T::kASf ? (BM + BM / 8) * kSfRow : 0;
  static constexpr int kBSfBytes = T::kBSf ? (BN + BN / 8 + 8) * kSfRow : 0;
  static constexpr int kAOff = 0;
  static constexpr int kBOff = kAOff + BM * kARow;
  static constexpr int kASfOff = kBOff + BN * kBRow;
  static constexpr int kBSfOff = kASfOff + kASfBytes;
  static constexpr int kStage = (kBSfOff + kBSfBytes + 127) / 128 * 128;
  static constexpr int kSmem = ST * kStage;
  // Patterns: homogeneous tiles are read by ldmatrix; the others by the loads described above.
  static constexpr int kAPat = T::kLdm ? 0 : (T::kABits == 16 ? 1 : 2);
  static constexpr int kBPat = T::kLdm ? 0 : (T::kBBits == 8 ? 3 : (kBRow == 32 ? 4 : 5));
  static_assert(BM % (16 * WM) == 0 && BN % (8 * WN) == 0, "warp tile");
  static_assert(T::kLdm == false || kNT % 2 == 0, "ldmatrix loads B in pairs of n8 tiles");
  static_assert(kSmem <= 101376, "shared memory");
  static_assert(RB == 128 || (RB == 64 && T::kLdm), "64-byte rows: homogeneous kinds only");
};

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }

// Scale of the f32 result before the epilogue, per kind.
template <int kKind>
__device__ __forceinline__ float row_col_scale(float row, float col, float global) {
  using T = Traits<kKind>;
  float s = 1.0f;
  if constexpr (T::kRowScale) s *= row;
  if constexpr (T::kColScale) s *= col;
  if constexpr (T::kGlobal) s *= global;
  return s;
}

// The kernel body. Prob: tile assignment and row addresses; Epi: what to do with the results.
template <int kKind, int kArch, int BM, int BN, int WM, int WN, int ST, int RB, class Prob, class Epi>
__device__ __forceinline__ void gemm_body(const typename Prob::Params& pp, const typename Epi::Params& ep,
                                          uint8_t* smem) {
  using C = Config<kKind, kArch, BM, BN, WM, WN, ST, RB>;
  using T = typename C::T;
  constexpr int kThreads = C::kThreads, kMT = C::kMT, kNT = C::kNT;
  using Acc = std::conditional_t<kKind == kI8, int, float>;

  Tile t;
  if (!Prob::template tile<BM, BN>(pp, t)) return;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int wm = warp / WN, wn = warp % WN;
  const int K = pp.K;
  const int KT = (K + C::kKTile - 1) / C::kKTile;
  const uint32_t s0 = mma::smem_addr(smem);

  // ---- loader: each thread copies fixed chunks of every tile, so its source rows are set up once ----
  constexpr int kAChunks = BM * C::kARow / 16, kBChunks = BN * C::kBRow / 16;
  constexpr int kAIt = (kAChunks + kThreads - 1) / kThreads, kBIt = (kBChunks + kThreads - 1) / kThreads;
  const uint8_t* a_src[kAIt];
  const uint8_t* b_src[kBIt];
  uint32_t a_dst[kAIt], b_dst[kBIt];
  int a_col[kAIt], b_col[kBIt];  // byte offset of the chunk within a K tile
#pragma unroll
  for (int i = 0; i < kAIt; ++i) {
    const int c = tid + i * kThreads;
    const int r = c / (C::kARow / 16), ch = c % (C::kARow / 16);
    a_src[i] = c < kAChunks && r < t.rows ? Prob::a_row(pp, t, r) : nullptr;
    a_dst[i] = C::kAOff + swz<C::kARow, C::kAPat>(r, ch);
    a_col[i] = ch * 16;
  }
#pragma unroll
  for (int i = 0; i < kBIt; ++i) {
    const int c = tid + i * kThreads;
    const int r = c / (C::kBRow / 16), ch = c % (C::kBRow / 16);
    b_src[i] = c < kBChunks && r < t.cols ? t.b + Prob::b_row(pp, t, r) * pp.ldb : nullptr;
    b_dst[i] = C::kBOff + swz<C::kBRow, C::kBPat>(r, ch);
    b_col[i] = ch * 16;
  }
  // Scale tiles: one copy of kSfRow bytes per row (16, 8 or 4 bytes).
  constexpr int kASfIt = T::kASf ? (BM + kThreads - 1) / kThreads : 0;
  constexpr int kBSfIt = T::kBSf ? (BN + kThreads - 1) / kThreads : 0;
  const uint8_t* asf_src[kASfIt > 0 ? kASfIt : 1];
  const uint8_t* bsf_src[kBSfIt > 0 ? kBSfIt : 1];
#pragma unroll
  for (int i = 0; i < kASfIt; ++i) {
    const int rr = tid + i * kThreads;
    asf_src[i] = rr < t.rows ? Prob::a_sf_row(pp, t, rr) : nullptr;
  }
#pragma unroll
  for (int i = 0; i < kBSfIt; ++i) {
    const int rr = tid + i * kThreads;
    bsf_src[i] = rr < t.cols ? t.b_sf + Prob::b_row(pp, t, rr) * pp.ld_bsf : nullptr;
  }
  const int k_bytes_a = K * T::kABits / 8, k_bytes_b = K * T::kBBits / 8, k_sf = K / 16;

  auto load_tile = [&](int stage, int kt) {
    const uint32_t base = s0 + stage * C::kStage;
    const int ka = kt * C::kARow, kb = kt * C::kBRow;
#pragma unroll
    for (int i = 0; i < kAIt; ++i) {
      if (kAChunks % kThreads == 0 || tid + i * kThreads < kAChunks) {
        const bool ok = a_src[i] && ka + a_col[i] < k_bytes_a;
        mma::cp_async16(base + a_dst[i], ok ? a_src[i] + ka + a_col[i] : t.b, ok);
      }
    }
#pragma unroll
    for (int i = 0; i < kBIt; ++i) {
      if (kBChunks % kThreads == 0 || tid + i * kThreads < kBChunks) {
        const bool ok = b_src[i] && kb + b_col[i] < k_bytes_b;
        mma::cp_async16(base + b_dst[i], ok ? b_src[i] + kb + b_col[i] : t.b, ok);
      }
    }
    const int ks = kt * C::kSfRow;
#pragma unroll
    for (int i = 0; i < kASfIt; ++i) {
      const int rr = tid + i * kThreads;
      if (rr < BM) {
        const bool ok = asf_src[i] != nullptr && ks < k_sf;
        const uint32_t dst = base + C::kASfOff + (rr + rr / 8) * C::kSfRow;
        if constexpr (C::kSfRow == 16) mma::cp_async16(dst, ok ? asf_src[i] + ks : t.b, ok);
        else mma::cp_async_small<C::kSfRow>(dst, ok ? asf_src[i] + ks : t.b, ok);
      }
    }
#pragma unroll
    for (int i = 0; i < kBSfIt; ++i) {
      const int rr = tid + i * kThreads;
      if (rr < BN) {
        const bool ok = bsf_src[i] != nullptr && ks < k_sf;
        const uint32_t dst = base + C::kBSfOff + (rr + rr / 8) * C::kSfRow;
        if constexpr (C::kSfRow == 16) mma::cp_async16(dst, ok ? bsf_src[i] + ks : t.b, ok);
        else mma::cp_async_small<C::kSfRow>(dst, ok ? bsf_src[i] + ks : t.b, ok);
      }
    }
  };

  Acc acc[kMT][kNT][4];
#pragma unroll
  for (int i = 0; i < kMT; ++i)
#pragma unroll
    for (int j = 0; j < kNT; ++j)
#pragma unroll
      for (int v = 0; v < 4; ++v) acc[i][j][v] = Acc(0);

  const int row0 = wm * C::kWTM, col0 = wn * C::kWTN;

  // ---- compute on one stage ----
  auto compute = [&](int stage) {
    const uint32_t base = s0 + stage * C::kStage;
    const uint32_t sA = base + C::kAOff, sB = base + C::kBOff;
    if constexpr (T::kLdm) {
      // ldmatrix lanes: A matrices (rows 0-7 | 8-15) x (bytes 0-15 | 16-31); B two n8 tiles x two halves.
      const int a_r = (lane & 7) + ((lane >> 3) & 1) * 8, a_c = lane >> 4;
      const int b_r = (lane & 7) + (lane >> 4) * 8, b_c = (lane >> 3) & 1;
#pragma unroll
      for (int ks = 0; ks < C::kARow / 32; ++ks) {
        uint32_t af[kMT][4], bfr[kNT][2];
#pragma unroll
        for (int mt = 0; mt < kMT; ++mt) mma::ldmatrix_x4(af[mt], sA + swz<C::kARow, 0>(row0 + mt * 16 + a_r, 2 * ks + a_c));
#pragma unroll
        for (int p = 0; p < kNT / 2; ++p) {
          uint32_t r[4];
          mma::ldmatrix_x4(r, sB + swz<C::kBRow, 0>(col0 + p * 16 + b_r, 2 * ks + b_c));
          bfr[2 * p][0] = r[0];
          bfr[2 * p][1] = r[1];
          bfr[2 * p + 1][0] = r[2];
          bfr[2 * p + 1][1] = r[3];
        }
        if constexpr (kKind == kF4) {
          // Scale registers: one covers two m16 tiles (A) or four n8 tiles (B); see mma.cuh.
          constexpr int kSA = (kMT + 1) / 2, kSB = (kNT + 3) / 4;
          uint32_t sfa[kSA], sfb[kSB];
          const int q = lane & 3;
#pragma unroll
          for (int g = 0; g < kSA; ++g) {
            int r = row0 + (2 * g + (q >> 1)) * 16 + (lane >> 2) + 8 * (q & 1);
            r = r < BM ? r : BM - 1;
            sfa[g] = mma::lds32(base + C::kASfOff + (r + r / 8) * C::kSfRow + 4 * ks);
          }
#pragma unroll
          for (int g = 0; g < kSB; ++g) {
            int r = col0 + (4 * g + q) * 8 + (lane >> 2);
            r = r < BN ? r : BN - 1;
            sfb[g] = mma::lds32(base + C::kBSfOff + (r + r / 8) * C::kSfRow + 4 * ks);
          }
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt) {
              const uint32_t sa = sfa[mt / 2], sb = sfb[nt / 4];
              if ((mt & 1) == 0) {
                switch (nt & 3) {
                  case 0: mma::mma_nvf4<0, 0>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  case 1: mma::mma_nvf4<0, 1>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  case 2: mma::mma_nvf4<0, 2>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  default: mma::mma_nvf4<0, 3>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                }
              } else {
                switch (nt & 3) {
                  case 0: mma::mma_nvf4<1, 0>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  case 1: mma::mma_nvf4<1, 1>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  case 2: mma::mma_nvf4<1, 2>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                  default: mma::mma_nvf4<1, 3>(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1], sa, sb); break;
                }
              }
            }
        } else {
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt)
#pragma unroll
            for (int nt = 0; nt < kNT; ++nt) {
              if constexpr (kKind == kBF16) mma::mma_bf16(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1]);
              else if constexpr (kKind == kI8) mma::mma_s8(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1]);
              else if constexpr (kArch >= 120) mma::mma_e4m3_mx(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1]);
              else mma::mma_e4m3(acc[mt][nt], af[mt], bfr[nt][0], bfr[nt][1]);
            }
        }
      }
    } else {
      const int r = lane >> 2, q = lane & 3;
      if constexpr (T::kABits == 16) {
        // bf16 x expanded weight: two k32 steps per tile, two MMAs per step.
        uint32_t sfw[kNT];  // NVFP4: the row's four block scales of this k tile
        if constexpr (kKind == kW4A16) {
#pragma unroll
          for (int nt = 0; nt < kNT; ++nt) {
            const int row = col0 + nt * 8 + r;
            sfw[nt] = mma::lds32(base + C::kBSfOff + (row + row / 8) * C::kSfRow);
          }
        }
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          uint32_t a0[kMT][4], a1[kMT][4];
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt) {
            const uint4 v = mma::lds128(sA + swz<128, 1>(row0 + mt * 16 + r, 4 * j + q));
            const uint4 u = mma::lds128(sA + swz<128, 1>(row0 + mt * 16 + r + 8, 4 * j + q));
            a0[mt][0] = v.x; a0[mt][1] = u.x; a0[mt][2] = v.y; a0[mt][3] = u.y;
            a1[mt][0] = v.z; a1[mt][1] = u.z; a1[mt][2] = v.w; a1[mt][3] = u.w;
          }
#pragma unroll
          for (int nt = 0; nt < kNT; ++nt) {
            const int row = col0 + nt * 8 + r;
            uint32_t b[4];
            if constexpr (kKind == kW4A16) {
              const uint32_t w = mma::lds32(sB + swz<C::kBRow, C::kBPat>(row, j) + 4 * q);
              const uint32_t s8 = (sfw[nt] >> (8 * (2 * j + (q >> 1)))) & 0xffu;
              const float sf = mma::e4m3_to_f32(s8);
              const uint32_t s2 = mma::pack_bf16(sf, sf);
              mma::e2m1x8_to_bf16(w, s2, b);
            } else {
              const uint2 w = mma::lds64(sB + swz<C::kBRow, C::kBPat>(row, 2 * j + (q >> 1)) + 8 * (q & 1));
              if constexpr (kKind == kW8I8) {
                mma::i8x4_to_bf16(w.x, b[0], b[1]);
                mma::i8x4_to_bf16(w.y, b[2], b[3]);
              } else {
                mma::e4m3x4_to_bf16<kArch>(w.x, b[0], b[1]);
                mma::e4m3x4_to_bf16<kArch>(w.y, b[2], b[3]);
              }
            }
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
              mma::mma_bf16(acc[mt][nt], a0[mt], b[0], b[1]);
              mma::mma_bf16(acc[mt][nt], a1[mt], b[2], b[3]);
            }
          }
        }
      } else {
        // W4A8: four k32 steps per tile; per step, one MMA per 16-block of the weight's scales.
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t af[kMT][4];
#pragma unroll
          for (int mt = 0; mt < kMT; ++mt) {
            const uint2 v = mma::lds64(sA + swz<128, 2>(row0 + mt * 16 + r, 2 * j + (q >> 1)) + 8 * (q & 1));
            const uint2 u = mma::lds64(sA + swz<128, 2>(row0 + mt * 16 + r + 8, 2 * j + (q >> 1)) + 8 * (q & 1));
            af[mt][0] = v.x; af[mt][1] = u.x; af[mt][2] = v.y; af[mt][3] = u.y;
          }
#pragma unroll
          for (int nt = 0; nt < kNT; ++nt) {
            const int row = col0 + nt * 8 + r;
            const uint32_t w = mma::lds32(sB + swz<C::kBRow, C::kBPat>(row, j) + 4 * q);
            const uint32_t b0 = mma::e2m1x4_to_e4m3(w), b1 = mma::e2m1x4_to_e4m3(w >> 16);
            const bool lo = q < 2;
            // the scales of this thread's two result columns, for the step's two blocks
            const int c0 = col0 + nt * 8 + 2 * q;
            const uint32_t sw0 = mma::lds32(base + C::kBSfOff + (c0 + c0 / 8) * C::kSfRow + 4 * (j >> 1));
            const uint32_t sw1 = mma::lds32(base + C::kBSfOff + (c0 + 1 + (c0 + 1) / 8) * C::kSfRow + 4 * (j >> 1));
            const int sh = 16 * (j & 1);
            const float s00 = mma::e4m3_to_f32((sw0 >> sh) & 0xffu), s01 = mma::e4m3_to_f32((sw0 >> (sh + 8)) & 0xffu);
            const float s10 = mma::e4m3_to_f32((sw1 >> sh) & 0xffu), s11 = mma::e4m3_to_f32((sw1 >> (sh + 8)) & 0xffu);
#pragma unroll
            for (int mt = 0; mt < kMT; ++mt) {
              float d0[4], d1[4];
              mma::mma_e4m3_mx_zero(d0, af[mt], lo ? b0 : 0u, lo ? b1 : 0u);
              mma::mma_e4m3_mx_zero(d1, af[mt], lo ? 0u : b0, lo ? 0u : b1);
              acc[mt][nt][0] = fmaf(d1[0], s01, fmaf(d0[0], s00, acc[mt][nt][0]));
              acc[mt][nt][1] = fmaf(d1[1], s11, fmaf(d0[1], s10, acc[mt][nt][1]));
              acc[mt][nt][2] = fmaf(d1[2], s01, fmaf(d0[2], s00, acc[mt][nt][2]));
              acc[mt][nt][3] = fmaf(d1[3], s11, fmaf(d0[3], s10, acc[mt][nt][3]));
            }
          }
        }
      }
    }
  };

  // ---- the pipeline ----
#pragma unroll
  for (int s = 0; s < ST - 1; ++s) {
    if (s < KT) load_tile(s, s);
    mma::cp_async_commit();
  }
  for (int kt = 0; kt < KT; ++kt) {
    mma::cp_async_wait<ST - 2>();
    __syncthreads();
    const int nk = kt + ST - 1;
    if (nk < KT) load_tile(nk % ST, nk);
    mma::cp_async_commit();
    compute(kt % ST);
  }
  mma::cp_async_wait<0>();

  // ---- epilogue: per-kind scaling, then the policy ----
  const int r = lane >> 2, q = lane & 3;
  const float global = pp.b_global;
#pragma unroll
  for (int mt = 0; mt < kMT; ++mt) {
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int tr = row0 + mt * 16 + r + 8 * h;
      if (tr >= t.rows) continue;
      float rs = 1.0f;
      if constexpr (T::kRowScale) rs = Prob::a_scale(pp, t, tr);
#pragma unroll
      for (int nt = 0; nt < kNT; ++nt) {
        const int tc = col0 + nt * 8 + 2 * q;
        if (tc >= t.cols) continue;
        float cs0 = 1.0f, cs1 = 1.0f;
        if constexpr (T::kColScale) {
          cs0 = t.b_scale[Prob::b_row(pp, t, tc)];
          cs1 = tc + 1 < t.cols ? t.b_scale[Prob::b_row(pp, t, tc + 1)] : 0.0f;
        }
        const float v0 = static_cast<float>(acc[mt][nt][2 * h]) * row_col_scale<kKind>(rs, cs0, global);
        const float v1 = static_cast<float>(acc[mt][nt][2 * h + 1]) * row_col_scale<kKind>(rs, cs1, global);
        Epi::store2(ep, t, tr, tc, v0, v1);
      }
    }
  }
}

}  // namespace lx::tc
