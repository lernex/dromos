// Lossless compressed BF16 (kernels.h, "Lossless BF16"; layout in bf16x.cuh): the encoder, a decoder,
// the decode GEMV and the decode experts.
//
// Decode is bound by weight bytes, and this format moves 12.1 to 12.3 bits a weight instead of 16 while
// reproducing every bf16 bit, so a product is faster by the byte ratio with nothing changed in its
// result. The kernels are built to be bit-identical to the bf16 kernels they replace, not merely close:
//   lx_gemv_bf16x          the lane mapping, split-K choice and summation order of lx_gemv on bf16
//                          weights (gemv.cu), so y matches it bit for bit.
//   lx_moe_experts_bf16x   the per-lane chunking and expression of lx_moe_experts (moe.cu), so out
//                          matches it bit for bit.
// A model that swaps every bf16 decode product for these produces the same logits to the last bit.
//
// Decoding a lane's four values costs about 13 integer operations (codes to exponents, one patch
// check, two byte permutes); the weight stream stays the limit. The base exponent of each block is a
// uniform byte load issued with the block's data, and a row's patches are read once, from its header,
// before the first step, so no load depends on another inside the loop.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cub/block/block_scan.cuh>
#include <type_traits>

#include "cuda/bf16x.cuh"
#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

namespace {

using bf16 = __nv_bfloat16;
namespace fx = lx::bf16x;

constexpr int kThreads = 256;  // 8 warps per block
constexpr int kWarps = kThreads / 32;
constexpr int kMaxM = 8;
constexpr int kSmallK = 256;       // gemv.cu: rows this short take the scalar kernel
constexpr int kWarpTarget = 2048;  // gemv.cu: warps wanted before K is split across warps

// ---- shared pieces ---------------------------------------------------------------------------

// The bf16 bits of value k of a row, one value at a time (the scalar paths). `pc` is the row's
// patch cursor, positioned at or before k; scalar readers walk k upwards.
__device__ __forceinline__ uint32_t value_bits(const uint8_t* row, int K, int k, fx::Patches& pc) {
  const uint8_t* blk = row + fx::header_bytes(K) + (k >> 7) * fx::kBlockBytes;
  const int off = k & 127;
  const uint32_t sm = blk[off];
  const uint32_t cb = blk[fx::kCodes + 4 * (off >> 3) + (off & 3)];
  const uint32_t code = (off & 4) ? cb >> 4 : cb & 15u;
  uint32_t e = (row[fx::base_offset(K) + (k >> 7)] - code) & 0xffu;
  while (pc.pos() < static_cast<uint32_t>(k)) pc.advance();
  if (pc.pos() == static_cast<uint32_t>(k)) e = pc.next & 0xffu;
  return ((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu);
}

__device__ __forceinline__ float bits_f32(uint32_t b) { return __uint_as_float(b << 16); }

// gemv.cu's input loads and output store, so x and y are handled identically.
__device__ __forceinline__ void decode4(uint2 v, float* f) {
  f[0] = __uint_as_float(v.x << 16);
  f[1] = __uint_as_float(v.x & 0xffff0000u);
  f[2] = __uint_as_float(v.y << 16);
  f[3] = __uint_as_float(v.y & 0xffff0000u);
}

template <typename TX>
__device__ __forceinline__ void load_x4(const TX* p, float* f) {
  if constexpr (std::is_same_v<TX, float>) {
    const float4 t = __ldg(reinterpret_cast<const float4*>(p));
    f[0] = t.x;
    f[1] = t.y;
    f[2] = t.z;
    f[3] = t.w;
  } else {
    decode4(__ldg(reinterpret_cast<const uint2*>(p)), f);
  }
}

__device__ __forceinline__ float bias_at(const void* bias, bool bias_bf16, int64_t n) {
  if (bias == nullptr) return 0.0f;
  return bias_bf16 ? lx::to_f32(static_cast<const bf16*>(bias)[n]) : static_cast<const float*>(bias)[n];
}

__device__ __forceinline__ void store_y(void* y, bool y_bf16, int64_t idx, float t, float alpha, float beta) {
  float v = alpha * t;
  if (beta != 0.0f) {
    const float old = y_bf16 ? lx::to_f32(static_cast<const bf16*>(y)[idx]) : static_cast<const float*>(y)[idx];
    v = fmaf(beta, old, v);
  }
  if (y_bf16) static_cast<bf16*>(y)[idx] = lx::from_f32<bf16>(v);
  else static_cast<float*>(y)[idx] = v;
}

cudaLaunchConfig_t pdl_config(unsigned blocks, int threads, cudaStream_t stream, cudaLaunchAttribute& attr) {
  cudaLaunchConfig_t cfg{};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(threads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = stream;
  attr.id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attr.val.programmaticStreamSerializationAllowed = 1;
  cfg.attrs = &attr;
  cfg.numAttrs = 1;
  return cfg;
}

// ---- encoder ---------------------------------------------------------------------------------
// Three passes, one warp per row: (1) each block's base and the row's patch count, (2) where each
// overflowing row's record goes, (3) everything else. Lane l of a block holds values 4l .. 4l + 3.

__device__ __forceinline__ uint32_t exponent_of(uint32_t u) { return (u >> 7) & 0xffu; }

__global__ void __launch_bounds__(kThreads) encode_bases(const uint16_t* __restrict__ w, int64_t rows, int K,
                                                         uint8_t* __restrict__ out, uint32_t* __restrict__ counts) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint16_t* src = w + r * K;
  uint8_t* base = out ? out + r * fx::row_bytes(K) + fx::base_offset(K) : nullptr;
  uint32_t total = 0;
  for (int j = 0; j < fx::blocks(K); ++j) {
    uint32_t e[4], mx = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k = fx::kBlock * j + 4 * lane + t;
      e[t] = k < K ? exponent_of(src[k]) : 0u;
      mx = max(mx, e[t]);
    }
    const uint32_t e0 = __reduce_max_sync(0xffffffffu, mx);
    uint32_t n = 0;
#pragma unroll
    for (int t = 0; t < 4; ++t) n += (fx::kBlock * j + 4 * lane + t < K && e0 - e[t] > 15u) ? 1u : 0u;
    total += __reduce_add_sync(0xffffffffu, n);
    if (base && lane == 0) base[j] = static_cast<uint8_t>(e0);
  }
  if (lane == 0) counts[r] = total;
}

// Bytes of a row's overflow record: count, the patches, the terminator; 8-byte aligned.
__host__ __device__ inline int64_t overflow_bytes(uint32_t count, int slots) {
  return count > static_cast<uint32_t>(slots) ? (4 * (static_cast<int64_t>(count) + 2) + 7) / 8 * 8 : 0;
}

// Exclusive prefix of the overflow record sizes over rows, in one block; total[0] is the sum.
__global__ void __launch_bounds__(1024) encode_offsets(const uint32_t* __restrict__ counts, int64_t rows, int slots,
                                                       int64_t* __restrict__ offsets, int64_t* __restrict__ total) {
  using Scan = cub::BlockScan<int64_t, 1024>;
  __shared__ typename Scan::TempStorage tmp;
  const int64_t per = (rows + 1023) / 1024;
  const int64_t r0 = threadIdx.x * per, r1 = min(rows, r0 + per);
  int64_t mine = 0;
  for (int64_t r = r0; r < r1; ++r) mine += overflow_bytes(counts[r], slots);
  int64_t before, all;
  Scan(tmp).ExclusiveSum(mine, before, all);
  for (int64_t r = r0; r < r1; ++r) {
    offsets[r] = before;
    before += overflow_bytes(counts[r], slots);
  }
  if (threadIdx.x == 0) total[0] = all;
}

__global__ void __launch_bounds__(kThreads) encode_rows(const uint16_t* __restrict__ w, int64_t rows, int K,
                                                        uint8_t* __restrict__ out, const uint32_t* __restrict__ counts,
                                                        const int64_t* __restrict__ offsets) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const int P = fx::patch_slots(K), B = fx::blocks(K), H = fx::header_bytes(K), bo = fx::base_offset(K);
  const int64_t rb = fx::row_bytes(K);
  const uint16_t* src = w + r * K;
  uint8_t* row = out + r * rb;
  uint32_t* slots = reinterpret_cast<uint32_t*>(row);
  const uint32_t count = counts[r];

  // Where this row's patches go: its header slots, or its overflow record after the rows.
  uint32_t* dst = slots;
  for (int s = lane; s < P; s += 32) slots[s] = fx::kNoPatch;
  for (int b = bo + B + lane; b < H; b += 32) row[b] = 0;  // header padding, so the encoding is deterministic
  if (count > static_cast<uint32_t>(P)) {
    const int64_t region = (rows * rb + 7) / 8 * 8;
    uint32_t* rec = reinterpret_cast<uint32_t*>(out + region + offsets[r]);
    const int64_t units = (region + offsets[r] - r * rb) / fx::kOverflowAlign;
    __syncwarp();
    if (lane == 0) {
      slots[0] = 0x80000000u | static_cast<uint32_t>(units);
      rec[0] = count;
      rec[1 + count] = fx::kNoPatch;
      if ((count & 1) != 0) rec[2 + count] = 0;  // the record's padding to 8 bytes
    }
    dst = rec + 1;
  }
  __syncwarp();

  uint32_t written = 0;
  for (int j = 0; j < B; ++j) {
    const uint32_t e0 = row[bo + j];
    uint32_t sm = 0, cn = 0, n = 0;
    uint32_t pk[4];
#pragma unroll
    for (int t = 0; t < 4; ++t) {
      const int k = fx::kBlock * j + 4 * lane + t;
      pk[t] = fx::kNoPatch;
      if (k < K) {
        const uint32_t u = src[k];
        const uint32_t e = exponent_of(u);
        sm |= (((u >> 8) & 0x80u) | (u & 0x7fu)) << (8 * t);
        if (e0 - e > 15u) {
          pk[t] = (static_cast<uint32_t>(k) << 8) | e;
          ++n;
        } else {
          cn |= (e0 - e) << (8 * t);
        }
      }
    }
    uint8_t* blk = row + H + static_cast<int64_t>(j) * fx::kBlockBytes;
    reinterpret_cast<uint32_t*>(blk)[lane] = sm;
    // Lane 2g holds values 8g .. 8g + 3 (low nibbles), lane 2g + 1 values 8g + 4 .. 8g + 7 (high).
    const uint32_t other = __shfl_xor_sync(0xffffffffu, cn, 1);
    if ((lane & 1) == 0) reinterpret_cast<uint32_t*>(blk + fx::kCodes)[lane >> 1] = cn | (other << 4);
    // Patches in k order: lanes in order, values in order within a lane.
    uint32_t before = n;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
      const uint32_t v = __shfl_up_sync(0xffffffffu, before, o);
      if (lane >= o) before += v;
    }
    const uint32_t block_total = __shfl_sync(0xffffffffu, before, 31);
    uint32_t idx = written + before - n;
#pragma unroll
    for (int t = 0; t < 4; ++t)
      if (pk[t] != fx::kNoPatch) dst[idx++] = pk[t];
    written += block_total;
  }
}

// ---- decoder ---------------------------------------------------------------------------------

// One warp per row, lane l decoding values 4l .. 4l + 3 of each block, with the same device functions
// the products use.
__global__ void __launch_bounds__(kThreads) decode_rows(const uint8_t* __restrict__ data, int64_t rows, int K,
                                                        uint16_t* __restrict__ w) {
  const int64_t r = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (r >= rows) return;
  const int lane = threadIdx.x & 31;
  const uint8_t* row = data + r * fx::row_bytes(K);
  const int H = fx::header_bytes(K), bo = fx::base_offset(K);
  fx::Patches pc;
  pc.start(row, fx::patch_slots(K));
  for (int j = 0; j < fx::blocks(K); ++j) {
    const uint8_t* blk = row + H + static_cast<int64_t>(j) * fx::kBlockBytes;
    const uint32_t k0 = fx::kBlock * j + 4 * lane;
    const uint32_t sm = reinterpret_cast<const uint32_t*>(blk)[lane];
    const uint32_t cd = reinterpret_cast<const uint32_t*>(blk + fx::kCodes)[lane >> 1];
    uint32_t e = fx::exponents(row[bo + j], cd >> (4 * (lane & 1)));
    while (pc.pos() < k0 + 4) {
      e = fx::patch(e, pc.next, k0);
      pc.advance();
    }
    const uint2 b = fx::assemble(sm, e);
    const uint16_t v[4] = {static_cast<uint16_t>(b.x), static_cast<uint16_t>(b.x >> 16), static_cast<uint16_t>(b.y),
                           static_cast<uint16_t>(b.y >> 16)};
#pragma unroll
    for (int t = 0; t < 4; ++t)
      if (static_cast<int>(k0) + t < K) w[r * K + k0 + t] = v[t];
  }
}

// ---- decode GEMV: vec kernel -----------------------------------------------------------------
// gemv.cu's vec kernel with the bf16 word replaced by this format's quad: lane l takes values
// 4l .. 4l + 3 of each 128-k step (one block), the S warps of a row group split the steps the same
// way, and the partial sums meet in the same order, so the result is lx_gemv's to the bit. A step's
// loads per lane: the sign|mantissa word (4 bytes), the codes word it shares with its neighbour
// lane, and the block's base byte (one address for the whole warp). kX = false runs the same kernel
// on plain bf16 rows (the lane's 8-byte word): the like-for-like baseline for the format.

template <bool kX>
struct Step;
template <>
struct Step<true> {
  uint32_t sm, cd, base;
};
template <>
struct Step<false> {
  uint2 w;
};

// Words for steps u_begin + u (u < KU) of the RG rows at rows[r] (null past N). Steps past u_end and
// k past K read as zero, so the consumer needs no load-side bounds handling.
template <bool kX, int KU, int RG>
__device__ __forceinline__ void load_steps(const uint8_t* const (&rows)[RG], int H, int bo, int K, int lane,
                                           int u_begin, int u_end, Step<kX> (&s)[KU][RG]) {
#pragma unroll
  for (int u = 0; u < KU; ++u) {
    const int j = u_begin + u;
    const bool live = j < u_end && j * fx::kBlock + 4 * lane < K;
    const int off = H + j * fx::kBlockBytes;
#pragma unroll
    for (int r = 0; r < RG; ++r) {
      s[u][r] = Step<kX>{};
      if (live && rows[r] != nullptr) {
        // Streamed once per token: evict-first, leaving L1 to x and the headers.
        if constexpr (kX) {
          s[u][r].sm = __ldcs(reinterpret_cast<const unsigned int*>(rows[r] + off) + lane);
          s[u][r].cd = __ldcs(reinterpret_cast<const unsigned int*>(rows[r] + off + fx::kCodes) + (lane >> 1));
          s[u][r].base = __ldg(rows[r] + bo + j);
        } else {
          s[u][r].w = __ldcs(reinterpret_cast<const uint2*>(rows[r]) + j * 32 + lane);
        }
      }
    }
  }
}

// One iteration's KU steps of the RG rows: decode, then the products. kGuard is for the last
// iteration, whose steps may run past the part's end or past K; every other iteration is free of
// per-step conditions, so the compiler interleaves its steps. Patches are checked once per row per
// iteration (a row rarely has any); only a row with one in range walks its steps.
template <bool kX, typename TX, int RG, int KU, int MT, bool kGuard>
__device__ __forceinline__ void consume(const Step<kX> (&cur)[KU][RG], fx::Patches (&pc)[RG], float (&acc)[RG][MT],
                                        const TX* __restrict__ x, int ld_x, int M, int K, int j, int u1, int lane,
                                        int shift) {
  // With one row of x, the iteration's x values are loaded before the decode so their L1 latency
  // overlaps it; more rows would hold too many registers, and load beside their products.
  constexpr bool kEarlyX = MT == 1;
  float xf[kEarlyX ? KU : 1][MT][4];
  if constexpr (kEarlyX) {
#pragma unroll
    for (int u = 0; u < KU; ++u) {
      const int k0 = (j + u) * fx::kBlock + 4 * lane;
      const bool live = !kGuard || (j + u < u1 && k0 < K);
#pragma unroll
      for (int m = 0; m < MT; ++m)
        if (live && (MT == 1 || m < M)) load_x4<TX>(x + static_cast<int64_t>(m) * ld_x + k0, xf[u][m]);
    }
  }
  uint32_t e[KU][RG];
  if constexpr (kX) {
#pragma unroll
    for (int u = 0; u < KU; ++u)
#pragma unroll
      for (int r = 0; r < RG; ++r) e[u][r] = fx::exponents(cur[u][r].base, cur[u][r].cd >> shift);
  }
  const uint32_t iter_end = static_cast<uint32_t>((j + KU) * fx::kBlock);
#pragma unroll
  for (int r = 0; r < RG; ++r) {
    if (kX && pc[r].pos() < iter_end) {
#pragma unroll
      for (int u = 0; u < KU; ++u) {
        const uint32_t k0 = static_cast<uint32_t>((j + u) * fx::kBlock + 4 * lane);
        const uint32_t end = static_cast<uint32_t>((j + u + 1) * fx::kBlock);
        while (pc[r].pos() < end) {
          e[u][r] = fx::patch(e[u][r], pc[r].next, k0);
          pc[r].advance();
        }
      }
    }
  }
#pragma unroll
  for (int u = 0; u < KU; ++u) {
    const int k0 = (j + u) * fx::kBlock + 4 * lane;
    if (!kGuard || (j + u < u1 && k0 < K)) {
      float wf[RG][4];
#pragma unroll
      for (int r = 0; r < RG; ++r) {
        if constexpr (kX) fx::to_f32(fx::assemble(cur[u][r].sm, e[u][r]), wf[r]);
        else decode4(cur[u][r].w, wf[r]);
      }
#pragma unroll
      for (int m = 0; m < MT; ++m) {
        if (MT == 1 || m < M) {
          float xl[4];
          if constexpr (kEarlyX) {
#pragma unroll
            for (int t = 0; t < 4; ++t) xl[t] = xf[u][m][t];
          } else {
            load_x4<TX>(x + static_cast<int64_t>(m) * ld_x + k0, xl);
          }
#pragma unroll
          for (int r = 0; r < RG; ++r)
#pragma unroll
            for (int t = 0; t < 4; ++t) acc[r][m] = fmaf(wf[r][t], xl[t], acc[r][m]);
        }
      }
    }
  }
}

// MT is the M bucket the kernel is built for (1, 2, 4, 8); M <= MT rows of x are used. Building per
// bucket keeps the M = 1 kernel free of the predicated-off work of seven unused rows, which would
// otherwise take most of the issue slots.
template <bool kX, typename TX, int RG, int S, int KU, int MT>
__global__ void __launch_bounds__(kThreads)
gemv_vec_kernel(const TX* __restrict__ x, int ld_x, const uint8_t* __restrict__ w, int K,
                const void* __restrict__ bias, bool bias_bf16, void* __restrict__ y, bool y_bf16, int ld_y, int M,
                int N, float alpha, float beta) {
  constexpr int kGroups = kWarps / S;
  __shared__ float partial[kWarps][RG * MT];

  cudaTriggerProgrammaticLaunchCompletion();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int group = warp / S, part = warp % S;
  const int64_t first = (static_cast<int64_t>(blockIdx.x) * kGroups + group) * RG;
  const int steps = fx::blocks(K);
  const int u0 = (part * steps) / S;
  const int u1 = ((part + 1) * steps) / S;
  const int64_t rb = kX ? fx::row_bytes(K) : 2 * static_cast<int64_t>(K);
  const int H = fx::header_bytes(K), bo = fx::base_offset(K);
  const int shift = 4 * (lane & 1);
  // Steps before this are whole (all their k < K): K % 128 == 0 leaves none partial.
  const int full = min(u1, K / fx::kBlock);

  const uint8_t* rows[RG];
#pragma unroll
  for (int r = 0; r < RG; ++r) rows[r] = first + r < N ? w + (first + r) * rb : nullptr;

  float acc[RG][MT];
#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m) acc[r][m] = 0.0f;

  // Weights do not depend on the previous kernel: the first steps and the patch lists are fetched
  // before the dependency wait.
  Step<kX> cur[KU][RG];
  load_steps<kX, KU, RG>(rows, H, bo, K, lane, u0, u1, cur);
  fx::Patches pc[RG];
#pragma unroll
  for (int r = 0; r < RG; ++r) {
    pc[r].p = nullptr;
    pc[r].left = 0;
    pc[r].next = fx::kNoPatch;
    if (kX && rows[r] != nullptr) pc[r].prefetch(rows[r]);
  }
  cudaGridDependencySynchronize();

  if (first < N) {
    for (int j = u0; j < u1; j += KU) {
      Step<kX> nxt[KU][RG];
      load_steps<kX, KU, RG>(rows, H, bo, K, lane, j + KU, u1, nxt);
      if (kX && j == u0) {  // the patch lists arrive with the first steps: read them once those are needed
#pragma unroll
        for (int r = 0; r < RG; ++r) {
          if (rows[r] != nullptr) {
            pc[r].resolve(fx::patch_slots(K));
            pc[r].skip_to(static_cast<uint32_t>(u0 * fx::kBlock));
          }
        }
      }
      if (j + KU <= full) consume<kX, TX, RG, KU, MT, false>(cur, pc, acc, x, ld_x, M, K, j, u1, lane, shift);
      else consume<kX, TX, RG, KU, MT, true>(cur, pc, acc, x, ld_x, M, K, j, u1, lane, shift);
#pragma unroll
      for (int u = 0; u < KU; ++u)
#pragma unroll
        for (int r = 0; r < RG; ++r) cur[u][r] = nxt[u][r];
    }
  }

#pragma unroll
  for (int r = 0; r < RG; ++r)
#pragma unroll
    for (int m = 0; m < MT; ++m)
      if (MT == 1 || m < M) acc[r][m] = lx::warp_sum(acc[r][m]);

  if constexpr (S == 1) {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RG; ++r) {
        const int64_t n = first + r;
        if (n >= N) continue;
        const float b = bias_at(bias, bias_bf16, n);
#pragma unroll
        for (int m = 0; m < MT; ++m)
          if (MT == 1 || m < M) store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, acc[r][m] + b, alpha, beta);
      }
    }
  } else {
    if (lane == 0) {
#pragma unroll
      for (int r = 0; r < RG; ++r)
#pragma unroll
        for (int m = 0; m < MT; ++m)
          if (MT == 1 || m < M) partial[warp][r * MT + m] = acc[r][m];
    }
    __syncthreads();
    for (int o = threadIdx.x; o < kGroups * RG * M; o += kThreads) {
      const int g = o / (RG * M), rem = o % (RG * M), r = rem / M, m = rem % M;
      const int64_t n = (static_cast<int64_t>(blockIdx.x) * kGroups + g) * RG + r;
      if (n >= N) continue;
      float sum = 0.0f;
      for (int p = 0; p < S; ++p) sum += partial[g * S + p][r * MT + m];
      store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, sum + bias_at(bias, bias_bf16, n), alpha, beta);
    }
  }
}

// ---- decode GEMV: small kernel ---------------------------------------------------------------
// gemv.cu's small kernel (K <= 256, or K % 4 != 0, or misaligned x): lpr lanes per row, lane c reading
// the e consecutive inputs from c * e, multiplied one at a time in k order, then the row's lanes summed
// by xor shuffles. The arithmetic is per lane, so the block shape is free: kSmallWarps warps a block
// (gemv.cu runs one) keep enough rows in flight to cover DRAM latency. When each lane takes exactly 8
// inputs (K = 64, 128, 256) and x is 16-byte aligned, a lane's weights are one load (two quads of this
// format, or 16 bytes of bf16) issued before the dependency wait, and its x rows are vector loads.

constexpr int kSmallWarps = 4;

template <bool kX>
__device__ __forceinline__ void small_octet(const uint8_t* __restrict__ w, int64_t n, int K, int c, float (&wv)[8]) {
  const int q = 8 * c;
  if constexpr (kX) {
    const uint8_t* row = w + n * fx::row_bytes(K);
    const uint8_t* blk = row + fx::header_bytes(K) + (q >> 7) * fx::kBlockBytes;
    const int off = q & 127;
    fx::Patches pc;
    pc.prefetch(row);
    const uint2 sm = *reinterpret_cast<const uint2*>(blk + off);
    const uint32_t cd = *reinterpret_cast<const uint32_t*>(blk + fx::kCodes + (off >> 1));  // word off / 8
    const uint32_t base = row[fx::base_offset(K) + (q >> 7)];
    pc.resolve(fx::patch_slots(K));
    uint32_t e0 = fx::exponents(base, cd), e1 = fx::exponents(base, cd >> 4);
    while (pc.pos() < static_cast<uint32_t>(q + 8)) {  // patches below q are other lanes' and change nothing
      e0 = fx::patch(e0, pc.next, static_cast<uint32_t>(q));
      e1 = fx::patch(e1, pc.next, static_cast<uint32_t>(q + 4));
      pc.advance();
    }
    fx::to_f32(fx::assemble(sm.x, e0), wv);
    fx::to_f32(fx::assemble(sm.y, e1), wv + 4);
  } else {
    const uint4 v = __ldcs(reinterpret_cast<const uint4*>(reinterpret_cast<const bf16*>(w) + n * K + q));
    decode4(make_uint2(v.x, v.y), wv);
    decode4(make_uint2(v.z, v.w), wv + 4);
  }
}

template <bool kX, typename TX>
__global__ void __launch_bounds__(32 * kSmallWarps)
gemv_small_kernel(const TX* __restrict__ x, int ld_x, const uint8_t* __restrict__ w, const void* __restrict__ bias,
                  bool bias_bf16, void* __restrict__ y, bool y_bf16, int ld_y, int M, int N, int K, int lpr, int e,
                  bool octet, float alpha, float beta) {
  cudaTriggerProgrammaticLaunchCompletion();
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int slot = lane / lpr, c = lane % lpr;
  const int64_t n = (static_cast<int64_t>(blockIdx.x) * kSmallWarps + warp) * (32 / lpr) + slot;

  float acc[kMaxM];
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) acc[m] = 0.0f;
  if (octet) {
    // Weights do not depend on the previous kernel: fetch them before the wait.
    float wv[8];
    const bool live = n < N && 8 * c < K;
    if (live) small_octet<kX>(w, n, K, c, wv);
    cudaGridDependencySynchronize();
    if (live) {
#pragma unroll
      for (int m = 0; m < kMaxM; ++m) {
        if (m < M) {
          float xf[8];
          const TX* xr = x + static_cast<int64_t>(m) * ld_x + 8 * c;
          load_x4<TX>(xr, xf);
          load_x4<TX>(xr + 4, xf + 4);
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[m] = fmaf(wv[i], xf[i], acc[m]);
        }
      }
    }
  } else if (n < N && !kX) {
    cudaGridDependencySynchronize();
    const bf16* row = reinterpret_cast<const bf16*>(w) + n * K;
    for (int i = 0; i < e; ++i) {
      const int k = c * e + i;
      if (k >= K) break;
      const float wv = lx::to_f32(row[k]);
#pragma unroll
      for (int m = 0; m < kMaxM; ++m)
        if (m < M) acc[m] = fmaf(wv, lx::to_f32(x[static_cast<int64_t>(m) * ld_x + k]), acc[m]);
    }
  } else if (n < N) {
    cudaGridDependencySynchronize();
    const uint8_t* row = w + n * fx::row_bytes(K);
    fx::Patches pc;
    pc.start(row, fx::patch_slots(K));
    if (e % 4 == 0 && K % 4 == 0) {
      // Whole quads (e and every run start are multiples of 4): decode four values per load, then
      // multiply them in order, exactly as the one-at-a-time loop below would.
      const int H = fx::header_bytes(K), bo = fx::base_offset(K);
      for (int q = c * e; q < c * e + e && q < K; q += 4) {
        const uint8_t* blk = row + H + (q >> 7) * fx::kBlockBytes;
        const int off = q & 127;
        const uint32_t sm = *reinterpret_cast<const uint32_t*>(blk + off);
        const uint32_t cd = *reinterpret_cast<const uint32_t*>(blk + fx::kCodes + 4 * (off >> 3));
        uint32_t ex = fx::exponents(row[bo + (q >> 7)], cd >> (off & 4));
        while (pc.pos() < static_cast<uint32_t>(q + 4)) {
          ex = fx::patch(ex, pc.next, static_cast<uint32_t>(q));
          pc.advance();
        }
        float wv[4];
        fx::to_f32(fx::assemble(sm, ex), wv);
#pragma unroll
        for (int t = 0; t < 4; ++t)
#pragma unroll
          for (int m = 0; m < kMaxM; ++m)
            if (m < M) acc[m] = fmaf(wv[t], lx::to_f32(x[static_cast<int64_t>(m) * ld_x + q + t]), acc[m]);
      }
    } else {
      for (int i = 0; i < e; ++i) {
        const int k = c * e + i;
        if (k >= K) break;
        const float wv = bits_f32(value_bits(row, K, k, pc));
#pragma unroll
        for (int m = 0; m < kMaxM; ++m)
          if (m < M) acc[m] = fmaf(wv, lx::to_f32(x[static_cast<int64_t>(m) * ld_x + k]), acc[m]);
      }
    }
  } else {
    cudaGridDependencySynchronize();
  }
  for (int off = lpr / 2; off > 0; off >>= 1)
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) acc[m] += __shfl_xor_sync(0xffffffffu, acc[m], off);

  if (c == 0 && n < N) {
    const float b = bias_at(bias, bias_bf16, n);
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      if (m < M) store_y(y, y_bf16, static_cast<int64_t>(m) * ld_y + n, acc[m] + b, alpha, beta);
  }
}

// ---- decode GEMV: host side ------------------------------------------------------------------

struct Args {
  const void* x;
  int ld_x;
  const uint8_t* w;
  const void* bias;
  bool bias_bf16;
  void* y;
  bool y_bf16;
  int ld_y;
  int M, N, K;
  float alpha, beta;
  cudaStream_t stream;
};

template <bool kX, typename TX>
cudaError_t launch_small(const Args& a) {
  int lpr = 1;
  while (lpr < 32 && lpr * 8 < a.K) lpr *= 2;
  const int e = (a.K + lpr - 1) / lpr;
  const int rows_per_block = kSmallWarps * (32 / lpr);
  const unsigned blocks = static_cast<unsigned>((a.N + rows_per_block - 1) / rows_per_block);
  // Whole octets per lane, x rows 16-byte aligned, and (bf16) rows 16-byte aligned.
  const bool octet =
      e == 8 && a.K % 8 == 0 &&
      ((reinterpret_cast<uintptr_t>(a.x) | (static_cast<uint64_t>(a.ld_x) * sizeof(TX))) % 16 == 0) &&
      (kX || reinterpret_cast<uintptr_t>(a.w) % 16 == 0);
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(blocks, 32 * kSmallWarps, a.stream, attr);
  cudaLaunchKernelEx(&cfg, gemv_small_kernel<kX, TX>, static_cast<const TX*>(a.x), a.ld_x, a.w, a.bias, a.bias_bf16, a.y,
                     a.y_bf16, a.ld_y, a.M, a.N, a.K, lpr, e, octet, a.alpha, a.beta);
  return cudaGetLastError();
}

template <bool kX, typename TX, int RG, int S, int KU>
cudaError_t run_vec(const Args& a) {
  constexpr int kGroups = kWarps / S;
  const int64_t rows_per_block = static_cast<int64_t>(kGroups) * RG;
  const unsigned blocks = static_cast<unsigned>((a.N + rows_per_block - 1) / rows_per_block);
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(blocks, kThreads, a.stream, attr);
  const TX* x = static_cast<const TX*>(a.x);
  if (a.M == 1)
    cudaLaunchKernelEx(&cfg, gemv_vec_kernel<kX, TX, RG, S, KU, 1>, x, a.ld_x, a.w, a.K, a.bias, a.bias_bf16, a.y,
                       a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  else if (a.M == 2)
    cudaLaunchKernelEx(&cfg, gemv_vec_kernel<kX, TX, RG, S, KU, 2>, x, a.ld_x, a.w, a.K, a.bias, a.bias_bf16, a.y,
                       a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  else if (a.M <= 4)
    cudaLaunchKernelEx(&cfg, gemv_vec_kernel<kX, TX, RG, S, KU, 4>, x, a.ld_x, a.w, a.K, a.bias, a.bias_bf16, a.y,
                       a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  else
    cudaLaunchKernelEx(&cfg, gemv_vec_kernel<kX, TX, RG, S, KU, 8>, x, a.ld_x, a.w, a.K, a.bias, a.bias_bf16, a.y,
                       a.y_bf16, a.ld_y, a.M, a.N, a.alpha, a.beta);
  return cudaGetLastError();
}

// Variants: rows per warp and steps in flight. Neither changes the summation order.
struct Variant {
  int RG, KU;
};
constexpr Variant kVariants[] = {{1, 2}, {1, 4}, {2, 2}, {2, 4}};
const lx::TunableRegistration kTune("gemv_bf16x", {"rg1_ku2", "rg1_ku4", "rg2_ku2", "rg2_ku4"});
const lx::TunableRegistration kTuneBf16("gemv_decode_bf16", {"rg1_ku2", "rg1_ku4", "rg2_ku2", "rg2_ku4"});

// The split S must be gemv.cu's choice for bf16 weights (steps of 128 k), or the partial sums would
// meet in a different order. Rows per warp and steps in flight are free.
int choose_split(int N, int K) {
  const int steps = (K + 127) / 128;
  int S = 1;
  while (S < 8 && static_cast<int64_t>(N) * S < kWarpTarget && steps / (2 * S) >= 1) S *= 2;
  return S;
}

template <bool kX, typename TX>
const void* vec_kernel_ptr(int RG, int S, int KU, int MT);

// Blocks of a kernel that fit on the GPU at once, cached per (device, kernel).
int resident_blocks(const void* kernel) {
  struct Entry {
    const void* kernel;
    int dev, blocks;
  };
  static thread_local Entry cache[32];
  static thread_local int used = 0;
  int dev = 0;
  cudaGetDevice(&dev);
  for (int i = 0; i < used; ++i)
    if (cache[i].kernel == kernel && cache[i].dev == dev) return cache[i].blocks;
  int sms = 0, per_sm = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, kernel, kThreads, 0);
  const int blocks = sms * (per_sm > 0 ? per_sm : 1);
  cache[used < 32 ? used++ : 31] = Entry{kernel, dev, blocks};
  return blocks;
}

// From the sweeps on the 5070 Ti (bench/bench_bf16x.py): at M = 1, two rows per warp with four steps
// in flight is within 6% of the best variant at every Praxis shape. At M = 2 to 4, one row a warp,
// with four steps in flight when that is all of a warp's steps: a grid that fits in one wave keeps
// every block it can (fewer, two-row blocks only lose parallelism), and so does one of three or more
// waves, whose tail is a small part of it. Only a grid of just over one wave, whose second wave would
// be mostly idle, takes two rows a warp to fit in one (3072 x 2048: 12.6 us against 15.7). At M > 4
// two rows share each x load, which pays once N is large.
template <bool kX>
Variant choose_variant(int N, int K, int M, int S, bool x_bf16) {
  if (M == 1) return Variant{2, 4};
  const Variant two_rows{2, 2};
  if (M > 4) return N >= 2048 ? two_rows : Variant{1, 2};
  const int per_warp = (fx::blocks(K) + S - 1) / S;
  const Variant one_row{1, per_warp <= 4 ? 4 : 2};
  const int MT = M == 2 ? 2 : 4;
  const int64_t rows_per_block = kWarps / S;
  const int64_t blocks = (N + rows_per_block - 1) / rows_per_block;
  const int slots = resident_blocks(x_bf16 ? vec_kernel_ptr<kX, bf16>(one_row.RG, S, one_row.KU, MT)
                                           : vec_kernel_ptr<kX, float>(one_row.RG, S, one_row.KU, MT));
  const int64_t waves = (blocks + slots - 1) / slots;
  if (waves == 1 || waves >= 3 || 4 * blocks >= 3 * waves * slots) return one_row;
  return two_rows;
}

template <bool kX>
Variant pick_variant(int N, int K, int M, int S, bool x_bf16) {
  const int tuned = lx_tune_get(kX ? "gemv_bf16x" : "gemv_decode_bf16", M, N, K);
  if (tuned >= 0 && tuned < static_cast<int>(sizeof(kVariants) / sizeof(kVariants[0]))) return kVariants[tuned];
  return choose_variant<kX>(N, K, M, S, x_bf16);
}

template <bool kX, typename TX, int MT>
const void* vec_kernel_mt(int RG, int S, int KU) {
#define LX_PTR(S_, RG_, KU_) \
  if (S == S_ && RG == RG_ && KU == KU_) return reinterpret_cast<const void*>(gemv_vec_kernel<kX, TX, RG_, S_, KU_, MT>);
#define LX_PTRS(S_) LX_PTR(S_, 1, 1) LX_PTR(S_, 1, 2) LX_PTR(S_, 1, 4) LX_PTR(S_, 2, 1) LX_PTR(S_, 2, 2) LX_PTR(S_, 2, 4)
  LX_PTRS(1) LX_PTRS(2) LX_PTRS(4) LX_PTRS(8)
#undef LX_PTRS
#undef LX_PTR
  return nullptr;
}

template <bool kX, typename TX>
const void* vec_kernel_ptr(int RG, int S, int KU, int MT) {
  if (MT == 1) return vec_kernel_mt<kX, TX, 1>(RG, S, KU);
  if (MT == 2) return vec_kernel_mt<kX, TX, 2>(RG, S, KU);
  if (MT == 4) return vec_kernel_mt<kX, TX, 4>(RG, S, KU);
  return vec_kernel_mt<kX, TX, 8>(RG, S, KU);
}

template <bool kX, typename TX>
cudaError_t launch_vec(const Args& a) {
  const int S = choose_split(a.N, a.K);
  Variant v = pick_variant<kX>(a.N, a.K, a.M, S, std::is_same_v<TX, bf16>);
  const int steps = (a.K + 127) / 128;
  while (v.KU > 1 && v.KU > steps / S) v.KU /= 2;
#define LX_CASE(S_, RG_, KU_) \
  if (S == S_ && v.RG == RG_ && v.KU == KU_) return run_vec<kX, TX, RG_, S_, KU_>(a);
#define LX_SPLIT(S_) \
  LX_CASE(S_, 1, 1) LX_CASE(S_, 1, 2) LX_CASE(S_, 1, 4) LX_CASE(S_, 2, 1) LX_CASE(S_, 2, 2) LX_CASE(S_, 2, 4)
  LX_SPLIT(1) LX_SPLIT(2) LX_SPLIT(4) LX_SPLIT(8)
#undef LX_SPLIT
#undef LX_CASE
  return cudaErrorInvalidValue;
}

template <bool kX, typename TX>
cudaError_t launch_gemv(const Args& a) {
  // gemv.cu's choice of kernel for the same x and weight (this format's rows are 8-byte aligned by
  // construction; a bf16 weight must be 16-byte aligned for the vector loads, as in gemv.cu).
  const bool aligned =
      ((reinterpret_cast<uintptr_t>(a.x) | (static_cast<uint64_t>(a.ld_x) * sizeof(TX))) % 16 == 0) &&
      (kX || reinterpret_cast<uintptr_t>(a.w) % 16 == 0);
  if (a.K <= kSmallK || a.K % 4 != 0 || !aligned) return launch_small<kX, TX>(a);
  return launch_vec<kX, TX>(a);
}

bool valid_dtype(lx_dtype d) { return d == LX_F32 || d == LX_BF16; }

// ---- decode experts --------------------------------------------------------------------------
// moe.cu's two stages with this format's weights, and the same arithmetic: a warp's dot product
// over a row takes, per lane, the 8 consecutive values at lane * 8 + 256 v and adds them pairwise
// (acc += w0 * x0 + w1 * x1), then sums the warp. Stage 1 writes bf16(silu(gate) * up) per (pair,
// intermediate row); stage 2 sums weight * dot over the token's experts per output row.
//
// Built around the decode stream instead of moe.cu's shared-memory staging: every chunk of a warp's
// rows is loaded before any is used, x comes from L1, and stage 2 fetches its routing and down rows
// before the programmatic-dependent-launch wait, so they stream while stage 1 runs. kX selects this
// format (true) or plain bf16 rows (false): the same kernels on bf16 are the baseline the format is
// measured against, and are themselves bit-identical to lx_moe_experts.

// A lane's 8 values of one chunk, as four words of bf16 pairs (values 0, 1 in the first).
template <bool kX>
struct Chunk;

template <>
struct Chunk<true> {
  uint2 sm;
  uint32_t cd, base;
  __device__ __forceinline__ void load(const uint8_t* row, int K, int c) {
    const uint8_t* blk = row + fx::header_bytes(K) + (c >> 7) * fx::kBlockBytes;
    sm = __ldcs(reinterpret_cast<const uint2*>(blk + (c & 127)));
    cd = __ldcs(reinterpret_cast<const unsigned int*>(blk + fx::kCodes + 4 * ((c & 127) >> 3)));
    base = __ldg(row + fx::base_offset(K) + (c >> 7));
  }
  // Values c .. c + 7, with every patch below `end` (the chunk's end for the whole warp) applied.
  __device__ __forceinline__ uint4 bits(int c, uint32_t end, fx::Patches& pc) const {
    uint32_t lo = fx::exponents(base, cd), hi = fx::exponents(base, cd >> 4);
    while (pc.pos() < end) {
      lo = fx::patch(lo, pc.next, static_cast<uint32_t>(c));
      hi = fx::patch(hi, pc.next, static_cast<uint32_t>(c + 4));
      pc.advance();
    }
    const uint2 a = fx::assemble(sm.x, lo), b = fx::assemble(sm.y, hi);
    return make_uint4(a.x, a.y, b.x, b.y);
  }
};

template <>
struct Chunk<false> {
  uint4 w;
  __device__ __forceinline__ void load(const uint8_t* row, int K, int c) {
    (void)K;
    w = __ldcs(reinterpret_cast<const uint4*>(row + 2 * static_cast<int64_t>(c)));
  }
  __device__ __forceinline__ uint4 bits(int, uint32_t, fx::Patches&) const { return w; }
};

template <bool kX>
__host__ __device__ inline int64_t expert_row_bytes(int K) {
  return kX ? fx::row_bytes(K) : 2 * static_cast<int64_t>(K);
}

// moe.cu's warp_dot step for one chunk: 8 weights (four bf16 pairs) against 8 activations.
__device__ __forceinline__ float dot_chunk(float acc, const uint4& w, const uint4& x) {
  const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
  const uint32_t xv[4] = {x.x, x.y, x.z, x.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 fa = make_float2(__uint_as_float(wv[j] << 16), __uint_as_float(wv[j] & 0xffff0000u));
    const float2 fb = make_float2(__uint_as_float(xv[j] << 16), __uint_as_float(xv[j] & 0xffff0000u));
    acc += fa.x * fb.x + fa.y * fb.y;
  }
  return acc;
}

constexpr int kChunks = 4;  // chunks of a row a lane holds in registers at once (rows of up to 1024)

// The warp's dot product of one row (n values, n % 8 == 0) with x, as moe.cu's warp_dot computes it.
// Loads are issued kChunks at a time ahead of their use.
// Gate and up rows (or any two rows against the same x): every load of both is issued before any
// is used.
template <bool kX>
__device__ __forceinline__ void row_dot2(const uint8_t* row_a, const uint8_t* row_b, const bf16* x, int n, int lane,
                                         float& dot_a, float& dot_b) {
  fx::Patches pa, pb;
  pa.next = pb.next = fx::kNoPatch;
  if constexpr (kX) {
    pa.prefetch(row_a);
    pb.prefetch(row_b);
  }
  float acc_a = 0.0f, acc_b = 0.0f;
  const int nv = (n + 255) / 256;
  for (int v0 = 0; v0 < nv; v0 += kChunks) {
    Chunk<kX> ca[kChunks], cb[kChunks];
    uint4 xs[kChunks];
#pragma unroll
    for (int v = 0; v < kChunks; ++v) {
      const int c = lane * 8 + 256 * (v0 + v);
      if (v0 + v < nv && c < n) {
        ca[v].load(row_a, n, c);
        cb[v].load(row_b, n, c);
        xs[v] = __ldg(reinterpret_cast<const uint4*>(x + c));
      }
    }
    if constexpr (kX) {
      if (v0 == 0) {
        pa.resolve(fx::patch_slots(n));
        pb.resolve(fx::patch_slots(n));
      }
    }
#pragma unroll
    for (int v = 0; v < kChunks; ++v) {
      const int c = lane * 8 + 256 * (v0 + v);
      if (v0 + v < nv && c < n) {
        const uint32_t end = static_cast<uint32_t>(256 * (v0 + v + 1));
        acc_a = dot_chunk(acc_a, ca[v].bits(c, end, pa), xs[v]);
        acc_b = dot_chunk(acc_b, cb[v].bits(c, end, pb), xs[v]);
      }
    }
  }
  dot_a = lx::warp_sum(acc_a);
  dot_b = lx::warp_sum(acc_b);
}

// The same for rows with n % 8 != 0: moe.cu's scalar warp_dot (lane strided, one value at a time).
template <bool kX>
__device__ __forceinline__ float row_dot_scalar(const uint8_t* row, const bf16* x, int n, int lane) {
  float acc = 0.0f;
  if constexpr (kX) {
    fx::Patches pc;
    pc.start(row, fx::patch_slots(n));
    for (int c = lane; c < n; c += 32) acc += bits_f32(value_bits(row, n, c, pc)) * lx::to_f32(x[c]);
  } else {
    const bf16* r = reinterpret_cast<const bf16*>(row);
    for (int c = lane; c < n; c += 32) acc += lx::to_f32(r[c]) * lx::to_f32(x[c]);
  }
  return lx::warp_sum(acc);
}

// Stage 1: block b covers intermediate rows 8b' .. 8b' + 7 of pair p (b = p * row_blocks + b').
template <bool kX, bool kVec>
__global__ void __launch_bounds__(kThreads) experts_gate_up(const bf16* __restrict__ latent,
                                                            const int32_t* __restrict__ ids,
                                                            const uint8_t* __restrict__ gate_up,
                                                            bf16* __restrict__ inter, int L, int I, int top_k,
                                                            int row_blocks) {
  // The routing comes from the previous kernel. Triggering only after the wait lets stage 2 read
  // the routing in its own prologue: once it starts, every block here has seen the routing complete.
  cudaGridDependencySynchronize();
  cudaTriggerProgrammaticLaunchCompletion();
  const int64_t p = blockIdx.x / row_blocks;
  const int i = (blockIdx.x % row_blocks) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int64_t e = ids[p];
  const int64_t rb = expert_row_bytes<kX>(L);
  const bf16* x = latent + (p / top_k) * L;
  const uint8_t* gate_row = gate_up + (e * 2 * I + i) * rb;
  const uint8_t* up_row = gate_up + (e * 2 * I + I + i) * rb;
  float gate, up;
  if constexpr (kVec) {
    row_dot2<kX>(gate_row, up_row, x, L, lane, gate, up);
  } else {
    gate = row_dot_scalar<kX>(gate_row, x, L, lane);
    up = row_dot_scalar<kX>(up_row, x, L, lane);
  }
  if (lane == 0) inter[p * I + i] = lx::from_f32<bf16>(lx::silu(gate) * up);
}

// Stage 2: block b covers output rows 8b' .. 8b' + 7 of token t (b = t * row_blocks + b'). The
// token's routing and down rows (up to KMAX experts, chunks up to NV per row) are loaded before the
// wait; anything beyond is loaded after it.
template <bool kX, int KMAX, int NV>
__global__ void __launch_bounds__(kThreads) experts_down(const bf16* __restrict__ inter,
                                                         const int32_t* __restrict__ ids,
                                                         const float* __restrict__ weights,
                                                         const uint8_t* __restrict__ down, float* __restrict__ out,
                                                         int L, int I, int top_k, int row_blocks) {
  cudaTriggerProgrammaticLaunchCompletion();  // the next kernel waits for this one before reading `out`
  const int64_t t = blockIdx.x / row_blocks;
  const int l = (blockIdx.x % row_blocks) * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const bool live = l < L;
  const int64_t rb = expert_row_bytes<kX>(I);
  const int nv = (I + 255) / 256;

  const uint8_t* rows[KMAX];
  float wk[KMAX];
  Chunk<kX> ch[KMAX][NV];
  fx::Patches pc[KMAX];
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    rows[k] = nullptr;
    wk[k] = 0.0f;
    pc[k].next = fx::kNoPatch;
    if (k < top_k && live) {
      const int64_t p = t * top_k + k;
      rows[k] = down + (static_cast<int64_t>(ids[p]) * L + l) * rb;
      wk[k] = weights[p];
      if constexpr (kX) pc[k].prefetch(rows[k]);
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int c = lane * 8 + 256 * v;
        if (v < nv && c < I) ch[k][v].load(rows[k], I, c);
      }
    }
  }
  if constexpr (kX) {
#pragma unroll
    for (int k = 0; k < KMAX; ++k)
      if (k < top_k && live) pc[k].resolve(fx::patch_slots(I));
  }
  cudaGridDependencySynchronize();  // stage 1 has written the intermediates
  if (!live) return;

  float acc = 0.0f;
#pragma unroll
  for (int k = 0; k < KMAX; ++k) {
    if (k < top_k) {
      const bf16* h = inter + (t * top_k + k) * I;
      float part = 0.0f;
#pragma unroll
      for (int v = 0; v < NV; ++v) {
        const int c = lane * 8 + 256 * v;
        if (v < nv && c < I)
          part = dot_chunk(part, ch[k][v].bits(c, static_cast<uint32_t>(256 * (v + 1)), pc[k]),
                           __ldg(reinterpret_cast<const uint4*>(h + c)));
      }
      for (int v = NV; v < nv; ++v) {
        const int c = lane * 8 + 256 * v;
        if (c < I) {
          Chunk<kX> late;
          late.load(rows[k], I, c);
          part = dot_chunk(part, late.bits(c, static_cast<uint32_t>(256 * (v + 1)), pc[k]),
                           __ldg(reinterpret_cast<const uint4*>(h + c)));
        }
      }
      acc += wk[k] * lx::warp_sum(part);
    }
  }
  if (lane == 0) out[t * L + l] = acc;
}

// Stage 2 for rows with I % 8 != 0 (moe.cu's scalar dot), nothing loaded ahead.
template <bool kX>
__global__ void __launch_bounds__(kThreads) experts_down_scalar(const bf16* __restrict__ inter,
                                                                const int32_t* __restrict__ ids,
                                                                const float* __restrict__ weights,
                                                                const uint8_t* __restrict__ down,
                                                                float* __restrict__ out, int L, int I, int top_k,
                                                                int row_blocks) {
  cudaTriggerProgrammaticLaunchCompletion();
  cudaGridDependencySynchronize();
  const int64_t t = blockIdx.x / row_blocks;
  const int l = (blockIdx.x % row_blocks) * kWarps + (threadIdx.x >> 5);
  if (l >= L) return;
  const int lane = threadIdx.x & 31;
  const int64_t rb = expert_row_bytes<kX>(I);
  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int64_t p = t * top_k + k;
    const uint8_t* row = down + (static_cast<int64_t>(ids[p]) * L + l) * rb;
    acc += weights[p] * row_dot_scalar<kX>(row, inter + p * I, I, lane);
  }
  if (lane == 0) out[t * L + l] = acc;
}

template <typename... Params, typename... Args>
cudaError_t launch_pdl(void (*kernel)(Params...), unsigned grid, cudaStream_t s, Args... args) {
  cudaLaunchAttribute attr{};
  const cudaLaunchConfig_t cfg = pdl_config(grid, kThreads, s, attr);
  return cudaLaunchKernelEx(&cfg, kernel, args...);
}

template <bool kX>
cudaError_t experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                    const void* down, float* out, void* workspace, int T, int L, int I, int top_k,
                    cudaStream_t s) {
  const int64_t pairs = static_cast<int64_t>(T) * top_k;
  const int rows_a = (I + kWarps - 1) / kWarps, rows_b = (L + kWarps - 1) / kWarps;
  if (pairs * rows_a > INT32_MAX || static_cast<int64_t>(T) * rows_b > INT32_MAX) return cudaErrorInvalidValue;
  auto* inter = static_cast<bf16*>(workspace);
  const auto* lat = static_cast<const bf16*>(latent);
  const auto* gu = static_cast<const uint8_t*>(gate_up);
  const auto* dn = static_cast<const uint8_t*>(down);
  const auto grid_a = static_cast<unsigned>(pairs * rows_a);
  cudaError_t err = L % 8 == 0
                        ? launch_pdl(experts_gate_up<kX, true>, grid_a, s, lat, ids, gu, inter, L, I, top_k, rows_a)
                        : launch_pdl(experts_gate_up<kX, false>, grid_a, s, lat, ids, gu, inter, L, I, top_k, rows_a);
  if (err != cudaSuccess) return err;
  const auto grid_b = static_cast<unsigned>(static_cast<int64_t>(T) * rows_b);
  const bf16* in = inter;
  if (I % 8 != 0) return launch_pdl(experts_down_scalar<kX>, grid_b, s, in, ids, weights, dn, out, L, I, top_k, rows_b);
  const int nv = (I + 255) / 256;
  if (top_k <= 4 && nv <= 2)
    return launch_pdl(experts_down<kX, 4, 2>, grid_b, s, in, ids, weights, dn, out, L, I, top_k, rows_b);
  if (top_k <= 4 && nv <= 4)
    return launch_pdl(experts_down<kX, 4, 4>, grid_b, s, in, ids, weights, dn, out, L, I, top_k, rows_b);
  if (nv <= 2) return launch_pdl(experts_down<kX, 8, 2>, grid_b, s, in, ids, weights, dn, out, L, I, top_k, rows_b);
  return launch_pdl(experts_down<kX, 8, 1>, grid_b, s, in, ids, weights, dn, out, L, I, top_k, rows_b);
}

}  // namespace

// ---- C ABI -----------------------------------------------------------------------------------

extern "C" int64_t lx_bf16x_row_bytes(int K) { return K > 0 && K < fx::kMaxK ? fx::row_bytes(K) : -1; }

extern "C" int64_t lx_bf16x_bytes(const void* w, int64_t rows, int K, void* stream) {
  if (rows <= 0 || K <= 0 || K >= fx::kMaxK || !w) return -static_cast<int64_t>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  uint32_t* counts = nullptr;
  int64_t* scratch = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&counts), rows * sizeof(uint32_t), s);
  if (err == cudaSuccess) err = cudaMallocAsync(reinterpret_cast<void**>(&scratch), (rows + 1) * sizeof(int64_t), s);
  if (err != cudaSuccess) return -static_cast<int64_t>(err);
  const auto grid = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  encode_bases<<<grid, kThreads, 0, s>>>(static_cast<const uint16_t*>(w), rows, K, nullptr, counts);
  encode_offsets<<<1, 1024, 0, s>>>(counts, rows, fx::patch_slots(K), scratch, scratch + rows);
  int64_t overflow = 0;
  cudaMemcpyAsync(&overflow, scratch + rows, sizeof(int64_t), cudaMemcpyDeviceToHost, s);
  cudaFreeAsync(counts, s);
  cudaFreeAsync(scratch, s);
  err = cudaStreamSynchronize(s);
  if (err == cudaSuccess) err = cudaGetLastError();
  if (err != cudaSuccess) return -static_cast<int64_t>(err);
  return (rows * fx::row_bytes(K) + 7) / 8 * 8 + overflow;
}

extern "C" int lx_bf16x_encode(const void* w, int64_t rows, int K, void* out, int64_t out_bytes, void* stream) {
  if (rows <= 0) return 0;
  if (K <= 0 || K >= fx::kMaxK || !w || !out || (reinterpret_cast<uintptr_t>(out) & 7) != 0 ||
      out_bytes < rows * fx::row_bytes(K))
    return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  uint32_t* counts = nullptr;
  int64_t* offsets = nullptr;
  cudaError_t err = cudaMallocAsync(reinterpret_cast<void**>(&counts), rows * sizeof(uint32_t), s);
  if (err == cudaSuccess) err = cudaMallocAsync(reinterpret_cast<void**>(&offsets), (rows + 1) * sizeof(int64_t), s);
  if (err != cudaSuccess) return static_cast<int>(err);
  const auto grid = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  auto* o = static_cast<uint8_t*>(out);
  const auto* src = static_cast<const uint16_t*>(w);
  encode_bases<<<grid, kThreads, 0, s>>>(src, rows, K, o, counts);
  encode_offsets<<<1, 1024, 0, s>>>(counts, rows, fx::patch_slots(K), offsets, offsets + rows);
  int64_t overflow = 0;
  cudaMemcpyAsync(&overflow, offsets + rows, sizeof(int64_t), cudaMemcpyDeviceToHost, s);
  err = cudaStreamSynchronize(s);
  if (err == cudaSuccess && (rows * fx::row_bytes(K) + 7) / 8 * 8 + overflow > out_bytes) err = cudaErrorInvalidValue;
  if (err == cudaSuccess) {
    encode_rows<<<grid, kThreads, 0, s>>>(src, rows, K, o, counts, offsets);
    err = cudaGetLastError();
  }
  cudaFreeAsync(counts, s);
  cudaFreeAsync(offsets, s);
  return static_cast<int>(err);
}

extern "C" int lx_bf16x_decode(const void* data, int64_t rows, int K, void* w, void* stream) {
  if (rows <= 0) return 0;
  if (K <= 0 || K >= fx::kMaxK || !data || !w) return static_cast<int>(cudaErrorInvalidValue);
  const auto grid = static_cast<unsigned>((rows + kWarps - 1) / kWarps);
  decode_rows<<<grid, kThreads, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint8_t*>(data), rows, K,
                                                                         static_cast<uint16_t*>(w));
  return lx::launch_status();
}

namespace {
template <bool kX>
int gemv_entry(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias, lx_dtype bias_dtype,
               void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta, void* stream) {
  if (M == 0 || N == 0) return 0;
  if (M < 0 || M > kMaxM || K <= 0 || K >= fx::kMaxK || ld_x < K || ld_y < N)
    return static_cast<int>(cudaErrorInvalidValue);
  if (!valid_dtype(x_dtype) || !valid_dtype(y_dtype) || !valid_dtype(bias_dtype))
    return static_cast<int>(cudaErrorInvalidValue);
  if (!x || !w || !y || (kX && (reinterpret_cast<uintptr_t>(w) & 7) != 0)) return static_cast<int>(cudaErrorInvalidValue);
  const Args a{
      .x = x,
      .ld_x = ld_x,
      .w = static_cast<const uint8_t*>(w),
      .bias = bias,
      .bias_bf16 = bias_dtype == LX_BF16,
      .y = y,
      .y_bf16 = y_dtype == LX_BF16,
      .ld_y = ld_y,
      .M = M,
      .N = N,
      .K = K,
      .alpha = alpha,
      .beta = beta,
      .stream = static_cast<cudaStream_t>(stream),
  };
  const cudaError_t e = x_dtype == LX_BF16 ? launch_gemv<kX, bf16>(a) : launch_gemv<kX, float>(a);
  return static_cast<int>(e);
}
}  // namespace

extern "C" int lx_gemv_bf16x(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias,
                             lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                             float beta, void* stream) {
  return gemv_entry<true>(x, x_dtype, ld_x, w, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
}

extern "C" int lx_gemv_decode_bf16(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias,
                                   lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K,
                                   float alpha, float beta, void* stream) {
  return gemv_entry<false>(x, x_dtype, ld_x, w, bias, bias_dtype, y, y_dtype, ld_y, M, N, K, alpha, beta, stream);
}

namespace {
int check_experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up, const void* down,
                  float* out, void* workspace, int T, int L, int I, int top_k) {
  if (L <= 0 || I <= 0 || top_k < 1 || top_k > 8 || !latent || !ids || !weights || !gate_up || !down || !out ||
      !workspace || (reinterpret_cast<uintptr_t>(workspace) & 15) != 0 || (reinterpret_cast<uintptr_t>(latent) & 15) != 0 ||
      (reinterpret_cast<uintptr_t>(gate_up) & 15) != 0 || (reinterpret_cast<uintptr_t>(down) & 15) != 0 ||
      L >= fx::kMaxK || I >= fx::kMaxK)
    return static_cast<int>(cudaErrorInvalidValue);
  (void)T;
  return 0;
}
}  // namespace

extern "C" int lx_moe_experts_bf16x(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                                    const void* down, float* out, void* workspace, int T, int L, int I, int top_k,
                                    void* stream) {
  if (T <= 0) return 0;
  if (const int bad = check_experts(latent, ids, weights, gate_up, down, out, workspace, T, L, I, top_k)) return bad;
  return static_cast<int>(experts<true>(latent, ids, weights, gate_up, down, out, workspace, T, L, I, top_k,
                                        static_cast<cudaStream_t>(stream)));
}

extern "C" int lx_moe_experts_decode_bf16(const void* latent, const int32_t* ids, const float* weights,
                                          const void* gate_up, const void* down, float* out, void* workspace, int T,
                                          int L, int I, int top_k, void* stream) {
  if (T <= 0) return 0;
  if (const int bad = check_experts(latent, ids, weights, gate_up, down, out, workspace, T, L, I, top_k)) return bad;
  return static_cast<int>(experts<false>(latent, ids, weights, gate_up, down, out, workspace, T, L, I, top_k,
                                         static_cast<cudaStream_t>(stream)));
}

extern "C" int64_t lx_moe_bf16x_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  return (2 * static_cast<int64_t>(T) * top_k * I + 255) / 256 * 256;
}
