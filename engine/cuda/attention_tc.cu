// Causal GQA over the new tokens of long sequences, on the tensor cores: the prefill half of
// lx_attention_varlen (attention.cu runs it beside the SIMT kernel, which keeps the short sequences).
//
// FlashAttention-2 on mma.sync. A CTA takes kRows (token, head) rows: BQ = kRows / G consecutive new
// tokens of one sequence times the G = H / KVH query heads of one kv head, 16 rows per warp, so every K
// and V tile staged from the cache serves all of them. Scores and the output accumulate in f32 and the
// online softmax runs in the log2 domain (scale * log2(e) folded into q). q enters the MMA as two bf16
// halves, hi and the rounding error of hi, so the scores keep about 16 bits of q, close to the SIMT
// kernel's f32 q (k is bf16 in the cache either way). The probabilities enter P V as bf16, as in
// FlashAttention; the softmax denominator sums those same rounded values, so each output is an exact
// convex combination of the values.
//
// Persistent CTAs walk the (sequence, q block, kv head) items, latest q blocks first (they see the most
// keys), counting the items from cu_seqlens on the device, since the C ABI carries no token count.
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <climits>

#include "cuda/attention_tc.h"
#include "cuda/common.cuh"
#include "cuda/mma.cuh"

namespace {

using bf16 = __nv_bfloat16;
namespace mma = lx::mma;

constexpr int kThreads = 256;  // 8 warps
constexpr int kRows = 128;     // (token, head) rows per CTA, 16 per warp
constexpr int kBK = 64;        // keys per tile
constexpr int kMinTokens = 16; // sequences with fewer new tokens stay on the SIMT kernel

template <int Dh>
struct Layout {
  static constexpr int kTile = kBK * Dh * 2;  // bytes of one K or V tile
  static constexpr int kStage = 2 * kTile;    // K then V
  static constexpr int kSmem = 2 * kStage;    // two stages
};

// Byte offset of 16-byte chunk c of key row r: the chunk index is XORed with the row's low three bits,
// so the eight rows an ldmatrix reads at one logical chunk fall in eight different bank groups.
template <int Dh>
__device__ __forceinline__ uint32_t toff(int r, int c) {
  return static_cast<uint32_t>(r * (Dh * 2) + ((c ^ (r & 7)) << 4));
}

__device__ __forceinline__ float ex2(float x) {
  float y;
  asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
  return y;
}

__device__ __forceinline__ float bf16_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

// Items of sequence s: its q blocks times the kv heads, or none when it is short.
__device__ __forceinline__ int items_of(const int32_t* cu, int s, int BQ, int KVH) {
  const int n = cu[s + 1] - cu[s];
  return n >= kMinTokens ? (n + BQ - 1) / BQ * KVH : 0;
}

template <int Dh>
__global__ void __launch_bounds__(kThreads, 1)
    attention_tc_kernel(const float* __restrict__ qkv, int ld_qkv, const bf16* __restrict__ k_cache,
                        const bf16* __restrict__ v_cache, const int32_t* __restrict__ slot_ids,
                        const int32_t* __restrict__ start_pos, const int32_t* __restrict__ cu, int n_seq,
                        int capacity, bf16* __restrict__ out, int H, int KVH, float scale_log2) {
  constexpr int kKS = Dh / 16;  // k16 steps over the head dimension
  constexpr int kND = Dh / 8;   // n8 tiles of the output
  constexpr int kChunks = Dh / 8;
  using LY = Layout<Dh>;
  extern __shared__ __align__(128) uint8_t smem[];
  const uint32_t s0 = mma::smem_addr(smem);
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int fr = lane >> 2, fq = lane & 3;
  const int b_r = (lane & 7) + (lane >> 4) * 8, b_c = (lane >> 3) & 1;         // K: ldmatrix lanes
  const int v_r = (lane & 7) + ((lane >> 3) & 1) * 8, v_c = lane >> 4;         // V: ldmatrix.trans lanes
  const int G = H / KVH, BQ = kRows / G;

  int total = 0;
  for (int s = 0; s < n_seq; ++s) total += items_of(cu, s, BQ, KVH);

  for (int item = blockIdx.x; item < total; item += gridDim.x) {
    // ---- the item: sequence s, q block b (latest first), kv head g ----
    int s = 0, rest = item;
    for (; s < n_seq; ++s) {
      const int k = items_of(cu, s, BQ, KVH);
      if (rest < k) break;
      rest -= k;
    }
    const int n = cu[s + 1] - cu[s];
    const int nb = (n + BQ - 1) / BQ;
    const int b = nb - 1 - rest / KVH, g = rest % KVH;
    const int j0 = b * BQ;
    const int nq = min(BQ, n - j0);
    const int t0 = cu[s] + j0;
    const int p0 = start_pos[s] + j0;  // cache position of the block's first token
    const int L = p0 + nq;             // keys 0 .. L - 1 are visible to its last token
    const int n_tiles = (L + kBK - 1) / kBK;
    const int64_t kv_stride = static_cast<int64_t>(KVH) * Dh;
    const int64_t kv_base = (static_cast<int64_t>(slot_ids[s]) * capacity * KVH + g) * Dh;

    // ---- this thread's two rows (fr and fr + 8 of the warp's 16) and their q fragments ----
    int tq[2], hq[2], prow[2];
    bool live[2];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const int r = warp * 16 + fr + 8 * i;
      tq[i] = r / G;
      hq[i] = r % G;
      live[i] = tq[i] < nq;
      prow[i] = p0 + min(tq[i], nq - 1);  // a dead row computes as the last live token and is not written
    }
    uint32_t qh[kKS][4], ql[kKS][4];
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      const float* q = qkv + static_cast<int64_t>(t0 + min(tq[i], nq - 1)) * ld_qkv + (g * G + hq[i]) * Dh;
#pragma unroll
      for (int kk = 0; kk < kKS; ++kk)
#pragma unroll
        for (int j = 0; j < 2; ++j) {
          float2 v = make_float2(0.f, 0.f);  // q rows may sit at any float offset of a wider buffer
          if (live[i]) v = make_float2(q[16 * kk + 8 * j + 2 * fq], q[16 * kk + 8 * j + 2 * fq + 1]);
          v.x *= scale_log2;
          v.y *= scale_log2;
          const uint32_t hi = mma::pack_bf16(v.x, v.y);
          qh[kk][i + 2 * j] = hi;
          ql[kk][i + 2 * j] = mma::pack_bf16(v.x - bf16_lo(hi), v.y - bf16_hi(hi));
        }
    }

    float o[kND][4], m[2] = {-INFINITY, -INFINITY}, l[2] = {0.f, 0.f};
#pragma unroll
    for (int d = 0; d < kND; ++d)
#pragma unroll
      for (int e = 0; e < 4; ++e) o[d][e] = 0.f;

    // Keys at or past L are zero-filled: masked out, and zero values keep 0 * garbage out of P V.
    auto stage = [&](int kt, int st) {
      const uint32_t sk = s0 + st * LY::kStage, sv = sk + LY::kTile;
      for (int c = tid; c < kBK * kChunks; c += kThreads) {
        const int r = c / kChunks, ch = c % kChunks;
        const int p = kt * kBK + r;
        const bool ok = p < L;
        const int64_t at = kv_base + (ok ? p : 0) * kv_stride + ch * 8;
        mma::cp_async16(sk + toff<Dh>(r, ch), k_cache + at, ok);
        mma::cp_async16(sv + toff<Dh>(r, ch), v_cache + at, ok);
      }
    };

    stage(0, 0);
    mma::cp_async_commit();
    for (int kt = 0; kt < n_tiles; ++kt) {
      if (kt + 1 < n_tiles) stage(kt + 1, (kt + 1) & 1);
      mma::cp_async_commit();
      mma::cp_async_wait<1>();
      __syncthreads();
      const uint32_t sk = s0 + (kt & 1) * LY::kStage, sv = sk + LY::kTile;

      // ---- S = q k^T over the tile's 64 keys (8 n8 tiles), q in two halves ----
      float sc[8][4];
#pragma unroll
      for (int nn = 0; nn < 8; ++nn)
#pragma unroll
        for (int e = 0; e < 4; ++e) sc[nn][e] = 0.f;
#pragma unroll
      for (int kk = 0; kk < kKS; ++kk)
#pragma unroll
        for (int j = 0; j < 4; ++j) {
          uint32_t kb[4];
          mma::ldmatrix_x4(kb, sk + toff<Dh>(j * 16 + b_r, 2 * kk + b_c));
          mma::mma_bf16(sc[2 * j], qh[kk], kb[0], kb[1]);
          mma::mma_bf16(sc[2 * j + 1], qh[kk], kb[2], kb[3]);
          mma::mma_bf16(sc[2 * j], ql[kk], kb[0], kb[1]);
          mma::mma_bf16(sc[2 * j + 1], ql[kk], kb[2], kb[3]);
        }

      // ---- the causal mask, on tiles that reach past the block's first position ----
      const int kb0 = kt * kBK;
      if (kb0 + kBK - 1 > p0) {
#pragma unroll
        for (int nn = 0; nn < 8; ++nn)
#pragma unroll
          for (int e = 0; e < 4; ++e)
            if (kb0 + nn * 8 + 2 * fq + (e & 1) > prow[e >> 1]) sc[nn][e] = -INFINITY;
      }

      // ---- online softmax; P rounded to bf16 as the A operand of P V ----
      uint32_t pa[4][4];
#pragma unroll
      for (int i = 0; i < 2; ++i) {
        float mx = -INFINITY;
#pragma unroll
        for (int nn = 0; nn < 8; ++nn) mx = fmaxf(mx, fmaxf(sc[nn][2 * i], sc[nn][2 * i + 1]));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
        mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
        const float mn = fmaxf(m[i], mx);  // finite: key 0 is visible to every row
        const float corr = ex2(m[i] - mn);
        m[i] = mn;
        l[i] *= corr;
#pragma unroll
        for (int d = 0; d < kND; ++d) {
          o[d][2 * i] *= corr;
          o[d][2 * i + 1] *= corr;
        }
#pragma unroll
        for (int nn = 0; nn < 8; ++nn) {
          const uint32_t p = mma::pack_bf16(ex2(sc[nn][2 * i] - mn), ex2(sc[nn][2 * i + 1] - mn));
          l[i] += bf16_lo(p) + bf16_hi(p);
          // C layout of n8 tiles 2c, 2c + 1 is the A layout of k16 step c
          pa[nn >> 1][i + 2 * (nn & 1)] = p;
        }
      }

      // ---- O += P V ----
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int jd = 0; jd < kND / 2; ++jd) {
          uint32_t vb[4];
          mma::ldmatrix_x4_trans(vb, sv + toff<Dh>(c * 16 + v_r, 2 * jd + v_c));
          mma::mma_bf16(o[2 * jd], pa[c], vb[0], vb[1]);
          mma::mma_bf16(o[2 * jd + 1], pa[c], vb[2], vb[3]);
        }
      __syncthreads();  // every warp is done with this stage before the next issue overwrites it
    }
    mma::cp_async_wait<0>();

    // ---- out = O / l ----
#pragma unroll
    for (int i = 0; i < 2; ++i) {
      float li = l[i];
      li += __shfl_xor_sync(0xffffffffu, li, 1);
      li += __shfl_xor_sync(0xffffffffu, li, 2);
      if (!live[i]) continue;
      const float inv = 1.0f / li;
      bf16* dst = out + static_cast<int64_t>(t0 + tq[i]) * H * Dh + (g * G + hq[i]) * Dh + 2 * fq;
#pragma unroll
      for (int d = 0; d < kND; ++d)
        *reinterpret_cast<uint32_t*>(dst + 8 * d) = mma::pack_bf16(o[d][2 * i] * inv, o[d][2 * i + 1] * inv);
    }
  }
}

template <int Dh>
int launch(const float* qkv, int ld_qkv, const bf16* k_cache, const bf16* v_cache, const int32_t* slot_ids,
           const int32_t* start_pos, const int32_t* cu, int n_seq, int capacity, bf16* out, int H, int KVH, float scale,
           cudaStream_t s) {
  static const int grid = [] {
    int dev = 0, sms = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    return 2 * sms;
  }();
  constexpr int smem = Layout<Dh>::kSmem;
  if constexpr (smem > 48 * 1024) {
    static const cudaError_t attr =
        cudaFuncSetAttribute(attention_tc_kernel<Dh>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    if (attr != cudaSuccess) return static_cast<int>(attr);
  }
  attention_tc_kernel<Dh><<<grid, kThreads, smem, s>>>(qkv, ld_qkv, k_cache, v_cache, slot_ids, start_pos, cu, n_seq,
                                                       capacity, out, H, KVH, scale * 1.4426950408889634f);
  return lx::launch_status();
}

}  // namespace

namespace lx {

int attention_tc_min_tokens(int H, int KVH, int Dh) {
  const int G = KVH > 0 ? H / KVH : 0;
  const bool ok = (Dh == 64 || Dh == 128) && G > 0 && G <= 16 && kRows % G == 0;
  return ok ? kMinTokens : INT_MAX;
}

int attention_tc(const float* qkv, int ld_qkv, const void* k_cache, const void* v_cache, const int32_t* slot_ids,
                 const int32_t* start_pos, const int32_t* cu_seqlens, int n_seq, int capacity, void* out, int H, int KVH,
                 int Dh, float scale, cudaStream_t s) {
  if (attention_tc_min_tokens(H, KVH, Dh) == INT_MAX) return 0;
  auto kc = static_cast<const bf16*>(k_cache);
  auto vc = static_cast<const bf16*>(v_cache);
  auto o = static_cast<bf16*>(out);
  if (Dh == 64) return launch<64>(qkv, ld_qkv, kc, vc, slot_ids, start_pos, cu_seqlens, n_seq, capacity, o, H, KVH, scale, s);
  return launch<128>(qkv, ld_qkv, kc, vc, slot_ids, start_pos, cu_seqlens, n_seq, capacity, o, H, KVH, scale, s);
}

}  // namespace lx
