// Causal grouped-query attention with no positional encoding, over varlen sequences.
//
// write_kv_kernel rounds this step's k and v to bf16 into the cache before attention_kernel runs,
// so prefill and decode read the same values. attention_kernel takes one (token, kv head) at a
// time and serves all G = H / KVH query heads of that kv head from one pass over the cache, so
// each cache row is read once per group. Cache tiles are staged through shared memory by a
// cp.async pipeline. Inside a tile, each group of 8 lanes owns one key and each lane 8 of its
// dims (one 16-byte load), so a dot product is a 3-step shuffle. Every group keeps its own
// online-softmax state, merged in registers within a warp and through shared memory across warps.
//
// Few (token, kv head) items, as in decode: the 8 blocks of a cluster share each item. Block r
// takes cache tiles r, r + 8, ... and the 8 partial softmax states are merged through distributed
// shared memory, so a single decode token runs on 8 SMs per kv head instead of one. Many items, as
// in prefill: each block takes whole items and the cluster never synchronizes. The choice depends
// on T, which is device data, so every block of a cluster makes the same one.
//
// The C ABI has no token count, so both kernels read T = cu_seqlens[n_seq] on the device and
// grid-stride over the work. Launch sizes are constants; the grid is a multiple of the cluster size.
//
// Sequences with many new tokens (a prefill) go to the tensor-core kernel in attention_tc.cu instead,
// which runs beside this one; this kernel skips their tokens.
#include <cooperative_groups.h>
#include <cuda_pipeline.h>

#include "cuda/attention_tc.h"
#include "cuda/common.cuh"

namespace {

namespace cg = cooperative_groups;

constexpr int kThreads = 256;                              // 8 warps
constexpr int kWarps = kThreads / 32;
constexpr int kLanesPerKey = 8;                            // lanes sharing one key, 8 dims each
constexpr int kKeysPerTileStep = kThreads / kLanesPerKey;  // keys handled per step of a tile
constexpr int kSplit = 8;                                  // blocks per item in split mode; the cluster size
constexpr int kSplitBelow = 64;                            // fewer items than this: split each item
constexpr int kTile = 4;                                   // query heads per pass over the cache
constexpr int kBlockN = 64;                                // cache positions per shared-memory tile
constexpr int kStages = 3;                                 // cp.async pipeline depth
constexpr int kWriteBlocks = 256;
constexpr int kAttnBlocks = 1024;                          // multiple of kSplit

static_assert(kBlockN % kKeysPerTileStep == 0, "every group takes the same number of keys per tile");
static_assert(kAttnBlocks % kSplit == 0, "clusters must tile the grid");

// Sequence of token t: the last one whose first token is <= t. Empty sequences share their start
// with the next sequence, so the search lands on the one that actually holds t.
__device__ __forceinline__ int seq_of_token(const int32_t* cu, int n_seq, int t) {
  int lo = 1, hi = n_seq;
  while (lo < hi) {
    const int mid = (lo + hi) >> 1;
    if (cu[mid] > t) hi = mid;
    else lo = mid + 1;
  }
  return lo - 1;
}

// Eight bf16 values, 16 bytes, to float. Zeros when `on` is false (chunk past the row or the tile).
__device__ __forceinline__ void load8(const __nv_bfloat16* p, bool on, float* f) {
  if (!on) {
    for (int e = 0; e < 8; ++e) f[e] = 0.f;
    return;
  }
  const uint4 raw = *reinterpret_cast<const uint4*>(p);
  const __nv_bfloat162* h = reinterpret_cast<const __nv_bfloat162*>(&raw);
  for (int i = 0; i < 4; ++i) {
    const float2 v = __bfloat1622float2(h[i]);
    f[2 * i] = v.x;
    f[2 * i + 1] = v.y;
  }
}

// Round this step's k and v into the cache rows start_pos[s] + j of each sequence's slot.
__global__ void write_kv_kernel(const float* __restrict__ qkv, int ld_qkv,
                                __nv_bfloat16* __restrict__ k_cache, __nv_bfloat16* __restrict__ v_cache,
                                const int32_t* __restrict__ slot_ids, const int32_t* __restrict__ start_pos,
                                const int32_t* __restrict__ cu_seqlens, int n_seq, int capacity, int H,
                                int KVH, int Dh) {
  const int T = cu_seqlens[n_seq];
  const int width = KVH * Dh;
  for (int t = blockIdx.x; t < T; t += gridDim.x) {
    const int s = seq_of_token(cu_seqlens, n_seq, t);
    const int64_t pos = start_pos[s] + (t - cu_seqlens[s]);
    const int64_t row = (static_cast<int64_t>(slot_ids[s]) * capacity + pos) * width;
    const float* k = qkv + static_cast<int64_t>(t) * ld_qkv + H * Dh;
    const float* v = k + width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
      k_cache[row + i] = lx::from_f32<__nv_bfloat16>(k[i]);
      v_cache[row + i] = lx::from_f32<__nv_bfloat16>(v[i]);
    }
  }
}

// NPL is the number of 8-dim chunks each lane owns: ceil(Dh / 64). The cluster size in the
// attribute below is kSplit; split mode shares each item across the cluster (see the file comment).
template <int NPL>
__global__ void __cluster_dims__(8, 1, 1) __launch_bounds__(kThreads) attention_kernel(
    const float* __restrict__ qkv, int ld_qkv, const __nv_bfloat16* __restrict__ k_cache,
    const __nv_bfloat16* __restrict__ v_cache, const int32_t* __restrict__ slot_ids,
    const int32_t* __restrict__ start_pos, const int32_t* __restrict__ cu_seqlens, int n_seq, int capacity,
    __nv_bfloat16* __restrict__ out, int H, int KVH, int Dh, float scale, int tc_min) {
  extern __shared__ __align__(16) unsigned char smem_raw[];
  __nv_bfloat16* sK = reinterpret_cast<__nv_bfloat16*>(smem_raw);  // [stage][kBlockN][Dh]
  __nv_bfloat16* sV = sK + kStages * kBlockN * Dh;                 // [stage][kBlockN][Dh]
  // After the tile loop, the same bytes hold each warp's state for the merge within the block...
  float* sAcc = reinterpret_cast<float*>(smem_raw);  // [kWarps][kTile][Dh]
  float* sM = sAcc + kWarps * kTile * Dh;            // [kWarps][kTile]
  float* sL = sM + kWarps * kTile;
  // ...and the block's merged state, which the other blocks of the cluster read in split mode.
  float* sBlk = sL + kWarps * kTile;  // [kTile][Dh]
  float* sBlkM = sBlk + kTile * Dh;   // [kTile]
  float* sBlkL = sBlkM + kTile;       // [kTile]

  const cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int tid = threadIdx.x;
  const int lane = tid & 31;
  const int warp = tid >> 5;
  const int chunk = tid & (kLanesPerKey - 1);  // this lane's 8-dim chunk within a key
  const int key = tid / kLanesPerKey;          // this lane's key group within the block
  const int G = H / KVH;
  const int NCH = Dh / 8;                      // 8-dim chunks per row
  const int T = cu_seqlens[n_seq];

  // Split mode: the cluster's blocks share each item; block `rank` takes tiles rank, rank + S, ...
  // Otherwise each block takes whole items and tiles 0, 1, ... Uniform across the cluster.
  const bool split = T * KVH < kSplitBelow;
  const int S = split ? kSplit : 1;
  const int r0 = split ? rank : 0;
  const int first = split ? blockIdx.x / kSplit : blockIdx.x;
  const int stride = split ? gridDim.x / kSplit : gridDim.x;

  for (int item = first; item < T * KVH; item += stride) {
    const int t = item / KVH, kvh = item % KVH;
    const int s = seq_of_token(cu_seqlens, n_seq, t);
    if (cu_seqlens[s + 1] - cu_seqlens[s] >= tc_min) continue;  // the tensor-core kernel's (uniform per item)
    const int64_t slot = slot_ids[s];
    const int L = start_pos[s] + (t - cu_seqlens[s]) + 1;  // visible cache positions: 0 .. L - 1
    const int n_tiles = (L + kBlockN - 1) / kBlockN;
    const int n_local = n_tiles > r0 ? (n_tiles - r0 + S - 1) / S : 0;
    const float* q_row = qkv + static_cast<int64_t>(t) * ld_qkv;

    for (int ht = 0; ht < G; ht += kTile) {
      const int nh = min(kTile, G - ht);  // query heads in this pass (block-uniform)
      const int hq0 = kvh * G + ht;       // query head of the pass's first slot

      // Per lane: its chunks of each head's scaled q, the output accumulator, and the group's
      // online-softmax state (running max m, running sum l). Identical across a key group.
      float qv[kTile][NPL][8], acc[kTile][NPL][8], m[kTile], l[kTile];
#pragma unroll
      for (int g = 0; g < kTile; ++g) {
        m[g] = -INFINITY;
        l[g] = 0.f;
#pragma unroll
        for (int k = 0; k < NPL; ++k) {
          const int c = chunk + kLanesPerKey * k;
#pragma unroll
          for (int e = 0; e < 8; ++e) {
            acc[g][k][e] = 0.f;
            // Scale folded into q: f32, not rounded (the reference keeps q in f32 too).
            qv[g][k][e] = (g < nh && c < NCH) ? q_row[(hq0 + g) * Dh + c * 8 + e] * scale : 0.f;
          }
        }
      }

      // Copies this block's local tile kk (global tile r0 + kk * S) of K and V into its stage,
      // 16 bytes at a time. Rows past L are skipped: they are never read.
      auto stage = [&](int kk) {
        const int buf = kk % kStages;
        const int tile = r0 + kk * S;
        for (int c = tid; c < kBlockN * NCH; c += kThreads) {
          const int r = c / NCH, ch = c % NCH;
          const int p = tile * kBlockN + r;
          if (p < L) {
            const int64_t src = ((slot * capacity + p) * KVH + kvh) * Dh + ch * 8;
            __pipeline_memcpy_async(sK + (buf * kBlockN + r) * Dh + ch * 8, k_cache + src, 16);
            __pipeline_memcpy_async(sV + (buf * kBlockN + r) * Dh + ch * 8, v_cache + src, 16);
          }
        }
      };

      // Prefetch kStages - 1 tiles, then each step issues the next tile before waiting on the
      // current one, so the cache loads overlap the math. Every commit is issued, even empty, so
      // wait_prior counts the same groups on every step.
#pragma unroll
      for (int i = 0; i < kStages - 1; ++i) {
        if (i < n_local) stage(i);
        __pipeline_commit();
      }
      for (int kk = 0; kk < n_local; ++kk) {
        // The stage this writes was read by the previous step, which ended with a barrier.
        if (kk + kStages - 1 < n_local) stage(kk + kStages - 1);
        __pipeline_commit();
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const int tile = r0 + kk * S;
        const __nv_bfloat16* kb = sK + (kk % kStages) * kBlockN * Dh;
        const __nv_bfloat16* vb = sV + (kk % kStages) * kBlockN * Dh;
#pragma unroll
        for (int i = 0; i < kBlockN / kKeysPerTileStep; ++i) {
          const int r = key + i * kKeysPerTileStep;  // this group's key within the tile
          const bool valid = tile * kBlockN + r < L;
          float kf[NPL][8], vf[NPL][8];
#pragma unroll
          for (int k = 0; k < NPL; ++k) {
            const int c = chunk + kLanesPerKey * k;
            load8(kb + r * Dh + c * 8, valid && c < NCH, kf[k]);
            load8(vb + r * Dh + c * 8, valid && c < NCH, vf[k]);
          }

          // Partial dot products for each head, then the key's full dot over its 8 lanes. Every
          // lane of the warp takes part, so the shuffles never diverge.
          float dot[kTile];
#pragma unroll
          for (int g = 0; g < kTile; ++g) {
            float d = 0.f;
#pragma unroll
            for (int k = 0; k < NPL; ++k)
#pragma unroll
              for (int e = 0; e < 8; ++e) d += qv[g][k][e] * kf[k][e];
            dot[g] = d;
          }
#pragma unroll
          for (int g = 0; g < kTile; ++g)
#pragma unroll
            for (int off = kLanesPerKey / 2; off > 0; off >>= 1)
              dot[g] += __shfl_xor_sync(0xffffffffu, dot[g], off);

          if (valid) {
#pragma unroll
            for (int g = 0; g < kTile; ++g) {
              if (g < nh) {
                // Online softmax. With m = -inf the first key's correction is exp(-inf) = 0.
                const float mn = fmaxf(m[g], dot[g]);
                const float corr = __expf(m[g] - mn);
                const float w = __expf(dot[g] - mn);
                l[g] = l[g] * corr + w;
#pragma unroll
                for (int k = 0; k < NPL; ++k)
#pragma unroll
                  for (int e = 0; e < 8; ++e) acc[g][k][e] = acc[g][k][e] * corr + w * vf[k][e];
                m[g] = mn;
              }
            }
          }
        }
        __syncthreads();
      }
      __pipeline_wait_prior(0);

      // Merge the four key groups of each warp. Lanes that differ only in bits 3 and 4 hold the
      // same chunk, so the butterfly is a plain xor exchange. A group with no keys has m = -inf
      // and weight 0, which is why the weights are guarded.
#pragma unroll
      for (int g = 0; g < kTile; ++g) {
        if (g < nh) {
#pragma unroll
          for (int off = kLanesPerKey; off < 32; off <<= 1) {
            const float m2 = __shfl_xor_sync(0xffffffffu, m[g], off);
            const float l2 = __shfl_xor_sync(0xffffffffu, l[g], off);
            float a2[NPL][8];
#pragma unroll
            for (int k = 0; k < NPL; ++k)
#pragma unroll
              for (int e = 0; e < 8; ++e) a2[k][e] = __shfl_xor_sync(0xffffffffu, acc[g][k][e], off);
            const float mn = fmaxf(m[g], m2);
            const float w1 = m[g] == -INFINITY ? 0.f : __expf(m[g] - mn);
            const float w2 = m2 == -INFINITY ? 0.f : __expf(m2 - mn);
            l[g] = l[g] * w1 + l2 * w2;
#pragma unroll
            for (int k = 0; k < NPL; ++k)
#pragma unroll
              for (int e = 0; e < 8; ++e) acc[g][k][e] = acc[g][k][e] * w1 + a2[k][e] * w2;
            m[g] = mn;
          }
        }
      }
      if (lane < kLanesPerKey) {  // lanes 0..7 of each warp hold its merged state
#pragma unroll
        for (int g = 0; g < kTile; ++g) {
          if (g < nh) {
#pragma unroll
            for (int k = 0; k < NPL; ++k) {
              const int c = chunk + kLanesPerKey * k;
              if (c < NCH)
#pragma unroll
                for (int e = 0; e < 8; ++e) sAcc[((warp * kTile) + g) * Dh + c * 8 + e] = acc[g][k][e];
            }
            if (chunk == 0) {
              sM[warp * kTile + g] = m[g];
              sL[warp * kTile + g] = l[g];
            }
          }
        }
      }
      __syncthreads();

      // Merge the warps. In split mode the block's state goes to shared memory for the cluster;
      // otherwise the output is final. M is -inf for a block that had no tiles in split mode, and
      // then every weight is 0.
      for (int idx = tid; idx < nh * Dh; idx += kThreads) {
        const int g = idx / Dh, d = idx % Dh;
        float M = -INFINITY;
        for (int w = 0; w < kWarps; ++w) M = fmaxf(M, sM[w * kTile + g]);
        float Z = 0.f, A = 0.f;
        if (M != -INFINITY) {
          for (int w = 0; w < kWarps; ++w) {
            const float mw = sM[w * kTile + g];
            const float wt = mw == -INFINITY ? 0.f : __expf(mw - M);
            Z += sL[w * kTile + g] * wt;
            A += sAcc[(w * kTile + g) * Dh + d] * wt;
          }
        }
        if (split) {
          sBlk[g * Dh + d] = A;
          if (d == 0) {
            sBlkM[g] = M;
            sBlkL[g] = Z;
          }
        } else {
          out[static_cast<int64_t>(t) * H * Dh + (hq0 + g) * Dh + d] = lx::from_f32<__nv_bfloat16>(A / Z);
        }
      }

      if (split) {
        cluster.sync();  // every block's state is in its shared memory
        // Cluster merge: block `rank` writes a strided share of the (head, dim) outputs, reading
        // every block's state across the cluster.
        for (int idx = rank * kThreads + tid; idx < nh * Dh; idx += kSplit * kThreads) {
          const int g = idx / Dh, d = idx % Dh;
          float M = -INFINITY;
          for (int r2 = 0; r2 < kSplit; ++r2) M = fmaxf(M, cluster.map_shared_rank(sBlkM, r2)[g]);
          float Z = 0.f, A = 0.f;
          for (int r2 = 0; r2 < kSplit; ++r2) {
            const float mr = cluster.map_shared_rank(sBlkM, r2)[g];
            if (mr == -INFINITY) continue;  // this block had no tiles of the item
            const float wt = __expf(mr - M);
            Z += cluster.map_shared_rank(sBlkL, r2)[g] * wt;
            A += cluster.map_shared_rank(sBlk, r2)[g * Dh + d] * wt;
          }
          out[static_cast<int64_t>(t) * H * Dh + (hq0 + g) * Dh + d] = lx::from_f32<__nv_bfloat16>(A / Z);
        }
        cluster.sync();  // nobody leaves, or restages its stages, while another block still reads them
      } else {
        __syncthreads();  // the next pass stages into the same shared memory
      }
    }
  }
}

}  // namespace

extern "C" int lx_attention_varlen(const float* qkv, int ld_qkv, void* k_cache, void* v_cache,
                                   const int32_t* slot_ids, const int32_t* start_pos, const int32_t* cu_seqlens,
                                   int n_seq, int capacity, void* out, int H, int KVH, int Dh, float scale,
                                   void* stream) {
  // Rows of the cache are read as 16-byte chunks, and a row fits the 8-lane key layout at up to
  // 128 dims. Anything else is a bad call, not a case to round around.
  if (n_seq <= 0 || capacity <= 0 || KVH <= 0 || H <= 0 || H % KVH != 0 || Dh <= 0 || Dh % 8 != 0 ||
      Dh > 128 || ld_qkv < (H + 2 * KVH) * Dh)
    return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  auto kc = static_cast<__nv_bfloat16*>(k_cache);
  auto vc = static_cast<__nv_bfloat16*>(v_cache);
  auto o = static_cast<__nv_bfloat16*>(out);
  write_kv_kernel<<<kWriteBlocks, kThreads, 0, s>>>(qkv, ld_qkv, kc, vc, slot_ids, start_pos, cu_seqlens,
                                                    n_seq, capacity, H, KVH, Dh);
  const int tc_min = lx::attention_tc_min_tokens(H, KVH, Dh);
  if (int e = lx::attention_tc(qkv, ld_qkv, kc, vc, slot_ids, start_pos, cu_seqlens, n_seq, capacity, o, H, KVH, Dh,
                               scale, s))
    return e;
  // Three stages of K and V tiles: kStages * 2 * kBlockN * Dh bf16. Above 48 KB needs the opt-in.
  const int smem = kStages * 2 * kBlockN * Dh * static_cast<int>(sizeof(__nv_bfloat16));
  if (Dh <= 64) {
    cudaFuncSetAttribute(attention_kernel<1>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attention_kernel<1><<<kAttnBlocks, kThreads, smem, s>>>(qkv, ld_qkv, kc, vc, slot_ids, start_pos,
                                                            cu_seqlens, n_seq, capacity, o, H, KVH, Dh, scale, tc_min);
  } else {
    cudaFuncSetAttribute(attention_kernel<2>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    attention_kernel<2><<<kAttnBlocks, kThreads, smem, s>>>(qkv, ld_qkv, kc, vc, slot_ids, start_pos,
                                                            cu_seqlens, n_seq, capacity, o, H, KVH, Dh, scale, tc_min);
  }
  return lx::launch_status();
}
