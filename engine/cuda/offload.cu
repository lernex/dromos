// Offloading (kernels.h, "Offload"): the GPU halves of running a model larger than VRAM.
//
// The expert kernels read every expert through a per-pair lx_expert_ref, so where an expert lives
// (a VRAM cache slot, or pinned host memory read across PCIe) is invisible to the math: the same
// bytes give the same result. They follow moe_w8.cu's decode structure, generalized over formats:
//   1. gate/up: one warp per intermediate row i of a pair; the warp loads gate row i and up row i
//      (16 bytes per lane per load, 512 contiguous bytes per warp), writes bf16(silu(g) * u).
//   2. down: one warp per output row l of a token, summed over the token's pairs in registers, so
//      every output is written once (no atomics).
// All of a warp's loads for a row are issued before any is used, which is what keeps enough bytes
// in flight to cover the latency of a read that crosses PCIe (about a microsecond).
//
// A pair with a fill destination writes what it reads into that VRAM slot: a miss costs one trip
// across the bus and is a hit the next time (pass 2 of the same token reuses most of pass 1's
// experts). The plan kernel picks the slots so that no slot a pair reads in this step is filled.
//
// The CPU lane is a mailbox in mapped host memory: the plan posts the job and the latent rows,
// fences at system scope and rings the doorbell; the runtime's CPU worker computes and sets `done`;
// lx_offload_join waits for it and adds the result. Nothing returns to the host between the two,
// so a decode step with offloading is still one graph launch.
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include <algorithm>

#include "cuda/common.cuh"

namespace {

using bf16 = __nv_bfloat16;

constexpr int kThreads = 256;  // expert kernels: 8 warps per block, a warp per row
constexpr int kWarps = kThreads / 32;
constexpr int kMaxTopK = 8;
constexpr int kMaxChunks = 4;  // 16-byte chunks per lane held in registers: rows up to 2048 bytes
constexpr unsigned kFull = 0xffffffffu;

template <int F>
struct Fmt;
template <>
struct Fmt<LX_W_BF16> {
  static constexpr int kElem = 2;    // bytes per weight
  static constexpr int kPer16 = 8;   // weights per 16-byte chunk
  static constexpr bool kScaled = false;
};
template <>
struct Fmt<LX_W_INT8> {
  static constexpr int kElem = 1;
  static constexpr int kPer16 = 16;
  static constexpr bool kScaled = true;
};
template <>
struct Fmt<LX_W_FP8> {
  static constexpr int kElem = 1;
  static constexpr int kPer16 = 16;
  static constexpr bool kScaled = true;
};

__device__ __forceinline__ float bf_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }

__device__ __forceinline__ uint4 ld16(const void* p) { return __ldg(reinterpret_cast<const uint4*>(p)); }
__device__ __forceinline__ void st16(void* p, const uint4& v) { *reinterpret_cast<uint4*>(p) = v; }

// Two fp8 e4m3 values (the low and high byte of v) as floats, through the hardware conversion.
__device__ __forceinline__ float2 fp8x2(uint32_t v) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<__nv_fp8x2_storage_t>(v & 0xffffu), __NV_E4M3);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

// The dot product of one 16-byte chunk of weights with its activations: 8 bf16 activations in xa
// (BF16), or 16 in xa then xb (8-bit formats). Weight n pairs with activation n.
template <int F>
__device__ __forceinline__ float chunk_dot(const uint4& w, const uint4& xa, const uint4& xb);

template <>
__device__ __forceinline__ float chunk_dot<LX_W_BF16>(const uint4& w, const uint4& xa, const uint4&) {
  float acc = bf_lo(w.x) * bf_lo(xa.x);
  acc = fmaf(bf_hi(w.x), bf_hi(xa.x), acc);
  acc = fmaf(bf_lo(w.y), bf_lo(xa.y), acc);
  acc = fmaf(bf_hi(w.y), bf_hi(xa.y), acc);
  acc = fmaf(bf_lo(w.z), bf_lo(xa.z), acc);
  acc = fmaf(bf_hi(w.z), bf_hi(xa.z), acc);
  acc = fmaf(bf_lo(w.w), bf_lo(xa.w), acc);
  acc = fmaf(bf_hi(w.w), bf_hi(xa.w), acc);
  return acc;
}

template <>
__device__ __forceinline__ float chunk_dot<LX_W_INT8>(const uint4& w, const uint4& xa, const uint4& xb) {
  const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
  const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    acc = fmaf(static_cast<float>(static_cast<int8_t>(wv[j])), bf_lo(xv[2 * j]), acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(wv[j] >> 8)), bf_hi(xv[2 * j]), acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(wv[j] >> 16)), bf_lo(xv[2 * j + 1]), acc);
    acc = fmaf(static_cast<float>(static_cast<int8_t>(wv[j] >> 24)), bf_hi(xv[2 * j + 1]), acc);
  }
  return acc;
}

template <>
__device__ __forceinline__ float chunk_dot<LX_W_FP8>(const uint4& w, const uint4& xa, const uint4& xb) {
  const uint32_t wv[4] = {w.x, w.y, w.z, w.w};
  const uint32_t xv[8] = {xa.x, xa.y, xa.z, xa.w, xb.x, xb.y, xb.z, xb.w};
  float acc = 0.0f;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float2 lo = fp8x2(wv[j]);
    const float2 hi = fp8x2(wv[j] >> 16);
    acc = fmaf(lo.x, bf_lo(xv[2 * j]), acc);
    acc = fmaf(lo.y, bf_hi(xv[2 * j]), acc);
    acc = fmaf(hi.x, bf_lo(xv[2 * j + 1]), acc);
    acc = fmaf(hi.y, bf_hi(xv[2 * j + 1]), acc);
  }
  return acc;
}

// Activations for chunk `idx` of a row: chunk idx covers weights [idx * kPer16, (idx + 1) * kPer16).
template <int F>
__device__ __forceinline__ void load_x(const bf16* x, int idx, uint4& xa, uint4& xb) {
  const bf16* p = x + static_cast<int64_t>(idx) * Fmt<F>::kPer16;
  xa = *reinterpret_cast<const uint4*>(p);
  xb = Fmt<F>::kPer16 == 16 ? *reinterpret_cast<const uint4*>(p + 8) : make_uint4(0, 0, 0, 0);
}

template <int F>
__device__ __forceinline__ float row_scale(const void* scale, int64_t row) {
  if constexpr (Fmt<F>::kScaled) {
    return static_cast<const float*>(scale)[row];
  } else {
    return 1.0f;
  }
}

// ---- decode form ----------------------------------------------------------------------------

// Stage 1. Block b covers rows 8b..8b+7 of pair p = blockIdx.x / rows; warp w takes row i.
template <int F>
__global__ void __launch_bounds__(kThreads) ox_gate_up(const bf16* __restrict__ latent, const lx_expert_ref* src,
                                                       const lx_expert_ref* fill, bf16* __restrict__ inter, int L,
                                                       int I, int top_k, int rows) {
  const int p = blockIdx.x / rows;
  const lx_expert_ref s = src[p];
  if (!s.gate_up) return;
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int64_t rb = static_cast<int64_t>(L) * Fmt<F>::kElem;
  const int nch = static_cast<int>(rb / 16);
  const char* g = static_cast<const char*>(s.gate_up) + i * rb;
  const char* u = g + static_cast<int64_t>(I) * rb;

  uint4 wg[kMaxChunks], wu[kMaxChunks];
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    const int idx = lane + 32 * c;
    if (idx < nch) {
      wg[c] = ld16(g + 16 * idx);
      wu[c] = ld16(u + 16 * idx);
    }
  }
  lx_expert_ref f{};
  if (fill) f = fill[p];
  if (f.gate_up) {
    char* fg = static_cast<char*>(const_cast<void*>(f.gate_up)) + i * rb;
    char* fu = fg + static_cast<int64_t>(I) * rb;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
      const int idx = lane + 32 * c;
      if (idx < nch) {
        st16(fg + 16 * idx, wg[c]);
        st16(fu + 16 * idx, wu[c]);
      }
    }
  }

  const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;
  float ga = 0.0f, ua = 0.0f;
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    const int idx = lane + 32 * c;
    if (idx < nch) {
      uint4 xa, xb;
      load_x<F>(x, idx, xa, xb);
      ga += chunk_dot<F>(wg[c], xa, xb);
      ua += chunk_dot<F>(wu[c], xa, xb);
    }
  }
  // rows longer than the register window
  for (int idx = lane + 32 * kMaxChunks; idx < nch; idx += 32) {
    const uint4 a = ld16(g + 16 * idx), b = ld16(u + 16 * idx);
    if (f.gate_up) {
      char* fg = static_cast<char*>(const_cast<void*>(f.gate_up)) + i * rb;
      st16(fg + 16 * idx, a);
      st16(fg + static_cast<int64_t>(I) * rb + 16 * idx, b);
    }
    uint4 xa, xb;
    load_x<F>(x, idx, xa, xb);
    ga += chunk_dot<F>(a, xa, xb);
    ua += chunk_dot<F>(b, xa, xb);
  }
  ga = lx::warp_sum(ga);
  ua = lx::warp_sum(ua);
  if (lane == 0) {
    const float sg = row_scale<F>(s.gate_up_scale, i), su = row_scale<F>(s.gate_up_scale, I + i);
    if constexpr (Fmt<F>::kScaled) {
      if (f.gate_up) {
        float* fs = static_cast<float*>(const_cast<void*>(f.gate_up_scale));
        fs[i] = sg;
        fs[I + i] = su;
      }
    }
    inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * ga) * (su * ua));
  }
}

// Stage 2. Block b covers output rows 8b..8b+7 of token t = blockIdx.x / rows; warp w takes row l.
// The token's intermediates are staged in shared memory once per block.
template <int F>
__global__ void __launch_bounds__(kThreads) ox_down(const bf16* __restrict__ inter, const float* __restrict__ weights,
                                                    const lx_expert_ref* src, const lx_expert_ref* fill,
                                                    float* __restrict__ out, int L, int I, int top_k, int rows) {
  extern __shared__ uint4 smem[];
  const int t = blockIdx.x / rows;
  const int l = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  const int lane = threadIdx.x & 31;
  const uint4* in = reinterpret_cast<const uint4*>(inter + static_cast<int64_t>(t) * top_k * I);
  for (int c = threadIdx.x; c < top_k * I / 8; c += kThreads) smem[c] = in[c];
  __syncthreads();
  if (l >= L) return;
  const bf16* xs = reinterpret_cast<const bf16*>(smem);
  const int64_t rb = static_cast<int64_t>(I) * Fmt<F>::kElem;
  const int nch = static_cast<int>(rb / 16);

  float acc = 0.0f;
  for (int k = 0; k < top_k; ++k) {
    const int p = t * top_k + k;
    const lx_expert_ref s = src[p];
    if (!s.gate_up) continue;
    const char* row = static_cast<const char*>(s.down) + l * rb;
    uint4 w[kMaxChunks];
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
      const int idx = lane + 32 * c;
      if (idx < nch) w[c] = ld16(row + 16 * idx);
    }
    lx_expert_ref f{};
    if (fill) f = fill[p];
    char* frow = f.gate_up ? static_cast<char*>(const_cast<void*>(f.down)) + l * rb : nullptr;
    if (frow) {
#pragma unroll
      for (int c = 0; c < kMaxChunks; ++c) {
        const int idx = lane + 32 * c;
        if (idx < nch) st16(frow + 16 * idx, w[c]);
      }
    }
    const bf16* x = xs + k * I;
    float d = 0.0f;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
      const int idx = lane + 32 * c;
      if (idx < nch) {
        uint4 xa, xb;
        load_x<F>(x, idx, xa, xb);
        d += chunk_dot<F>(w[c], xa, xb);
      }
    }
    for (int idx = lane + 32 * kMaxChunks; idx < nch; idx += 32) {
      const uint4 a = ld16(row + 16 * idx);
      if (frow) st16(frow + 16 * idx, a);
      uint4 xa, xb;
      load_x<F>(x, idx, xa, xb);
      d += chunk_dot<F>(a, xa, xb);
    }
    d = lx::warp_sum(d);
    const float sc = row_scale<F>(s.down_scale, l);
    if constexpr (Fmt<F>::kScaled) {
      if (frow && lane == 0) static_cast<float*>(const_cast<void*>(f.down_scale))[l] = sc;
    }
    acc = fmaf(weights[p], sc * d, acc);
  }
  if (lane == 0) out[static_cast<int64_t>(t) * L + l] = acc;
}

// ---- grouped form (prefill) -----------------------------------------------------------------

// Stage 1: block b covers rows 8b..8b+7 of group g; the warp holds gate and up row i in registers
// and applies them to every pair of the group.
template <int F>
__global__ void __launch_bounds__(kThreads) ox_gate_up_grouped(const bf16* __restrict__ latent,
                                                               const lx_expert_ref* groups,
                                                               const int32_t* __restrict__ offsets,
                                                               const int32_t* __restrict__ pairs,
                                                               bf16* __restrict__ inter, int L, int I, int top_k,
                                                               int rows) {
  const int g = blockIdx.x / rows;
  const int begin = offsets[g], end = offsets[g + 1];
  if (begin == end) return;
  const lx_expert_ref s = groups[g];
  const int i = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (i >= I) return;
  const int lane = threadIdx.x & 31;
  const int64_t rb = static_cast<int64_t>(L) * Fmt<F>::kElem;
  const int nch = static_cast<int>(rb / 16);
  const char* gr = static_cast<const char*>(s.gate_up) + i * rb;
  const char* ur = gr + static_cast<int64_t>(I) * rb;
  const bool fits = nch <= 32 * kMaxChunks;
  uint4 wg[kMaxChunks], wu[kMaxChunks];
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    const int idx = lane + 32 * c;
    if (idx < nch) {
      wg[c] = ld16(gr + 16 * idx);
      wu[c] = ld16(ur + 16 * idx);
    }
  }
  const float sg = row_scale<F>(s.gate_up_scale, i), su = row_scale<F>(s.gate_up_scale, I + i);
  for (int j = begin; j < end; ++j) {
    const int p = pairs[j];
    const bf16* x = latent + static_cast<int64_t>(p / top_k) * L;
    float ga = 0.0f, ua = 0.0f;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
      const int idx = lane + 32 * c;
      if (idx < nch) {
        uint4 xa, xb;
        load_x<F>(x, idx, xa, xb);
        ga += chunk_dot<F>(wg[c], xa, xb);
        ua += chunk_dot<F>(wu[c], xa, xb);
      }
    }
    if (!fits) {
      for (int idx = lane + 32 * kMaxChunks; idx < nch; idx += 32) {
        uint4 xa, xb;
        load_x<F>(x, idx, xa, xb);
        ga += chunk_dot<F>(ld16(gr + 16 * idx), xa, xb);
        ua += chunk_dot<F>(ld16(ur + 16 * idx), xa, xb);
      }
    }
    ga = lx::warp_sum(ga);
    ua = lx::warp_sum(ua);
    if (lane == 0) inter[static_cast<int64_t>(p) * I + i] = lx::from_f32<bf16>(lx::silu(sg * ga) * (su * ua));
  }
}

// Stage 2: the warp holds down row l of group g and writes weight * (row . inter[p]) for every
// pair of the group into partial [P, L].
template <int F>
__global__ void __launch_bounds__(kThreads) ox_down_grouped(const bf16* __restrict__ inter,
                                                            const float* __restrict__ weights,
                                                            const lx_expert_ref* groups,
                                                            const int32_t* __restrict__ offsets,
                                                            const int32_t* __restrict__ pairs,
                                                            float* __restrict__ partial, int L, int I, int rows) {
  const int g = blockIdx.x / rows;
  const int begin = offsets[g], end = offsets[g + 1];
  if (begin == end) return;
  const lx_expert_ref s = groups[g];
  const int l = (blockIdx.x % rows) * kWarps + (threadIdx.x >> 5);
  if (l >= L) return;
  const int lane = threadIdx.x & 31;
  const int64_t rb = static_cast<int64_t>(I) * Fmt<F>::kElem;
  const int nch = static_cast<int>(rb / 16);
  const char* row = static_cast<const char*>(s.down) + l * rb;
  const bool fits = nch <= 32 * kMaxChunks;
  uint4 w[kMaxChunks];
#pragma unroll
  for (int c = 0; c < kMaxChunks; ++c) {
    const int idx = lane + 32 * c;
    if (idx < nch) w[c] = ld16(row + 16 * idx);
  }
  const float sc = row_scale<F>(s.down_scale, l);
  for (int j = begin; j < end; ++j) {
    const int p = pairs[j];
    const bf16* x = inter + static_cast<int64_t>(p) * I;
    float d = 0.0f;
#pragma unroll
    for (int c = 0; c < kMaxChunks; ++c) {
      const int idx = lane + 32 * c;
      if (idx < nch) {
        uint4 xa, xb;
        load_x<F>(x, idx, xa, xb);
        d += chunk_dot<F>(w[c], xa, xb);
      }
    }
    if (!fits) {
      for (int idx = lane + 32 * kMaxChunks; idx < nch; idx += 32) {
        uint4 xa, xb;
        load_x<F>(x, idx, xa, xb);
        d += chunk_dot<F>(ld16(row + 16 * idx), xa, xb);
      }
    }
    d = lx::warp_sum(d);
    if (lane == 0) partial[static_cast<int64_t>(p) * L + l] = weights[p] * (sc * d);
  }
}

// Stage 3: out[t, l] = sum over k of partial[t * top_k + k, l], in k order.
__global__ void ox_sum_pairs(const float* __restrict__ partial, float* __restrict__ out, int T, int L, int top_k) {
  const int64_t n = static_cast<int64_t>(T) * L;
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i / L, l = i % L;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k) acc += partial[(t * top_k + k) * L + l];
    out[i] = acc;
  }
}

// ---- copies ---------------------------------------------------------------------------------

__device__ __forceinline__ void copy_bytes(const void* src, void* dst, int64_t bytes, int part, int parts) {
  const uint4* s = static_cast<const uint4*>(src);
  uint4* d = static_cast<uint4*>(dst);
  const int64_t n = bytes / 16;
  const int64_t stride = static_cast<int64_t>(parts) * blockDim.x;
  int64_t i = static_cast<int64_t>(part) * blockDim.x + threadIdx.x;
  // four loads in flight per thread before the stores
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = ld16(s + i), b = ld16(s + i + stride), c = ld16(s + i + 2 * stride), e = ld16(s + i + 3 * stride);
    d[i] = a;
    d[i + stride] = b;
    d[i + 2 * stride] = c;
    d[i + 3 * stride] = e;
  }
  for (; i < n; i += stride) d[i] = ld16(s + i);
}

__global__ void ox_copy(const lx_expert_ref* src, const lx_expert_ref* dst, const int32_t* count, int64_t gu_bytes,
                        int64_t gus_bytes, int64_t dn_bytes, int64_t dns_bytes) {
  const int j = blockIdx.y;
  if (j >= *count) return;
  const lx_expert_ref s = src[j], d = dst[j];
  copy_bytes(s.gate_up, const_cast<void*>(d.gate_up), gu_bytes, blockIdx.x, gridDim.x);
  copy_bytes(s.down, const_cast<void*>(d.down), dn_bytes, blockIdx.x, gridDim.x);
  if (gus_bytes && blockIdx.x == 0) {
    for (int64_t i = threadIdx.x; i < gus_bytes / 4; i += blockDim.x)
      static_cast<float*>(const_cast<void*>(d.gate_up_scale))[i] = static_cast<const float*>(s.gate_up_scale)[i];
    for (int64_t i = threadIdx.x; i < dns_bytes / 4; i += blockDim.x)
      static_cast<float*>(const_cast<void*>(d.down_scale))[i] = static_cast<const float*>(s.down_scale)[i];
  }
}

// ---- the cache ------------------------------------------------------------------------------

__device__ __forceinline__ lx_expert_ref pool_ref(const lx_offload_state& st, const lx_expert_ref& pool, int64_t slot) {
  lx_expert_ref r;
  r.gate_up = static_cast<const char*>(pool.gate_up) + slot * st.gate_up_stride;
  r.down = static_cast<const char*>(pool.down) + slot * st.down_stride;
  r.gate_up_scale = pool.gate_up_scale ? static_cast<const char*>(pool.gate_up_scale) + slot * st.gate_up_scale_stride
                                       : nullptr;
  r.down_scale = pool.down_scale ? static_cast<const char*>(pool.down_scale) + slot * st.down_scale_stride : nullptr;
  return r;
}

// The least recently used dynamic slot of a layer not yet used at clock `now`, chosen by the whole
// warp; -1 when every dynamic slot was used at `now`. Ties go to the lower slot.
__device__ int pick_victim(const lx_offload_state& st, int layer, uint32_t now) {
  const int lane = threadIdx.x & 31;
  const int first = st.layer_first[layer] + st.layer_pinned[layer], last = st.layer_first[layer + 1];
  uint32_t best = 0xffffffffu;
  int best_s = -1;
  for (int s = first + lane; s < last; s += 32) {
    const uint32_t v = st.slot_stamp[s];
    if (v < now && (v < best || (v == best && s < best_s))) {
      best = v;
      best_s = s;
    }
  }
  for (int o = 16; o > 0; o >>= 1) {
    const uint32_t ov = __shfl_xor_sync(kFull, best, o);
    const int os = __shfl_xor_sync(kFull, best_s, o);
    if (os >= 0 && (best_s < 0 || ov < best || (ov == best && os < best_s))) {
      best = ov;
      best_s = os;
    }
  }
  return best_s;
}

// Moves `expert` of `layer` into slot v (lane 0), evicting the slot's previous expert.
__device__ void claim_slot(const lx_offload_state& st, int layer, int expert, int v, uint32_t now, bool prefetched) {
  const int base = layer * st.n_experts;
  const int old = st.slot_expert[v];
  if (old >= 0) st.vram_slot[base + old] = -1;
  st.vram_slot[base + expert] = v;
  st.slot_expert[v] = expert;
  st.slot_stamp[v] = now;
  st.slot_prefetched[v] = prefetched ? 1 : 0;
}

__device__ __forceinline__ uint64_t global_ns() {
  uint64_t t;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
  return t;
}

__device__ __forceinline__ uint32_t load_acquire_sys(const volatile uint32_t* p) {
  uint32_t v;
  asm volatile("ld.acquire.sys.global.u32 %0, [%1];" : "=r"(v) : "l"(p) : "memory");
  return v;
}

__global__ void ox_plan(const int32_t* __restrict__ ids, const float* __restrict__ weights,
                        const bf16* __restrict__ latent, int layer, lx_offload_state* sp, lx_expert_ref* src,
                        lx_expert_ref* fill, int T) {
  __shared__ int miss_expert[LX_OFFLOAD_MAX_PAIRS], miss_lane[LX_OFFLOAD_MAX_PAIRS], miss_slot[LX_OFFLOAD_MAX_PAIRS];
  __shared__ int pair_miss[LX_OFFLOAD_MAX_PAIRS];  // index into the miss list, -1 for a hit
  const lx_offload_state st = *sp;
  const int lane = threadIdx.x;
  const int K = st.top_k, P = T * K, E = st.n_experts, base = layer * E;
  uint32_t now = 0;
  if (lane == 0) now = atomicAdd(st.clock, 1u) + 1u;
  now = __shfl_sync(kFull, now, 0);

  // Hits first, so their slots carry `now` before any victim is chosen.
  int hits = 0, prefetch_hits = 0;
  for (int p = lane; p < P; p += 32) {
    const int e = ids[p];
    const int s = st.vram_slot[base + e];
    pair_miss[p] = s >= 0 ? -1 : 0;
    if (s >= 0) {
      st.slot_stamp[s] = now;
      if (st.slot_prefetched[s]) {
        st.slot_prefetched[s] = 0;
        ++prefetch_hits;
      }
      src[p] = pool_ref(st, st.vram_pool, s);
      fill[p] = lx_expert_ref{};
      ++hits;
    }
  }
  __syncwarp();

  // The distinct missed experts, in pair order (lane 0; at most 64 pairs).
  int n_miss = 0;
  if (lane == 0) {
    for (int p = 0; p < P; ++p) {
      if (pair_miss[p] < 0) continue;
      const int e = ids[p];
      int m = 0;
      while (m < n_miss && miss_expert[m] != e) ++m;
      if (m == n_miss) {
        miss_expert[m] = e;
        miss_slot[m] = -1;
        ++n_miss;
      }
      pair_miss[p] = m;
    }
    // Lanes: experts only on disk go to the CPU; then the CPU takes its share of the rest, cold
    // experts first (a hot one read by the GPU lands in VRAM and is a hit next time).
    int cpu = 0;
    for (int m = 0; m < n_miss; ++m) {
      const bool disk = st.host_slot[base + miss_expert[m]] < 0;
      miss_lane[m] = disk ? 1 : 0;
      cpu += disk;
    }
    int target = static_cast<int>(st.cpu_fraction * n_miss + 0.5f);
    target = min(target, st.max_cpu_pairs);
    for (int pass = 0; pass < 2 && cpu < target; ++pass) {
      for (int m = n_miss - 1; m >= 0 && cpu < target; --m) {
        if (miss_lane[m]) continue;
        if (pass == 0 && !st.cpu_first[base + miss_expert[m]]) continue;
        miss_lane[m] = 1;
        ++cpu;
      }
    }
  }
  n_miss = __shfl_sync(kFull, n_miss, 0);
  __syncwarp();

  // Victims for the GPU's misses, one warp-wide choice each.
  int fills = 0;
  if (st.fill) {
    for (int m = 0; m < n_miss; ++m) {
      if (miss_lane[m]) continue;
      const int v = pick_victim(st, layer, now);
      if (v < 0) break;
      if (lane == 0) {
        claim_slot(st, layer, miss_expert[m], v, now, false);
        miss_slot[m] = v;
      }
      ++fills;
      __syncwarp();
    }
  }
  __syncwarp();

  // Refs for the misses: the first pair of a filled expert writes its slot; any other pair of the
  // same expert reads host memory without writing.
  int gpu_pairs = 0, cpu_pairs = 0;
  for (int p = lane; p < P; p += 32) {
    const int m = pair_miss[p];
    if (m < 0) continue;
    if (miss_lane[m]) {
      src[p] = lx_expert_ref{};
      fill[p] = lx_expert_ref{};
      ++cpu_pairs;
      continue;
    }
    const int e = miss_expert[m];
    src[p] = pool_ref(st, st.host_pool, st.host_slot[base + e]);
    bool lead = true;
    for (int q = 0; q < p; ++q) lead &= pair_miss[q] != m;
    fill[p] = (lead && miss_slot[m] >= 0) ? pool_ref(st, st.vram_pool, miss_slot[m]) : lx_expert_ref{};
    ++gpu_pairs;
  }
  for (int o = 16; o > 0; o >>= 1) {
    hits += __shfl_xor_sync(kFull, hits, o);
    prefetch_hits += __shfl_xor_sync(kFull, prefetch_hits, o);
    gpu_pairs += __shfl_xor_sync(kFull, gpu_pairs, o);
    cpu_pairs += __shfl_xor_sync(kFull, cpu_pairs, o);
  }

  // Post the CPU's job: its pairs, then the latent rows, then (after a system fence) the doorbell.
  if (cpu_pairs > 0) {
    lx_offload_mailbox* mb = st.mailbox;
    if (lane == 0) {
      int n = 0;
      for (int p = 0; p < P; ++p) {
        const int m = pair_miss[p];
        if (m < 0 || !miss_lane[m]) continue;
        mb->token[n] = p / K;
        mb->expert[n] = ids[p];
        mb->weight[n] = weights[p];
        ++n;
      }
      mb->layer = layer;
      mb->T = T;
      mb->n = n;
    }
    const int chunks = T * st.L / 8;
    const uint4* lsrc = reinterpret_cast<const uint4*>(latent);
    uint4* ldst = static_cast<uint4*>(st.cpu_latent);
    for (int c = lane; c < chunks; c += 32) ldst[c] = lsrc[c];
    __threadfence_system();
    __syncwarp();
    if (lane == 0) {
      const uint32_t seq = *st.job_seq + 1;
      *st.job_seq = seq;
      *st.job_pending = 1;
      mb->doorbell = seq;
      __threadfence_system();
    }
  } else if (lane == 0) {
    *st.job_pending = 0;
  }

  if (lane == 0) {
    unsigned long long* c = st.counters + static_cast<int64_t>(layer) * LX_OFFLOAD_COUNTERS;
    atomicAdd(c + LX_OC_HITS, static_cast<unsigned long long>(hits));
    atomicAdd(c + LX_OC_GPU_MISSES, static_cast<unsigned long long>(gpu_pairs));
    atomicAdd(c + LX_OC_CPU_MISSES, static_cast<unsigned long long>(cpu_pairs));
    atomicAdd(c + LX_OC_FILLS, static_cast<unsigned long long>(fills));
    atomicAdd(c + LX_OC_PREFETCH_HITS, static_cast<unsigned long long>(prefetch_hits));
    atomicAdd(c + LX_OC_CALLS, 1ull);
  }
}

__global__ void ox_join(lx_offload_state* sp, float* __restrict__ out, int T, int L) {
  __shared__ int ready;
  if (threadIdx.x == 0) {
    ready = 0;
    if (*sp->job_pending) {
      const uint32_t want = *sp->job_seq;
      const uint64_t start = global_ns();
      for (;;) {
        if (load_acquire_sys(&sp->mailbox->done) == want) {
          ready = 1;
          break;
        }
        if (global_ns() - start > 10'000'000'000ull) {  // 10 s: the CPU worker is gone
          *sp->error = 1;
          break;
        }
      }
      *sp->job_pending = 0;
    }
  }
  __syncthreads();
  if (!ready) return;
  const float* r = sp->cpu_result;
  for (int i = threadIdx.x; i < T * L; i += blockDim.x) out[i] += __ldcv(r + i);
}

__global__ void ox_prefetch(const int32_t* __restrict__ predicted, int n, int layer, lx_offload_state* sp,
                            lx_expert_ref* src, lx_expert_ref* dst, int32_t* count) {
  const lx_offload_state st = *sp;
  const int lane = threadIdx.x;
  const int base = layer * st.n_experts;
  // One past the clock: the slots the last plan used (another layer's) are no business of this one,
  // and the next plan of this layer sees the prefetched slots as used now, so it keeps them.
  const uint32_t now = *reinterpret_cast<volatile uint32_t*>(st.clock) + 1u;
  // Protect what is already resident among the predictions first.
  for (int j = lane; j < n; j += 32) {
    const int s = st.vram_slot[base + predicted[j]];
    if (s >= 0 && st.slot_stamp[s] < now) st.slot_stamp[s] = now;
  }
  __syncwarp();
  int c = 0;
  for (int j = 0; j < n; ++j) {
    const int e = predicted[j];
    if (st.vram_slot[base + e] >= 0) continue;  // resident, or claimed by an earlier prediction
    const int h = st.host_slot[base + e];
    if (h < 0) continue;
    const int v = pick_victim(st, layer, now);
    if (v < 0) break;
    if (lane == 0) {
      claim_slot(st, layer, e, v, now, true);
      src[c] = pool_ref(st, st.host_pool, h);
      dst[c] = pool_ref(st, st.vram_pool, v);
    }
    ++c;
    __syncwarp();
  }
  if (lane == 0) {
    *count = c;
    atomicAdd(st.counters + static_cast<int64_t>(layer) * LX_OFFLOAD_COUNTERS + LX_OC_PREFETCHES,
              static_cast<unsigned long long>(c));
  }
}

__global__ void ox_read_probe(const uint4* __restrict__ src, int64_t n, unsigned long long* sink) {
  uint32_t acc = 0;
  const int64_t stride = static_cast<int64_t>(gridDim.x) * blockDim.x;
  int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  for (; i + 3 * stride < n; i += 4 * stride) {
    const uint4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
    acc ^= a.x ^ a.y ^ a.z ^ a.w ^ b.x ^ b.y ^ b.z ^ b.w ^ c.x ^ c.y ^ c.z ^ c.w ^ d.x ^ d.y ^ d.z ^ d.w;
  }
  for (; i < n; i += stride) {
    const uint4 a = src[i];
    acc ^= a.x ^ a.y ^ a.z ^ a.w;
  }
  if (acc == 0x9e3779b9u) atomicAdd(sink, 1ull);  // keeps the loads; almost never taken
}

template <int F>
int launch_decode(const void* latent, const float* weights, const lx_expert_ref* src, const lx_expert_ref* fill,
                  float* out, void* workspace, int T, int L, int I, int top_k, cudaStream_t s) {
  bf16* inter = static_cast<bf16*>(workspace);
  const int P = T * top_k;
  const int rows1 = (I + kWarps - 1) / kWarps;
  ox_gate_up<F><<<P * rows1, kThreads, 0, s>>>(static_cast<const bf16*>(latent), src, fill, inter, L, I, top_k, rows1);
  const int rows2 = (L + kWarps - 1) / kWarps;
  const size_t smem = static_cast<size_t>(top_k) * I * sizeof(bf16);
  ox_down<F><<<T * rows2, kThreads, smem, s>>>(inter, weights, src, fill, out, L, I, top_k, rows2);
  return lx::launch_status();
}

// Stages 1 and 2 of the grouped form for G groups, into the workspace's intermediates and partials.
template <int F>
int launch_grouped(const void* latent, const float* weights, const lx_expert_ref* groups, const int32_t* offsets,
                   const int32_t* pairs, int G, void* workspace, int L, int I, int top_k, int T, cudaStream_t s) {
  const int P = T * top_k;
  bf16* inter = static_cast<bf16*>(workspace);
  float* partial = reinterpret_cast<float*>(static_cast<char*>(workspace) + (static_cast<int64_t>(P) * I * 2 + 15) / 16 * 16);
  const int rows1 = (I + kWarps - 1) / kWarps;
  ox_gate_up_grouped<F><<<G * rows1, kThreads, 0, s>>>(static_cast<const bf16*>(latent), groups, offsets, pairs, inter,
                                                       L, I, top_k, rows1);
  const int rows2 = (L + kWarps - 1) / kWarps;
  ox_down_grouped<F><<<G * rows2, kThreads, 0, s>>>(inter, weights, groups, offsets, pairs, partial, L, I, rows2);
  return lx::launch_status();
}

bool shape_ok(int L, int I, int top_k) { return L % 16 == 0 && I % 16 == 0 && top_k >= 1 && top_k <= kMaxTopK; }

}  // namespace

extern "C" {

int lx_rt_host_alloc_mapped(void** host, void** device, int64_t bytes) {
  *host = *device = nullptr;
  if (bytes <= 0) return 0;
  if (cudaError_t e = cudaHostAlloc(host, static_cast<size_t>(bytes), cudaHostAllocMapped | cudaHostAllocPortable);
      e != cudaSuccess)
    return static_cast<int>(e);
  return static_cast<int>(cudaHostGetDevicePointer(device, *host, 0));
}

int lx_rt_host_register(void* host, int64_t bytes, void** device) {
  *device = nullptr;
  if (cudaError_t e = cudaHostRegister(host, static_cast<size_t>(bytes), cudaHostRegisterMapped | cudaHostRegisterPortable);
      e != cudaSuccess)
    return static_cast<int>(e);
  return static_cast<int>(cudaHostGetDevicePointer(device, host, 0));
}

int lx_rt_host_unregister(void* host) { return static_cast<int>(cudaHostUnregister(host)); }

int lx_rt_mem_info(int64_t* free_bytes, int64_t* total_bytes) {
  size_t f = 0, t = 0;
  const cudaError_t e = cudaMemGetInfo(&f, &t);
  *free_bytes = static_cast<int64_t>(f);
  *total_bytes = static_cast<int64_t>(t);
  return static_cast<int>(e);
}

int lx_rt_stream_wait(void* stream, void* event) {
  return static_cast<int>(cudaStreamWaitEvent(static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0));
}

int64_t lx_offload_workspace_bytes(int T, int L, int I, int top_k) {
  const int64_t P = static_cast<int64_t>(T) * top_k;
  return (P * I * 2 + 15) / 16 * 16 + P * L * 4;
}

int lx_offload_experts(const void* latent, const float* weights, const lx_expert_ref* src, const lx_expert_ref* fill,
                       lx_wtype type, float* out, void* workspace, int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return 0;
  if (!shape_ok(L, I, top_k)) return static_cast<int>(cudaErrorInvalidValue);
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  switch (type) {
    case LX_W_BF16: return launch_decode<LX_W_BF16>(latent, weights, src, fill, out, workspace, T, L, I, top_k, s);
    case LX_W_INT8: return launch_decode<LX_W_INT8>(latent, weights, src, fill, out, workspace, T, L, I, top_k, s);
    case LX_W_FP8: return launch_decode<LX_W_FP8>(latent, weights, src, fill, out, workspace, T, L, I, top_k, s);
    default: return LX_ERR_NOT_SUPPORTED;
  }
}

int lx_offload_experts_grouped(const void* latent, const float* weights, const lx_expert_ref* groups,
                               const int32_t* offsets, const int32_t* pairs, int G, lx_wtype type, float* out,
                               void* workspace, int T, int L, int I, int top_k, int flags, void* stream) {
  if (T <= 0) return 0;
  if (!shape_ok(L, I, top_k)) return static_cast<int>(cudaErrorInvalidValue);
  if (type != LX_W_BF16 && type != LX_W_INT8 && type != LX_W_FP8) return LX_ERR_NOT_SUPPORTED;
  cudaStream_t s = static_cast<cudaStream_t>(stream);
  const int64_t partial_off = (static_cast<int64_t>(T) * top_k * I * 2 + 15) / 16 * 16;
  float* partial = reinterpret_cast<float*>(static_cast<char*>(workspace) + partial_off);
  if (flags & LX_OG_BEGIN) cudaMemsetAsync(partial, 0, static_cast<size_t>(T) * top_k * L * 4, s);
  if (G > 0) {
    int r = 0;
    switch (type) {
      case LX_W_BF16: r = launch_grouped<LX_W_BF16>(latent, weights, groups, offsets, pairs, G, workspace, L, I, top_k, T, s); break;
      case LX_W_INT8: r = launch_grouped<LX_W_INT8>(latent, weights, groups, offsets, pairs, G, workspace, L, I, top_k, T, s); break;
      default: r = launch_grouped<LX_W_FP8>(latent, weights, groups, offsets, pairs, G, workspace, L, I, top_k, T, s); break;
    }
    if (r) return r;
  }
  if (flags & LX_OG_END) {
    const int64_t n = static_cast<int64_t>(T) * L;
    const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 4096));
    ox_sum_pairs<<<blocks, 256, 0, s>>>(partial, out, T, L, top_k);
  }
  return lx::launch_status();
}

int lx_offload_copy(const lx_expert_ref* src, const lx_expert_ref* dst, const int32_t* count, int max_count,
                    lx_wtype type, int L, int I, void* stream) {
  if (max_count <= 0) return 0;
  const int elem = type == LX_W_BF16 ? 2 : 1;
  const bool scaled = type == LX_W_INT8 || type == LX_W_FP8;
  if (type != LX_W_BF16 && !scaled) return LX_ERR_NOT_SUPPORTED;
  const int64_t gu = 2ll * I * L * elem, dn = static_cast<int64_t>(L) * I * elem;
  if (gu % 16 || dn % 16) return static_cast<int>(cudaErrorInvalidValue);
  // 24 blocks per expert keep ~400 KB in flight, enough to cover PCIe latency, while leaving most
  // of the SMs to the work it overlaps.
  dim3 grid(24, max_count);
  ox_copy<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(src, dst, count, gu, scaled ? 2ll * I * 4 : 0, dn,
                                                                scaled ? static_cast<int64_t>(L) * 4 : 0);
  return lx::launch_status();
}

int lx_offload_plan(const int32_t* ids, const float* weights, const void* latent, int layer, lx_offload_state* state,
                    lx_expert_ref* src, lx_expert_ref* fill, int T, void* stream) {
  if (T <= 0) return 0;
  ox_plan<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(ids, weights, static_cast<const bf16*>(latent), layer, state,
                                                           src, fill, T);
  return lx::launch_status();
}

int lx_offload_join(lx_offload_state* state, float* out, int T, int L, void* stream) {
  if (T <= 0) return 0;
  ox_join<<<1, 1024, 0, static_cast<cudaStream_t>(stream)>>>(state, out, T, L);
  return lx::launch_status();
}

int lx_offload_prefetch(const int32_t* predicted, int n, int layer, lx_offload_state* state, lx_expert_ref* src,
                        lx_expert_ref* dst, int32_t* count, void* stream) {
  ox_prefetch<<<1, 32, 0, static_cast<cudaStream_t>(stream)>>>(predicted, n, layer, state, src, dst, count);
  return lx::launch_status();
}

int lx_offload_read_probe(const void* src, int64_t bytes, unsigned long long* sink, int blocks, void* stream) {
  ox_read_probe<<<blocks, 256, 0, static_cast<cudaStream_t>(stream)>>>(static_cast<const uint4*>(src), bytes / 16, sink);
  return lx::launch_status();
}

}  // extern "C"
