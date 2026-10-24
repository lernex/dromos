// The persistent decode kernel (kernels.h, "Decode programs").
//
// One block per SM runs its list of tasks in program order, with three roles. Warp 17 is the loader:
// it walks the same list ahead of the others and copies each task's weight tile into a ring of
// shared-memory pages (bulk copies, or cp.async on sm_80; an mbarrier per page) as soon as a page is
// free. Static tiles (weights, the Mamba state) need nothing from the step, so they stream ahead
// across op boundaries and keep DRAM busy while the others wait on a dependency; expert tiles go
// through a second ring and wait for the routing op. The 16 compute warps run each task: products
// only take the dot products and hand each tile's partial sums to warp 16, the epilogue warp, which
// writes the outputs while the compute warps start on the next tile. Before a run of an op's tasks
// the block waits for the ops it reads (per-op counters in global memory, acquire); after it, the
// block counts itself into the op's counter (release).
//
// Activations written by other SMs are read through L2 (ld.cg), never L1, so a line an SM cached
// earlier in the step can never be stale.
#include <cuda_bf16.h>

#include "cuda/common.cuh"
#include "cuda/decode_ops.h"
#include "cuda/decode_sync.cuh"
#include "cuda/wformat.cuh"

namespace lx::decode {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kCT = kComputeThreads;
constexpr int kF32 = static_cast<int>(LX_W_F32);

// The compute warps (named barrier 1).
__device__ __forceinline__ void csync() { named_sync<1, kCT>(); }

__device__ __forceinline__ float bf(const void* p, int64_t i) { return __bfloat162float(static_cast<const bf16*>(p)[i]); }
__device__ __forceinline__ float round_bf16(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }
__device__ __forceinline__ float4 ldcg4(const float* p) { return __ldcg(reinterpret_cast<const float4*>(p)); }
__device__ __forceinline__ float dot4(const float4& a, const float4& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}
__device__ __forceinline__ float4 bf16x4_bits(uint2 u) {
  return make_float4(__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u), __uint_as_float(u.y << 16),
                     __uint_as_float(u.y & 0xffff0000u));
}
// Four bf16 at p (8-byte aligned) as f32: weights (cached) and activations (through L2).
__device__ __forceinline__ float4 bf16x4(const void* p) { return bf16x4_bits(*reinterpret_cast<const uint2*>(p)); }
__device__ __forceinline__ float4 bf16x4_cg(const void* p) {
  return bf16x4_bits(__ldcg(reinterpret_cast<const uint2*>(p)));
}

// ---- Weight formats -----------------------------------------------------------------------------

// f32 weights (the routers and the depth head): two values per 8-byte word.
struct WordF32 {
  static constexpr int kValues = 2;
  __device__ __forceinline__ static void decode(uint2 v, float, float* f) {
    f[0] = __uint_as_float(v.x);
    f[1] = __uint_as_float(v.y);
  }
};
template <int kType>
struct Word {
  using F = WordFormat<kType>;
};
template <>
struct Word<kF32> {
  using F = WordF32;
};
// int8 to f32 through the exponent trick (0x4B000000 | (q ^ 0x80) is 2^23 + 128 + q as a float):
// exact, like a conversion, but on the integer and FMA pipes instead of the slower conversion unit.
struct WordI8 {
  static constexpr int kValues = 8;
  __device__ __forceinline__ static void decode(uint2 v, float, float* f) {
    const uint32_t a = v.x ^ 0x80808080u, b = v.y ^ 0x80808080u;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      f[i] = __uint_as_float(__byte_perm(a, 0x4B000000u, 0x7540 + i)) - 8388736.0f;
      f[4 + i] = __uint_as_float(__byte_perm(b, 0x4B000000u, 0x7540 + i)) - 8388736.0f;
    }
  }
};
template <>
struct Word<LX_W_INT8> {
  using F = WordI8;
};
// Formats whose tiles carry one f32 scale per row after the rows.
template <int kType>
constexpr bool kRowScaled = kType == LX_W_INT8 || kType == LX_W_FP8;
__device__ __forceinline__ bool row_scaled(int wtype) { return wtype == LX_W_INT8 || wtype == LX_W_FP8; }

// ---- Shared memory ------------------------------------------------------------------------------
//
// A fixed layout, so every task body addresses shared memory through dsm at offsets the compiler
// knows (plain shared loads and stores, never generic ones): the launch context, the current op's
// descriptor, reduction scratch, the page barriers, then the x buffer and the pages after it.

extern __shared__ __align__(128) char dsm[];

struct Ctx {
  const int32_t* tokens;
  const int32_t* slots;
  const int32_t* has_state;
  const int32_t* start_pos;
  int64_t* trace;  // or null
  int M, pages_off, ring_static, ring_dynamic, trace_base;
};
static_assert(sizeof(Ctx) <= kCtxBytes, "the launch context must fit its slot");

__device__ __forceinline__ const Ctx& ctx() { return *reinterpret_cast<const Ctx*>(dsm + kCtxOff); }
__device__ __forceinline__ const OpDesc& desc() { return *reinterpret_cast<const OpDesc*>(dsm + kDescOff); }
__device__ __forceinline__ float* scratch() { return reinterpret_cast<float*>(dsm + kScratchOff); }
__device__ __forceinline__ uint64_t* full_bar(int i) { return reinterpret_cast<uint64_t*>(dsm + kBarOff) + i; }
__device__ __forceinline__ uint64_t* empty_bar(int i) {
  return reinterpret_cast<uint64_t*>(dsm + kBarOff) + kMaxPages + i;
}
// A product tile's partial sums (two slots, alternating): ready when all 16 compute warps wrote
// theirs, free when the epilogue warp has read them.
__device__ __forceinline__ uint64_t* red_ready_bar(int b) {
  return reinterpret_cast<uint64_t*>(dsm + kBarOff) + 2 * kMaxPages + b;
}
__device__ __forceinline__ uint64_t* red_free_bar(int b) {
  return reinterpret_cast<uint64_t*>(dsm + kBarOff) + 2 * kMaxPages + 2 + b;
}
// A page is free once each compute warp has arrived (or one thread for all of them).
__device__ __forceinline__ void release_page(int p) {
  for (int k = 0; k < kComputeWarps; ++k) mbar_arrive(empty_bar(p));
}
// The compute and epilogue warps together (named barrier 3).
__device__ __forceinline__ void work_sync() { named_sync<3, kComputeThreads + 32>(); }
// What the x buffer holds: a key per op (and token), or -1.
__device__ __forceinline__ int& misc() { return *reinterpret_cast<int*>(dsm + kMiscOff); }
__device__ __forceinline__ float* xbuf() { return reinterpret_cast<float*>(dsm + kXOff); }
// The loader's own copies of the descriptors its cursors are on (0 static, 1 dynamic, 2 prefetch).
__device__ __forceinline__ const OpDesc& loader_desc(int slot) {
  return *reinterpret_cast<const OpDesc*>(dsm + kLoaderDescOff + slot * kDescBytes);
}

// A task's place in its ring: seq counts the pages the pool has handed out so far.
struct Pages {
  int p0, p1;            // page indices
  uint32_t ph0, ph1;     // the parity of their fill for this use
};
__device__ __forceinline__ Pages ring_pages(int pool, uint32_t seq) {
  const Ctx& c = ctx();
  const int ring = pool == kStatic ? c.ring_static : c.ring_dynamic;
  const int first = pool == kStatic ? 0 : c.ring_static;
  return {first + static_cast<int>(seq % ring), first + static_cast<int>((seq + 1) % ring), (seq / ring) & 1,
          ((seq + 1) / ring) & 1};
}
__device__ __forceinline__ int page_offset(int page) { return ctx().pages_off + page * kPageBytes; }

__device__ __forceinline__ void trace_task(int index, int op, int task, int64_t t0, int64_t t1) {
  const Ctx& c = ctx();
  if (c.trace && threadIdx.x == 0) {
    int64_t* rec = c.trace + static_cast<int64_t>(c.trace_base + index) * 4;
    rec[0] = (static_cast<int64_t>(op) << 32) | (static_cast<int64_t>(task) << 16) | blockIdx.x;
    rec[1] = t0;
    rec[2] = t1;
    rec[3] = global_ns();
  }
}
__device__ __forceinline__ int64_t trace_now() { return ctx().trace && threadIdx.x == 0 ? global_ns() : 0; }
// The same, recorded by the epilogue warp's lane 0 (product tiles).
__device__ __forceinline__ int64_t trace_now32() {
  return ctx().trace && threadIdx.x == kEpilogueWarp * 32 ? global_ns() : 0;
}
__device__ __forceinline__ void trace_task32(int index, int op, int task, int64_t t0, int64_t t1) {
  const Ctx& c = ctx();
  if (c.trace && threadIdx.x == kEpilogueWarp * 32) {
    int64_t* rec = c.trace + static_cast<int64_t>(c.trace_base + index) * 4;
    rec[0] = (static_cast<int64_t>(op) << 32) | (static_cast<int64_t>(task) << 16) | blockIdx.x;
    rec[1] = t0;
    rec[2] = t1;
    rec[3] = global_ns();
  }
}

// Sums n values held by every compute thread (vals[0..n)) over the block in a fixed order, so every
// block that runs the same reduction gets the same bits. The totals land in out[0..n). n <= 40.
__device__ __forceinline__ void block_sums(float* vals, int n, float* red, float* out) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int i = 0; i < n; ++i) vals[i] = lx::warp_sum(vals[i]);
  if (lane == 0)
    for (int i = 0; i < n; ++i) red[warp * 40 + i] = vals[i];
  csync();
  if (threadIdx.x < n) {
    float t = 0.0f;
    for (int w = 0; w < kComputeWarps; ++w) t += red[w * 40 + threadIdx.x];
    out[threadIdx.x] = t;
  }
  csync();
}

// ---- Tile dot products: the weight reader -------------------------------------------------------

// The x values for one weight word: V consecutive k from the staged buffer (f32 or bf16 storage).
template <int V, bool kBf16>
__device__ __forceinline__ void xs_word(const float* xs, int idx, float* f) {
  if constexpr (kBf16) {
    const bf16* p = reinterpret_cast<const bf16*>(xs) + idx;
    if constexpr (V == 2) {
      const uint32_t u = *reinterpret_cast<const uint32_t*>(p);
      f[0] = __uint_as_float(u << 16);
      f[1] = __uint_as_float(u & 0xffff0000u);
    } else {
#pragma unroll
      for (int i = 0; i < V / 4; ++i) {
        const float4 v = bf16x4(p + 4 * i);
        f[4 * i] = v.x;
        f[4 * i + 1] = v.y;
        f[4 * i + 2] = v.z;
        f[4 * i + 3] = v.w;
      }
    }
  } else {
    const float* p = xs + idx;
    if constexpr (V == 2) {
      const float2 v = *reinterpret_cast<const float2*>(p);
      f[0] = v.x;
      f[1] = v.y;
    } else {
#pragma unroll
      for (int i = 0; i < V / 4; ++i) {
        const float4 v = *reinterpret_cast<const float4*>(p + 4 * i);
        f[4 * i] = v.x;
        f[4 * i + 1] = v.y;
        f[4 * i + 2] = v.z;
        f[4 * i + 3] = v.w;
      }
    }
  }
}

// Dot products of the rows of a tile in shared memory with staged x rows: row r (W words starting at
// words + r * row_words) times x rows xr0 .. xr0 + mc - 1, where xr0 = r / per_x (or 0 when per_x is
// 0). The warps split each row into wpr parts (fewer rows than warps) or take whole rows; part p of
// row r leaves red[(r * wpr + p) * mc + j] (added to it with accumulate, which a later K chunk of the
// same rows uses: the split is the same, so each slot has one writer). Returns wpr.
//
// This is the only place the products decode a weight word. A new weight format plugs in through
// its WordFormat (and its tile layout in the builder and the loader); prologue and epilogue stay.
template <class F, bool kXsBf16, int kM>
__device__ __forceinline__ int tile_dots(const uint2* words, int rows, int row_words, int W, const float* xs, int x_stride, int mc,
                         int per_x, float* red, bool accumulate) {
  constexpr int kV = F::kValues;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  int wpr = 1;
  while (wpr < kComputeWarps && 2 * wpr * rows <= kComputeWarps) wpr *= 2;
  const int groups = kComputeWarps / wpr, group = warp / wpr, part = warp % wpr;
  const int w0 = part * W / wpr, w1 = (part + 1) * W / wpr;
  for (int r = group; r < rows; r += groups) {
    const uint2* row = words + static_cast<int64_t>(r) * row_words;
    const int xr0 = per_x > 0 ? r / per_x : 0;
    float acc[kM];
#pragma unroll
    for (int j = 0; j < kM; ++j) acc[j] = 0.0f;
#pragma unroll 2
    for (int w = w0 + lane; w < w1; w += 32) {
      float wf[kV];
      F::decode(row[w], 1.0f, wf);
#pragma unroll
      for (int j = 0; j < kM; ++j) {
        if (kM == 1 || j < mc) {
          float xf[kV];
          xs_word<kV, kXsBf16>(xs, (xr0 + j) * x_stride + w * kV, xf);
#pragma unroll
          for (int t = 0; t < kV; ++t) acc[j] = fmaf(wf[t], xf[t], acc[j]);
        }
      }
    }
#pragma unroll
    for (int j = 0; j < kM; ++j)
      if (kM == 1 || j < mc) acc[j] = lx::warp_sum(acc[j]);
    if (lane == 0)
#pragma unroll
      for (int j = 0; j < kM; ++j)
        if (kM == 1 || j < mc) {
          float* slot = red + (r * wpr + part) * mc + j;
          *slot = accumulate ? *slot + acc[j] : acc[j];
        }
  }
  return wpr;
}

template <class F, bool kXsBf16>
__device__ __forceinline__ int tile_dots_m(const uint2* words, int rows, int row_words, int W, const float* xs,
                                           int x_stride, int mc, int per_x, float* red, bool accumulate = false) {
  // built per bucket of tokens, so few tokens never pay for the predicated-off work of eight
  if (mc == 1) return tile_dots<F, kXsBf16, 1>(words, rows, row_words, W, xs, x_stride, 1, per_x, red, accumulate);
  if (mc == 2) return tile_dots<F, kXsBf16, 2>(words, rows, row_words, W, xs, x_stride, 2, per_x, red, accumulate);
  if (mc <= 4) return tile_dots<F, kXsBf16, 4>(words, rows, row_words, W, xs, x_stride, mc, per_x, red, accumulate);
  return tile_dots<F, kXsBf16, kMaxM>(words, rows, row_words, W, xs, x_stride, mc, per_x, red, accumulate);
}

__device__ __forceinline__ float sum_parts(const float* red, int r, int wpr, int mc, int j) {
  float v = 0.0f;
  for (int p = 0; p < wpr; ++p) v += red[(r * wpr + p) * mc + j];
  return v;
}

// ---- Products: prologue (x staging) and epilogues -----------------------------------------------

// Stages the product's x [M, k1 - k0] into the x buffer (bf16 or f32 storage), four values at a time.
template <bool kXsBf16>
__device__ __forceinline__ void stage_gemv_x(const Gemv& g, int M) {
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int Kc = g.k1 - g.k0;
  float* scale = scratch();      // [M] (RMS) or [M * groups] (GATED)
  float* tmp = scratch() + 128;  // [kComputeWarps][kMaxM]
  int groups = 1;
  if (g.x_kind == LX_FX_RMS) {
    float part[kMaxM];
#pragma unroll
    for (int m = 0; m < kMaxM; ++m) part[m] = 0.0f;
    for (int j = tid; j < g.n_parts; j += kCT)
#pragma unroll
      for (int m = 0; m < kMaxM; ++m)
        if (m < M) part[m] += __ldcg(g.parts + static_cast<int64_t>(m) * g.n_parts + j);
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      if (m < M) part[m] = lx::warp_sum(part[m]);
    if (lane == 0)
      for (int m = 0; m < M; ++m) tmp[warp * kMaxM + m] = part[m];
    csync();
    if (tid < M) {
      float s = 0.0f;
      for (int w = 0; w < kComputeWarps; ++w) s += tmp[w * kMaxM + tid];
      scale[tid] = rsqrtf(s / static_cast<float>(g.K) + g.eps);
    }
    csync();
  } else if (g.x_kind == LX_FX_GATED) {
    groups = g.K / g.group;
    const int per = g.n_parts / groups;
    for (int e = tid; e < M * groups; e += kCT) {
      const int m = e / groups, gg = e % groups;
      float s = 0.0f;
      for (int j = 0; j < per; ++j) s += __ldcg(g.parts + static_cast<int64_t>(m) * g.n_parts + gg * per + j);
      scale[e] = rsqrtf(s / static_cast<float>(g.group) + g.eps);
    }
    csync();
  }
  const int quads = Kc / 4;
  for (int e = tid; e < M * quads; e += kCT) {
    const int m = e / quads, j = (e % quads) * 4, k = g.k0 + j;
    float4 v;
    if (g.x_kind == LX_FX_PLAIN) {
      const int64_t at = static_cast<int64_t>(m) * g.ld_x + k;
      v = g.x_dtype == LX_BF16 ? bf16x4_cg(static_cast<const bf16*>(g.x) + at)
                               : ldcg4(static_cast<const float*>(g.x) + at);
    } else if (g.x_kind == LX_FX_SWIGLU) {
      const float* row = static_cast<const float*>(g.x) + static_cast<int64_t>(m) * g.ld_x;
      const float4 gt = ldcg4(row + k), up = ldcg4(row + g.K + k);
      v = make_float4(round_bf16(lx::silu(gt.x) * up.x), round_bf16(lx::silu(gt.y) * up.y),
                      round_bf16(lx::silu(gt.z) * up.z), round_bf16(lx::silu(gt.w) * up.w));
    } else if (g.x_kind == LX_FX_CONCAT) {
      int si = 0;
      while (k >= g.seg_end[si]) ++si;
      const int base = si ? g.seg_end[si - 1] : 0;
      v = ldcg4(g.seg[si] + static_cast<int64_t>(m) * (g.seg_end[si] - base) + (k - base));
      if (g.round_bf16) v = make_float4(round_bf16(v.x), round_bf16(v.y), round_bf16(v.z), round_bf16(v.w));
    } else {
      const float4 src = ldcg4(static_cast<const float*>(g.x) + static_cast<int64_t>(m) * g.ld_x + k);
      const float4 w = bf16x4(static_cast<const bf16*>(g.norm_w) + k);
      const float s = g.x_kind == LX_FX_RMS ? scale[m] : scale[m * groups + k / g.group];
      v = make_float4(round_bf16(src.x * s * w.x), round_bf16(src.y * s * w.y), round_bf16(src.z * s * w.z),
                      round_bf16(src.w * s * w.w));
    }
    if constexpr (kXsBf16) {
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(xbuf()) + m * Kc + j);
      dst[0] = __floats2bfloat162_rn(v.x, v.y);
      dst[1] = __floats2bfloat162_rn(v.z, v.w);
    } else {
      *reinterpret_cast<float4*>(xbuf() + m * Kc + j) = v;
    }
  }
  csync();
}

// What an output's epilogue reads besides the product itself (the bias, the old value, the LoRA
// term, the streams it mixes into), loaded before the product is ready so the epilogue warp's L2
// round trip overlaps the dot products instead of following them.
struct OutInputs {
  float a[kMaxStreams];  // STORE: bias, old out, LoRA dot; WRITE: the streams' values
};

__device__ __forceinline__ OutInputs load_out(const Gemv& g, int m, int64_t n, bool write) {
  OutInputs in;
  if (write) {
    const float* st = g.streams + static_cast<int64_t>(m) * g.S * g.N + n;
#pragma unroll
    for (int i = 0; i < kMaxStreams; ++i) in.a[i] = i < g.S ? __ldcg(st + i * g.N) : 0.0f;
    return in;
  }
  in.a[0] = g.bias ? (g.bias_dtype == LX_BF16 ? bf(g.bias, n) : static_cast<const float*>(g.bias)[n]) : 0.0f;
  const int64_t at = static_cast<int64_t>(m) * g.ld_out + n;
  in.a[1] = g.beta != 0.0f ? (g.out_dtype == LX_BF16 ? __bfloat162float(__ldcg(static_cast<const bf16*>(g.out) + at))
                                                     : __ldcg(static_cast<const float*>(g.out) + at))
                           : 0.0f;
  float l = 0.0f;
  if (g.lora_up)
    for (int j = 0; j < g.rank; ++j)
      l = fmaf(bf(g.lora_up, n * g.rank + j), __ldcg(g.lora + static_cast<int64_t>(m) * g.rank + j), l);
  in.a[2] = l;
  in.a[3] = 0.0f;
  return in;
}

__device__ __forceinline__ void store_out(const Gemv& g, int m, int64_t n, float y, const OutInputs& in) {
  float v = y;
  if (g.bias) v += in.a[0];
  v *= g.alpha;
  const int64_t at = static_cast<int64_t>(m) * g.ld_out + n;
  if (g.beta != 0.0f) v = fmaf(g.beta, in.a[1], v);
  if (g.lora_up) v = fmaf(g.lora_gate, in.a[2], v);
  if (g.act == 1) v = tanhf(v);
  else if (g.act == 2) v = v * sigmoid(v);
  if (g.out_dtype == LX_BF16) static_cast<bf16*>(g.out)[at] = __float2bfloat16_rn(v);
  else static_cast<float*>(g.out)[at] = v;
  if (g.out_bf) static_cast<bf16*>(g.out_bf)[static_cast<int64_t>(m) * g.ld_bf + n] = __float2bfloat16_rn(v);
}

// The hyper-connection write for column n of token m (in.a: the streams there before it); returns
// next_src^2 (0 without next_read).
__device__ __forceinline__ float write_out(const Gemv& g, int m, int64_t n, float y, const OutInputs& in) {
  const int S = g.S;
  const int64_t N = g.N;
  float* st = g.streams + static_cast<int64_t>(m) * S * N + n;
  float nw[kMaxStreams];
#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s) {
    nw[s] = 0.0f;
    if (s < S) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) acc += __ldg(g.matrix + s * S + i) * in.a[i];
      nw[s] = acc + __ldg(g.write + s) * y;
      st[s * N] = nw[s];
    }
  }
  if (g.state) {
    float a = 0.0f;
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s)
      if (s < S) a += nw[s];
    g.state[static_cast<int64_t>(m) * N + n] = a / S;
  }
  if (!g.next_read) return 0.0f;
  float src = 0.0f;
#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s)
    if (s < S) src += __ldg(g.next_read + s) * nw[s];
  g.next_src[static_cast<int64_t>(m) * N + n] = src;
  return src * src;
}

// A block's run of tasks of one product, split over warps: the compute warps stage x once, then per
// tile wait for its page, leave their partial sums (and the rows' scales) in a reduction slot, and
// free the page; the epilogue warp turns each slot into outputs and frees it. No block-wide barrier
// per tile: the warps hand tiles on through mbarriers, so a tile's epilogue overlaps the next
// tiles' dot products. gt counts the block's product tiles so far (the slots' and their barriers'
// phases); seq is the static ring's position.
struct RunState {
  uint32_t seq, gt;
};

template <int kType, bool kXsBf16>
__device__ __forceinline__ RunState gemv_compute_run(const uint32_t* entries, int count, RunState st) {
  using F = typename Word<kType>::F;
  const Gemv& g = desc().gemv;
  const int M = ctx().M, Kc = g.k1 - g.k0, W = Kc / F::kValues, row_words = g.row_bytes / 8;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const Ctx& c = ctx();
  int p = static_cast<int>(st.seq % c.ring_static);
  uint32_t ph = (st.seq / c.ring_static) & 1;
  uint32_t next = __ldg(entries);
  for (int i = 0; i < count; ++i) {
    const int task = static_cast<int>(next & 0xffff);
    if (i + 1 < count) next = __ldg(entries + i + 1);
    const int b = st.gt & 1;
    mbar_wait(red_free_bar(b), ((st.gt >> 1) & 1) ^ 1);
    mbar_wait(full_bar(p), ph);
    const char* page = dsm + c.pages_off + p * kPageBytes;
    const int rows = min(g.rows, g.N - task * g.rows);
    float* red = scratch() + b * 256;
    const int wpr = tile_dots_m<F, kXsBf16>(reinterpret_cast<const uint2*>(page), rows, row_words, W, xbuf(), Kc, M, 0,
                                            red);
    if constexpr (kRowScaled<kType>) {
      // the warps on part 0 of a row hand its scale to the epilogue (the page is about to be freed)
      float* scl = scratch() + 512 + b * 64;
      const int groups = kComputeWarps / wpr;
      if (warp % wpr == 0)
        for (int r = warp / wpr + lane * groups; r < rows; r += 32 * groups)
          scl[r] = reinterpret_cast<const float*>(page + g.scale_off)[r];
    }
    __syncwarp();
    if (lane == 0) {
      mbar_arrive(red_ready_bar(b));
      mbar_arrive(empty_bar(p));
    }
    ++st.gt;
    ++st.seq;
    if (++p == c.ring_static) {
      p = 0;
      ph ^= 1;
    }
  }
  return st;
}

__device__ __noinline__ RunState gemv_compute(int op, const uint32_t* entries, int count, RunState st,
                                              int64_t* run_rec) {
  const Gemv& g = desc().gemv;
  // Every thread reads the key before the staging's barriers; thread 0 rewrites it after them.
  if (misc() != op * 16) {
    if (g.xs_bf16) stage_gemv_x<true>(g, ctx().M);
    else stage_gemv_x<false>(g, ctx().M);
    if (threadIdx.x == 0) misc() = op * 16;
  }
  if (run_rec) run_rec[2] = global_ns();
  switch (g.wtype) {
    case LX_W_BF16:
      return g.xs_bf16 ? gemv_compute_run<LX_W_BF16, true>(entries, count, st)
                       : gemv_compute_run<LX_W_BF16, false>(entries, count, st);
    case LX_W_INT8:
      return g.xs_bf16 ? gemv_compute_run<LX_W_INT8, true>(entries, count, st)
                       : gemv_compute_run<LX_W_INT8, false>(entries, count, st);
    case kF32: return gemv_compute_run<kF32, false>(entries, count, st);
    default: return st;
  }
}

// The epilogue warp's side of a product run: per tile, the outputs from the reduction slot. A lane
// takes elements lane, lane + 32, ... of the tile's rows x tokens (at most 256). The inputs of the
// first 64 are loaded before the products are ready (enough for one token's tile, and few
// registers); a larger tile (several tokens) loads the rest's after.
__device__ __forceinline__ void epilogue_element(const Gemv& g, int e, int M, int n0, float y, const OutInputs& in,
                                                 bool write, float* sqs) {
  const int r = e / M, m = e - r * M;
  if (!write) store_out(g, m, n0 + r, y, in);
  else {
    const float q = write_out(g, m, n0 + r, y, in);
    if (g.next_parts) sqs[e] = q;
  }
}

__device__ __forceinline__ float epilogue_add(const Gemv& g, int e, int M, int n0, int64_t MN) {
  const int r = e / M, m = e - r * M;
  float add = 0.0f;
  for (int c = 0; c < g.n_add; ++c) add += __ldcg(g.y_add + c * MN + static_cast<int64_t>(m) * g.N + n0 + r);
  return add;
}

__device__ __noinline__ uint32_t gemv_epilogue(int op, const uint32_t* entries, int count, int index, uint32_t gt) {
  constexpr int kPre = 2;
  const Gemv& g = desc().gemv;
  const int M = ctx().M, ntasks = desc().ntasks, lane = threadIdx.x & 31;
  const int64_t MN = static_cast<int64_t>(M) * g.N;
  const bool write = g.y_kind == LX_FY_WRITE, parts = write && g.next_parts;
  const bool scaled = row_scaled(g.wtype);
  float* sqs = scratch() + 640;
  for (int i = 0; i < count; ++i) {
    const int task = static_cast<int>(__ldg(entries + i) & 0xffff);
    const int64_t t0 = trace_now32();
    const int b = gt & 1;
    const int n0 = task * g.rows, rows = min(g.rows, g.N - n0), n = rows * M;
    OutInputs in[kPre];
    float add[kPre];
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
      const int e = lane + 32 * j;
      if (e < n) {
        const int r = e / M, m = e - r * M;
        in[j] = load_out(g, m, n0 + r, write);
        add[j] = epilogue_add(g, e, M, n0, MN);
      }
    }
    mbar_wait(red_ready_bar(b), (gt >> 1) & 1);
    const int64_t t1 = trace_now32();
    int wpr = 1;
    while (wpr < kComputeWarps && 2 * wpr * rows <= kComputeWarps) wpr *= 2;
    const float* red = scratch() + b * 256;
    const float* scl = scratch() + 512 + b * 64;
    // the products out of the slot first, so it frees as early as possible
    float v[256 / 32];
#pragma unroll
    for (int j = 0; j < 256 / 32; ++j) {
      const int e = lane + 32 * j;
      v[j] = 0.0f;
      if (e < n) {
        const int r = e / M, m = e - r * M;
        v[j] = sum_parts(red, r, wpr, M, m);
        if (scaled) v[j] *= scl[r];
      }
    }
    __syncwarp();
    if (lane == 0) mbar_arrive(red_free_bar(b));  // the compute warps may refill the slot
#pragma unroll
    for (int j = 0; j < kPre; ++j) {
      const int e = lane + 32 * j;
      if (e < n) epilogue_element(g, e, M, n0, v[j] + add[j], in[j], write, sqs);
    }
#pragma unroll
    for (int j = kPre; j < 256 / 32; ++j) {
      const int e = lane + 32 * j;
      if (e < n) {
        const int r = e / M, m = e - r * M;
        epilogue_element(g, e, M, n0, v[j] + epilogue_add(g, e, M, n0, MN), load_out(g, m, n0 + r, write), write, sqs);
      }
    }
    if (parts) {
      // the sum of next_src^2 over the tile's rows, per token, in row order
      __syncwarp();
      if (lane < M) {
        float sum = 0.0f;
        for (int rr = 0; rr < rows; ++rr) sum += sqs[rr * M + lane];
        g.next_parts[static_cast<int64_t>(lane) * ntasks + task] = sum;
      }
      __syncwarp();
    }
    trace_task32(index + i, op, task, t0, t1);
    ++gt;
  }
  return gt;
}

// ---- Embedding ----------------------------------------------------------------------------------

__device__ __noinline__ void embed_task(int task) {
  const Embed& a = desc().embed;
  const int m = task / a.chunks, c = task % a.chunks;
  const int D = a.D, S = a.S;
  const int per = (D + a.chunks - 1) / a.chunks;
  const int d0 = c * per, d1 = min(D, d0 + per);
  const int64_t tok = ctx().tokens[m];
  for (int d = d0 + threadIdx.x; d < d1; d += kCT) {
    const float v = bf(a.table, tok * D + d);
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) {
      const float x = v + bf(a.stream_emb, s * D + d);
      a.streams[(static_cast<int64_t>(m) * S + s) * D + d] = x;
      acc += x;
    }
    a.previous[static_cast<int64_t>(m) * D + d] = acc / S;
  }
  if (c == 0) {
    for (int r = threadIdx.x; r < a.R; r += kCT) a.history[static_cast<int64_t>(m) * a.R + r] = 0.0f;
    if (threadIdx.x == 0) a.continuation[m] = 0.0f;
  }
}

// ---- Depth-memory read --------------------------------------------------------------------------

__device__ __noinline__ void memread_task(int m) {
  const MemRead& a = desc().mem;
  const int M = ctx().M;
  constexpr int kCh = 2;  // float4 columns per thread: D <= 4 * 2 * 512
  const int tid = threadIdx.x;
  const int S = a.S, D = a.D, ne = a.n_entries, chunks = D / 4;
  float* red = scratch();
  float* fin = scratch() + kComputeWarps * 40;
  float* streams = a.streams + static_cast<int64_t>(m) * S * D;
  float4 x[kCh][kMaxStreams];
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    const int c = tid + j * kCT;
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s)
      x[j][s] = (c < chunks && s < S) ? ldcg4(streams + s * D + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (ne > 0) {
    const float* bank = a.bank + static_cast<int64_t>(m) * a.token_stride;
    // Scores x[s] . U_e and the gate dots x[s] . gate[s] (on the streams before the update).
    float dots[kMaxStreams * (kMaxEntries + 1)];
#pragma unroll
    for (int i = 0; i < kMaxStreams * (kMaxEntries + 1); ++i) dots[i] = 0.0f;
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      const int c = tid + j * kCT;
      if (c < chunks) {
#pragma unroll
        for (int e = 0; e < kMaxEntries; ++e) {
          if (e < ne) {
            const float4 u = ldcg4(bank + static_cast<int64_t>(e) * a.entry_stride + 4 * c);
#pragma unroll
            for (int s = 0; s < kMaxStreams; ++s)
              if (s < S) dots[s * (kMaxEntries + 1) + e] += dot4(x[j][s], u);
          }
        }
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s)
          if (s < S)
            dots[s * (kMaxEntries + 1) + kMaxEntries] +=
                dot4(x[j][s], bf16x4(static_cast<const bf16*>(a.gate) + s * D + 4 * c));
      }
    }
    block_sums(dots, S * (kMaxEntries + 1), red, fin);
    // Softmax over the entries, and the gate, per stream.
    const float inv_m = rsqrtf(static_cast<float>(a.mem_dim));
    const float inv_d = rsqrtf(static_cast<float>(D));
    float w[kMaxStreams][kMaxEntries], norm[kMaxStreams], gs[kMaxStreams];
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s) {
      norm[s] = 0.0f;
      gs[s] = 0.0f;
#pragma unroll
      for (int e = 0; e < kMaxEntries; ++e) w[s][e] = 0.0f;
      if (s < S) {
        float top = -INFINITY;
#pragma unroll
        for (int e = 0; e < kMaxEntries; ++e) {
          if (e < ne) {
            const float c_e = __ldcg(bank + static_cast<int64_t>(e) * a.entry_stride + 2 * D);
            w[s][e] = (fin[s * (kMaxEntries + 1) + e] + c_e) * inv_m;
            top = fmaxf(top, w[s][e]);
          }
        }
        float total = 0.0f;
#pragma unroll
        for (int e = 0; e < kMaxEntries; ++e) {
          if (e < ne) {
            w[s][e] = expf(w[s][e] - top);
            total += w[s][e];
          }
        }
        norm[s] = 1.0f / total;
        gs[s] = sigmoid(fin[s * (kMaxEntries + 1) + kMaxEntries] * inv_d + bf(a.gate_bias, s));
      }
    }
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      const int c = tid + j * kCT;
      if (c < chunks) {
        float4 pr[kMaxStreams];
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s) pr[s] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int e = 0; e < kMaxEntries; ++e) {
          if (e < ne) {
            const float4 p = ldcg4(bank + static_cast<int64_t>(e) * a.entry_stride + D + 4 * c);
#pragma unroll
            for (int s = 0; s < kMaxStreams; ++s) {
              pr[s].x += w[s][e] * p.x;
              pr[s].y += w[s][e] * p.y;
              pr[s].z += w[s][e] * p.z;
              pr[s].w += w[s][e] * p.w;
            }
          }
        }
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s) {
          if (s < S) {
            const float4 p = make_float4(pr[s].x * norm[s], pr[s].y * norm[s], pr[s].z * norm[s], pr[s].w * norm[s]);
            x[j][s].x += gs[s] * p.x;
            x[j][s].y += gs[s] * p.y;
            x[j][s].z += gs[s] * p.z;
            x[j][s].w += gs[s] * p.w;
            acc.x += p.x;
            acc.y += p.y;
            acc.z += p.z;
            acc.w += p.w;
            *reinterpret_cast<float4*>(streams + s * D + 4 * c) = x[j][s];
          }
        }
        *reinterpret_cast<float4*>(a.summary + static_cast<int64_t>(m) * D + 4 * c) =
            make_float4(acc.x / S, acc.y / S, acc.z / S, acc.w / S);
      }
    }
  } else {
    for (int c = tid; c < chunks; c += kCT)
      *reinterpret_cast<float4*>(a.summary + static_cast<int64_t>(m) * D + 4 * c) = make_float4(0.f, 0.f, 0.f, 0.f);
  }

  // The mean over the streams, its sum of squares, and the end-of-pass difference.
  float sums[2] = {0.0f, 0.0f};
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    const int c = tid + j * kCT;
    if (c < chunks) {
      float4 st = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < kMaxStreams; ++s) {
        if (s < S) {
          st.x += x[j][s].x;
          st.y += x[j][s].y;
          st.z += x[j][s].z;
          st.w += x[j][s].w;
        }
      }
      st = make_float4(st.x / S, st.y / S, st.z / S, st.w / S);
      *reinterpret_cast<float4*>(a.state + static_cast<int64_t>(m) * D + 4 * c) = st;
      sums[0] += dot4(st, st);
      if (a.previous) {
        float* prev = a.previous + static_cast<int64_t>(m) * D + 4 * c;
        const float4 p = ldcg4(prev);
        const float4 df = make_float4(st.x - p.x, st.y - p.y, st.z - p.z, st.w - p.w);
        *reinterpret_cast<float4*>(a.difference + static_cast<int64_t>(m) * D + 4 * c) = df;
        *reinterpret_cast<float4*>(prev) = st;
        sums[1] += dot4(df, df);
      }
    }
  }
  if (a.rms) {
    block_sums(sums, 2, red, fin);
    if (tid == 0) a.rms[m] = fin[0];
    if (tid == 1 && a.previous) a.rms[M + m] = fin[1];
  }

  // The next read's normalized input, and the attention block's per-pass LoRA down product.
  if (a.h) {
    float ss[1] = {0.0f};
    float4 src[kCh];
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      src[j] = make_float4(0.f, 0.f, 0.f, 0.f);
      const int c = tid + j * kCT;
      if (c < chunks) {
#pragma unroll
        for (int s = 0; s < kMaxStreams; ++s) {
          if (s < S) {
            const float r = __ldg(a.read + s);
            src[j].x += r * x[j][s].x;
            src[j].y += r * x[j][s].y;
            src[j].z += r * x[j][s].z;
            src[j].w += r * x[j][s].w;
          }
        }
        ss[0] += dot4(src[j], src[j]);
      }
    }
    block_sums(ss, 1, red, fin);
    const float scale = rsqrtf(fin[0] / D + a.eps);
    float lora[kMaxEntries];
#pragma unroll
    for (int r = 0; r < kMaxEntries; ++r) lora[r] = 0.0f;
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      const int c = tid + j * kCT;
      if (c < chunks) {
        const float4 w = bf16x4(static_cast<const bf16*>(a.norm_w) + 4 * c);
        const float4 hv = make_float4(round_bf16(src[j].x * scale * w.x), round_bf16(src[j].y * scale * w.y),
                                      round_bf16(src[j].z * scale * w.z), round_bf16(src[j].w * scale * w.w));
        __nv_bfloat162* dst =
            reinterpret_cast<__nv_bfloat162*>(static_cast<bf16*>(a.h) + static_cast<int64_t>(m) * D + 4 * c);
        dst[0] = __floats2bfloat162_rn(hv.x, hv.y);
        dst[1] = __floats2bfloat162_rn(hv.z, hv.w);
        if (a.lora_down)
#pragma unroll
          for (int r = 0; r < kMaxEntries; ++r)
            if (r < a.rank)
              lora[r] += dot4(hv, bf16x4(static_cast<const bf16*>(a.lora_down) + static_cast<int64_t>(r) * D + 4 * c));
      }
    }
    if (a.lora_down) {
      block_sums(lora, a.rank, red, fin);
      if (tid < a.rank) a.lora[static_cast<int64_t>(m) * a.rank + tid] = round_bf16(fin[tid]);
    }
  }
}

// ---- Mamba-2 step -------------------------------------------------------------------------------

// One (token, head): the conv on the head's x channels and its group's B and C channels, the scan of
// the head's [P, N] state (prefetched into one or two pages by the loader), and v = y * silu(z).
template <int NI>
__device__ __noinline__ void ssm_task(int task, int pg0, int pg1) {
  const Ssm& a = desc().ssm;
  const char* page0 = dsm + pg0;
  const char* page1 = dsm + pg1;
  const int tid = threadIdx.x;
  const int H = a.H, Pd = a.P, N = a.N, G = a.G, I = a.I, W = a.W;
  const int m = task / H, h = task % H;
  const int slot = ctx().slots[m];
  const bool has = ctx().has_state[m] != 0;
  const int g = h / (H / G), gn = G * N, conv_dim = I + 2 * gn;
  const float* zrow = a.zxbcdt + static_cast<int64_t>(m) * a.ld;
  float* xc = scratch();  // [Pd] conv outputs of the x channels, then [N] B, [N] C
  float* Bc = xc + Pd;
  float* Cc = Bc + N;
  float* vv = Cc + N;  // [Pd] v^2
  float* cst = a.conv_state + static_cast<int64_t>(slot) * (W - 1) * conv_dim;
  const int nch = Pd + 2 * N;
  for (int j = tid; j < nch; j += kCT) {
    const int c = j < Pd ? h * Pd + j : (j < Pd + N ? I + g * N + (j - Pd) : I + gn + g * N + (j - Pd - N));
    const float xn = __ldcg(zrow + I + c);
    float acc = bf(a.conv_b, c);
    float old[8];
#pragma unroll
    for (int k = 0; k < 7; ++k) {
      if (k < W - 1) {
        old[k] = has ? __ldcg(cst + k * conv_dim + c) : 0.0f;
        acc = fmaf(bf(a.conv_w, static_cast<int64_t>(c) * W + k), old[k], acc);
      }
    }
    acc = fmaf(bf(a.conv_w, static_cast<int64_t>(c) * W + W - 1), xn, acc);
    const float out = lx::silu(acc);
    if (j < Pd) {
      xc[j] = out;
      // The x channels belong to this head alone, so their conv state moves on here; the group's B and
      // C channels move on in the op after the step, once every head of the group has read them.
#pragma unroll
      for (int k = 0; k < 6; ++k)
        if (k < W - 2) cst[k * conv_dim + c] = old[k + 1];
      cst[(W - 2) * conv_dim + c] = xn;
    } else if (j < Pd + N) {
      Bc[j - Pd] = out;
    } else {
      Cc[j - Pd - N] = out;
    }
  }
  csync();
  const float dt = lx::softplus(__ldcg(zrow + I + conv_dim + h) + a.dt_bias[h]);
  const float decay = expf(dt * -expf(a.A_log[h]));
  const float Dh = a.D[h];
  const int lpr = kCT / Pd;  // lanes per state row
  const int p = tid / lpr, q = tid % lpr;
  const int rpp = kPageData / (N * 4);  // state rows per page
  const float* srow = reinterpret_cast<const float*>(p < rpp ? page0 + p * N * 4 : page1 + (p - rpp) * N * 4);
  float S[NI][4];
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const float4 v = has ? *reinterpret_cast<const float4*>(srow + 4 * (q + lpr * i)) : make_float4(0.f, 0.f, 0.f, 0.f);
    S[i][0] = v.x;
    S[i][1] = v.y;
    S[i][2] = v.z;
    S[i][3] = v.w;
  }
  const float xp = xc[p], dtx = dt * xp;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int c0 = 4 * (q + lpr * i);
    const float4 b4 = *reinterpret_cast<const float4*>(Bc + c0);
    const float4 c4 = *reinterpret_cast<const float4*>(Cc + c0);
    const float bv[4] = {b4.x, b4.y, b4.z, b4.w};
    const float cv[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      S[i][e] = fmaf(decay, S[i][e], dtx * bv[e]);
      acc[e] = fmaf(S[i][e], cv[e], acc[e]);
    }
  }
  float y = (acc[0] + acc[1]) + (acc[2] + acc[3]);
  for (int o = 1; o < lpr; o <<= 1) y += __shfl_xor_sync(0xffffffffu, y, o);
  y = y + Dh * xp;
  float* gst = a.ssm_state + (static_cast<int64_t>(slot) * H + h) * Pd * N + static_cast<int64_t>(p) * N;
#pragma unroll
  for (int i = 0; i < NI; ++i)
    __stcg(reinterpret_cast<float4*>(gst + 4 * (q + lpr * i)), make_float4(S[i][0], S[i][1], S[i][2], S[i][3]));
  if (q == 0) {
    const float v = y * lx::silu(__ldcg(zrow + h * Pd + p));
    a.v[static_cast<int64_t>(m) * I + h * Pd + p] = v;
    vv[p] = v * v;
  }
  csync();
  if (tid == 0) {
    float s = 0.0f;
    for (int i = 0; i < Pd; ++i) s += vv[i];
    a.head_ss[m * H + h] = s;
  }
}

// The conv state of a token's B and C channels moves on by one input, after every head has read it.
__device__ __noinline__ void conv_bc_task(int m) {
  const Ssm& a = desc().ssm;
  const int gn = a.G * a.N, conv_dim = a.I + 2 * gn, W = a.W;
  const bool has = ctx().has_state[m] != 0;
  float* cst = a.conv_state + static_cast<int64_t>(ctx().slots[m]) * (W - 1) * conv_dim;
  const float* zrow = a.zxbcdt + static_cast<int64_t>(m) * a.ld;
  for (int c = a.I + threadIdx.x; c < conv_dim; c += kCT) {
    float old[8];
#pragma unroll
    for (int k = 0; k < 7; ++k)
      if (k < W - 1) old[k] = has ? __ldcg(cst + k * conv_dim + c) : 0.0f;
#pragma unroll
    for (int k = 0; k < 6; ++k)
      if (k < W - 2) cst[k * conv_dim + c] = old[k + 1];
    cst[(W - 2) * conv_dim + c] = __ldcg(zrow + a.I + c);
  }
}

// ---- Routing ------------------------------------------------------------------------------------

// lx_moe_route for every token (one warp each), then the expert summary and the history.
__device__ __noinline__ void select_task() {
  const Select& a = desc().sel;
  const int M = ctx().M;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int* picks = reinterpret_cast<int*>(scratch());  // [kMaxM][32]
  float* pw = scratch() + kMaxM * 32;               // [kMaxM][32]
  const int E = a.E, top_k = a.top_k;
  if (warp < M) {
    const int t = warp;
    const float* row = a.logits + static_cast<int64_t>(t) * E;
    float mx = -INFINITY;
    for (int i = lane; i < E; i += 32) mx = fmaxf(mx, __ldcg(row + i));
    mx = lx::warp_max(mx);
    float z = 0.0f;
    for (int i = lane; i < E; i += 32) z += expf(__ldcg(row + i) - mx);
    z = lx::warp_sum(z);
    float hs = 0.0f;
    for (int i = lane; i < E; i += 32) {
      const float pr = expf(__ldcg(row + i) - mx) / z;
      hs -= pr * logf(fmaxf(pr, 1e-9f));
    }
    hs = lx::warp_sum(hs);
    if (lane == 0) {
      a.entropy[t] = hs;
      a.confidence[t] = 1.0f / z;
    }
    int* pick = picks + t * 32;
    for (int r = 0; r < top_k; ++r) {
      float best = -INFINITY;
      int best_i = E;
      for (int i = lane; i < E; i += 32) {
        bool taken = false;
        for (int j = 0; j < r; ++j) taken |= pick[j] == i;
        if (taken) continue;
        const float v = __ldcg(row + i) + (a.selection_bias ? a.selection_bias[i] : 0.0f);
        if (v > best || (v == best && i < best_i)) {
          best = v;
          best_i = i;
        }
      }
      for (int o = 16; o > 0; o >>= 1) {
        const float ov = __shfl_xor_sync(0xffffffffu, best, o);
        const int oi = __shfl_xor_sync(0xffffffffu, best_i, o);
        if (ov > best || (ov == best && oi < best_i)) {
          best = ov;
          best_i = oi;
        }
      }
      if (lane == 0) pick[r] = best_i;
      __syncwarp();
    }
    const bool live = lane < top_k;
    const int id = live ? pick[lane] : 0;
    const float l = live ? __ldcg(row + id) : -INFINITY;
    const float lmax = lx::warp_max(l);
    const float e = live ? expf(l - lmax) : 0.0f;
    const float esum = lx::warp_sum(e);
    if (live) {
      pw[t * 32 + lane] = e / esum;
      a.ids[static_cast<int64_t>(t) * top_k + lane] = id;
      a.weights[static_cast<int64_t>(t) * top_k + lane] = e / esum;
    }
  }
  csync();
  const int R = a.R;
  for (int idx = tid; idx < M * R; idx += kCT) {
    const int t = idx / R, r = idx % R;
    float acc = 0.0f;
    for (int k = 0; k < top_k; ++k)
      acc += pw[t * 32 + k] * bf(a.expert_emb, static_cast<int64_t>(picks[t * 32 + k]) * R + r);
    a.summary[idx] = acc;
    a.history[idx] = 0.5f * __ldcg(a.history + idx) + 0.5f * acc;
  }
}

// ---- Experts ------------------------------------------------------------------------------------

// Stages count bf16 values (16-byte aligned) into the x buffer.
__device__ __forceinline__ void stage_bf16(const void* src, int count) {
  const uint4* s = static_cast<const uint4*>(src);
  uint4* d = reinterpret_cast<uint4*>(xbuf());
  for (int i = threadIdx.x; i < count / 8; i += kCT) d[i] = __ldcg(s + i);
  csync();
}

// Gate and up rows [i0, i0 + rows) of one expert: routed (task = pair * blocks_per_pair + row block,
// x = the pair's token) or shared (task = row block, every token).
template <int kType>
__device__ __noinline__ void expup_task(int op, int task, int pg) {
  using F = typename Word<kType>::F;
  const ExpUp& a = desc().up;
  const int M = ctx().M;
  const char* page = dsm + pg;
  const int rb = task % a.blocks_per_pair, pair = task / a.blocks_per_pair, t = pair / a.top_k;
  const int i0 = rb * a.rows, rows = a.rows;
  const int mc = 1;
  (void)M;
  if (misc() != op * 16 + t + 1) {  // the pair's token's latent row
    stage_bf16(static_cast<const bf16*>(a.latent) + static_cast<int64_t>(t) * a.L, a.L);
    if (threadIdx.x == 0) misc() = op * 16 + t + 1;
  }
  const float* xs = xbuf();
  float* red = scratch();
  const int wpr = tile_dots_m<F, true>(reinterpret_cast<const uint2*>(page), 2 * rows, a.row_bytes / 8,
                                       a.L / F::kValues, xs, a.L, mc, 0, red);
  csync();
  for (int e = threadIdx.x; e < rows * mc; e += kCT) {
    const int i = e / mc, j = e % mc;
    float gt = sum_parts(red, i, wpr, mc, j), up = sum_parts(red, rows + i, wpr, mc, j);
    if constexpr (kRowScaled<kType>) {
      const float* sc = reinterpret_cast<const float*>(page + a.scale_off);
      gt *= sc[i];
      up *= sc[rows + i];
    }
    (void)j;
    static_cast<bf16*>(a.inter)[static_cast<int64_t>(pair) * a.I + i0 + i] = __float2bfloat16_rn(lx::silu(gt) * up);
  }
}

// Down rows [l0, l0 + rows) of each of token t's experts, weighted and summed (plus the shared expert).
template <int kType>
__device__ __noinline__ void expdown_task(int op, int task, int pg) {
  using F = typename Word<kType>::F;
  const ExpDown& a = desc().down;
  const char* page = dsm + pg;
  const int t = task / a.blocks_per_token, rb = task % a.blocks_per_token;
  const int l0 = rb * a.rows, rows = a.rows, K = a.top_k;
  if (misc() != op * 16 + t + 1) {
    stage_bf16(static_cast<const bf16*>(a.inter) + static_cast<int64_t>(t) * K * a.I, K * a.I);
    if (threadIdx.x == 0) misc() = op * 16 + t + 1;
  }
  float* red = scratch();
  const int wpr = tile_dots_m<F, true>(reinterpret_cast<const uint2*>(page), K * rows, a.row_bytes / 8,
                                       a.I / F::kValues, xbuf(), a.I, 1, rows, red);
  csync();
  for (int i = threadIdx.x; i < rows; i += kCT) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float c = __ldcg(a.route_w + static_cast<int64_t>(t) * K + k);
      if constexpr (kRowScaled<kType>) c *= reinterpret_cast<const float*>(page + a.scale_off)[k * rows + i];
      acc += c * sum_parts(red, k * rows + i, wpr, 1, 0);
    }
    const int64_t at = static_cast<int64_t>(t) * a.L + l0 + i;
    if (a.add) acc += __ldcg(a.add + at);
    a.out[at] = acc;
  }
}

// ---- Attention ----------------------------------------------------------------------------------

// One (token, kv head): writes the token's k and v to the caches, then attends over positions
// 0 .. start_pos in chunks of kCT positions with an online softmax across chunks.
__device__ __noinline__ void attention_task(int task) {
  const Attention& a = desc().attn;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int KVH = a.KVH, Dh = a.Dh, H = a.H, G = H / KVH;
  const int m = task / KVH, kvh = task % KVH;
  const int pos = ctx().start_pos[m], L = pos + 1;
  const int64_t slot = ctx().slots[m];
  const float* row = a.qkv + static_cast<int64_t>(m) * a.ld_qkv;
  const int64_t pstride = static_cast<int64_t>(KVH) * Dh;
  bf16* kc = static_cast<bf16*>(a.k_cache) + slot * a.capacity * pstride + kvh * Dh;
  bf16* vc = static_cast<bf16*>(a.v_cache) + slot * a.capacity * pstride + kvh * Dh;
  float* qs = xbuf();               // [G][Dh], scaled
  float* sc = qs + G * Dh;        // [kCT][G] scores, then weights; reused by the final merge
  float* run = scratch() + 256;  // running max [G], running sum [G], new max [G], correction [G]
  for (int i = tid; i < G * Dh; i += kCT) qs[i] = __ldcg(row + kvh * G * Dh + i) * a.scale;
  for (int i = tid; i < Dh; i += kCT) {
    kc[pos * pstride + i] = __float2bfloat16_rn(__ldcg(row + H * Dh + kvh * Dh + i));
    vc[pos * pstride + i] = __float2bfloat16_rn(__ldcg(row + H * Dh + KVH * Dh + kvh * Dh + i));
  }
  if (tid < G) {
    run[tid] = -INFINITY;
    run[G + tid] = 0.0f;
  }
  // Other threads of this block read the cache rows written above: the fence and barrier order them.
  __threadfence_block();
  csync();
  const int pg_count = kCT / Dh;  // position groups for the value sum
  const int d = tid % Dh, pg = tid / Dh;
  float acc[8];
#pragma unroll
  for (int g = 0; g < 8; ++g) acc[g] = 0.0f;
  float* red = scratch();
  for (int p0 = 0; p0 < L; p0 += kCT) {
    const int len = min(kCT, L - p0);
    float mx[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) mx[g] = -INFINITY;
    if (tid < len) {
      const bf16* kr = kc + static_cast<int64_t>(p0 + tid) * pstride;
      float s[8];
#pragma unroll
      for (int g = 0; g < 8; ++g) s[g] = 0.0f;
      for (int i = 0; i < Dh; i += 8) {
        const uint4 u = __ldcg(reinterpret_cast<const uint4*>(kr + i));
        const float kv[8] = {__uint_as_float(u.x << 16), __uint_as_float(u.x & 0xffff0000u),
                             __uint_as_float(u.y << 16), __uint_as_float(u.y & 0xffff0000u),
                             __uint_as_float(u.z << 16), __uint_as_float(u.z & 0xffff0000u),
                             __uint_as_float(u.w << 16), __uint_as_float(u.w & 0xffff0000u)};
#pragma unroll
        for (int g = 0; g < 8; ++g)
          if (g < G)
#pragma unroll
            for (int e = 0; e < 8; ++e) s[g] = fmaf(qs[g * Dh + i + e], kv[e], s[g]);
      }
#pragma unroll
      for (int g = 0; g < 8; ++g)
        if (g < G) {
          sc[tid * G + g] = s[g];
          mx[g] = s[g];
        }
    }
#pragma unroll
    for (int g = 0; g < 8; ++g)
      if (g < G) mx[g] = lx::warp_max(mx[g]);
    if (lane == 0)
      for (int g = 0; g < G; ++g) red[warp * 8 + g] = mx[g];
    csync();
    if (tid < G) {
      float cm = -INFINITY;
      for (int w = 0; w < kComputeWarps; ++w) cm = fmaxf(cm, red[w * 8 + tid]);
      const float nm = fmaxf(run[tid], cm);
      run[2 * G + tid] = nm;
      run[3 * G + tid] = run[tid] == -INFINITY ? 0.0f : expf(run[tid] - nm);
    }
    csync();
    float ps[8];
#pragma unroll
    for (int g = 0; g < 8; ++g) ps[g] = 0.0f;
    if (tid < len)
      for (int g = 0; g < G; ++g) {
        const float wv = expf(sc[tid * G + g] - run[2 * G + g]);
        sc[tid * G + g] = wv;
        ps[g] = wv;
      }
#pragma unroll
    for (int g = 0; g < 8; ++g)
      if (g < G) ps[g] = lx::warp_sum(ps[g]);
    if (lane == 0)
      for (int g = 0; g < G; ++g) red[warp * 8 + g] = ps[g];
    csync();
    for (int g = 0; g < G; ++g) acc[g] *= run[3 * G + g];
    for (int p = pg; p < len; p += pg_count) {
      const float v = __bfloat162float(__ldcg(vc + static_cast<int64_t>(p0 + p) * pstride + d));
#pragma unroll
      for (int g = 0; g < 8; ++g)
        if (g < G) acc[g] = fmaf(sc[p * G + g], v, acc[g]);
    }
    csync();
    if (tid < G) {
      float cs = 0.0f;
      for (int w = 0; w < kComputeWarps; ++w) cs += red[w * 8 + tid];
      run[G + tid] = run[G + tid] * run[3 * G + tid] + cs;
      run[tid] = run[2 * G + tid];
    }
    csync();
  }
  float* mg = sc;  // [pg_count][G][Dh]
  for (int g = 0; g < G; ++g) mg[(pg * G + g) * Dh + d] = acc[g];
  csync();
  bf16* out = static_cast<bf16*>(a.out) + static_cast<int64_t>(m) * H * Dh + kvh * G * Dh;
  for (int i = tid; i < G * Dh; i += kCT) {
    const int g = i / Dh, dd = i % Dh;
    float s = 0.0f;
    for (int q = 0; q < pg_count; ++q) s += mg[(q * G + g) * Dh + dd];
    out[i] = __float2bfloat16_rn(s / run[G + g]);
  }
}

// ---- Depth-memory entry and the depth head ------------------------------------------------------

template <int kType>
__device__ __noinline__ void entry_task(int op, int task, int pg) {
  using F = typename Word<kType>::F;
  const Entry& a = desc().entry;
  const int M = ctx().M;
  const char* page = dsm + pg;
  const int D = a.D, Km = a.R + 4, kc = a.kc;
  float* meta = xbuf();         // [M, Km], bf16 values in f32
  float* xs = xbuf() + M * Km;  // [M, kc] f32: the state, a chunk at a time
  const bool whole = kc == D;
  const bool cached = misc() == op * 16;
  if (!cached) {
    for (int i = threadIdx.x; i < M * Km; i += kCT) {
      const int m = i / Km, r = i % Km;
      float v;
      if (r < a.R) v = __ldcg(a.summary + static_cast<int64_t>(m) * a.R + r);
      else if (r == a.R) v = a.mean_k;
      else if (r == a.R + 1) v = __ldcg(a.entropy + m) * a.inv_log_e;
      else if (r == a.R + 2) v = __ldcg(a.confidence + m);
      else v = __ldcg(a.continuation + m);
      meta[i] = round_bf16(v);
    }
  }
  const int r0 = task * a.rows, rows = min(a.rows, a.mem_dim - r0);
  float* red = scratch();
  float* red2 = scratch() + 256;
  int wpr1 = 1;
  for (int c0 = 0; c0 < D; c0 += kc) {
    if (!(whole && cached)) {
      if (c0 > 0) csync();  // the previous chunk's dots are done with the buffer
      const int q4 = kc / 4;
      for (int i = threadIdx.x; i < M * q4; i += kCT) {
        const int m = i / q4, j = (i % q4) * 4;
        *reinterpret_cast<float4*>(xs + m * kc + j) = ldcg4(a.state + static_cast<int64_t>(m) * D + c0 + j);
      }
      csync();
    }
    wpr1 = tile_dots_m<F, false>(reinterpret_cast<const uint2*>(page) + c0 / F::kValues, rows, a.state_bytes / 8,
                                 kc / F::kValues, xs, kc, M, 0, red, c0 > 0);
  }
  if (whole && threadIdx.x == 0) misc() = op * 16;
  const int wpr2 = tile_dots_m<WordFormat<LX_W_BF16>, false>(reinterpret_cast<const uint2*>(page + a.meta_off), rows,
                                                             Km * 2 / 8, Km / 4, meta, Km, M, 0, red2);
  csync();
  for (int e = threadIdx.x; e < rows * M; e += kCT) {
    const int i = e / M, m = e % M, r = r0 + i;
    float ds = sum_parts(red, i, wpr1, M, m);
    if constexpr (kRowScaled<kType>) ds *= reinterpret_cast<const float*>(page + a.scale_off)[i];
    const float v = ds + bf(a.b_state, r);
    const float vm = sum_parts(red2, i, wpr2, M, m) + bf(a.b_meta, r);
    float en = vm + v;
    en += bf(a.pass_emb, r);
    en += bf(a.anchor_emb, r);
    a.entry[static_cast<int64_t>(m) * a.mem_dim + r] = en;
  }
}

__device__ __noinline__ void continuation_task() {
  const Continuation& a = desc().cont;
  const int M = ctx().M;
  float* red = scratch();
  float* fin = scratch() + kComputeWarps * 40;
  float part[kMaxM];
#pragma unroll
  for (int m = 0; m < kMaxM; ++m) part[m] = 0.0f;
  for (int j = threadIdx.x; j < a.R; j += kCT) {
    const float w = a.w_out[j];
#pragma unroll
    for (int m = 0; m < kMaxM; ++m)
      if (m < M) part[m] += w * __ldcg(a.hidden + static_cast<int64_t>(m) * a.R + j);
  }
  block_sums(part, M, red, fin);
  if (threadIdx.x < M) {
    const int m = threadIdx.x;
    const float logit = fin[m] + a.b_out[0];
    const float scale = fmaxf(sqrtf(__ldcg(a.rms + m) / a.D), 1e-6f);
    const float innovation = sqrtf(__ldcg(a.rms + M + m) / a.D);
    const float z = fminf(fmaxf(logit + tanhf(innovation / scale) - 0.5f, -a.limit), a.limit);
    a.out[m] = sigmoid(z);
  }
}

// ---- The loader warp ----------------------------------------------------------------------------

// Calls copy(src, page, offset, bytes, count, src_stride, dst_stride) for each copy that fills a
// task's pages: count rows of bytes each, src_stride apart in global memory and dst_stride apart in
// the page. The tile layouts here are the ones the task bodies above read.
template <class Copy>
__device__ __forceinline__ void tile_copies(const OpDesc& d, int task, Copy&& copy) {
  switch (d.kind) {
    case kGemv: {
      const Gemv& g = d.gemv;
      const int n0 = task * g.rows, rows = min(g.rows, g.N - n0);
      const int bpv = g.wtype == LX_W_BF16 ? 2 : g.wtype == kF32 ? 4 : 1;
      const char* base = static_cast<const char*>(g.w.data);
      if (g.k0 == 0 && g.k1 == g.K) copy(base + static_cast<int64_t>(n0) * g.row_bytes, 0, 0, rows * g.row_bytes, 1, 0, 0);
      else
        copy(base + (static_cast<int64_t>(n0) * g.K + g.k0) * bpv, 0, 0, g.row_bytes, rows,
             static_cast<int64_t>(g.K) * bpv, g.row_bytes);
      if (row_scaled(g.wtype))
        copy(static_cast<const char*>(g.w.scale) + static_cast<int64_t>(n0) * 4, 0, g.scale_off, rows * 4, 1, 0, 0);
      break;
    }
    case kSsm: {
      const Ssm& a = d.ssm;
      const int m = task / a.H, h = task % a.H;
      if (ctx().has_state[m]) {
        const int bytes = a.P * a.N * 4;
        const char* src =
            reinterpret_cast<const char*>(a.ssm_state + (static_cast<int64_t>(ctx().slots[m]) * a.H + h) * a.P * a.N);
        copy(src, 0, 0, min(bytes, kPageData), 1, 0, 0);
        if (bytes > kPageData) copy(src + kPageData, 1, 0, bytes - kPageData, 1, 0, 0);
      }
      break;
    }
    case kExpUp: {
      const ExpUp& a = d.up;
      const int rb = task % a.blocks_per_pair;
      const int e = __ldcg(a.ids + task / a.blocks_per_pair);
      const int64_t row0 = static_cast<int64_t>(e) * 2 * a.I + rb * a.rows;  // the first gate row
      // gate rows, then the up rows I further on
      copy(static_cast<const char*>(a.w.data) + row0 * a.row_bytes, 0, 0, a.rows * a.row_bytes, 2,
           static_cast<int64_t>(a.I) * a.row_bytes, a.rows * a.row_bytes);
      if (row_scaled(a.wtype))
        copy(static_cast<const char*>(a.w.scale) + row0 * 4, 0, a.scale_off, a.rows * 4, 2, static_cast<int64_t>(a.I) * 4,
             a.rows * 4);
      break;
    }
    case kExpDown: {
      const ExpDown& a = d.down;
      const int t = task / a.blocks_per_token, l0 = (task % a.blocks_per_token) * a.rows;
      for (int k = 0; k < a.top_k; ++k) {
        const int64_t row0 = static_cast<int64_t>(__ldcg(a.ids + static_cast<int64_t>(t) * a.top_k + k)) * a.L + l0;
        copy(static_cast<const char*>(a.w.data) + row0 * a.row_bytes, 0, k * a.rows * a.row_bytes, a.rows * a.row_bytes,
             1, 0, 0);
        if (row_scaled(a.wtype))
          copy(static_cast<const char*>(a.w.scale) + row0 * 4, 0, a.scale_off + k * a.rows * 4, a.rows * 4, 1, 0, 0);
      }
      break;
    }
    case kEntry: {
      const Entry& a = d.entry;
      const int r0 = task * a.rows, rows = min(a.rows, a.mem_dim - r0);
      copy(static_cast<const char*>(a.w_state.data) + static_cast<int64_t>(r0) * a.state_bytes, 0, 0,
           rows * a.state_bytes, 1, 0, 0);
      if (row_scaled(a.wtype))
        copy(static_cast<const char*>(a.w_state.scale) + static_cast<int64_t>(r0) * 4, 0, a.scale_off, rows * 4, 1, 0, 0);
      const int mb = (a.R + 4) * 2;
      copy(static_cast<const char*>(a.w_meta) + static_cast<int64_t>(r0) * mb, 0, a.meta_off, rows * mb, 1, 0, 0);
      break;
    }
    default: break;
  }
}

__device__ __forceinline__ bool aligned16(const char* src, int offset, int bytes, int64_t src_stride, int dst_stride) {
  return ((reinterpret_cast<uintptr_t>(src) | static_cast<uintptr_t>(offset) | static_cast<uintptr_t>(bytes) |
           static_cast<uintptr_t>(src_stride) | static_cast<uintptr_t>(dst_stride)) & 15) == 0;
}

template <int kG>
__device__ __forceinline__ void cp_rows(char* dst, const char* src, int bytes, int count, int64_t src_stride,
                                        int dst_stride, int lane, uint64_t policy) {
  const int per = bytes / kG;
  for (int r = 0; r < count; ++r)
    for (int c = lane; c < per; c += 32)
      cp_async<kG>(dst + r * dst_stride + c * kG, src + r * src_stride + c * kG, policy);
}

// Fills a task's pages (all 32 lanes). With bulk copies: lane 0 first tells each page's barrier how
// many bytes to expect, then the lanes issue one bulk copy per row. Copies a bulk copy cannot take
// (not 16-byte aligned), and every copy on sm_80, go through cp.async, which the barrier waits for
// too. Lane 0's arrival completes the setup. d is the op's descriptor in global memory.
__device__ __noinline__ void issue_tile(int slot, int task, int page0, int page1, uint64_t policy, int bulk_on) {
  const OpDesc& d = loader_desc(slot);
  const int lane = threadIdx.x & 31;
  const bool bulk = bulk_supported() && bulk_on;
  const int npages = d.npages;
  uint64_t* bar0 = full_bar(page0);
  uint64_t* bar1 = full_bar(page1);
  const int pages_off = ctx().pages_off;
  if (bulk) {
    if (lane == 0) {
      uint32_t b0 = 0, b1 = 0;
      tile_copies(d, task, [&](const char* src, int page, int off, int bytes, int count, int64_t ss, int ds) {
        if (aligned16(src, off, bytes, ss, ds)) (page ? b1 : b0) += static_cast<uint32_t>(bytes) * count;
      });
      if (b0) mbar_expect_tx(bar0, b0);
      if (b1) mbar_expect_tx(bar1, b1);
    }
    __syncwarp();
  }
  bool any_cp = false;
  tile_copies(d, task, [&](const char* src, int page, int off, int bytes, int count, int64_t ss, int ds) {
    char* dst = dsm + pages_off + (page ? page1 : page0) * kPageBytes + off;
    if (bulk && aligned16(src, off, bytes, ss, ds)) {
      for (int r = lane; r < count; r += 32)
        bulk_copy(dst + r * ds, src + r * ss, static_cast<uint32_t>(bytes), page ? bar1 : bar0, policy);
      return;
    }
    any_cp = true;
    const uintptr_t al = reinterpret_cast<uintptr_t>(src) | static_cast<uintptr_t>(off) | static_cast<uintptr_t>(bytes) |
                         static_cast<uintptr_t>(ss) | static_cast<uintptr_t>(ds);
    if (al % 16 == 0) cp_rows<16>(dst, src, bytes, count, ss, ds, lane, policy);
    else if (al % 8 == 0) cp_rows<8>(dst, src, bytes, count, ss, ds, lane, policy);
    else cp_rows<4>(dst, src, bytes, count, ss, ds, lane, policy);
  });
  if (any_cp) {
    cp_async_arrive(bar0);
    if (npages > 1) cp_async_arrive(bar1);
  }
  __syncwarp();
  if (lane == 0) {
    mbar_arrive(bar0);
    if (npages > 1) mbar_arrive(bar1);
  }
}

// Asks L2 for a static tile's bytes ahead of its turn in the ring, so DRAM keeps streaming while the
// ring is full and the compute warps wait on a dependency: when the tile's page frees up, its copy
// is served from L2. Rows go out across the lanes, one bulk prefetch each (or a prefetch per line).
__device__ __noinline__ void prefetch_tile(int task, int bulk_on) {
  const OpDesc& d = loader_desc(2);
  const int lane = threadIdx.x & 31;
  const bool bulk = bulk_supported() && bulk_on;
  tile_copies(d, task, [&](const char* src, int, int, int bytes, int count, int64_t ss, int) {
    for (int r = lane; r < count; r += 32) {
      const char* row = src + r * ss;
      const uintptr_t a0 = reinterpret_cast<uintptr_t>(row) & ~static_cast<uintptr_t>(127);
      const uintptr_t a1 = reinterpret_cast<uintptr_t>(row) + bytes;
      if (bulk) bulk_prefetch_l2(reinterpret_cast<const void*>(a0), static_cast<uint32_t>((a1 - a0 + 15) & ~15u));
      else
        for (uintptr_t a = a0; a < a1; a += 128) prefetch_l2_line(reinterpret_cast<const void*>(a));
    }
  });
}

// Copies op's descriptor into the loader's slot (all 32 lanes, 16 bytes each). The lanes meet
// first: a lane may still be reading the slot's previous descriptor (issue_tile, prefetch_tile).
__device__ __forceinline__ void load_desc(const Program& P, int op, int slot) {
  const int lane = threadIdx.x & 31;
  const int4* src = reinterpret_cast<const int4*>(P.ops + op);
  int4* dst = reinterpret_cast<int4*>(dsm + kLoaderDescOff + slot * kDescBytes);
  __syncwarp();
  for (int j = lane; j < static_cast<int>(sizeof(OpDesc) / 16); j += 32) dst[j] = __ldg(src + j);
  __syncwarp();
}

// Walks the block's static and dynamic tiles in order, each into its own ring, as pages free up.
// Static tiles go as soon as a page is free; a dynamic op's tiles also wait for the op that routes
// them (dyn_dep).
__device__ __forceinline__ void loader(const Program& P, int block) {
  const int lane = threadIdx.x & 31;
  const uint32_t* sl = P.static_lists + P.static_off[block];
  const uint32_t* dl = P.dynamic_lists + P.dynamic_off[block];
  const int ns = P.static_off[block + 1] - P.static_off[block];
  const int nd = P.dynamic_off[block + 1] - P.dynamic_off[block];
  const int PS = P.pages_static, PD = P.pages_dynamic;
  const uint64_t policy = evict_first_policy();
  int si = 0, di = 0, s_op = -1, d_op = -1, s_np = 1, d_np = 1;
  int pi = 0, p_op = -1;  // the prefetch cursor, up to P.prefetch tiles ahead of the static one
  bool d_ready = false;
  uint32_t sseq = 0, dseq = 0;
  while (si < ns || di < nd) {
    bool progressed = false;
    if (si < ns) {
      const uint32_t e = __ldg(sl + si);
      const int op = static_cast<int>(e >> 16);
      if (op != s_op) {
        load_desc(P, op, 0);
        s_op = op;
        s_np = loader_desc(0).npages;
      }
      const int p0 = static_cast<int>(sseq % PS), p1 = static_cast<int>((sseq + 1) % PS);
      int ok = 1;
      if (lane == 0) {
        ok = mbar_test(empty_bar(p0), ((sseq / PS) & 1) ^ 1);
        if (s_np > 1) ok &= mbar_test(empty_bar(p1), (((sseq + 1) / PS) & 1) ^ 1);
      }
      if (__shfl_sync(0xffffffffu, ok, 0)) {
        issue_tile(0, e & 0xffff, p0, p1, policy, P.bulk);
        sseq += s_np;
        ++si;
        progressed = true;
      } else if (pi < si + P.prefetch) {
        // The ring is full. If even its newest tile has landed, this SM's DRAM stream has stopped
        // (the compute warps are waiting on a dependency): have L2 fetch the next tiles meanwhile,
        // so their copies are quick when pages free up. While copies are still in flight, prefetches
        // would only compete with them.
        int idle = 0;
        if (lane == 0) idle = mbar_test(full_bar(static_cast<int>((sseq - 1) % PS)), ((sseq - 1) / PS) & 1);
        if (__shfl_sync(0xffffffffu, idle, 0)) {
          if (pi < si) pi = si;
          for (int k = 0; k < 2 && pi < ns && pi < si + P.prefetch; ++k, ++pi) {
            const uint32_t pe = __ldg(sl + pi);
            const int pop = static_cast<int>(pe >> 16);
            if (pop != p_op) {
              load_desc(P, pop, 2);
              p_op = pop;
            }
            prefetch_tile(pe & 0xffff, P.bulk);
          }
        }
      }
    }
    if (di < nd) {
      const uint32_t e = __ldg(dl + di);
      const int op = static_cast<int>(e >> 16);
      if (op != d_op) {
        load_desc(P, op, 1);
        d_op = op;
        d_np = loader_desc(1).npages;
        d_ready = false;
      }
      const int p0 = PS + static_cast<int>(dseq % PD), p1 = PS + static_cast<int>((dseq + 1) % PD);
      int ok = 1;
      if (lane == 0) {
        ok = mbar_test(empty_bar(p0), ((dseq / PD) & 1) ^ 1);
        if (d_np > 1) ok &= mbar_test(empty_bar(p1), (((dseq + 1) / PD) & 1) ^ 1);
      }
      ok = __shfl_sync(0xffffffffu, ok, 0);
      // Every lane acquires the routing op itself before reading the expert ids it published.
      if (ok && !d_ready) {
        const int dep = loader_desc(1).dyn_dep;
        const uint32_t need = static_cast<uint32_t>(__ldg(&P.ops[dep].nblocks));
        ok = __all_sync(0xffffffffu, ld_acquire(&P.done[dep]) >= need);
        d_ready = ok;
      }
      if (ok) {
        issue_tile(1, e & 0xffff, p0, p1, policy, P.bulk);
        dseq += d_np;
        ++di;
        progressed = true;
      }
    }
    if (!progressed) __nanosleep(32);
  }
}

// ---- The compute warps --------------------------------------------------------------------------

// Runs a block's run of tasks of one op (its descriptor in shared memory): products through their
// own loop, everything else one task at a time (wait for the task's pages, run it, free them).
// Returns the pool's page count after the run.
__device__ __noinline__ uint32_t run_segment(int op, const uint32_t* entries, int count, int index, uint32_t seq) {
  const OpDesc& d = desc();
  const int pool = d.pool, np = pool != kNoPool ? d.npages : 0;
  for (int i = 0; i < count; ++i) {
    const int task = static_cast<int>(__ldg(entries + i) & 0xffff);
    const int64_t t0 = trace_now();
    Pages pg = {0, 0, 0, 0};
    if (np) {
      pg = ring_pages(pool, seq);
      mbar_wait(full_bar(pg.p0), pg.ph0);
      if (np > 1) mbar_wait(full_bar(pg.p1), pg.ph1);
    }
    const int64_t t1 = trace_now();
    const int pg0 = page_offset(pg.p0), pg1 = page_offset(pg.p1);
    switch (d.kind) {
      case kEmbed: embed_task(task); break;
      case kMemRead: memread_task(task); break;
      case kSsm: {
        const int lpr = kCT / d.ssm.P, ni = d.ssm.N / (4 * lpr);
        if (ni == 1) ssm_task<1>(task, pg0, pg1);
        else if (ni == 2) ssm_task<2>(task, pg0, pg1);
        else if (ni == 4) ssm_task<4>(task, pg0, pg1);
        else ssm_task<8>(task, pg0, pg1);
        break;
      }
      case kConvBC: conv_bc_task(task); break;
      case kSelect: select_task(); break;
      case kExpUp:
        if (d.up.wtype == LX_W_INT8) expup_task<LX_W_INT8>(op, task, pg0);
        else expup_task<LX_W_BF16>(op, task, pg0);
        break;
      case kExpDown:
        if (d.down.wtype == LX_W_INT8) expdown_task<LX_W_INT8>(op, task, pg0);
        else expdown_task<LX_W_BF16>(op, task, pg0);
        break;
      case kAttention:
        attention_task(task);
        if (threadIdx.x == 0) misc() = -1;  // the x buffer was its scratch
        break;
      case kEntry:
        if (d.entry.wtype == LX_W_INT8) entry_task<LX_W_INT8>(op, task, pg0);
        else entry_task<LX_W_BF16>(op, task, pg0);
        break;
      case kContinuation: continuation_task(); break;
      default: break;
    }
    csync();
    if (np && threadIdx.x == 0) {
      release_page(pg.p0);
      if (np > 1) release_page(pg.p1);
    }
    trace_task(index + i, op, task, t0, t1);
    seq += np;
  }
  return seq;
}

// The compute warps and the epilogue warp walk the block's list together, one run of an op at a
// time. Before a run: everyone is done with the previous one (the barrier), which is also when the
// previous op's counter is released; then the descriptor, and the wait for the ops this one reads.
__device__ __forceinline__ void work(const Program& P, int block) {
  const int tid = threadIdx.x;
  const bool epilogue = tid >= kComputeThreads;
  const int base = P.list_off[block];
  const uint32_t* list = P.lists + base;
  const int n = P.list_off[block + 1] - base;
  RunState st = {0, 0};
  uint32_t dseq = 0;
  int prev = -1;
  for (int i = 0; i < n;) {
    const int op = static_cast<int>(__ldg(list + i) >> 16);
    int j = i + 1;
    while (j < n && static_cast<int>(__ldg(list + j) >> 16) == op) ++j;
    work_sync();
    // Everything the previous run wrote is visible to this thread through the barrier; the release
    // carries it to the SMs that acquire the counter (CUTLASS's semaphore pattern).
    if (tid == 0 && prev >= 0) red_release_add(&P.done[prev], 1u);
    int64_t* run_rec = P.trace && tid == 0 ? P.trace + static_cast<int64_t>(P.list_off[gridDim.x] + base + i) * 4 : nullptr;
    if (run_rec) run_rec[0] = global_ns();
    if (!epilogue) {
      const int4* src = reinterpret_cast<const int4*>(P.ops + op);
      int4* dst = reinterpret_cast<int4*>(dsm + kDescOff);
      for (int k = tid; k < static_cast<int>(sizeof(OpDesc) / 16); k += kCT) dst[k] = __ldg(src + k);
    }
    if (tid < 32) {
      // a lane per dependency, so they are waited for together; the barrier after it hands what
      // each lane acquired to the whole block
      const OpDesc& od = P.ops[op];
      for (int k = tid; k < od.ndeps; k += 32) {
        const int dep = __ldg(&od.deps[k]);
        const uint32_t need = static_cast<uint32_t>(__ldg(&P.ops[dep].nblocks));
        while (ld_acquire(&P.done[dep]) < need) {
        }
      }
      __syncwarp();
      if (run_rec) run_rec[1] = global_ns();
    }
    work_sync();
    const int kind = desc().kind, pool = desc().pool;
    if (kind == kGemv) {
      if (epilogue) st.gt = gemv_epilogue(op, list + i, j - i, i, st.gt);
      else st = gemv_compute(op, list + i, j - i, st, run_rec);
    } else if (!epilogue) {
      if (pool == kDynamic) dseq = run_segment(op, list + i, j - i, i, dseq);
      else st.seq = run_segment(op, list + i, j - i, i, st.seq);
    }
    prev = op;
    i = j;
  }
  work_sync();
  if (tid == 0) {
    if (prev >= 0) red_release_add(&P.done[prev], 1u);
    // The last block out resets the counters for the next step.
    __threadfence();
    const uint32_t blocks_done = atomicAdd(&P.done[P.nops], 1u);
    if (blocks_done == gridDim.x - 1) {
      for (int k = 0; k <= P.nops; ++k) P.done[k] = 0;
      __threadfence();
    }
  }
}

}  // namespace

__global__ void __launch_bounds__(kBlockThreads, 1) decode_step_kernel(const __grid_constant__ Program P) {
  const int npages = P.pages_static + P.pages_dynamic;
  if (threadIdx.x == 0) {
    Ctx* c = reinterpret_cast<Ctx*>(dsm + kCtxOff);
    c->tokens = P.tokens;
    c->slots = P.slots;
    c->has_state = P.has_state;
    c->start_pos = P.start_pos;
    c->trace = P.trace;
    c->M = P.M;
    c->pages_off = pages_offset(P.x_bytes);
    c->ring_static = P.pages_static;
    c->ring_dynamic = P.pages_dynamic > 0 ? P.pages_dynamic : 1;
    c->trace_base = P.list_off[blockIdx.x];
    for (int i = 0; i < npages; ++i) {
      mbar_init(full_bar(i), 1);
      mbar_init(empty_bar(i), kComputeWarps);
    }
    for (int b = 0; b < 2; ++b) {
      mbar_init(red_ready_bar(b), kComputeWarps);
      mbar_init(red_free_bar(b), 1);
    }
    misc() = -1;
#if __CUDA_ARCH__ >= 900
    // the barriers' initialization, visible to the bulk copies (the async proxy) too
    asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    asm volatile("fence.proxy.async.shared::cta;" ::: "memory");
#endif
  }
  __syncthreads();
  if (threadIdx.x >= kLoaderWarp * 32) loader(P, blockIdx.x);
  else work(P, blockIdx.x);
}

size_t step_smem_bytes(int npages, int x_bytes) {
  return static_cast<size_t>(pages_offset(x_bytes)) + static_cast<size_t>(npages) * kPageBytes;
}

int step_configure(size_t smem) {
  cudaFuncSetAttribute(decode_step_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
  int per_sm = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&per_sm, decode_step_kernel, kBlockThreads, smem);
  const cudaError_t e = cudaGetLastError();
  if (e != cudaSuccess) return static_cast<int>(e);
  return per_sm >= 1 ? 0 : static_cast<int>(cudaErrorInvalidConfiguration);
}

// Cooperative, so every block is resident at once: tasks wait on other blocks' tasks.
int step_launch(const Program& P, int blocks, size_t smem, cudaStream_t stream) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(blocks);
  cfg.blockDim = dim3(kBlockThreads);
  cfg.dynamicSmemBytes = smem;
  cfg.stream = stream;
  cudaLaunchAttribute attr[1];
  attr[0].id = cudaLaunchAttributeCooperative;
  attr[0].val.cooperative = 1;
  cfg.attrs = attr;
  cfg.numAttrs = 1;
  return static_cast<int>(cudaLaunchKernelEx(&cfg, decode_step_kernel, P));
}

}  // namespace lx::decode
