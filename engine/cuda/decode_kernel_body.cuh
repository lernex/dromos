// The persistent decode kernel (kernels.h, "Decode programs").
//
// One block of 16 warps per SM runs its list of tasks in program order. Products read their weight
// rows straight from global memory on the tensor cores (dots_mma: 16 rows per warp at a time, several
// steps per lane in flight; dots_mma_lines for the products, the same with whole-line loads) or, for
// f32 weights and odd shapes, with FMAs (dots_rows), and finish the outputs themselves: x staging and
// norms before the reader, the epilogue after it. At the start of
// each run of an op's tasks, warp 1 has L2 fetch the block's coming weights up to a fixed number of
// bytes ahead (spans the builder lists), so DRAM keeps streaming while warp 0 waits for the ops the
// run reads (per-op counters in global memory, acquire); after the run the block counts itself into
// the op's counter (release).
//
// Activations written by other SMs are read through L2 (ld.cg), never L1, so a line an SM cached
// earlier in the step can never be stale.
#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "cuda/common.cuh"
#include "cuda/decode_ops.h"
#include "cuda/decode_sync.cuh"
#include "cuda/bf16x.cuh"
#include "cuda/wformat.cuh"

namespace lx::decode {
namespace LX_DECODE_NS {
namespace {

using bf16 = __nv_bfloat16;
constexpr int kCT = kComputeThreads;
constexpr int kF32 = static_cast<int>(LX_W_F32);

// The compute warps (named barrier 1).
__device__ __forceinline__ void csync() { named_sync<1, kCT>(); }

__device__ __forceinline__ float bf(const void* p, int64_t i) {
  return __bfloat162float(static_cast<const bf16*>(p)[i]);
}
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
// BF16X's quad, the copy's own. bf16x::patched() loops over a row's patch slots (and an overflow record); a row without
// patches takes the same first compare. Inline: the four family copies compile about five times faster than with this
// out of line (633 s against 3446 s at -j2), for 1 to 2% of decode. The bits are the same as bf16x::quad's.
__device__ __forceinline__ uint32_t bf16x_patched(const uint8_t* row, int K, uint32_t k, uint32_t e) {
  return bf16x::patched(row, K, k, e);
}
__device__ __forceinline__ uint2 bf16x_quad(const uint8_t* row, int K, int k) {
  const int blk = k / bf16x::kBlock, off = k - blk * bf16x::kBlock;
  const uint8_t* b = row + bf16x::header_bytes(K) + static_cast<int64_t>(blk) * bf16x::kBlockBytes;
  const uint32_t sm = *reinterpret_cast<const uint32_t*>(b + off);
  const uint32_t codes = *reinterpret_cast<const uint32_t*>(b + bf16x::kCodes + (off & ~7) / 2);
  const uint32_t base = row[bf16x::base_offset(K) + blk];
  const uint32_t e = bf16x::exponents(base, (off & 4) ? codes >> 4 : codes);
  return bf16x::assemble(sm, bf16x_patched(row, K, static_cast<uint32_t>(k), e));
}
__device__ __forceinline__ uint4 bf16x_chunk8(const uint8_t* row, int K, int k) {
  const uint2 lo = bf16x_quad(row, K, k), hi = bf16x_quad(row, K, k + 4);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}
// BF16X words are decoded as they are fetched (WordLoad): the bf16 bits of four values, then the BF16 word.
struct WordBf16X : WordFormat<LX_W_BF16> {};
template <>
struct Word<LX_W_BF16X> {
  using F = WordBf16X;
};
// The formats this copy of the kernel reads: its family's (decode_family.h), defined by the copy's TU. No default: a copy
// that read every format would be the generic kernel, which no program needs.
#ifndef LX_DECODE_FMTS
#error "a decode kernel copy defines LX_DECODE_FMTS (decode_family.h)"
#endif
// A format as a type, so a dispatched call can take it as a template argument.
template <int V>
struct WTag {
  static constexpr int value = V;
};
// fn(WTag<f>) for the format wt, among the formats this copy reads. A format it does not read has no code in it, so
// the hot loops of a copy hold one format's arithmetic and nothing else. kWithF32: the f32 weights (routers, the depth
// head) are dispatched too; the quantized formats are dispatched in any case.
template <bool kWithF32, class Fn>
__device__ __forceinline__ void dispatch_w(int wt, Fn&& fn) {
  if constexpr ((LX_DECODE_FMTS >> LX_W_INT8) & 1u) {
    if (wt == LX_W_INT8) return fn(WTag<LX_W_INT8>{});
  }
  if constexpr ((LX_DECODE_FMTS >> LX_W_BF16) & 1u) {
    if (wt == LX_W_BF16) return fn(WTag<LX_W_BF16>{});
  }
  if constexpr ((LX_DECODE_FMTS >> LX_W_FP8) & 1u) {
    if (wt == LX_W_FP8) return fn(WTag<LX_W_FP8>{});
  }
  if constexpr ((LX_DECODE_FMTS >> LX_W_NVFP4) & 1u) {
    if (wt == LX_W_NVFP4) return fn(WTag<LX_W_NVFP4>{});
  }
  if constexpr ((LX_DECODE_FMTS >> LX_W_BF16X) & 1u) {
    if (wt == LX_W_BF16X) return fn(WTag<LX_W_BF16X>{});
  }
  if constexpr (kWithF32 && ((LX_DECODE_FMTS >> kF32) & 1u)) {
    if (wt == kF32) return fn(WTag<kF32>{});
  }
  // a format no program of this copy holds never reaches here; the bf16 reader is the default, as it always was
  if constexpr ((LX_DECODE_FMTS >> LX_W_BF16) & 1u) fn(WTag<LX_W_BF16>{});
}
// fn<kType>(args) for the format wt: the expert and entry readers are templates over the formats they read.
#define LX_DISPATCH_W(wt, fn, ...) \
  dispatch_w<false>((wt), [&](auto lx_tag) { fn<decltype(lx_tag)::value>(__VA_ARGS__); })
// Formats whose tiles carry one f32 scale per row after the rows.
template <int kType>
constexpr bool kRowScaled = kType == LX_W_INT8 || kType == LX_W_FP8;
__device__ __forceinline__ bool row_scaled(int wtype) { return wtype == LX_W_INT8 || wtype == LX_W_FP8; }

// ---- Shared memory ------------------------------------------------------------------------------
//
// A fixed layout, so every task body addresses shared memory through dsm at offsets the compiler
// knows (plain shared loads and stores, never generic ones): the launch context, the current op's
// descriptor, a few words of state, the run's task numbers, reduction scratch, and the x buffer.

extern __shared__ __align__(128) char dsm[];

struct Ctx {
  const int32_t* tokens;
  const int32_t* slots;
  const int32_t* has_state;
  const int32_t* start_pos;
  int64_t* trace;            // or null
  int M, trace_base;
};
static_assert(sizeof(Ctx) <= kCtxBytes, "the launch context must fit its slot");

__device__ __forceinline__ const Ctx& ctx() { return *reinterpret_cast<const Ctx*>(dsm + kCtxOff); }
__device__ __forceinline__ const OpDesc& desc() { return *reinterpret_cast<const OpDesc*>(dsm + kDescOff); }
__device__ __forceinline__ float* scratch() { return reinterpret_cast<float*>(dsm + kScratchOff); }
__device__ __forceinline__ float* xbuf() { return reinterpret_cast<float*>(dsm + kXOff); }
// The task numbers of the run of a product the compute warps are on (gemv_run).
__device__ __forceinline__ uint16_t* run_tasks() { return reinterpret_cast<uint16_t*>(dsm + kRunOff); }
// What the x buffer holds: a key per op (and token), or -1.
__device__ __forceinline__ int& misc() { return *reinterpret_cast<int*>(dsm + kMiscOff); }


__device__ __forceinline__ void trace_task(int index, int op, int task, int64_t t0, int64_t t1, int64_t t2) {
  const Ctx& c = ctx();
  if (c.trace && threadIdx.x == 0) {
    int64_t* rec = c.trace + static_cast<int64_t>(c.trace_base + index) * 4;
    rec[0] = (static_cast<int64_t>(op) << 32) | (static_cast<int64_t>(task) << 16) | blockIdx.x;
    rec[1] = t0;
    rec[2] = t1;
    rec[3] = t2;
  }
}
__device__ __forceinline__ int64_t trace_now() { return ctx().trace && threadIdx.x == 0 ? global_ns() : 0; }

// Sums n values held by every compute thread (vals[0..n)) over the block in a fixed order, so every
// block that runs the same reduction gets the same bits. The totals land in out[0..n). n <= 40.
// Each of N values (N a power of two, <= 32) summed over a warp's lanes in N - 1 + 5 - log2(N) shuffles
// instead of 5 N: each round hands half of a lane's values to its partner, so after log2(N) rounds
// lane l holds value (l >> (5 - log2 N)) summed over its group of lanes, and butterflies finish the
// groups. Returns the lane's value; its index is warp_spread_index<N>().
template <int N>
__device__ __forceinline__ float warp_spread(float (&v)[N]) {
  static_assert(N >= 1 && N <= 32 && (N & (N - 1)) == 0, "a power of two up to 32");
  const int lane = threadIdx.x & 31;
#pragma unroll
  for (int o = 16, n = N; o >= 1; o >>= 1) {
    if (n > 1) {
      const bool up = lane & o;
#pragma unroll
      for (int i = 0; i < n / 2; ++i) {
        const float send = up ? v[i] : v[i + n / 2], keep = up ? v[i + n / 2] : v[i];
        v[i] = keep + __shfl_xor_sync(0xffffffffu, send, o);
      }
      n >>= 1;
    } else {
      v[0] += __shfl_xor_sync(0xffffffffu, v[0], o);
    }
  }
  return v[0];
}
template <int N>
__device__ __forceinline__ int warp_spread_index() {
  constexpr int kBits = N >= 32 ? 5 : N >= 16 ? 4 : N >= 8 ? 3 : N >= 4 ? 2 : N >= 2 ? 1 : 0;
  return (threadIdx.x & 31) >> (5 - kBits);
}
// Whether this lane writes its group's value (one lane per value).
template <int N>
__device__ __forceinline__ bool warp_spread_writer() {
  return ((threadIdx.x & 31) & ((32 / N) - 1)) == 0;
}

// red[warp * 40 + i] = the warp's sum of vals[i], for i < N (N a power of two).
template <int N>
__device__ __forceinline__ void warp_sums_to(float (&vals)[N], float* red) {
  const float v = warp_spread(vals);
  if (warp_spread_writer<N>()) red[(threadIdx.x >> 5) * 40 + warp_spread_index<N>()] = v;
}

// out[i] = the block's sum of vals[i] (N <= 32): each warp's sum, then the warps in order.
template <int N>
__device__ __forceinline__ void block_sums(float (&vals)[N], float* red, float* out) {
  constexpr int P = N <= 1 ? 1 : N <= 2 ? 2 : N <= 4 ? 4 : N <= 8 ? 8 : N <= 16 ? 16 : 32;
  static_assert(N <= 32, "up to 32 values");
  float v[P];
#pragma unroll
  for (int i = 0; i < P; ++i) v[i] = i < N ? vals[i] : 0.0f;
  warp_sums_to(v, red);
  csync();
  if (threadIdx.x < N) {
    float t = 0.0f;
    for (int w = 0; w < kComputeWarps; ++w) t += red[w * 40 + threadIdx.x];
    out[threadIdx.x] = t;
  }
  csync();
}


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

// ---- The weight reader --------------------------------------------------------------------------

// One weight word (8 bytes) from global memory, past L1: weights are read once per step.
__device__ __forceinline__ uint2 ldg_word(const uint2* p) {
  uint2 v;
  asm("ld.global.nc.L1::no_allocate.v2.u32 {%0, %1}, [%2];" : "=r"(v.x), "=r"(v.y) : "l"(p));
  return v;
}

// A virtual row of a product: words [0, words) at p (null: none, it reads as zeros) times the staged
// x rows xrow .. xrow + mc - 1, from word xw of the x row.
struct RowRef {
  const uint2* p;       // the row's first word (formats read by words); null past the row's range
  int words, xw, xrow;  // words in the range, the x offset of its first word, and the x row
  const uint8_t* sc;    // NVFP4: the block scales of the range's first word (a byte a word); else null
  const uint8_t* row;   // BF16X: the row record, whose words are decoded as they are read; else null
  int kb, K;            // BF16X: the value index of the range's first word, and the row's K
};

// The bits a value takes in each format (NVFP4's 4 bits are its code; its block scales come apart).
template <int kType>
constexpr int kValueBits = kType == LX_W_BF16 || kType == LX_W_BF16X ? 16 : kType == LX_W_NVFP4 ? 4
                           : kType == kF32                          ? 32
                                                                    : 8;

// One word of a row for the FMA reader: its bits, and the block scale its decode applies (NVFP4; 1 otherwise).
template <class F>
struct WordLoad {
  __device__ __forceinline__ static void load(const RowRef& r, int w, uint2& word, float& bs) {
    word = ldg_word(r.p + w);
    bs = 1.0f;
  }
};
template <>
struct WordLoad<WordFormat<LX_W_NVFP4>> {
  __device__ __forceinline__ static void load(const RowRef& r, int w, uint2& word, float& bs) {
    word = ldg_word(r.p + w);
    bs = e4m3_to_f32(__ldg(r.sc + w));
  }
};
template <>
struct WordLoad<WordBf16X> {
  __device__ __forceinline__ static void load(const RowRef& r, int w, uint2& word, float& bs) {
    word = bf16x_quad(r.row, r.K, r.kb + 4 * w);
    bs = 1.0f;
  }
};

// The RowRef of the words from value kb of row `row` of a weight whose rows are row_bytes apart (full records).
template <int kType>
__device__ __forceinline__ RowRef row_ref(const uint8_t* data, const lx_weight& w, int64_t row, int64_t row_bytes,
                                          int kb, int K, int words, int xrow) {
  RowRef r{nullptr, words, 0, xrow, nullptr, nullptr, kb, K};
  const uint8_t* rec = data + row * row_bytes;
  if constexpr (kType == LX_W_BF16X) {
    r.row = rec;
  } else {
    r.p = reinterpret_cast<const uint2*>(rec + (static_cast<int64_t>(kb) * kValueBits<kType>) / 8);
    if constexpr (kType == LX_W_NVFP4) r.sc = static_cast<const uint8_t*>(w.scale) + row * (K / 16) + kb / 16;
  }
  return r;
}

// Dot products of virtual rows with staged x rows, straight from global memory (from L2 where the
// block fetched them ahead): warp w takes virtual rows w, w + 16, ... < nv; a warp step reads 32
// lanes' consecutive words (256 contiguous bytes), and kU steps per lane are in flight, across the
// ends of rows. Virtual row v leaves red[v * mc + j] (added to it with accumulate).
//
// The loads never wait on anything but shared memory: row_of must not read global memory (a load
// that hits L1 still queues behind the warp's outstanding weight misses, and the pipeline drains).
// Each word in flight carries where its x is and whether it ends a row, so the consuming side keeps
// no row state of its own; the next row's reference is worked out a row early.
//
// This is the only place the products decode a weight word: a new weight format plugs in through
// its Word (F) and the row pointers; the prologue (staging) and the epilogues stay as they are.
template <class F, bool kXsBf16, int kM, int kU, class RowOf>
__device__ __forceinline__ void dots_rows(int nv, RowOf&& row_of, const float* xs, int x_stride, int mc, float* red,
                                          bool accumulate) {
  constexpr int kV = F::kValues;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int mine = nv > warp ? (nv - warp + kComputeWarps - 1) / kComputeWarps : 0;
  if (mine == 0) return;
  int lv = 0, ls = 0;
  RowRef lr = row_of(warp);
  int lsteps = max(1, (lr.words + 31) >> 5);
  RowRef ln = mine > 1 ? row_of(warp + kComputeWarps) : lr;
  // the next word of the warp's rows: the word and its block scale, its x offset (-1: none), the slot of the row it
  // ends (-1: none)
  auto produce = [&](uint2& word, float& bs, int& xo, int& end) {
    word = make_uint2(0u, 0u);
    bs = 1.0f;
    xo = -1;
    end = -1;
    if (lv < mine) {
      const int w = ls * 32 + lane;
      if ((lr.p != nullptr || lr.row != nullptr) && w < lr.words) {
        WordLoad<F>::load(lr, w, word, bs);
        xo = lr.xrow * x_stride + (lr.xw + w) * kV;
      }
      if (++ls == lsteps) {
        end = (warp + lv * kComputeWarps) * mc;
        ls = 0;
        if (++lv < mine) {
          lr = ln;
          lsteps = max(1, (lr.words + 31) >> 5);
          if (lv + 1 < mine) ln = row_of(warp + (lv + 1) * kComputeWarps);
        }
      }
    }
  };
  uint2 buf[kU];
  float bsb[kU];
  int bx[kU], be[kU];
#pragma unroll
  for (int u = 0; u < kU; ++u) produce(buf[u], bsb[u], bx[u], be[u]);
  float acc[kM];
#pragma unroll
  for (int j = 0; j < kM; ++j) acc[j] = 0.0f;
  for (int left = mine; left > 0;) {
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const uint2 word = buf[u];
      const float bs = bsb[u];
      const int xo = bx[u], end = be[u];
      produce(buf[u], bsb[u], bx[u], be[u]);
      if (xo >= 0) {
        float wf[kV];
        F::decode(word, bs, wf);
#pragma unroll
        for (int j = 0; j < kM; ++j)
          if (kM == 1 || j < mc) {
            float xf[kV];
            xs_word<kV, kXsBf16>(xs, xo + j * x_stride, xf);
#pragma unroll
            for (int t = 0; t < kV; ++t) acc[j] = fmaf(wf[t], xf[t], acc[j]);
          }
      }
      if (end >= 0) {  // warp-uniform
#pragma unroll
        for (int j = 0; j < kM; ++j)
          if (kM == 1 || j < mc) acc[j] = lx::warp_sum(acc[j]);
        if (lane == 0) {
#pragma unroll
          for (int j = 0; j < kM; ++j)
            if (kM == 1 || j < mc) red[end + j] = accumulate ? red[end + j] + acc[j] : acc[j];
        }
#pragma unroll
        for (int j = 0; j < kM; ++j) acc[j] = 0.0f;
        --left;
      }
    }
  }
}

// The same, built per bucket of tokens, so few tokens never pay for the predicated-off work of eight.
template <class F, bool kXsBf16, class RowOf>
__device__ __forceinline__ void dots_rows_m(int nv, RowOf&& row_of, const float* xs, int x_stride, int mc, float* red,
                                            bool accumulate) {
  constexpr int kU = 8;
  if (mc == 1) dots_rows<F, kXsBf16, 1, kU>(nv, row_of, xs, x_stride, 1, red, accumulate);
  else if (mc == 2) dots_rows<F, kXsBf16, 2, kU>(nv, row_of, xs, x_stride, 2, red, accumulate);
  else if (mc <= 4) dots_rows<F, kXsBf16, 4, kU>(nv, row_of, xs, x_stride, mc, red, accumulate);
  else dots_rows<F, kXsBf16, kMaxM, kU>(nv, row_of, xs, x_stride, mc, red, accumulate);
}

// ---- The tensor-core reader ---------------------------------------------------------------------
//
// mma.m16n8k16 with the weights as A and x as B: a warp takes 16 rows at a time, lane (g, t)
// (g = lane / 4, t = lane % 4) reading 16-byte chunks of rows g and g + 8 straight from global memory,
// and the same k of token g's x. A dot product does not care in which order it visits k, so the
// values of a lane's chunk (contiguous k) stand in for the k slots mma gives the lane, and x is read
// in the same order: the four lanes of a row cover 64 contiguous bytes per step. The products come
// out already summed over the lanes, rows g and g + 8 by tokens 2t and 2t + 1.

__device__ __forceinline__ uint4 ldg_chunk(const uint4* p) {
  uint4 v;
  asm("ld.global.nc.L1::no_allocate.v4.u32 {%0, %1, %2, %3}, [%4];"
      : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w)
      : "l"(p));
  return v;
}

template <bool kF16>
__device__ __forceinline__ void mma16816(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0,
                                         uint32_t b1) {
  if constexpr (kF16)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
  else
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, "
        "{%0, %1, %2, %3};"
        : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

// A weight format for the tensor-core reader: a 16-byte chunk of a row, as kValues values (in pairs, the lower k in
// the low half), bf16 or f16. aux carries an NVFP4 chunk's two block scales as f16 pairs (scale_pair); other formats
// ignore it.
template <int kType>
constexpr bool kXsF16 = kType != LX_W_BF16 && kType != LX_W_BF16X;  // the x of the reader: f16 (else bf16)
template <int kType>
struct Chunk;
template <>
struct Chunk<LX_W_BF16> {
  static constexpr int kValues = 8;
  static constexpr bool kF16 = false;
  __device__ __forceinline__ static void decode(uint4 v, uint2 aux, uint32_t (&r)[4]) {
    (void)aux;
    r[0] = v.x;
    r[1] = v.y;
    r[2] = v.z;
    r[3] = v.w;
  }
};
// BF16X chunks are decoded as they are fetched (chunk_load), so the chunk is the bf16 one.
template <>
struct Chunk<LX_W_BF16X> : Chunk<LX_W_BF16> {};
// int8 to f16 exactly: 0x6400 | (q ^ 0x80) is the f16 1024 + 128 + q, and 1152 comes off in pairs.
template <>
struct Chunk<LX_W_INT8> {
  static constexpr int kValues = 16;
  static constexpr bool kF16 = true;
  __device__ __forceinline__ static void decode(uint4 v, uint2 aux, uint32_t (&r)[8]) {
    (void)aux;
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint32_t u = w[i] ^ 0x80808080u;
      const uint32_t lo = __byte_perm(u, 0x64646464u, 0x5140), hi = __byte_perm(u, 0x64646464u, 0x5342);
      asm("sub.f16x2 %0, %1, %2;" : "=r"(r[2 * i]) : "r"(lo), "r"(0x64806480u));
      asm("sub.f16x2 %0, %1, %2;" : "=r"(r[2 * i + 1]) : "r"(hi), "r"(0x64806480u));
    }
  }
};
// e4m3 to f16, exact (every e4m3 value is an f16): the hardware converts a byte pair, the lower byte the lower k.
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t pair) {
  const __half2_raw h = __nv_cvt_fp8x2_to_halfraw2(static_cast<uint16_t>(pair), __NV_E4M3);
  return *reinterpret_cast<const uint32_t*>(&h);
}
template <>
struct Chunk<LX_W_FP8> {
  static constexpr int kValues = 16;
  static constexpr bool kF16 = true;
  __device__ __forceinline__ static void decode(uint4 v, uint2 aux, uint32_t (&r)[8]) {
    (void)aux;
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      r[2 * i] = e4m3x2_to_f16x2(w[i] & 0xffffu);
      r[2 * i + 1] = e4m3x2_to_f16x2(w[i] >> 16);
    }
  }
};
// The two block scales of a chunk, e4m3 bytes at k / 16 and k / 16 + 1 (low byte first), as f16 pairs: .x holds the
// first block's scale in both halves, .y the second block's.
__device__ __forceinline__ uint2 scale_pair(uint16_t s) {
  const __half2 h = __half2(__nv_cvt_fp8x2_to_halfraw2(s, __NV_E4M3));
  const __half2 a = __half2half2(__low2half(h)), b = __half2half2(__high2half(h));
  return make_uint2(*reinterpret_cast<const uint32_t*>(&a), *reinterpret_cast<const uint32_t*>(&b));
}
// NVFP4: 32 values in 16 bytes. Byte i holds values 2i (low nibble) and 2i + 1 (high), so word i is the pair 2i, 2i + 1
// and belongs to block i / 8. e2m1 times its e4m3 block scale is exact in f16 (at most 6 significant bits, in range).
template <>
struct Chunk<LX_W_NVFP4> {
  static constexpr int kValues = 32;
  static constexpr bool kF16 = true;
  __device__ __forceinline__ static void decode(uint4 v, uint2 aux, uint32_t (&r)[16]) {
    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
    const __half2 s0 = *reinterpret_cast<const __half2*>(&aux.x), s1 = *reinterpret_cast<const __half2*>(&aux.y);
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint8_t byte = static_cast<uint8_t>(w[i >> 2] >> (8 * (i & 3)));
      const __half2 e = __half2(__nv_cvt_fp4x2_to_halfraw2(byte, __NV_E2M1));
      const __half2 p = __hmul2(e, i < 8 ? s0 : s1);
      r[i] = *reinterpret_cast<const uint32_t*>(&p);
    }
  }
};

// A unit of the tensor-core reader, for one lane: its first chunks in rows g and g + 8 (null: the row reads as zeros),
// the NVFP4 block scales of those rows, the unit's first value (local to the staged x, and in the row record), its
// steps (a step: four chunks per row), the first of the eight x rows its products' columns are (xr + j for column j),
// and the row's K.
struct UnitRef {
  const uint8_t* a;
  const uint8_t* b;
  const uint8_t* sa;
  const uint8_t* sb;
  int kb, kabs;
  int steps, xr, K;
};

// Two rows of a unit of experts: row ra and row rb of the stack (rows row_bytes apart), the unit's first value kb.
template <int kType>
__device__ __forceinline__ UnitRef unit_rows(const uint8_t* data, const lx_weight& w, int64_t ra, int64_t rb,
                                             int64_t row_bytes, int kb, int steps, int xr, int K) {
  UnitRef u{};
  u.a = data + ra * row_bytes;
  u.b = data + rb * row_bytes;
  if constexpr (kType == LX_W_NVFP4) {
    u.sa = static_cast<const uint8_t*>(w.scale) + ra * (K / 16);
    u.sb = static_cast<const uint8_t*>(w.scale) + rb * (K / 16);
  }
  u.kb = kb;
  u.kabs = kb;
  u.steps = steps;
  u.xr = xr;
  u.K = K;
  return u;
}

// The 16-byte chunk of a row record at value k (a multiple of the chunk's values), as stored, and for NVFP4 also the
// chunk's two block scales in aux.
template <int kType>
__device__ __forceinline__ uint4 chunk_load(const uint8_t* row, const uint8_t* sc, int K, int k, uint2& aux) {
  if constexpr (kType == LX_W_BF16X) {
    aux = make_uint2(0u, 0u);
    (void)sc;
    return bf16x_chunk8(row, K, k);
  }
  (void)K;
  if constexpr (kType == LX_W_NVFP4) {
    aux = scale_pair(__ldg(reinterpret_cast<const uint16_t*>(sc + k / 16)));
    return ldg_chunk(reinterpret_cast<const uint4*>(row + k / 2));
  } else {
    (void)sc;
    (void)aux;
    return ldg_chunk(reinterpret_cast<const uint4*>(row + (static_cast<int64_t>(k) * kValueBits<kType>) / 8));
  }
}

// Units of 16 rows by a run of steps: warp w takes units w, w + 16, ... < nunits; kU steps per lane
// are in flight, across the ends of units. unit_of(u) gives a unit; sink(u, c) takes its products.
// xs: the staged x, row r at r * xst (16-bit values), nx rows (past them x reads as zeros); with
// kPlanes 2, a second plane of nx rows follows. Only the first kCols columns read x (the others are
// zero), so a unit that wants one column (an expert's rows for one token) reads one row of x.
template <int kType, int kU, int kPlanes, int kCols, class UnitOf, class Sink>
__device__ __forceinline__ void dots_mma(int nunits, UnitOf&& unit_of, Sink&& sink, const uint16_t* xs, int xst,
                                         int nx) {
  using C = Chunk<kType>;
  constexpr int kV = C::kValues, kR = kV / 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const int mine = nunits > warp ? (nunits - warp + kComputeWarps - 1) / kComputeWarps : 0;
  if (mine == 0) return;
  int lv = 0, ls = 0;
  UnitRef lr = unit_of(warp);
  UnitRef ln = mine > 1 ? unit_of(warp + kComputeWarps) : lr;
  // the lane's next chunks and their aux, the x offset of the step (-1: no step, the same in every lane;
  // -2: this lane's column has no x), the unit it ends (-1: none)
  auto produce = [&](uint4& wa, uint4& wb, uint2& aa, uint2& ab, int& xo, int& end) {
    wa = make_uint4(0u, 0u, 0u, 0u);
    wb = wa;
    aa = make_uint2(0u, 0u);
    ab = aa;
    xo = -1;
    end = -1;
    if (lv < mine) {
      const int k = lr.kabs + (4 * ls + t) * kV;  // this lane's chunk, in the row record
      if (lr.a) wa = chunk_load<kType>(lr.a, lr.sa, lr.K, k, aa);
      if (lr.b) wb = chunk_load<kType>(lr.b, lr.sb, lr.K, k, ab);
      const int row = lr.xr + g;
      xo = g < kCols && row < nx ? row * xst + lr.kb + (4 * ls + t) * kV : -2;
      if (++ls == lr.steps) {
        end = warp + lv * kComputeWarps;
        ls = 0;
        if (++lv < mine) {
          lr = ln;
          if (lv + 1 < mine) ln = unit_of(warp + (lv + 1) * kComputeWarps);
        }
      }
    }
  };
  uint4 ba[kU], bb[kU];
  uint2 baa[kU], bab[kU];
  int bx[kU], be[kU];
#pragma unroll
  for (int u = 0; u < kU; ++u) produce(ba[u], bb[u], baa[u], bab[u], bx[u], be[u]);
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (int left = mine; left > 0;) {
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const uint4 wa = ba[u], wb = bb[u];
      const uint2 aa = baa[u], ab = bab[u];
      const int xo = bx[u], end = be[u];
      produce(ba[u], bb[u], baa[u], bab[u], bx[u], be[u]);
      if (xo != -1) {  // warp-uniform
        uint32_t ra[kR], rb[kR];
        C::decode(wa, aa, ra);
        C::decode(wb, ab, rb);
#pragma unroll
        for (int h = 0; h < kPlanes; ++h) {  // x, or its planes in order
          uint32_t rx[kR];
#pragma unroll
          for (int i = 0; i < kR / 4; ++i) {
            const uint4 q =
                xo >= 0 ? *reinterpret_cast<const uint4*>(xs + h * nx * xst + xo + 8 * i) : make_uint4(0u, 0u, 0u, 0u);
            rx[4 * i] = q.x;
            rx[4 * i + 1] = q.y;
            rx[4 * i + 2] = q.z;
            rx[4 * i + 3] = q.w;
          }
#pragma unroll
          for (int j = 0; j < kV / 4; ++j)
            mma16816<C::kF16>(c, ra[2 * j], rb[2 * j], ra[2 * j + 1], rb[2 * j + 1], rx[2 * j], rx[2 * j + 1]);
        }
      }
      if (end >= 0) {  // warp-uniform
        sink(end, c);
        c[0] = c[1] = c[2] = c[3] = 0.0f;
        --left;
      }
    }
  }
}

// The partner lane's value: lane ^ 4, the same t in row group g ^ 1.
__device__ __forceinline__ uint4 shfl_partner(uint4 v) {
  v.x = __shfl_xor_sync(0xffffffffu, v.x, 4);
  v.y = __shfl_xor_sync(0xffffffffu, v.y, 4);
  v.z = __shfl_xor_sync(0xffffffffu, v.z, 4);
  v.w = __shfl_xor_sync(0xffffffffu, v.w, 4);
  return v;
}
__device__ __forceinline__ uint2 shfl_partner(uint2 v) {
  v.x = __shfl_xor_sync(0xffffffffu, v.x, 4);
  v.y = __shfl_xor_sync(0xffffffffu, v.y, 4);
  return v;
}
// The partner lane's pointer (a row, or a row's scales; null when the partner has none).
template <class P>
__device__ __forceinline__ P shfl_partner_ptr(P p) {
  const uint64_t v = reinterpret_cast<uintptr_t>(p);
  const uint32_t lo = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v), 4);
  const uint32_t hi = __shfl_xor_sync(0xffffffffu, static_cast<uint32_t>(v >> 32), 4);
  return reinterpret_cast<P>(static_cast<uintptr_t>((static_cast<uint64_t>(hi) << 32) | lo));
}
// A lane's chunk of one row as chunk_load returns it: the 16 bytes and, for NVFP4, the chunk's scales.
struct LineChunk {
  uint4 w;
  uint2 aux;
};
__device__ __forceinline__ LineChunk shfl_partner_chunk(LineChunk c) {
  c.w = shfl_partner(c.w);
  c.aux = shfl_partner(c.aux);
  return c;
}

// dots_mma with every load reading whole 128-byte lines. In dots_mma a load covers 64 bytes of each of
// eight rows, half a line, and the other half follows in the next step's load: two requests per line,
// and DRAM serves that pattern worse the more of it is in flight (measured: a bare stream of it loses
// a quarter of its bandwidth going from 8 to 16 loads a lane in flight, where whole lines lose nothing).
// Here two steps of a unit go out together: row groups 2i and 2i + 1 both read rows 2i and 2i + 1 (and
// + 8), the even group at the first step and the odd group at the second, so each load covers 128
// contiguous bytes of four rows. One swap with the partner group (lane ^ 4) then gives every lane
// exactly the chunks dots_mma would have loaded for it (the chunk, its scales and its row pointers
// move together), and the steps are consumed in the same order, so the products are the same bits. A
// unit's last step goes out alone when its count is odd. kU pairs of steps per lane are in flight across
// the ends of units, and a pair's loads go out before the oldest pair is used.
template <int kType, int kU, int kPlanes, int kCols, class UnitOf, class Sink>
__device__ __forceinline__ void dots_mma_lines(int nunits, UnitOf&& unit_of, Sink&& sink, const uint16_t* xs,
                                               int xst, int nx) {
  using C = Chunk<kType>;
  constexpr int kV = C::kValues, kR = kV / 2;
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
  const bool odd = (g & 1) != 0;
  const int mine = nunits > warp ? (nunits - warp + kComputeWarps - 1) / kComputeWarps : 0;
  if (mine == 0) return;
  int lv = 0, ls = 0;
  UnitRef lr = unit_of(warp);
  UnitRef ln = mine > 1 ? unit_of(warp + kComputeWarps) : lr;
  // the current unit's rows 2i, 2i + 1, 2i + 8 and 2i + 9 (i = g / 2) and their scale rows, from the lane and its
  // partner (warp-uniform calls, so the shuffles see every lane)
  const uint8_t *q0 = nullptr, *q1 = nullptr, *q2 = nullptr, *q3 = nullptr;
  const uint8_t *s0 = nullptr, *s1 = nullptr, *s2 = nullptr, *s3 = nullptr;
  auto rows_of = [&]() {
    const uint8_t *pa = shfl_partner_ptr(lr.a), *pb = shfl_partner_ptr(lr.b);
    q0 = odd ? pa : lr.a;
    q1 = odd ? lr.a : pa;
    q2 = odd ? pb : lr.b;
    q3 = odd ? lr.b : pb;
    if constexpr (kType == LX_W_NVFP4) {
      const uint8_t *sa = shfl_partner_ptr(lr.sa), *sb = shfl_partner_ptr(lr.sb);
      s0 = odd ? sa : lr.sa;
      s1 = odd ? lr.sa : sa;
      s2 = odd ? sb : lr.sb;
      s3 = odd ? lr.sb : sb;
    }
  };
  rows_of();
  // the chunk of a row at this lane's index j of the unit (value k = kabs + j * kV), with its scales; zero for no row
  auto fetch = [&](const uint8_t* row, const uint8_t* sc, int j, LineChunk& out) {
    out.w = make_uint4(0u, 0u, 0u, 0u);
    out.aux = make_uint2(0u, 0u);
    if (row) out.w = chunk_load<kType>(row, sc, lr.K, lr.kabs + j * kV, out.aux);
  };
  // the lane's next loads: w[0..3] rows 2i, 2i + 1, 2i + 8, 2i + 9 at its step of a pair, or w[0] and w[2] its own
  // rows g and g + 8 for a lone step; the x offset of the (first) step (-1: none, the same in every lane; -2: this
  // lane's column has no x); the unit it ends (-1: none); whether it is a pair
  auto produce = [&](LineChunk (&w)[4], int& xo, int& end, int& two) {
#pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = LineChunk{make_uint4(0u, 0u, 0u, 0u), make_uint2(0u, 0u)};
    xo = -1;
    end = -1;
    two = 0;
    if (lv < mine) {
      two = ls + 1 < lr.steps;
      if (two) {
        const int j = 4 * ls + (odd ? 4 : 0) + t;
        fetch(q0, s0, j, w[0]);
        fetch(q1, s1, j, w[1]);
        fetch(q2, s2, j, w[2]);
        fetch(q3, s3, j, w[3]);
      } else {  // the lane's own rows g and g + 8
        const int j = 4 * ls + t;
        fetch(odd ? q1 : q0, odd ? s1 : s0, j, w[0]);
        fetch(odd ? q3 : q2, odd ? s3 : s2, j, w[2]);
      }
      const int row = lr.xr + g;
      xo = g < kCols && row < nx ? row * xst + lr.kb + (4 * ls + t) * kV : -2;
      ls += two ? 2 : 1;
      if (ls == lr.steps) {
        end = warp + lv * kComputeWarps;
        ls = 0;
        if (++lv < mine) {
          lr = ln;
          rows_of();
          if (lv + 1 < mine) ln = unit_of(warp + (lv + 1) * kComputeWarps);
        }
      }
    }
  };
  float c[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  // one step: the lane's chunks of rows g and g + 8 against x at xo, as in dots_mma
  auto step = [&](const LineChunk& wa, const LineChunk& wb, int xo) {
    uint32_t ra[kR], rb[kR];
    C::decode(wa.w, wa.aux, ra);
    C::decode(wb.w, wb.aux, rb);
#pragma unroll
    for (int h = 0; h < kPlanes; ++h) {  // x, or its planes in order
      uint32_t rx[kR];
#pragma unroll
      for (int i = 0; i < kR / 4; ++i) {
        const uint4 q =
            xo >= 0 ? *reinterpret_cast<const uint4*>(xs + h * nx * xst + xo + 8 * i) : make_uint4(0u, 0u, 0u, 0u);
        rx[4 * i] = q.x;
        rx[4 * i + 1] = q.y;
        rx[4 * i + 2] = q.z;
        rx[4 * i + 3] = q.w;
      }
#pragma unroll
      for (int j = 0; j < kV / 4; ++j)
        mma16816<C::kF16>(c, ra[2 * j], rb[2 * j], ra[2 * j + 1], rb[2 * j + 1], rx[2 * j], rx[2 * j + 1]);
    }
  };
  LineChunk bw[kU][4];
  int bx[kU], be[kU], b2[kU];
#pragma unroll
  for (int u = 0; u < kU; ++u) produce(bw[u], bx[u], be[u], b2[u]);
  for (int left = mine; left > 0;) {
#pragma unroll
    for (int u = 0; u < kU; ++u) {
      const LineChunk w0 = bw[u][0], w1 = bw[u][1], w2 = bw[u][2], w3 = bw[u][3];
      const int xo = bx[u], end = be[u], two = b2[u];
      produce(bw[u], bx[u], be[u], b2[u]);
      if (xo != -1) {  // warp-uniform
        if (two) {     // warp-uniform
          // the even lane holds its rows at the first step and the partner's at it; the odd lane the
          // partner's rows at the second step and its own: each hands the other what it holds of the
          // other's rows
          const LineChunk sa = shfl_partner_chunk(odd ? w0 : w1), sb = shfl_partner_chunk(odd ? w2 : w3);
          step(odd ? sa : w0, odd ? sb : w2, xo);
          step(odd ? w1 : sa, odd ? w3 : sb, xo >= 0 ? xo + 4 * kV : xo);
        } else {
          step(w0, w2, xo);
        }
      }
      if (end >= 0) {  // warp-uniform
        sink(end, c);
        c[0] = c[1] = c[2] = c[3] = 0.0f;
        --left;
      }
    }
  }
}

// ---- Products: prologue (x staging) and epilogues -----------------------------------------------

// x storage in the x buffer: f32, bf16, or f16 (for the int8 tensor-core reader).
enum XStore { kStoreF32 = 0, kStoreBf16 = 1, kStoreF16 = 2, kStoreBf16Planes = 3, kStoreF16Planes = 4 };

// Stages the product's x [M, k1 - k0] into the x buffer, rows xst values apart, four values at a time.
// Planes carry f32 x to a 16-bit product: plane 0 (rows [0, M)) is the value rounded to 16 bits, plane
// 1 (rows [M, 2M)) what that left, rounded again (16 bits of x in bf16, 22 in f16). A third plane
// gains nothing: past this the tensor cores' own accumulation is the larger error.
template <int kXs>
__device__ __forceinline__ void stage_gemv_x(const Gemv& g, int M, int xst) {
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
    if constexpr (kXs == kStoreBf16 || kXs == kStoreBf16Planes) {
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(reinterpret_cast<bf16*>(xbuf()) + m * xst + j);
#pragma unroll
      for (int pl = 0; pl < (kXs == kStoreBf16Planes ? 2 : 1); ++pl) {
        const __nv_bfloat162 h0 = __floats2bfloat162_rn(v.x, v.y), h1 = __floats2bfloat162_rn(v.z, v.w);
        dst[0] = h0;
        dst[1] = h1;
        const float2 f0 = __bfloat1622float2(h0), f1 = __bfloat1622float2(h1);
        v = make_float4(v.x - f0.x, v.y - f0.y, v.z - f1.x, v.w - f1.y);
        dst += M * xst / 2;
      }
    } else if constexpr (kXs == kStoreF16 || kXs == kStoreF16Planes) {
      // clamped to the f16 range: a value past it would turn the whole dot product into inf
      constexpr float kMax = 65504.0f;
      v = make_float4(fminf(fmaxf(v.x, -kMax), kMax), fminf(fmaxf(v.y, -kMax), kMax), fminf(fmaxf(v.z, -kMax), kMax),
                      fminf(fmaxf(v.w, -kMax), kMax));
      __half2* dst = reinterpret_cast<__half2*>(reinterpret_cast<__half*>(xbuf()) + m * xst + j);
      const __half2 h0 = __floats2half2_rn(v.x, v.y), h1 = __floats2half2_rn(v.z, v.w);
      dst[0] = h0;
      dst[1] = h1;
      if constexpr (kXs == kStoreF16Planes) {
        const float2 f0 = __half22float2(h0), f1 = __half22float2(h1);
        dst += M * xst / 2;
        dst[0] = __floats2half2_rn(v.x - f0.x, v.y - f0.y);
        dst[1] = __floats2half2_rn(v.z - f1.x, v.w - f1.y);
      }
    } else {
      *reinterpret_cast<float4*>(xbuf() + m * xst + j) = v;
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

// ---- Products -----------------------------------------------------------------------------------

// The K-split partial products of earlier chunks, for output (m, n).
__device__ __forceinline__ float partial_sums(const Gemv& g, int m, int n, int64_t MN) {
  float add = 0.0f;
  for (int c = 0; c < g.n_add; ++c) add += __ldcg(g.y_add + c * MN + static_cast<int64_t>(m) * g.N + n);
  return add;
}
__device__ __forceinline__ float row_factor(const Gemv& g, int n) {
  if (g.wtype == LX_W_NVFP4) return g.w.global;  // the matrix's global; the block scales are inside the words
  return row_scaled(g.wtype) ? __ldg(static_cast<const float*>(g.w.scale) + n) : 1.0f;
}
// The NVFP4 global of expert e of a stack: its own, or the stack's.
__device__ __forceinline__ float expert_global(const lx_weight& w, int e) {
  return w.expert_global ? __ldg(w.expert_global + e) : w.global;
}
// The output row of the run's local row lr (rows past N are the op's last task's padding).
__device__ __forceinline__ int run_row(const Gemv& g, int lr) {
  return static_cast<int>(run_tasks()[lr / g.rows]) * g.rows + lr % g.rows;
}

template <int kType, bool kXsBf16>
__device__ __forceinline__ void gemv_dots(const Gemv& g, int b0, int nb, int sh, float* red) {
  using F = typename Word<kType>::F;
  const int M = ctx().M, Kc = g.k1 - g.k0, W = Kc / F::kValues;
  const uint8_t* data = static_cast<const uint8_t*>(g.w.data);
  auto row_of = [&](int v) -> RowRef {
    const int p = v & ((1 << sh) - 1), n = run_row(g, b0 + (v >> sh));
    const int wb = (p * W) >> sh, we = ((p + 1) * W) >> sh;
    if (n >= g.N) return {nullptr, we - wb, wb, 0};
    RowRef r = row_ref<kType>(data, g.w, n, g.ld, g.k0 + wb * F::kValues, g.K, we - wb, 0);
    r.xw = wb;
    return r;
  };
  dots_rows_m<F, kXsBf16>(nb << sh, row_of, xbuf(), Kc, M, red, false);
}

// The tensor-core reader on a batch of a run's rows: units of 16 rows by one of S = 2^sh parts of k. A chunk is read
// from its row record at its absolute value (chunk_load); x is the staged rows, bf16 for the bf16 readers.
template <int kType, int kPlanes>
__device__ __forceinline__ void gemv_mma(const Gemv& g, int b0, int nb, int sh, float* red) {
  constexpr int kV = Chunk<kType>::kValues;
  const int M = ctx().M, Kc = g.k1 - g.k0, spr = Kc / (4 * kV), S = 1 << sh;
  const uint8_t* data = static_cast<const uint8_t*>(g.w.data);
  const int lane = threadIdx.x & 31, gr = lane >> 2, t = lane & 3;
  // the weight row of unit row r, or -1 when it reads as zeros
  auto row_n = [&](int r) -> int {
    if (r >= nb) return -1;
    const int n = run_row(g, b0 + r);
    return n < g.N ? n : -1;
  };
  auto unit_of = [&](int u) -> UnitRef {
    const int r0 = (u >> sh) * 16 + gr, p = u & (S - 1);
    const int s0 = (p * spr) >> sh, s1 = ((p + 1) * spr) >> sh, kb = s0 * 4 * kV;
    const int na = row_n(r0), nc = row_n(r0 + 8);
    UnitRef ur{};
    ur.a = na < 0 ? nullptr : data + na * g.ld;
    ur.b = nc < 0 ? nullptr : data + nc * g.ld;
    if constexpr (kType == LX_W_NVFP4) {
      const uint8_t* sc = static_cast<const uint8_t*>(g.w.scale);
      ur.sa = na < 0 ? nullptr : sc + na * (g.K / 16);
      ur.sb = nc < 0 ? nullptr : sc + nc * (g.K / 16);
    }
    ur.kb = kb;
    ur.kabs = g.k0 + kb;
    ur.steps = s1 - s0;
    ur.xr = 0;
    ur.K = g.K;
    return ur;
  };
  auto sink = [&](int u, const float (&c)[4]) {
    const int r0 = (u >> sh) * 16 + gr, p = u & (S - 1), m0 = 2 * t;
    if (r0 < nb) {
      if (m0 < M) red[(r0 * S + p) * M + m0] = c[0];
      if (m0 + 1 < M) red[(r0 * S + p) * M + m0 + 1] = c[1];
    }
    if (r0 + 8 < nb) {
      if (m0 < M) red[((r0 + 8) * S + p) * M + m0] = c[2];
      if (m0 + 1 < M) red[((r0 + 8) * S + p) * M + m0 + 1] = c[3];
    }
  };
  const int units = ((nb + 15) >> 4) << sh;
  const uint16_t* xs = reinterpret_cast<const uint16_t*>(xbuf());
  const int xst = Kc + (kXsF16<kType> ? 8 : 32);
  // Whole-line loads take two steps of a unit at a time: when every unit has at least two (most
  // shapes), they keep as many steps in flight as dots_mma does; units of one step keep dots_mma.
  if ((spr >> sh) >= 2) dots_mma_lines<kType, 2, kPlanes, 8>(units, unit_of, sink, xs, xst, M);
  else dots_mma<kType, 4, kPlanes, 8>(units, unit_of, sink, xs, xst, M);
}

// The tensor-core reader for the product's format (kPlanes 2: f32 x in two 16-bit planes).
// One format's products for a batch of a run's rows: the tensor-core reader when the product has mma, else the FMA reader.
// Inline, like bf16x_patched: gemv_run dispatches to one format's products per copy, and out of line they cost the
// copies about five times the compile.
template <int kType>
__device__ __forceinline__ void gemv_products(const Gemv& g, int b0, int nb, int sh, float* red) {
  if constexpr (kType != kF32) {
    if (g.mma) {
      if (g.mma <= 2) gemv_mma<kType, 1>(g, b0, nb, sh, red);
      else gemv_mma<kType, 2>(g, b0, nb, sh, red);
      return;
    }
  }
  if constexpr (kType == kF32) gemv_dots<kType, false>(g, b0, nb, sh, red);
  else if (g.xs_bf16) gemv_dots<kType, true>(g, b0, nb, sh, red);
  else gemv_dots<kType, false>(g, b0, nb, sh, red);
}

// A block's run of tasks of one product: stage x once, then all its rows at once across the warps
// (a row split S ways when there are fewer rows than warps), then every output through its epilogue,
// in batches of whole tasks when the partial sums outgrow the scratch. The first batch's epilogue
// inputs are loaded before anything else, so their round trip overlaps the staging and the products.
__device__ __noinline__ void gemv_run(int op, const uint32_t* entries, int count, int index) {
  const Gemv& g = desc().gemv;
  const int M = ctx().M, tid = threadIdx.x;
  const int R = count * g.rows;
  const bool write = g.y_kind == LX_FY_WRITE, parts = write && g.next_parts;
  const int Kc = g.k1 - g.k0, sh = g.sh;
  const int S = 1 << sh;
  const int cap = max(g.rows, kScratchFloats / (S * M + (parts ? M : 0)) / g.rows * g.rows);
  const int64_t MN = static_cast<int64_t>(M) * g.N;
  const int64_t t0 = trace_now();
  constexpr int kPre = 2;
  OutInputs in[kPre];
  float add[kPre], fac[kPre];
  const int n0 = min(cap, R) * M;
#pragma unroll
  for (int j = 0; j < kPre; ++j) {
    const int e = tid + j * kCT;
    if (e < n0) {
      const int m = e % M, n = run_row(g, e / M);
      if (n < g.N) {
        in[j] = load_out(g, m, n, write);
        add[j] = partial_sums(g, m, n, MN);
        fac[j] = row_factor(g, n);
      }
    }
  }
  if (misc() != op * 16) {
    if (g.mma == 1) stage_gemv_x<kStoreBf16>(g, M, Kc + 32);
    else if (g.mma == 2) stage_gemv_x<kStoreF16>(g, M, Kc + 8);
    else if (g.mma == 3) stage_gemv_x<kStoreBf16Planes>(g, M, Kc + 32);
    else if (g.mma == 4) stage_gemv_x<kStoreF16Planes>(g, M, Kc + 8);
    else if (g.xs_bf16) stage_gemv_x<kStoreBf16>(g, M, Kc);
    else stage_gemv_x<kStoreF32>(g, M, Kc);
    if (tid == 0) misc() = op * 16;
  }
  const int64_t t1 = trace_now();
  for (int b0 = 0; b0 < R; b0 += cap) {
    const int nb = min(cap, R - b0);
    float* red = scratch();
    float* sq = red + nb * S * M;
    dispatch_w<true>(g.wtype, [&](auto lx_tag) { gemv_products<decltype(lx_tag)::value>(g, b0, nb, sh, red); });
    csync();
    for (int e = tid, j = 0; e < nb * M; e += kCT, ++j) {
      const int r = e / M, m = e % M, n = run_row(g, b0 + r);
      if (n >= g.N) {
        if (parts) sq[e] = 0.0f;
        continue;
      }
      float v = 0.0f;
      for (int p = 0; p < S; ++p) v += red[(r * S + p) * M + m];
      const bool pre = b0 == 0 && j < kPre;
      const float y = v * (pre ? fac[j > 0] : row_factor(g, n)) + (pre ? add[j > 0] : partial_sums(g, m, n, MN));
      const OutInputs oi = pre ? in[j > 0] : load_out(g, m, n, write);
      if (!write) store_out(g, m, n, y, oi);
      else {
        const float q = write_out(g, m, n, y, oi);
        if (parts) sq[e] = q;
      }
    }
    if (parts) {
      // the sum of next_src^2 over each task's rows, per token, in row order
      csync();
      const int tb = nb / g.rows, ntasks = desc().ntasks;
      for (int q = tid; q < tb * M; q += kCT) {
        const int jt = q / M, m = q % M;
        float s = 0.0f;
        for (int r = 0; r < g.rows; ++r) s += sq[(jt * g.rows + r) * M + m];
        const int task = run_tasks()[b0 / g.rows + jt];
        g.next_parts[static_cast<int64_t>(m) * ntasks + task] = s;
      }
    }
    if (b0 + cap < R) csync();  // the next batch reuses the scratch
  }
  if (ctx().trace) {
    const int64_t t2 = global_ns();
    for (int i = 0; i < count; ++i)
      trace_task(index + i, op, static_cast<int>(__ldg(entries + i) & 0xffff), t0, t1, t2);
  }
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

// 16 bytes from global to shared memory without a register round trip (cp.async, sm_80).
__device__ __forceinline__ void cp_async16(void* dst, const void* src) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" ::"r"(smem_addr(dst)), "l"(src) : "memory");
}
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_all;" ::: "memory"); }

// Rows [0, n) of the token's entries at bank + off (D floats each, entry_stride apart) into the x
// buffer, all in flight at once.
__device__ __forceinline__ void stage_entries(const float* bank, int off, int n, int stride, int D) {
  const int per = D / 4;
  for (int e = 0; e < n; ++e)
    for (int c = threadIdx.x; c < per; c += kCT)
      cp_async16(xbuf() + e * D + 4 * c, bank + static_cast<int64_t>(e) * stride + off + 4 * c);
}

// One token's depth-memory read, in one block, so everything here is on the step's critical path: the
// entries are staged in shared memory with all of their loads in flight together (keys, then values
// over them), each phase's sums are one spread warp reduction, and what later phases read is loaded
// while earlier ones compute.
__device__ __noinline__ void memread_task(int m) {
  const MemRead& a = desc().mem;
  const int M = ctx().M;
  constexpr int kCh = 2;  // float4 columns per thread: D <= 4 * 2 * 512
  constexpr int kE = kMaxEntries, kS = kMaxStreams;
  static_assert(kE * kS == 32 && (kE + 1) * kS <= 40, "the scores fill one spread reduction, red holds 40");
  const int tid = threadIdx.x;
  const int S = a.S, D = a.D, ne = a.n_entries, chunks = D / 4;
  float* red = scratch();
  float* fin = scratch() + kComputeWarps * 40;
  float* wts = fin + 40;  // [kS][kE] softmax weights, then [kS] norms and [kS] gates
  float* streams = a.streams + static_cast<int64_t>(m) * S * D;
  const float* bank = a.bank + static_cast<int64_t>(m) * a.token_stride;
  if (ne > 0) stage_entries(bank, 0, ne, a.entry_stride, D);  // the keys U_e
  float4 x[kCh][kS];
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    const int c = tid + j * kCT;
#pragma unroll
    for (int s = 0; s < kS; ++s)
      x[j][s] = (c < chunks && s < S) ? ldcg4(streams + s * D + 4 * c) : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  float4 prev[kCh];  // the previous pass's mean, for the end-of-pass difference
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    const int c = tid + j * kCT;
    prev[j] = a.previous && c < chunks ? ldcg4(a.previous + static_cast<int64_t>(m) * D + 4 * c)
                                       : make_float4(0.f, 0.f, 0.f, 0.f);
  }
  if (ne > 0) {
    // Scores x[s] . U_e (values e * kS + s) and the gate dots x[s] . gate[s], on the streams before the
    // update; then every warp's sums in one spread reduction each.
    float dots[kE * kS], gates[kS];
#pragma unroll
    for (int i = 0; i < kE * kS; ++i) dots[i] = 0.0f;
#pragma unroll
    for (int s = 0; s < kS; ++s) gates[s] = 0.0f;
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      const int c = tid + j * kCT;
      if (c < chunks)
#pragma unroll
        for (int s = 0; s < kS; ++s)
          if (s < S) gates[s] += dot4(x[j][s], bf16x4(static_cast<const bf16*>(a.gate) + s * D + 4 * c));
    }
    float ce = 0.0f;  // entry tid's score offset, for the softmax
    if (tid < ne) ce = __ldcg(bank + static_cast<int64_t>(tid) * a.entry_stride + 2 * D);
    cp_async_wait_all();
    csync();
#pragma unroll
    for (int e = 0; e < kE; ++e) {
      if (e < ne) {
#pragma unroll
        for (int j = 0; j < kCh; ++j) {
          const int c = tid + j * kCT;
          if (c < chunks) {
            const float4 u = *reinterpret_cast<const float4*>(xbuf() + e * D + 4 * c);
#pragma unroll
            for (int s = 0; s < kS; ++s)
              if (s < S) dots[e * kS + s] += dot4(x[j][s], u);
          }
        }
      }
    }
    warp_sums_to(dots, red);
    {
      float g4[kS];
#pragma unroll
      for (int s = 0; s < kS; ++s) g4[s] = gates[s];
      const float v = warp_spread(g4);
      if (warp_spread_writer<kS>()) red[(tid >> 5) * 40 + kE * kS + warp_spread_index<kS>()] = v;
    }
    if (tid < ne) wts[2 * kS + kS * kE + tid] = ce;  // past the weights, norms and gates
    csync();  // every warp is done with the keys and has its sums in red
    stage_entries(bank, D, ne, a.entry_stride, D);  // the values P_e, over the keys
    if (tid < (kE + 1) * kS) {
      float t = 0.0f;
      for (int w = 0; w < kComputeWarps; ++w) t += red[w * 40 + tid];
      fin[tid] = t;
    }
    csync();
    // Softmax over the entries and the gate, per stream: one thread each.
    if (tid < S) {
      const int s = tid;
      const float inv_m = rsqrtf(static_cast<float>(a.mem_dim));
      const float inv_d = rsqrtf(static_cast<float>(D));
      float w[kE], top = -INFINITY;
#pragma unroll
      for (int e = 0; e < kE; ++e)
        if (e < ne) {
          w[e] = (fin[e * kS + s] + wts[2 * kS + kS * kE + e]) * inv_m;
          top = fmaxf(top, w[e]);
        }
      float total = 0.0f;
#pragma unroll
      for (int e = 0; e < kE; ++e)
        if (e < ne) {
          w[e] = expf(w[e] - top);
          total += w[e];
        }
#pragma unroll
      for (int e = 0; e < kE; ++e)
        if (e < ne) wts[s * kE + e] = w[e];
      wts[kS * kE + s] = 1.0f / total;
      wts[kS * kE + kS + s] = sigmoid(fin[kE * kS + s] * inv_d + bf(a.gate_bias, s));
    }
    cp_async_wait_all();
    csync();
#pragma unroll
    for (int j = 0; j < kCh; ++j) {
      const int c = tid + j * kCT;
      if (c < chunks) {
        float4 pr[kS];
#pragma unroll
        for (int s = 0; s < kS; ++s) pr[s] = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int e = 0; e < kE; ++e) {
          if (e < ne) {
            const float4 p = *reinterpret_cast<const float4*>(xbuf() + e * D + 4 * c);
#pragma unroll
            for (int s = 0; s < kS; ++s) {
              const float w = wts[s * kE + e];
              pr[s].x += w * p.x;
              pr[s].y += w * p.y;
              pr[s].z += w * p.z;
              pr[s].w += w * p.w;
            }
          }
        }
        float4 acc = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
        for (int s = 0; s < kS; ++s) {
          if (s < S) {
            const float norm = wts[kS * kE + s], gs = wts[kS * kE + kS + s];
            const float4 p = make_float4(pr[s].x * norm, pr[s].y * norm, pr[s].z * norm, pr[s].w * norm);
            x[j][s].x += gs * p.x;
            x[j][s].y += gs * p.y;
            x[j][s].z += gs * p.z;
            x[j][s].w += gs * p.w;
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

  // The mean over the streams (and its sum of squares), the end-of-pass difference (and its), and the
  // next read's input src (and its, for the norm): the three sums in one reduction.
  float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  float4 src[kCh];
#pragma unroll
  for (int j = 0; j < kCh; ++j) {
    src[j] = make_float4(0.f, 0.f, 0.f, 0.f);
    const int c = tid + j * kCT;
    if (c < chunks) {
      float4 st = make_float4(0.f, 0.f, 0.f, 0.f);
#pragma unroll
      for (int s = 0; s < kS; ++s) {
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
        const float4 p = prev[j];
        const float4 df = make_float4(st.x - p.x, st.y - p.y, st.z - p.z, st.w - p.w);
        *reinterpret_cast<float4*>(a.difference + static_cast<int64_t>(m) * D + 4 * c) = df;
        *reinterpret_cast<float4*>(a.previous + static_cast<int64_t>(m) * D + 4 * c) = st;
        sums[1] += dot4(df, df);
      }
      if (a.h) {
#pragma unroll
        for (int s = 0; s < kS; ++s) {
          if (s < S) {
            const float r = __ldg(a.read + s);
            src[j].x += r * x[j][s].x;
            src[j].y += r * x[j][s].y;
            src[j].z += r * x[j][s].z;
            src[j].w += r * x[j][s].w;
          }
        }
        sums[2] += dot4(src[j], src[j]);
      }
    }
  }
  if (a.rms || a.h) {
    block_sums(sums, red, fin);
    if (a.rms) {
      if (tid == 0) a.rms[m] = fin[0];
      if (tid == 1 && a.previous) a.rms[M + m] = fin[1];
    }
  }

  // The next read's normalized input, and the attention block's per-pass LoRA down product.
  if (a.h) {
    const float scale = rsqrtf(fin[2] / D + a.eps);
    float lora[kE];
#pragma unroll
    for (int r = 0; r < kE; ++r) lora[r] = 0.0f;
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
          for (int r = 0; r < kE; ++r)
            if (r < a.rank)
              lora[r] += dot4(hv, bf16x4(static_cast<const bf16*>(a.lora_down) + static_cast<int64_t>(r) * D + 4 * c));
      }
    }
    if (a.lora_down) {
      block_sums(lora, red, fin);
      if (tid < a.rank) a.lora[static_cast<int64_t>(m) * a.rank + tid] = round_bf16(fin[tid]);
    }
  }
}
// ---- Mamba-2 step -------------------------------------------------------------------------------

// One (token, head): the conv on the head's x channels and its group's B and C channels, the scan of
// the head's [P, N] state (which its block has L2 fetch ahead of time), and v = y * silu(z).
template <int NI>
__device__ __noinline__ void ssm_task(int task) {
  const Ssm& a = desc().ssm;
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
  float* gst = a.ssm_state + (static_cast<int64_t>(slot) * H + h) * Pd * N + static_cast<int64_t>(p) * N;
  float S[NI][4];
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const float4 v = has ? ldcg4(gst + 4 * (q + lpr * i)) : make_float4(0.f, 0.f, 0.f, 0.f);
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
// A token's channels are split over kConvParts tasks (each channel's state is its own).
__device__ __noinline__ void conv_bc_task(int task) {
  const Ssm& a = desc().ssm;
  const int gn = a.G * a.N, conv_dim = a.I + 2 * gn, W = a.W;
  const int m = task / kConvParts, chunk = (2 * gn + kConvParts - 1) / kConvParts;
  const int c0 = a.I + task % kConvParts * chunk, c1 = min(conv_dim, c0 + chunk);
  const bool has = ctx().has_state[m] != 0;
  float* cst = a.conv_state + static_cast<int64_t>(ctx().slots[m]) * (W - 1) * conv_dim;
  const float* zrow = a.zxbcdt + static_cast<int64_t>(m) * a.ld;
  for (int c = c0 + threadIdx.x; c < c1; c += kCT) {
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

// lx_moe_route for every token (one warp each), then the expert summary and the history. A token's
// logits and selection bias are read once, into registers (E <= 32 * kPer).
__device__ __noinline__ void select_task() {
  constexpr int kPer = 8;
  const Select& a = desc().sel;
  const int M = ctx().M;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  int* picks = reinterpret_cast<int*>(scratch());  // [kMaxM][32]
  float* pw = scratch() + kMaxM * 32;               // [kMaxM][32]
  const int E = a.E, top_k = a.top_k;
  if (warp < M) {
    const int t = warp;
    const float* row = a.logits + static_cast<int64_t>(t) * E;
    float v[kPer], sb[kPer];
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const int i = lane + 32 * j;
      v[j] = i < E ? __ldcg(row + i) : -INFINITY;
      sb[j] = i < E && a.selection_bias ? __ldg(a.selection_bias + i) : 0.0f;
    }
    float mx = -INFINITY;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (lane + 32 * j < E) mx = fmaxf(mx, v[j]);
    mx = lx::warp_max(mx);
    float z = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (lane + 32 * j < E) z += expf(v[j] - mx);
    z = lx::warp_sum(z);
    float hs = 0.0f;
#pragma unroll
    for (int j = 0; j < kPer; ++j)
      if (lane + 32 * j < E) {
        const float pr = expf(v[j] - mx) / z;
        hs -= pr * logf(fmaxf(pr, 1e-9f));
      }
    hs = lx::warp_sum(hs);
    if (lane == 0) {
      a.entropy[t] = hs;
      a.confidence[t] = 1.0f / z;
    }
    int* pick = picks + t * 32;
    unsigned taken = 0;  // this lane's values already picked, a bit per j
    for (int r = 0; r < top_k; ++r) {
      float best = -INFINITY;
      int best_i = E;
#pragma unroll
      for (int j = 0; j < kPer; ++j) {
        const int i = lane + 32 * j;
        if (i < E && !(taken >> j & 1u)) {
          const float sv = v[j] + sb[j];
          if (sv > best || (sv == best && i < best_i)) {
            best = sv;
            best_i = i;
          }
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
      if (best_i < E && (best_i & 31) == lane) taken |= 1u << (best_i >> 5);
      if (lane == 0) pick[r] = best_i;
    }
    __syncwarp();
    const bool live = lane < top_k;
    const int id = live ? pick[lane] : 0;
    float l = -INFINITY;  // logit id, from the lane that holds it
#pragma unroll
    for (int j = 0; j < kPer; ++j) {
      const float o = __shfl_sync(0xffffffffu, v[j], id & 31);
      if (live && (id >> 5) == j) l = o;
    }
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

// Gate and up rows [i0, i0 + rows) of one routed expert (task = pair * blocks_per_pair + row block,
// x = the pair's token), on the FMA reader.
template <int kType>
__device__ __noinline__ void expup_task(int op, int task) {
  using F = typename Word<kType>::F;
  const ExpUp& a = desc().up;
  const int rb = task % a.blocks_per_pair, pair = task / a.blocks_per_pair, t = pair / a.top_k;
  const int i0 = rb * a.rows, rows = a.rows;
  const int e = __ldcg(a.ids + pair);
  if (misc() != op * 16 + t + 1) {  // the pair's token's latent row
    stage_bf16(static_cast<const bf16*>(a.latent) + static_cast<int64_t>(t) * a.L, a.L);
    if (threadIdx.x == 0) misc() = op * 16 + t + 1;
  }
  const int64_t row0 = static_cast<int64_t>(e) * 2 * a.I + i0;  // the first gate row; up rows are I on
  const uint8_t* base = static_cast<const uint8_t*>(a.w.data);
  auto row_of = [&](int v) -> RowRef {
    const int64_t row = v < rows ? row0 + v : row0 + a.I + (v - rows);
    return row_ref<kType>(base, a.w, row, a.row_bytes, 0, a.L, a.L / F::kValues, 0);
  };
  float* red = scratch();
  dots_rows<F, true, 1, 8>(2 * rows, row_of, xbuf(), a.L, 1, red, false);
  csync();
  for (int i = threadIdx.x; i < rows; i += kCT) {
    float gt = red[i], up = red[rows + i];
    if constexpr (kRowScaled<kType>) {
      const float* sc = static_cast<const float*>(a.w.scale);
      gt *= __ldg(sc + row0 + i);
      up *= __ldg(sc + row0 + a.I + i);
    }
    if constexpr (kType == LX_W_NVFP4) {
      const float gl = expert_global(a.w, e);
      gt *= gl;
      up *= gl;
    }
    static_cast<bf16*>(a.inter)[static_cast<int64_t>(pair) * a.I + i0 + i] = __float2bfloat16_rn(lx::silu(gt) * up);
  }
}

// Down rows [l0, l0 + rows) of each of token t's experts, weighted and summed (plus the shared expert).
template <int kType>
__device__ __noinline__ void expdown_task(int op, int task) {
  using F = typename Word<kType>::F;
  const ExpDown& a = desc().down;
  const int t = task / a.blocks_per_token, rb = task % a.blocks_per_token;
  const int l0 = rb * a.rows, rows = a.rows, K = a.top_k;
  int* ids = reinterpret_cast<int*>(scratch() + kScratchFloats - 16);  // the token's experts
  if (threadIdx.x < K) ids[threadIdx.x] = __ldcg(a.ids + static_cast<int64_t>(t) * K + threadIdx.x);
  if (misc() != op * 16 + t + 1) {
    stage_bf16(static_cast<const bf16*>(a.inter) + static_cast<int64_t>(t) * K * a.I, K * a.I);
    if (threadIdx.x == 0) misc() = op * 16 + t + 1;
  } else {
    csync();
  }
  const uint8_t* base = static_cast<const uint8_t*>(a.w.data);
  auto row_of = [&](int v) -> RowRef {
    const int k = v / rows;
    const int64_t row = static_cast<int64_t>(ids[k]) * a.L + l0 + v % rows;
    return row_ref<kType>(base, a.w, row, a.row_bytes, 0, a.I, a.I / F::kValues, k);
  };
  float* red = scratch();
  dots_rows<F, true, 1, 8>(K * rows, row_of, xbuf(), a.I, 1, red, false);
  csync();
  for (int i = threadIdx.x; i < rows; i += kCT) {
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      float c = __ldcg(a.route_w + static_cast<int64_t>(t) * K + k);
      if constexpr (kRowScaled<kType>)
        c *= __ldg(static_cast<const float*>(a.w.scale) + static_cast<int64_t>(ids[k]) * a.L + l0 + i);
      if constexpr (kType == LX_W_NVFP4) c *= expert_global(a.w, ids[k]);
      acc += c * red[k * rows + i];
    }
    const int64_t at = static_cast<int64_t>(t) * a.L + l0 + i;
    if (a.add) acc += __ldcg(a.add + at);
    a.out[at] = acc;
  }
}

// ---- Experts on the tensor cores ----------------------------------------------------------------
//
// A block's whole run of expert tasks at once, so all of its weight bytes are in flight together:
// these rows are known only once the routing is, so nothing fetched them ahead. A unit is one
// expert's 16 rows by one of 2^sh parts of k (fixed by the shape), and only its token's x column is
// read. A block's tasks of an op are one ascending range (lx_program_build), so a run's tasks cover
// a range of tokens, whose x rows are staged and whose picks are read once.

constexpr int kExpReserve = 2 * kMaxM * 8;  // scratch floats at the end: the run's picks, route weights

// Rows [r0, r1) of a bf16 matrix of n values a row (n % 8 == 0, 16-byte aligned) into the x buffer,
// rows xst values apart: as bf16, or as f16 for the int8 reader (clamped to its range).
template <bool kF16>
__device__ __forceinline__ void stage_rows(const void* src, int r0, int r1, int n, int xst) {
  const int per = n / 8;
  const uint4* s = static_cast<const uint4*>(src) + static_cast<int64_t>(r0) * per;
  for (int e = threadIdx.x; e < (r1 - r0) * per; e += kCT) {
    const int r = e / per, c = e - r * per;
    uint4 v = __ldcg(s + e);
    if constexpr (kF16) {
      constexpr float kMax = 65504.0f;
      uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const float lo = fminf(fmaxf(__uint_as_float(w[i] << 16), -kMax), kMax);
        const float hi = fminf(fmaxf(__uint_as_float(w[i] & 0xffff0000u), -kMax), kMax);
        const __half2 h = __floats2half2_rn(lo, hi);
        w[i] = *reinterpret_cast<const uint32_t*>(&h);
      }
      v = make_uint4(w[0], w[1], w[2], w[3]);
    }
    *reinterpret_cast<uint4*>(reinterpret_cast<uint16_t*>(xbuf()) + r * xst + c * 8) = v;
  }
}

// Gate|up: a task is gate rows [i0, i0 + 8) of its pair's expert in A rows 0..7 and the same up rows
// in A rows 8..15, so lane (g, 0) ends with gate and up of row i0 + g side by side.
template <int kType>
__device__ __noinline__ void expup_run(int op, const uint32_t* entries, int count, int index) {
  constexpr int kV = Chunk<kType>::kValues;
  const ExpUp& a = desc().up;
  const int tid = threadIdx.x, lane = tid & 31, gr = lane >> 2, tq = lane & 3;
  const int K = a.top_k, bpp = a.blocks_per_pair, L = a.L, I = a.I, sh = a.sh, S = 1 << sh;
  const int xst = L + (kXsF16<kType> ? 8 : 32), spr = L / (4 * kV);
  const int64_t t0 = trace_now();
  const uint16_t* tasks = run_tasks();
  const int p0 = tasks[0] / bpp, p1 = tasks[count - 1] / bpp + 1;
  const int tlo = p0 / K, thi = (p1 - 1) / K + 1;
  int* eid = reinterpret_cast<int*>(scratch() + kScratchFloats - kExpReserve);  // pair p0 + i's expert
  for (int i = tid; i < p1 - p0; i += kCT) eid[i] = __ldcg(a.ids + p0 + i);
  stage_rows<kXsF16<kType>>(a.latent, tlo, thi, L, xst);
  if (tid == 0) misc() = -1;
  csync();
  const int64_t t1 = trace_now();
  const int cap = (kScratchFloats - kExpReserve) / (16 * S);  // tasks per batch
  // the first batch's row scales, loaded while its products are read
  float sg = 1.0f, su = 1.0f;
  if constexpr (kRowScaled<kType>) {
    if (tid < min(cap, count) * 8) {
      const int task = tasks[tid >> 3], pair = task / bpp;
      const int64_t row = static_cast<int64_t>(eid[pair - p0]) * 2 * I + (task % bpp) * 8 + (tid & 7);
      const float* sc = static_cast<const float*>(a.w.scale) + row;
      sg = __ldg(sc);
      su = __ldg(sc + I);
    }
  }
  const uint8_t* base = static_cast<const uint8_t*>(a.w.data);
  for (int b0 = 0; b0 < count; b0 += cap) {
    const int nb = min(cap, count - b0);
    float* red = scratch();
    auto unit_of = [&](int u) -> UnitRef {
      const int task = tasks[b0 + (u >> sh)], p = u & (S - 1), pair = task / bpp;
      const int64_t row = static_cast<int64_t>(eid[pair - p0]) * 2 * I + (task % bpp) * 8 + gr;
      const int s0 = (p * spr) >> sh, s1 = ((p + 1) * spr) >> sh, kb = s0 * 4 * kV;
      return unit_rows<kType>(base, a.w, row, row + I, a.row_bytes, kb, s1 - s0, pair / K - tlo, L);
    };
    auto sink = [&](int u, const float (&c)[4]) {
      if (tq == 0) {
        const int at = ((u >> sh) * 8 + gr) * 2 * S + (u & (S - 1));
        red[at] = c[0];
        red[at + S] = c[2];
      }
    };
    dots_mma<kType, 4, 1, 1>(nb << sh, unit_of, sink, reinterpret_cast<const uint16_t*>(xbuf()), xst, thi - tlo);
    csync();
    for (int e = tid; e < nb * 8; e += kCT) {
      const int task = tasks[b0 + (e >> 3)], pair = task / bpp, i = (task % bpp) * 8 + (e & 7);
      float gt = 0.0f, up = 0.0f;
      for (int p = 0; p < S; ++p) {
        gt += red[e * 2 * S + p];
        up += red[e * 2 * S + S + p];
      }
      if constexpr (kRowScaled<kType>) {
        if (b0 == 0 && e == tid) {
          gt *= sg;
          up *= su;
        } else {
          const float* sc = static_cast<const float*>(a.w.scale) + static_cast<int64_t>(eid[pair - p0]) * 2 * I + i;
          gt *= __ldg(sc);
          up *= __ldg(sc + I);
        }
      }
      if constexpr (kType == LX_W_NVFP4) {
        const float gl = expert_global(a.w, eid[pair - p0]);
        gt *= gl;
        up *= gl;
      }
      static_cast<bf16*>(a.inter)[static_cast<int64_t>(pair) * I + i] = __float2bfloat16_rn(lx::silu(gt) * up);
    }
    if (b0 + cap < count) csync();  // the next batch reuses the scratch
  }
  if (ctx().trace) {
    const int64_t t2 = global_ns();
    for (int i = 0; i < count; ++i)
      trace_task(index + i, op, static_cast<int>(__ldg(entries + i) & 0xffff), t0, t1, t2);
  }
}

// Down: a task is rows [l0, l0 + 16) of each of its token's top_k experts, a unit per expert, summed
// with the route weights in pick order, then the shared expert added.
template <int kType>
__device__ __noinline__ void expdown_run(int op, const uint32_t* entries, int count, int index) {
  constexpr int kV = Chunk<kType>::kValues;
  const ExpDown& a = desc().down;
  const int tid = threadIdx.x, lane = tid & 31, gr = lane >> 2, tq = lane & 3;
  const int K = a.top_k, bpt = a.blocks_per_token, L = a.L, I = a.I, sh = a.sh, S = 1 << sh;
  const int xst = I + (kXsF16<kType> ? 8 : 32), spr = I / (4 * kV);
  const int64_t t0 = trace_now();
  const uint16_t* tasks = run_tasks();
  const int tlo = tasks[0] / bpt, thi = tasks[count - 1] / bpt + 1;
  int* eid = reinterpret_cast<int*>(scratch() + kScratchFloats - kExpReserve);  // [(t - tlo) * K + k]
  float* rw = scratch() + kScratchFloats - kExpReserve / 2;
  for (int i = tid; i < (thi - tlo) * K; i += kCT) {
    eid[i] = __ldcg(a.ids + tlo * K + i);
    rw[i] = __ldcg(a.route_w + tlo * K + i);
  }
  stage_rows<kXsF16<kType>>(a.inter, tlo * K, thi * K, I, xst);
  if (tid == 0) misc() = -1;
  csync();
  const int64_t t1 = trace_now();
  const int cap = (kScratchFloats - kExpReserve) / (16 * K * S);  // tasks per batch
  // the first batch's epilogue inputs (the shared expert's value, the row scales), ahead of its products
  constexpr int kMaxK = 8;
  float pre_add = 0.0f, pre_sc[kMaxK];
#pragma unroll
  for (int k = 0; k < kMaxK; ++k) pre_sc[k] = 1.0f;
  if (tid < min(cap, count) * 16) {
    const int task = tasks[tid >> 4], t = task / bpt, l = (task % bpt) * 16 + (tid & 15);
    if (a.add) pre_add = __ldcg(a.add + static_cast<int64_t>(t) * L + l);
    if constexpr (kRowScaled<kType>) {
#pragma unroll
      for (int k = 0; k < kMaxK; ++k)
        if (k < K)
          pre_sc[k] =
              __ldg(static_cast<const float*>(a.w.scale) + static_cast<int64_t>(eid[(t - tlo) * K + k]) * L + l);
    }
  }
  const uint8_t* base = static_cast<const uint8_t*>(a.w.data);
  for (int b0 = 0; b0 < count; b0 += cap) {
    const int nb = min(cap, count - b0);
    float* red = scratch();
    auto unit_of = [&](int u) -> UnitRef {
      const int q = u >> sh, p = u & (S - 1), task = tasks[b0 + q / K], k = q % K;
      const int t = task / bpt, x = (t - tlo) * K + k;
      const int64_t row = static_cast<int64_t>(eid[x]) * L + (task % bpt) * 16 + gr;
      const int s0 = (p * spr) >> sh, s1 = ((p + 1) * spr) >> sh, kb = s0 * 4 * kV;
      return unit_rows<kType>(base, a.w, row, row + 8, a.row_bytes, kb, s1 - s0, x, I);
    };
    auto sink = [&](int u, const float (&c)[4]) {
      if (tq == 0) {
        const int at = ((u >> sh) * 16 + gr) * S + (u & (S - 1));
        red[at] = c[0];
        red[at + 8 * S] = c[2];
      }
    };
    dots_mma<kType, 4, 1, 1>((nb * K) << sh, unit_of, sink, reinterpret_cast<const uint16_t*>(xbuf()), xst,
                             (thi - tlo) * K);
    csync();
    for (int e = tid; e < nb * 16; e += kCT) {
      const int j = e >> 4, r = e & 15, task = tasks[b0 + j], t = task / bpt, l = (task % bpt) * 16 + r;
      const bool pre = b0 == 0 && e == tid;
      float acc = 0.0f;
#pragma unroll
      for (int k = 0; k < kMaxK; ++k) {
        if (k >= K) break;
        float c = rw[(t - tlo) * K + k];
        if constexpr (kRowScaled<kType>)
          c *= pre ? pre_sc[k]
                   : __ldg(static_cast<const float*>(a.w.scale) + static_cast<int64_t>(eid[(t - tlo) * K + k]) * L + l);
        if constexpr (kType == LX_W_NVFP4) c *= expert_global(a.w, eid[(t - tlo) * K + k]);
        float v = 0.0f;
        for (int p = 0; p < S; ++p) v += red[((j * K + k) * 16 + r) * S + p];
        acc += c * v;
      }
      const int64_t at = static_cast<int64_t>(t) * L + l;
      if (a.add) acc += pre ? pre_add : __ldcg(a.add + at);
      a.out[at] = acc;
    }
    if (b0 + cap < count) csync();
  }
  if (ctx().trace) {
    const int64_t t2 = global_ns();
    for (int i = 0; i < count; ++i)
      trace_task(index + i, op, static_cast<int>(__ldg(entries + i) & 0xffff), t0, t1, t2);
  }
}

// ---- Attention ----------------------------------------------------------------------------------

// One (token, query head): writes the token's k and v to the caches (every head's task writes the same
// bytes for its kv head, and reads back only its own write), then attends over positions 0 .. start_pos
// in chunks of kCT positions with an online softmax across chunks.
__device__ __noinline__ void attention_task(int task) {
  const Attention& a = desc().attn;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  const int KVH = a.KVH, Dh = a.Dh, H = a.H, G = 1;
  const int m = task / H, h = task % H, kvh = h / (H / KVH);
  const int pos = ctx().start_pos[m], L = pos + 1;
  const int64_t slot = ctx().slots[m];
  const float* row = a.qkv + static_cast<int64_t>(m) * a.ld_qkv;
  const int64_t pstride = static_cast<int64_t>(KVH) * Dh;
  bf16* kc = static_cast<bf16*>(a.k_cache) + slot * a.capacity * pstride + kvh * Dh;
  bf16* vc = static_cast<bf16*>(a.v_cache) + slot * a.capacity * pstride + kvh * Dh;
  float* qs = xbuf();               // [G][Dh], scaled
  float* sc = qs + G * Dh;        // [kCT][G] scores, then weights; reused by the final merge
  float* run = scratch() + 256;  // running max [G], running sum [G], new max [G], correction [G]
  for (int i = tid; i < Dh; i += kCT) qs[i] = __ldcg(row + h * Dh + i) * a.scale;
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
  bf16* out = static_cast<bf16*>(a.out) + static_cast<int64_t>(m) * H * Dh + h * Dh;
  for (int i = tid; i < G * Dh; i += kCT) {
    const int g = i / Dh, dd = i % Dh;
    float s = 0.0f;
    for (int q = 0; q < pg_count; ++q) s += mg[(q * G + g) * Dh + dd];
    out[i] = __float2bfloat16_rn(s / run[G + g]);
  }
}
// ---- Depth-memory entry and the depth head ------------------------------------------------------

// Rows [r0, r0 + rows) of a memory entry for every token: the state projection (the state staged a
// chunk of kc columns at a time when it does not fit at once) and the metadata projection.
template <int kType>
__device__ __noinline__ void entry_task(int op, int task) {
  using F = typename Word<kType>::F;
  const Entry& a = desc().entry;
  const int M = ctx().M;
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
  float* red2 = scratch() + a.rows * M;
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
    auto row_of = [&](int v) -> RowRef {
      if (v >= rows) return {nullptr, kc / F::kValues, 0, 0};
      return row_ref<kType>(static_cast<const uint8_t*>(a.w_state.data), a.w_state, r0 + v, a.state_bytes, c0, D,
                            kc / F::kValues, 0);
    };
    dots_rows_m<F, false>(rows, row_of, xs, kc, M, red, c0 > 0);
  }
  if (whole && threadIdx.x == 0) misc() = op * 16;
  const char* mbase = static_cast<const char*>(a.w_meta);
  auto meta_of = [&](int v) -> RowRef {
    if (v >= rows) return {nullptr, Km / 4, 0, 0};
    return {reinterpret_cast<const uint2*>(mbase + static_cast<int64_t>(r0 + v) * Km * 2), Km / 4, 0, 0};
  };
  dots_rows_m<WordFormat<LX_W_BF16>, false>(rows, meta_of, meta, Km, M, red2, false);
  csync();
  for (int e = threadIdx.x; e < rows * M; e += kCT) {
    const int i = e / M, m = e % M, r = r0 + i;
    float ds = red[i * M + m];
    if constexpr (kRowScaled<kType>) ds *= __ldg(static_cast<const float*>(a.w_state.scale) + r);
    if constexpr (kType == LX_W_NVFP4) ds *= a.w_state.global;
    const float v = ds + bf(a.b_state, r);
    const float vm = red2[i * M + m] + bf(a.b_meta, r);
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
  block_sums(part, red, fin);
  if (threadIdx.x < M) {
    const int m = threadIdx.x;
    const float logit = fin[m] + a.b_out[0];
    const float scale = fmaxf(sqrtf(__ldcg(a.rms + m) / a.D), 1e-6f);
    const float innovation = sqrtf(__ldcg(a.rms + M + m) / a.D);
    const float z = fminf(fmaxf(logit + tanhf(innovation / scale) - 0.5f, -a.limit), a.limit);
    a.out[m] = sigmoid(z);
  }
}
// ---- L2 prefetch ---------------------------------------------------------------------------------

// Has L2 fetch [p, p + bytes), from one lane: bulk prefetches of up to 4 KB, or a prefetch per line
// before sm_90.
__device__ __forceinline__ void prefetch_lane(uint64_t p, uint32_t bytes, int bulk) {
  const uint64_t a0 = p & ~static_cast<uint64_t>(127), a1 = (p + bytes + 127) & ~static_cast<uint64_t>(127);
  if (bulk) {
    for (uint64_t a = a0; a < a1; a += 4096)
      bulk_prefetch_l2(reinterpret_cast<const void*>(a),
                       static_cast<uint32_t>(min(static_cast<uint64_t>(4096), a1 - a)));
  } else {
    for (uint64_t a = a0; a < a1; a += 128) prefetch_l2_line(reinterpret_cast<const void*>(a));
  }
}

// One warp has L2 fetch the block's spans from *pf on, up to those starting past `limit` bytes, a
// span per lane; spans of entries before `start` are behind the compute warps and are skipped.
__device__ __forceinline__ void prefetch_spans(const Program& P, int block, int& pf, uint32_t start, uint32_t limit) {
  const int lane = threadIdx.x & 31;
  const PfSpan* spans = P.spans + P.span_off[block];
  const int ns = P.span_off[block + 1] - P.span_off[block];
  while (pf < ns) {
    const int s = pf + lane;
    PfSpan sp;
    bool due = false;
    if (s < ns) {
      sp = spans[s];
      due = sp.cum <= limit;
    }
    const unsigned took = __ballot_sync(0xffffffffu, due);  // a prefix: cum only grows
    if (due && sp.cum >= start) {
      uint64_t a = sp.addr;
      bool go = true;
      if (sp.m > 0) {
        go = ctx().has_state[sp.m - 1] != 0;
        a += static_cast<uint64_t>(ctx().slots[sp.m - 1]) * sp.stride;
      }
      if (go) prefetch_lane(a, sp.bytes, P.bulk);
    }
    pf += __popc(took);
    if (took != 0xffffffffu) break;
  }
}

// ---- The compute warps --------------------------------------------------------------------------

// A block's run of tasks of an op other than a product, one task at a time.
__device__ __noinline__ void run_segment(int op, const uint32_t* entries, int count, int index) {
  const OpDesc& d = desc();
  for (int i = 0; i < count; ++i) {
    const int task = static_cast<int>(__ldg(entries + i) & 0xffff);
    const int64_t t0 = trace_now();
    switch (d.kind) {
      case kEmbed: embed_task(task); break;
      case kMemRead:
        memread_task(task);
        if (threadIdx.x == 0) misc() = -1;  // the x buffer held its entries
        break;
      case kSsm: {
        const int lpr = kCT / d.ssm.P, ni = d.ssm.N / (4 * lpr);
        if (ni == 1) ssm_task<1>(task);
        else if (ni == 2) ssm_task<2>(task);
        else if (ni == 4) ssm_task<4>(task);
        else ssm_task<8>(task);
        break;
      }
      case kConvBC: conv_bc_task(task); break;
      case kSelect: select_task(); break;
      case kExpUp:
        LX_DISPATCH_W(d.up.wtype, expup_task, op, task);
        break;
      case kExpDown:
        LX_DISPATCH_W(d.down.wtype, expdown_task, op, task);
        break;
      case kAttention:
        attention_task(task);
        if (threadIdx.x == 0) misc() = -1;  // the x buffer was its scratch
        break;
      case kEntry:
        LX_DISPATCH_W(d.entry.wtype, entry_task, op, task);
        break;
      case kContinuation: continuation_task(); break;
      default: break;
    }
    csync();
    trace_task(index + i, op, task, t0, t0, trace_now());
  }
}

// The compute warps walk the block's list one run of an op at a time. Before a run: everyone is done
// with the previous one (the barrier), which is also when the previous op's counter is released;
// then the descriptor and the run's task numbers, and the wait for the ops this one reads. A run
// longer than kMaxRunTasks is cut in pieces; the pieces after the first neither release nor wait.
__device__ __forceinline__ void work(const Program& P, int block) {
  const int tid = threadIdx.x;
  const int base = P.list_off[block];
  const uint32_t* list = P.lists + base;
  const int n = P.list_off[block + 1] - base;
  const uint32_t* ahead = P.ahead + base;
  const uint32_t budget = static_cast<uint32_t>(P.prefetch_bytes);
  int prev = -1;
  int pf = 0;  // warp 1: the next of the block's prefetch spans
  for (int i = 0; i < n;) {
    const int op = static_cast<int>(__ldg(list + i) >> 16);
    // The run's end: the first entry past i of another op, or the run's limit. Each warp reads 32
    // entries a load and finds the first other op by ballot, so a run of R tasks costs R / 32 loads
    // in a row rather than R (a run of 118 product tasks spent 15 us here, the weights not yet asked for).
    int j = i + 1;
    for (int b = i + 1;; b += 32) {
      const int k = b + (tid & 31);
      bool stop = true;
      if (k < n && k - i < kMaxRunTasks) stop = static_cast<int>(__ldg(list + k) >> 16) != op;
      const unsigned m = __ballot_sync(0xffffffffu, stop);
      if (m) {
        j = b + __ffs(m) - 1;
        break;
      }
    }
    const bool more = op == prev;  // a further piece of the same run
    csync();
    // Everything the previous run wrote is visible to this thread through the barrier; the release
    // carries it to the SMs that acquire the counter (CUTLASS's semaphore pattern).
    // An op nothing waits on signals nothing: the descriptor still in shared memory is the previous op's.
    if (tid == 0 && prev >= 0 && !more && desc().waited) red_release_add(&P.done[prev], 1u);
    // While warp 0 waits for the dependencies, warp 1 has L2 fetch the weights of the coming runs, up
    // to the budget past this one's start, so DRAM keeps streaming through waits and small ops.
    if (tid >> 5 == 1 && budget) {
      const uint32_t start = __ldg(ahead + i);
      prefetch_spans(P, block, pf, start, start + budget);
    }
    int64_t* run_rec =
        P.trace && tid == 0 ? P.trace + static_cast<int64_t>(P.list_off[gridDim.x] + base + i) * 4 : nullptr;
    if (run_rec) run_rec[0] = global_ns();
    for (int k = tid; k < j - i; k += kCT) run_tasks()[k] = static_cast<uint16_t>(__ldg(list + i + k) & 0xffff);
    if (!more) {
      const int4* src = reinterpret_cast<const int4*>(P.ops + op);
      int4* dst = reinterpret_cast<int4*>(dsm + kDescOff);
      for (int k = tid; k < static_cast<int>(sizeof(OpDesc) / 16); k += kCT) dst[k] = __ldg(src + k);
      csync();  // the descriptor, its dependencies with their block counts included, is in shared memory
      if (tid < 32) {
        // a lane per dependency, so they are waited for together; the barrier after it hands what
        // each lane acquired to the whole block
        const OpDesc& od = desc();
        for (int k = tid; k < od.ndeps; k += 32) {
          const int dep = od.deps[k] & 0xffff;
          const uint32_t need = static_cast<uint32_t>(od.deps[k]) >> 16;
          while (ld_acquire(&P.done[dep]) < need) {
          }
        }
        __syncwarp();
      }
    }
    if (run_rec) run_rec[1] = global_ns();
    csync();
    const OpDesc& od = desc();
    if (od.kind == kGemv) gemv_run(op, list + i, j - i, i);
    else if (od.kind == kExpUp && od.up.mma) {
      LX_DISPATCH_W(od.up.wtype, expup_run, op, list + i, j - i, i);
    } else if (od.kind == kExpDown && od.down.mma) {
      LX_DISPATCH_W(od.down.wtype, expdown_run, op, list + i, j - i, i);
    } else {
      run_segment(op, list + i, j - i, i);
    }
    prev = op;
    i = j;
  }
  csync();
  if (tid == 0) {
    if (prev >= 0 && desc().waited) red_release_add(&P.done[prev], 1u);
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
  if (threadIdx.x == 0) {
    Ctx* c = reinterpret_cast<Ctx*>(dsm + kCtxOff);
    c->tokens = P.tokens;
    c->slots = P.slots;
    c->has_state = P.has_state;
    c->start_pos = P.start_pos;
    c->trace = P.trace;
    c->M = P.M;
    c->trace_base = P.list_off[blockIdx.x];
    misc() = -1;
  }
  __syncthreads();
  work(P, blockIdx.x);
}

size_t step_smem_bytes(int x_bytes) { return static_cast<size_t>(kXOff) + (x_bytes + 127) / 128 * 128; }

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

}  // namespace LX_DECODE_NS

}  // namespace lx::decode
