// Helpers every kernel uses. The library is one source: args.h, this file, then each kernel file.
#include <metal_stdlib>
using namespace metal;

// ---- bf16 -------------------------------------------------------------------------------------
// bf16 is stored as ushort: the top half of an f32. Widening is a shift; narrowing rounds to nearest
// even (as __float2bfloat16_rn does), and a NaN stays a quiet NaN.
inline float bf(ushort v) { return as_type<float>(uint(v) << 16); }
inline float bf_lo(uint w) { return as_type<float>(w << 16); }
inline float bf_hi(uint w) { return as_type<float>(w & 0xffff0000u); }
inline ushort to_bf(float f) {
  uint u = as_type<uint>(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return ushort((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return ushort(u >> 16);
}

// A value of dtype `is_bf16` (lx_dtype: 0 f32, 1 bf16) at index i.
inline float load_val(const device void* p, long i, int is_bf16) {
  return is_bf16 ? bf(((const device ushort*)p)[i]) : ((const device float*)p)[i];
}
inline void store_val(device void* p, long i, float v, int is_bf16) {
  if (is_bf16) ((device ushort*)p)[i] = to_bf(v);
  else ((device float*)p)[i] = v;
}

inline float silu(float v) { return v / (1.0f + exp(-v)); }
inline float sigmoid(float v) { return 1.0f / (1.0f + exp(-v)); }
// log(1 + x) without the cancellation of the plain form (Kahan): exact enough where 1 + x rounds.
inline float log1p_(float x) {
  const float u = 1.0f + x;
  return u == 1.0f ? x : log(u) * (x / (u - 1.0f));
}
inline float softplus(float v) { return v > 20.0f ? v : log1p_(exp(v)); }

// ---- 8-bit and 4-bit floats ------------------------------------------------------------------
// e4m3 (FP8, no infinities, 0x7f / 0xff are NaN) to f32, exact. A normal value's exponent and mantissa
// land in an f32 whose exponent field is rebased by adding 120; subnormals (exponent 0) are m * 2^-9.
// No f32 subnormal is ever formed, since the GPU flushes those to zero.
inline float e4m3(uint b) {
  const uint em = b & 0x7fu;
  const float mag = em >= 8u ? as_type<float>((em << 20) + (120u << 23)) : float(em) * 0x1p-9f;
  return (b & 0x80u) ? -mag : mag;
}
// e2m1 (FP4) code (low 4 bits) to f32, exact, by the same rebasing (+126); code 1 is the subnormal 0.5.
inline float e2m1(uint code) {
  const uint em = code & 7u;
  const float mag = em >= 2u ? as_type<float>((em << 22) + (126u << 23)) : float(em) * 0.5f;
  return (code & 8u) ? -mag : mag;
}

// The nearest e4m3 of x, ties to even, saturating to +-448 (as __nv_cvt_float_to_fp8 with
// __NV_SATFINITE). NaN maps to the NaN code.
inline uint e4m3_encode(float x) {
  const uint sign = as_type<uint>(x) & 0x80000000u ? 0x80u : 0u;
  float a = fabs(x);
  if (isnan(x)) return 0x7fu;
  if (a >= 448.0f) return sign | 0x7eu;  // 448 = the largest finite e4m3
  // Below 2^-6 the format is subnormal: steps of 2^-9. Round the magnitude to that grid.
  if (a < 0x1p-6f) {
    const float q = rint(a * 512.0f);  // rint: ties to even
    return sign | uint(q);             // 0..8; 8 is the smallest normal (exponent 1, mantissa 0)
  }
  // Normal: keep 3 mantissa bits of the f32, rounding to nearest even on the dropped 20 bits.
  uint u = as_type<uint>(a);
  u += 0x7ffffu + ((u >> 20) & 1u);
  const int e = int((u >> 23) & 0xffu) - 127 + 7;  // e4m3 exponent field
  const uint m = (u >> 20) & 7u;
  uint code = (uint(e) << 3) | m;
  if (code > 0x7eu) code = 0x7eu;  // rounding up past 448 saturates
  return sign | code;
}

// The nearest e2m1 code of x: 0, .5, 1, 1.5, 2, 3, 4, 6 and a sign bit (bit 3). A value on a
// midpoint goes to the even code; saturates at 6; zero, or anything that rounds to it, is +0.
inline uint e2m1_encode(float x) {
  const float mag = min(fabs(x), 6.0f);
  const float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  uint idx = 0;
  for (int i = 0; i < 7; ++i) idx += mag > mids[i] ? 1u : 0u;
  uint tie = 0;
  for (int i = 0; i < 7; ++i) tie |= (uint(i) == idx && mag == mids[i] && (i & 1)) ? 1u : 0u;
  idx += tie;
  const uint sign = (x < 0.0f && idx > 0) ? 8u : 0u;
  return idx | sign;
}

// ---- reductions ------------------------------------------------------------------------------
// Sum over a threadgroup of up to 1024 threads (simdgroups of 32); every thread gets the result.
// `scratch` holds 32 floats. The order is fixed, so the result is the same on every run.
inline float group_sum(float v, threadgroup float* scratch, uint tid, uint threads) {
  const uint lane = tid & 31u, sg = tid >> 5;
  v = simd_sum(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint groups = (threads + 31u) >> 5;
  float total = 0.0f;
  for (uint i = 0; i < groups; ++i) total += scratch[i];
  return total;
}

inline float group_max(float v, threadgroup float* scratch, uint tid, uint threads) {
  const uint lane = tid & 31u, sg = tid >> 5;
  v = simd_max(v);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) scratch[sg] = v;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint groups = (threads + 31u) >> 5;
  float m = -INFINITY;
  for (uint i = 0; i < groups; ++i) m = max(m, scratch[i]);
  return m;
}

// ---- runtime kernels -------------------------------------------------------------------------
// Byte copy and fill: 16 bytes per thread where the pointers allow, bytes otherwise.
kernel void copy_bytes(constant CopyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const long at = long(i) * 16;
  if (at >= a.bytes) return;
  const bool vec = ((ulong(a.src) | ulong(a.dst)) & 15ul) == 0 && at + 16 <= a.bytes;
  if (vec) {
    *(device uint4*)(a.dst + at) = *(device const uint4*)(a.src + at);
  } else {
    for (long j = at; j < min(at + 16, a.bytes); ++j) a.dst[j] = a.src[j];
  }
}

kernel void fill_bytes(constant FillArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const long at = long(i) * 16;
  if (at >= a.bytes) return;
  const uint v = a.value * 0x01010101u;
  if ((ulong(a.dst) & 15ul) == 0 && at + 16 <= a.bytes) {
    *(device uint4*)(a.dst + at) = uint4(v);
  } else {
    for (long j = at; j < min(at + 16, a.bytes); ++j) a.dst[j] = uchar(a.value);
  }
}

// ---- weight formats ----------------------------------------------------------------------------
// kernels.h, "Weight formats". The unit a lane loads is one 16-byte chunk of a row: 8 bf16, 16 int8
// or e4m3, or 32 e2m1 (two NVFP4 scale blocks). decode_chunk writes its values as f32 with the NVFP4
// block scales applied; the per-row factor (the INT8 / FP8 row scale, the NVFP4 global) is left out,
// so a dot product applies it once per row: sum_k x * (v * s) = s * sum_k x * v. Format 4 is f32
// (16 bytes, 4 values), used by the f32 GEMM. Format 5 is lossless bf16 (LX_W_BF16X, below), which a
// reader expands to the very bf16 chunk the plain matrix holds and then treats as bf16.
constant constexpr int W_BF16 = 0, W_INT8 = 1, W_FP8 = 2, W_NVFP4 = 3, W_F32 = 4, W_BF16X = 5;

// ---- lossless bf16 (LX_W_BF16X) ---------------------------------------------------------------
// The layout of kernels.h and cuda/bf16x.cuh. A row of K values is a header (u32 patch slots, one base
// exponent byte per block of 128, padding to 8 bytes), then per block 128 bytes of sign << 7 | mantissa
// and 64 bytes of 4-bit codes (exponent = base - code; u32 word g holds values 8g + b in the low nibble
// of byte b and 8g + 4 + b in the high one). Values more than 15 binades below their base are patches:
// (k << 8 | exponent) in the slots, ascending, or in an overflow record when slot 0 has bit 31 set.
constant constexpr uint kBf16xNoPatch = 0xffffffffu;
inline int bf16x_blocks(int K) { return (K + 127) / 128; }
inline int bf16x_header(int K) { return (4 * (2 + K / 2048) + bf16x_blocks(K) + 7) / 8 * 8; }
inline int bf16x_slots(int K) { return (bf16x_header(K) - bf16x_blocks(K)) / 4; }
inline int bf16x_base_offset(int K) { return 4 * bf16x_slots(K); }
inline long bf16x_row_bytes(int K) { return bf16x_header(K) + 192l * bf16x_blocks(K); }

// A row's patches in k order: `next` is the first one a reader moving up through k has not passed
// (kBf16xNoPatch, at position 0xffffff, once none are left).
struct Bf16xPatches {
  const device uint* p;  // the slot after `next`
  int left;              // slots readable after `next`
  uint next;
  void start(const device uchar* row, int K) {
    p = (const device uint*)row;
    next = p[0];
    int slots = bf16x_slots(K);
    if (next != kBf16xNoPatch && (next & 0x80000000u)) {  // the patches are in the row's overflow record
      p = (const device uint*)(row + long(next & 0x7fffffffu) * 8);
      slots = int(p[0]) + 1;  // the patches and their terminator
      ++p;
      next = p[0];
    }
    ++p;
    left = slots - 1;
  }
  uint pos() const { return next >> 8; }
  void advance() {
    next = left > 0 ? *p : kBf16xNoPatch;
    ++p;
    --left;
  }
};

// The bf16 bits of four values from their sign|mantissa bytes and exponents (one per byte): values 0 and
// 1 in .x, 2 and 3 in .y, low half first, as the plain matrix holds them.
inline uint2 bf16x_assemble(uint sm, uint e) {
  const uchar4 hi = as_type<uchar4>((sm & 0x80808080u) | ((e >> 1) & 0x7f7f7f7fu));
  const uchar4 lo = as_type<uchar4>((sm & 0x7f7f7f7fu) | ((e << 7) & 0x80808080u));
  return uint2(as_type<uint>(uchar4(lo.x, hi.x, lo.y, hi.y)), as_type<uint>(uchar4(lo.z, hi.z, lo.w, hi.w)));
}

// Applies one patch to the exponents of values k0 .. k0 + 3, if it falls among them.
inline uint bf16x_patch(uint e, uint slot, uint k0) {
  const uint d = (slot >> 8) - k0;
  if (d < 4u) {
    const uint sh = 8u * d;
    e = (e & ~(0xffu << sh)) | ((slot & 0xffu) << sh);
  }
  return e;
}

// Values k0 .. k0 + 7 of a row (k0 a multiple of 8) as the 16-byte chunk of the plain bf16 matrix.
// `pc` must not have passed k0; it ends past k0 + 7.
inline uint4 bf16x_chunk(const device uchar* row, int K, int k0, thread Bf16xPatches& pc) {
  const int o = k0 & 127;
  const device uchar* blk = row + bf16x_header(K) + long(k0 >> 7) * 192;
  const uint2 sm = *(const device uint2*)(blk + o);
  const uint codes = *(const device uint*)(blk + 128 + (o >> 1));
  const uint base = uint(row[bf16x_base_offset(K) + (k0 >> 7)]) * 0x01010101u;
  uint e0 = base - (codes & 0x0f0f0f0fu), e1 = base - ((codes >> 4) & 0x0f0f0f0fu);
  while (pc.pos() < uint(k0 + 8)) {
    e0 = bf16x_patch(e0, pc.next, uint(k0));
    e1 = bf16x_patch(e1, pc.next, uint(k0 + 4));
    pc.advance();
  }
  const uint2 a = bf16x_assemble(sm.x, e0), b = bf16x_assemble(sm.y, e1);
  return uint4(a.x, a.y, b.x, b.y);
}

// The bf16 bits of value k of a row, one value at a time; `pc` must not have passed k.
inline ushort bf16x_value(const device uchar* row, int K, int k, thread Bf16xPatches& pc) {
  const device uchar* blk = row + bf16x_header(K) + long(k >> 7) * 192;
  const int off = k & 127;
  const uint sm = blk[off];
  const uint cb = blk[128 + 4 * (off >> 3) + (off & 3)];
  const uint code = (off & 4) ? cb >> 4 : cb & 15u;
  uint e = (uint(row[bf16x_base_offset(K) + (k >> 7)]) - code) & 0xffu;
  while (pc.pos() < uint(k)) pc.advance();
  if (pc.pos() == uint(k)) e = pc.next & 0xffu;
  return ushort(((sm & 0x80u) << 8) | (e << 7) | (sm & 0x7fu));
}

template <int F> constexpr int chunk_values() {
  return F == W_BF16 || F == W_BF16X ? 8 : F == W_NVFP4 ? 32 : F == W_F32 ? 4 : 16;
}
// Bytes of row n's start, for a row of K values.
template <int F> inline long row_offset(long n, long K) {
  if (F == W_BF16X) return n * bf16x_row_bytes(int(K));
  return F == W_BF16 ? n * K * 2 : F == W_NVFP4 ? n * (K / 2) : F == W_F32 ? n * K * 4 : n * K;
}

template <int F> inline void decode_chunk(uint4 w, float s0, float s1, thread float* v);
template <> inline void decode_chunk<W_BF16>(uint4 w, float, float, thread float* v) {
  v[0] = bf_lo(w.x); v[1] = bf_hi(w.x); v[2] = bf_lo(w.y); v[3] = bf_hi(w.y);
  v[4] = bf_lo(w.z); v[5] = bf_hi(w.z); v[6] = bf_lo(w.w); v[7] = bf_hi(w.w);
}
template <> inline void decode_chunk<W_INT8>(uint4 w, float, float, thread float* v) {
  const char4 a = as_type<char4>(w.x), b = as_type<char4>(w.y), c = as_type<char4>(w.z), d = as_type<char4>(w.w);
  v[0] = a.x; v[1] = a.y; v[2] = a.z; v[3] = a.w; v[4] = b.x; v[5] = b.y; v[6] = b.z; v[7] = b.w;
  v[8] = c.x; v[9] = c.y; v[10] = c.z; v[11] = c.w; v[12] = d.x; v[13] = d.y; v[14] = d.z; v[15] = d.w;
}
template <> inline void decode_chunk<W_FP8>(uint4 w, float, float, thread float* v) {
  const uint words[4] = {w.x, w.y, w.z, w.w};
  for (int i = 0; i < 4; ++i)
    for (int b = 0; b < 4; ++b) v[4 * i + b] = e4m3((words[i] >> (8 * b)) & 0xffu);
}
template <> inline void decode_chunk<W_NVFP4>(uint4 w, float s0, float s1, thread float* v) {
  const uint words[4] = {w.x, w.y, w.z, w.w};
  for (int i = 0; i < 4; ++i) {
    const float s = i < 2 ? s0 : s1;
    for (int b = 0; b < 4; ++b) {
      const uint byte = (words[i] >> (8 * b)) & 0xffu;
      v[8 * i + 2 * b] = e2m1(byte & 15u) * s;  // the low nibble is the even k
      v[8 * i + 2 * b + 1] = e2m1(byte >> 4) * s;
    }
  }
}
template <> inline void decode_chunk<W_F32>(uint4 w, float, float, thread float* v) {
  v[0] = as_type<float>(w.x); v[1] = as_type<float>(w.y); v[2] = as_type<float>(w.z); v[3] = as_type<float>(w.w);
}

// The chunk of row `row` (a pointer to the row's first byte) starting at value k0, decoded. For
// NVFP4, `scales` points at the row's e4m3 block scales.
template <int F>
inline void load_chunk(const device uchar* row, const device uchar* scales, int k0, thread float* v) {
  constexpr int BYTES_PER_16 = F == W_BF16 ? 32 : F == W_NVFP4 ? 8 : F == W_F32 ? 64 : 16;
  const uint4 w = *(const device uint4*)(row + long(k0) * BYTES_PER_16 / 16);
  float s0 = 1.0f, s1 = 1.0f;
  if (F == W_NVFP4) {
    s0 = e4m3(scales[k0 / 16]);
    s1 = e4m3(scales[k0 / 16 + 1]);
  }
  decode_chunk<F>(w, s0, s1, v);
}

// One element (n, k) of a weight of format F (rows of K values), without the row factor.
template <int F>
inline float element(const device uchar* data, const device uchar* scale, long n, long k, long K) {
  if (F == W_BF16X) {  // a random access walks the row's patches from the start
    const device uchar* row = data + row_offset<W_BF16X>(n, K);
    Bf16xPatches pc;
    pc.start(row, int(K));
    return bf(bf16x_value(row, int(K), int(k), pc));
  }
  if (F == W_BF16) return bf(((const device ushort*)data)[n * K + k]);
  if (F == W_INT8) return float(((const device char*)data)[n * K + k]);
  if (F == W_FP8) return e4m3(data[n * K + k]);
  if (F == W_F32) return ((const device float*)data)[n * K + k];
  const uint byte = data[n * (K / 2) + k / 2];
  return e2m1((k & 1) ? (byte >> 4) : (byte & 15u)) * e4m3(scale[n * (K / 16) + k / 16]);
}

// The per-row factor of row n: the INT8 / FP8 row scale, the NVFP4 global, or 1.
template <int F>
inline float row_factor(const device uchar* scale, long n, float global) {
  if (F == W_INT8 || F == W_FP8) return ((const device float*)scale)[n];
  if (F == W_NVFP4) return global;
  return 1.0f;
}

// Expert e's slice of a stack [E, N, K]: its data, its scales and its global.
template <int F>
inline void expert_slice(WeightRef w, long e, long N, long K, const device uchar* thread& data,
                         const device uchar* thread& scale, thread float& global) {
  data = w.data + row_offset<F>(e * N, K);
  if (F == W_INT8 || F == W_FP8) scale = w.scale + e * N * 4;
  else if (F == W_NVFP4) scale = w.scale + e * N * (K / 16);
  else scale = w.scale;
  global = (F == W_NVFP4 && w.expert_global) ? w.expert_global[e] : w.global;
}

// ---- GEMM tiles -------------------------------------------------------------------------------
// The tile loaders of the GEMMs (gemm.metal on simdgroup matrices, nax/gemm.metal on the neural
// accelerators): 16 values of an activation or weight row, decoded exactly.

// Activation formats: bf16, f32 (rounded to bf16, as BF16 math sees it), int8 and e4m3 codes with a
// per-row scale, NVFP4 (codes, e4m3 block scales, a per-row global), and exact f32 (lx_gemm_f32).
constant constexpr int A_BF16 = 0, A_F32 = 1, A_INT8 = 2, A_E4M3 = 3, A_FP4 = 4, A_F32X = 5;
constant constexpr int kTileM = 64, kTileN = 64, kTileK = 32, kTileKP = kTileK + 4;

// 16 activation values [k, k + 16) of one row, zero past K and for rows past the end.
template <int AF>
inline void load_act16(const device uchar* row, const device uchar* blocks, int k, int K, bool vec, bool valid,
                       thread float* v) {
  if (!valid) {
    for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    return;
  }
  if (vec) {
    if (AF == A_BF16) {
      const device uint4* p = (const device uint4*)((const device ushort*)row + k);
      for (int h = 0; h < 2; ++h) {
        const uint4 w = p[h];
        v[8 * h] = bf_lo(w.x); v[8 * h + 1] = bf_hi(w.x); v[8 * h + 2] = bf_lo(w.y); v[8 * h + 3] = bf_hi(w.y);
        v[8 * h + 4] = bf_lo(w.z); v[8 * h + 5] = bf_hi(w.z); v[8 * h + 6] = bf_lo(w.w); v[8 * h + 7] = bf_hi(w.w);
      }
    } else if (AF == A_F32 || AF == A_F32X) {
      const device float4* p = (const device float4*)((const device float*)row + k);
      for (int q = 0; q < 4; ++q) {
        const float4 f = p[q];
        v[4 * q] = f.x; v[4 * q + 1] = f.y; v[4 * q + 2] = f.z; v[4 * q + 3] = f.w;
      }
      if (AF == A_F32)
        for (int i = 0; i < 16; ++i) v[i] = bf(to_bf(v[i]));
    } else if (AF == A_INT8) {
      decode_chunk<W_INT8>(*(const device uint4*)(row + k), 1.0f, 1.0f, v);
    } else if (AF == A_E4M3) {
      decode_chunk<W_FP8>(*(const device uint4*)(row + k), 1.0f, 1.0f, v);
    } else {
      const uint2 w = *(const device uint2*)(row + k / 2);
      const float s = e4m3(blocks[k / 16]);
      for (int i = 0; i < 4; ++i) {
        const uint byte0 = (w.x >> (8 * i)) & 0xffu, byte1 = (w.y >> (8 * i)) & 0xffu;
        v[2 * i] = e2m1(byte0 & 15u) * s;
        v[2 * i + 1] = e2m1(byte0 >> 4) * s;
        v[8 + 2 * i] = e2m1(byte1 & 15u) * s;
        v[8 + 2 * i + 1] = e2m1(byte1 >> 4) * s;
      }
    }
    return;
  }
  for (int i = 0; i < 16; ++i) {
    const int kk = k + i;
    float x = 0.0f;
    if (kk < K) {
      if (AF == A_BF16) x = bf(((const device ushort*)row)[kk]);
      else if (AF == A_F32) x = bf(to_bf(((const device float*)row)[kk]));
      else if (AF == A_F32X) x = ((const device float*)row)[kk];
      else if (AF == A_INT8) x = float(((const device char*)row)[kk]);
      else if (AF == A_E4M3) x = e4m3(row[kk]);
      else {
        const uint byte = row[kk / 2];
        x = e2m1((kk & 1) ? (byte >> 4) : (byte & 15u)) * e4m3(blocks[kk / 16]);
      }
    }
    v[i] = x;
  }
}

// 16 weight values [k, k + 16) of one row of format WF (W_F32 for lx_gemm_f32), without the row factor.
template <int WF>
inline void load_w16(const device uchar* row, const device uchar* scales, int k, int K, bool vec, bool valid,
                     thread float* v) {
  if (!valid) {
    for (int i = 0; i < 16; ++i) v[i] = 0.0f;
    return;
  }
  if (vec) {
    if (WF == W_BF16) {
      const device uint4* p = (const device uint4*)(row + long(k) * 2);
      decode_chunk<W_BF16>(p[0], 1.0f, 1.0f, v);
      decode_chunk<W_BF16>(p[1], 1.0f, 1.0f, v + 8);
    } else if (WF == W_INT8 || WF == W_FP8) {
      decode_chunk<WF>(*(const device uint4*)(row + k), 1.0f, 1.0f, v);
    } else if (WF == W_F32) {
      const device uint4* p = (const device uint4*)(row + long(k) * 4);
      for (int q = 0; q < 4; ++q) decode_chunk<W_F32>(p[q], 1.0f, 1.0f, v + 4 * q);
    } else {
      const uint2 w = *(const device uint2*)(row + k / 2);
      const float s = e4m3(scales[k / 16]);
      for (int i = 0; i < 4; ++i) {
        const uint byte0 = (w.x >> (8 * i)) & 0xffu, byte1 = (w.y >> (8 * i)) & 0xffu;
        v[2 * i] = e2m1(byte0 & 15u) * s;
        v[2 * i + 1] = e2m1(byte0 >> 4) * s;
        v[8 + 2 * i] = e2m1(byte1 & 15u) * s;
        v[8 + 2 * i + 1] = e2m1(byte1 >> 4) * s;
      }
    }
    return;
  }
  for (int i = 0; i < 16; ++i) {
    const int kk = k + i;
    v[i] = kk < K ? element<WF>(row, scales, 0, kk, K) : 0.0f;
  }
}

template <int AF> constexpr int act_bytes_shift() { return AF == A_BF16 ? 1 : (AF == A_F32 || AF == A_F32X) ? 2 : 0; }

// 16 int8 codes [k, k + 16) of a row as one 16-byte word, zero past K and for rows past the end.
inline uint4 load_codes16(const device uchar* row, int k, int K, bool vec, bool valid) {
  if (!valid) return uint4(0u);
  if (vec) return *(const device uint4*)(row + k);
  uint w[4] = {0u, 0u, 0u, 0u};
  for (int i = 0; i < 16; ++i)
    if (k + i < K) w[i >> 2] |= uint(row[k + i]) << (8 * (i & 3));
  return uint4(w[0], w[1], w[2], w[3]);
}

// 16 f32 values that bf16 holds exactly, as 16 bf16 in two words (dropping the low half is exact).
inline void pack_bf16(const thread float* v, thread uint4* w) {
  uint u[8];
  for (int i = 0; i < 8; ++i) u[i] = (as_type<uint>(v[2 * i]) >> 16) | (as_type<uint>(v[2 * i + 1]) & 0xffff0000u);
  w[0] = uint4(u[0], u[1], u[2], u[3]);
  w[1] = uint4(u[4], u[5], u[6], u[7]);
}

// One thread's 16 values of an activation row (A) or weight row (W) in staged form: copied as stored
// where that is the staged type (bf16, and int8 codes when CODES), else decoded and packed as bf16.
template <int AF, bool CODES>
inline void stage_act(const device uchar* row, const device uchar* blocks, int k, int K, bool vec, bool valid,
                      thread uint4* w) {
  if (CODES && AF == A_INT8) {
    w[0] = load_codes16(row, k, K, vec, valid);
    return;
  }
  if (AF == A_BF16 && vec && valid) {
    const device uint4* p = (const device uint4*)((const device ushort*)row + k);
    w[0] = p[0];
    w[1] = p[1];
    return;
  }
  float v[16];
  load_act16<AF>(row, blocks, k, K, vec, valid, v);
  pack_bf16(v, w);
}
template <int WF, bool CODES>
inline void stage_w(const device uchar* row, const device uchar* scales, int k, int K, bool vec, bool valid,
                    thread uint4* w) {
  if (CODES && WF == W_INT8) {
    w[0] = load_codes16(row, k, K, vec, valid);
    return;
  }
  if (WF == W_BF16 && vec && valid) {
    const device uint4* p = (const device uint4*)(row + long(k) * 2);
    w[0] = p[0];
    w[1] = p[1];
    return;
  }
  float v[16];
  load_w16<WF>(row, scales, k, K, vec, valid, v);
  pack_bf16(v, w);
}
template <typename T>
inline void stage_store(threadgroup T* dst, const thread uint4* w) {
  threadgroup uint4* d = (threadgroup uint4*)dst;
  d[0] = w[0];
  if (sizeof(T) == 2) d[1] = w[1];
}
