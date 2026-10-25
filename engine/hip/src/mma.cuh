// Tensor-core building blocks for the HIP port: the interface of engine/cuda/mma.cuh (the PTX forms),
// implemented portably. translate.py copies this file to src/mma.cuh in place of the PTX header.
//
// The layout contract is the one in engine/cuda/mma.cuh: a 32-lane group computes D = A B for one
// m16n8 tile, with the operand fragments in the registers the PTX instructions read. This port does
// not issue a matrix instruction. It fetches each lane's operand words from the lanes that hold them
// (__shfl_sync within the 32-lane group) and accumulates in f32 (s32 for int8), so the result is the
// same product the PTX form computes, up to the order the sums are taken in.
//
// Shared memory is addressed by its 32-bit offset, the low word of the shared window address. Every
// shared pointer on the NVIDIA shim and on AMD carries the window in its high word, so the window is
// recovered from a __shared__ probe and the offset is resolved against it (see shared_window()).
//
// Asynchronous copies are synchronous here: a cp.async completes before the next instruction, which
// is correct because every reader of a staged tile is behind a barrier. The commit and wait forms
// are therefore empty. The tensor-core families do not need overlap to be correct; the native
// matrix instructions (MFMA, WMMA) are the follow-up that restores it.
#pragma once

#include <stdint.h>

#include "lx_hip.h"

namespace lx::mma {

namespace detail {

// The 32-bit word a lane reads from lane `src` of its own 32-lane group.
__device__ __forceinline__ uint32_t shfl(uint32_t v, int src) { return __shfl_sync(LX_FULL_MASK, v, src, 32); }

// Shared memory window: the high word of every shared pointer. A __shared__ object's address carries it.
__device__ __forceinline__ uintptr_t shared_window() {
  __shared__ uint32_t probe;
  return reinterpret_cast<uintptr_t>(&probe) & ~static_cast<uintptr_t>(0xffffffffu);
}

__device__ __forceinline__ void* smem_ptr(uint32_t addr) {
  return reinterpret_cast<void*>(shared_window() | static_cast<uintptr_t>(addr));
}

__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  __builtin_memcpy(&v, smem_ptr(addr), 4);
  return v;
}

__device__ __forceinline__ uint32_t lds16(uint32_t addr) {
  uint16_t v;
  __builtin_memcpy(&v, smem_ptr(addr), 2);
  return v;
}

// e4m3 bits as f32: the 7 magnitude bits placed under an f32 exponent give value * 2^-120 for normals and
// subnormals alike; one multiply by 2^120 restores it (f32 keeps the subnormal intermediates).
__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  const float m = __uint_as_float(((b & 0x7fu) << 20) | ((b & 0x80u) << 24));
  return m * 0x1p120f;
}

__device__ __forceinline__ float bf16_to_f32(uint32_t bits16) { return __uint_as_float(bits16 << 16); }

// Round-to-nearest-even f32 to bf16 bits, NaN kept a NaN.
__device__ __forceinline__ uint16_t f32_to_bf16_bits(float f) {
  uint32_t u = __float_as_uint(f);
  if ((u & 0x7fffffffu) > 0x7f800000u) return static_cast<uint16_t>((u >> 16) | 0x40u);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

// Element e of a 32-bit operand word, as f32, for each element type. E elements per word; the byte
// order is the PTX one (element 0 in the low bits).
template <int Fmt>
__device__ __forceinline__ float dec(uint32_t w, int e);

template <>
__device__ __forceinline__ float dec<0>(uint32_t w, int e) {  // bf16, E = 2
  return bf16_to_f32((w >> (16 * e)) & 0xffffu);
}
template <>
__device__ __forceinline__ float dec<1>(uint32_t w, int e) {  // e4m3, E = 4
  return e4m3_to_f32(w >> (8 * e));
}
template <>
__device__ __forceinline__ float dec<2>(uint32_t w, int e) {  // e2m1, E = 8
  const uint32_t n = (w >> (4 * e)) & 0xfu;
  // magnitudes of the three low bits: 0, 0.5, 1, 1.5, 2, 3, 4, 6
  const float mag = (n & 4u) ? ((n & 2u) ? ((n & 1u) ? 6.0f : 4.0f) : ((n & 1u) ? 3.0f : 2.0f))
                             : ((n & 2u) ? ((n & 1u) ? 1.5f : 1.0f) : ((n & 1u) ? 0.5f : 0.0f));
  return (n & 8u) ? -mag : mag;
}

template <int Fmt>
struct Layout;
template <>
struct Layout<0> {  // bf16: k16, two elements per word
  static constexpr int E = 2, KH = 8;
};
template <>
struct Layout<1> {  // e4m3: k32, four elements per word
  static constexpr int E = 4, KH = 16;
};
template <>
struct Layout<2> {  // e2m1: k64, eight elements per word
  static constexpr int E = 8, KH = 32;
};

// Scale byte `byte` of a scale register: ue8m0 (2^(b-127)) for the 1X forms, ue4m3 otherwise.
template <bool Mx>
__device__ __forceinline__ float scale_of(uint32_t w, int byte) {
  const uint32_t b = (w >> (8 * byte)) & 0xffu;
  return Mx ? __uint_as_float(b << 23) : e4m3_to_f32(b);
}

// The m16n8 tile for one 32-lane group. Lane (r, q) holds row r and r+8 of A, column 2q and 2q+1 of
// the result. Its A words come from lanes 4r + j (j = 0..3, one K slice each), its B words from lanes
// 4n + j for the columns n it needs. For K slice j the A words of the lane's own row hold K elements
// j*E.. (word a0, row r) and KH + j*E.. (word a2), and the B words hold the same K elements.
//
// Block scales: for a row the scale register of lane 4r + 2 TA (row r) and 4r + 2 TA + 1 (row r + 8)
// applies, for a column the register of lane 4n + TB. Scale byte b covers K block b (16 K for the
// 4X form, 32 K for the 1X form, which uses byte 0 only).
template <int Fmt, bool Scaled, bool Mx, int TA, int TB>
__device__ __forceinline__ void tile(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                     uint32_t sfb) {
  using L = Layout<Fmt>;
  const int lane = threadIdx.x & 31;
  const int r = lane >> 2;
  const int q = lane & 3;
  const int base = lane & ~3;

  uint32_t sa_row = 0, sa_row8 = 0, sb_col = 0, sb_col1 = 0;
  if constexpr (Scaled) {
    sa_row = shfl(sfa, 4 * r + 2 * TA);
    sa_row8 = shfl(sfa, 4 * r + 2 * TA + 1);
    sb_col = shfl(sfb, 8 * q + TB);
    sb_col1 = shfl(sfb, 8 * q + 4 + TB);
  }

  float c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t x0 = shfl(a[0], base + j);  // row r, K slice 0
    const uint32_t x1 = shfl(a[1], base + j);  // row r + 8, K slice 0
    const uint32_t x2 = shfl(a[2], base + j);  // row r, K slice 1
    const uint32_t x3 = shfl(a[3], base + j);  // row r + 8, K slice 1
    const uint32_t y0 = shfl(b0, 8 * q + j);      // column 2q, K slice 0
    const uint32_t y1 = shfl(b1, 8 * q + j);      // column 2q, K slice 1
    const uint32_t z0 = shfl(b0, 8 * q + 4 + j);  // column 2q + 1, K slice 0
    const uint32_t z1 = shfl(b1, 8 * q + 4 + j);  // column 2q + 1, K slice 1

    // Scale bytes for this j: K block of slice 0 is (j E) / 16, of slice 1 is (KH + j E) / 16.
    const int blk0 = Mx ? 0 : (j * L::E) / 16;
    const int blk1 = Mx ? 0 : (L::KH + j * L::E) / 16;
    const float ra0 = Scaled ? scale_of<Mx>(sa_row, blk0) : 1.0f;
    const float ra1 = Scaled ? scale_of<Mx>(sa_row, blk1) : 1.0f;
    const float rh0 = Scaled ? scale_of<Mx>(sa_row8, blk0) : 1.0f;
    const float rh1 = Scaled ? scale_of<Mx>(sa_row8, blk1) : 1.0f;
    const float cc0 = Scaled ? scale_of<Mx>(sb_col, blk0) : 1.0f;
    const float cc1 = Scaled ? scale_of<Mx>(sb_col, blk1) : 1.0f;
    const float cd0 = Scaled ? scale_of<Mx>(sb_col1, blk0) : 1.0f;
    const float cd1 = Scaled ? scale_of<Mx>(sb_col1, blk1) : 1.0f;

#pragma unroll
    for (int e = 0; e < L::E; ++e) {
      const float ar0 = dec<Fmt>(x0, e) * ra0;
      const float ar1 = dec<Fmt>(x2, e) * ra1;
      const float ah0 = dec<Fmt>(x1, e) * rh0;
      const float ah1 = dec<Fmt>(x3, e) * rh1;
      const float bc0 = dec<Fmt>(y0, e) * cc0;
      const float bc1 = dec<Fmt>(y1, e) * cc1;
      const float bd0 = dec<Fmt>(z0, e) * cd0;
      const float bd1 = dec<Fmt>(z1, e) * cd1;
      c0 += ar0 * bc0 + ar1 * bc1;
      c1 += ar0 * bd0 + ar1 * bd1;
      c2 += ah0 * bc0 + ah1 * bc1;
      c3 += ah0 * bd0 + ah1 * bd1;
    }
  }
  c[0] = c0;
  c[1] = c1;
  c[2] = c2;
  c[3] = c3;
}

}  // namespace detail

// ---- addressing ---------------------------------------------------------------------------------

__device__ __forceinline__ uint32_t smem_addr(const void* p) { return static_cast<uint32_t>(reinterpret_cast<uintptr_t>(p)); }

// ---- asynchronous copies (synchronous here, see the header note) ----------------------------------

// 16 bytes global -> shared; when !pred the destination is zero-filled and nothing is read.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  uint4 v = make_uint4(0u, 0u, 0u, 0u);
  if (pred) v = *reinterpret_cast<const uint4*>(src);
  *reinterpret_cast<uint4*>(detail::smem_ptr(dst)) = v;
}
// 4 or 8 bytes (scale vectors), the same zero-fill rule.
template <int N>
__device__ __forceinline__ void cp_async_small(uint32_t dst, const void* src, bool pred) {
  static_assert(N == 4 || N == 8);
  if constexpr (N == 4) {
    uint32_t v = 0u;
    if (pred) v = *reinterpret_cast<const uint32_t*>(src);
    *reinterpret_cast<uint32_t*>(detail::smem_ptr(dst)) = v;
  } else {
    uint2 v = make_uint2(0u, 0u);
    if (pred) v = *reinterpret_cast<const uint2*>(src);
    *reinterpret_cast<uint2*>(detail::smem_ptr(dst)) = v;
  }
}
__device__ __forceinline__ void cp_async_commit() {}
template <int N>
__device__ __forceinline__ void cp_async_wait() {}

// ---- shared memory loads ------------------------------------------------------------------------

// ldmatrix: matrix i's row t / 4 (addressed by lane 8 i + t / 4) comes to lane t as the 32-bit word at
// column pair t % 4. The transposed form gives lane t the two elements of column t / 4 in rows
// 2 (t % 4) and 2 (t % 4) + 1, one per half word.
__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  const int lane = threadIdx.x & 31;
  const int row = lane >> 2, q = lane & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t ra = detail::shfl(addr, 8 * i + row);
    r[i] = detail::lds32(ra + 4u * static_cast<uint32_t>(q));
  }
}
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  const int lane = threadIdx.x & 31;
  const int col = lane >> 2, q = lane & 3;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t r0 = detail::shfl(addr, 8 * i + 2 * q);
    const uint32_t r1 = detail::shfl(addr, 8 * i + 2 * q + 1);
    const uint32_t lo = detail::lds16(r0 + 2u * static_cast<uint32_t>(col));
    const uint32_t hi = detail::lds16(r1 + 2u * static_cast<uint32_t>(col));
    r[i] = lo | (hi << 16);
  }
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  const int lane = threadIdx.x & 31;
  const int row = lane >> 2, q = lane & 3;
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    const uint32_t ra = detail::shfl(addr, 8 * i + row);
    r[i] = detail::lds32(ra + 4u * static_cast<uint32_t>(q));
  }
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  __builtin_memcpy(&v, detail::smem_ptr(addr), 16);
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  __builtin_memcpy(&v, detail::smem_ptr(addr), 8);
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) { return detail::lds32(addr); }

// ---- mma ----------------------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  detail::tile<0, false, false, 0, 0>(c, a, b0, b1, 0u, 0u);
}

// int8 x int8, s32 accumulation (k32).
__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  const int lane = threadIdx.x & 31;
  const int r = lane >> 2, q = lane & 3, base = lane & ~3;
  int c0 = c[0], c1 = c[1], c2 = c[2], c3 = c[3];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const uint32_t x0 = detail::shfl(a[0], base + j), x1 = detail::shfl(a[1], base + j);
    const uint32_t x2 = detail::shfl(a[2], base + j), x3 = detail::shfl(a[3], base + j);
    const uint32_t y0 = detail::shfl(b0, 8 * q + j), y1 = detail::shfl(b1, 8 * q + j);
    const uint32_t z0 = detail::shfl(b0, 8 * q + 4 + j), z1 = detail::shfl(b1, 8 * q + 4 + j);
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int ar0 = static_cast<int8_t>(x0 >> (8 * e)), ar1 = static_cast<int8_t>(x2 >> (8 * e));
      const int ah0 = static_cast<int8_t>(x1 >> (8 * e)), ah1 = static_cast<int8_t>(x3 >> (8 * e));
      const int bc0 = static_cast<int8_t>(y0 >> (8 * e)), bc1 = static_cast<int8_t>(y1 >> (8 * e));
      const int bd0 = static_cast<int8_t>(z0 >> (8 * e)), bd1 = static_cast<int8_t>(z1 >> (8 * e));
      c0 += ar0 * bc0 + ar1 * bc1;
      c1 += ar0 * bd0 + ar1 * bd1;
      c2 += ah0 * bc0 + ah1 * bc1;
      c3 += ah0 * bd0 + ah1 * bd1;
    }
  }
  c[0] = c0;
  c[1] = c1;
  c[2] = c2;
  c[3] = c3;
}

// e4m3 x e4m3, f32 accumulation.
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  detail::tile<1, false, false, 0, 0>(c, a, b0, b1, 0u, 0u);
}

// e4m3 x e4m3 through the block-scaled form with unit (2^0) scales.
__device__ __forceinline__ void mma_e4m3_mx(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  detail::tile<1, true, true, 0, 0>(c, a, b0, b1, 0x7f7f7f7fu, 0x7f7f7f7fu);
}

// The same into a fresh result (d = a * b), for the per-block products of W4A8.
__device__ __forceinline__ void mma_e4m3_mx_zero(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  const float zero[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  d[0] = zero[0];
  d[1] = zero[1];
  d[2] = zero[2];
  d[3] = zero[3];
  detail::tile<1, true, true, 0, 0>(d, a, b0, b1, 0x7f7f7f7fu, 0x7f7f7f7fu);
}

// e2m1 x e2m1 with one ue4m3 scale per 16 K on each side (NVFP4), k64. TA / TB select which lane's
// scale register applies (see detail::tile).
template <int TA, int TB>
__device__ __forceinline__ void mma_nvf4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint32_t sfb) {
  detail::tile<2, true, false, TA, TB>(c, a, b0, b1, sfa, sfb);
}

// ---- conversions --------------------------------------------------------------------------------
// All exact: every value of the source format is a bf16 (or e4m3) value.

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  return static_cast<uint32_t>(detail::f32_to_bf16_bits(lo)) | (static_cast<uint32_t>(detail::f32_to_bf16_bits(hi)) << 16);
}

// Four int8 (one word, element 0 in the low byte) to two bf16x2: (e0, e1), (e2, e3).
__device__ __forceinline__ void i8x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = pack_bf16(static_cast<float>(static_cast<int8_t>(w)), static_cast<float>(static_cast<int8_t>(w >> 8)));
  hi = pack_bf16(static_cast<float>(static_cast<int8_t>(w >> 16)), static_cast<float>(static_cast<int8_t>(w >> 24)));
}

__device__ __forceinline__ float e4m3_to_f32(uint32_t b) { return detail::e4m3_to_f32(b); }

// Four e4m3 (one word) to two bf16x2.
template <int kArch>
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = pack_bf16(detail::e4m3_to_f32(w), detail::e4m3_to_f32(w >> 8));
  hi = pack_bf16(detail::e4m3_to_f32(w >> 16), detail::e4m3_to_f32(w >> 24));
}

// Byte permute: result byte i is byte (sel nibble i) of {b, a}. The sign-replicate mode (nibble bit 3)
// is not used by the callers; they pass selectors masked to 0..7.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  const uint64_t x = static_cast<uint64_t>(a) | (static_cast<uint64_t>(b) << 32);
  uint32_t r = 0;
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const uint32_t s = (sel >> (4 * i)) & 7u;
    r |= static_cast<uint32_t>((x >> (8 * s)) & 0xffu) << (8 * i);
  }
  return r;
}

// Two bf16 products. An e2m1 times an e4m3 scale has at most 6 significant bits, so the rounded product
// is exact.
__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  const float lo = detail::bf16_to_f32(a & 0xffffu) * detail::bf16_to_f32(b & 0xffffu);
  const float hi = detail::bf16_to_f32(a >> 16) * detail::bf16_to_f32(b >> 16);
  return pack_bf16(lo, hi);
}

// Eight e2m1 (one word, element 0 in the low nibble) times a bf16 block scale, as four bf16x2. The 3-bit
// magnitudes index two byte tables (high and low halves of the bf16 values 0, 0.5, 1, 1.5, 2, 3, 4, 6)
// through prmt, the sign bits move to bit 15 of each half, and one bf16x2 multiply applies the scale.
__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t scale2, uint32_t (&out)[4]) {
  // bf16 of the magnitudes 0..7: high bytes 00 3f 3f 3f 40 40 40 40, low bytes 00 00 80 c0 00 40 80 c0.
  constexpr uint32_t kHi0 = 0x3f3f3f00u, kHi1 = 0x40404040u, kLo0 = 0xc0800000u, kLo1 = 0xc0804000u;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t nib = w >> (16 * h);
    const uint32_t sel = nib & 0x7777u;
    const uint32_t hi = prmt(kHi0, kHi1, sel);
    const uint32_t lo = prmt(kLo0, kLo1, sel);
    // signs: nibble bit 3 of v0 (bit 3), v1 (bit 7) to bits 15 and 31; v2 (bit 11), v3 (bit 15) likewise.
    const uint32_t s01 = ((nib & 0x88u) * 0x01001000u) & 0x80008000u;
    const uint32_t s23 = ((nib & 0x8800u) * 0x00010010u) & 0x80008000u;
    out[2 * h] = bf16x2_mul(prmt(lo, hi, 0x5140u) | s01, scale2);
    out[2 * h + 1] = bf16x2_mul(prmt(lo, hi, 0x7362u) | s23, scale2);
  }
}

// Four e2m1 (the low 16 bits of w) to four e4m3 bytes, exactly (every e2m1 value is an e4m3 value).
__device__ __forceinline__ uint32_t e2m1x4_to_e4m3(uint32_t w) {
  // e4m3 of the magnitudes 0..7: 00 30 38 3c 40 44 48 4c
  const uint32_t x = w & 0xffffu;
  const uint32_t mag = prmt(0x3c383000u, 0x4c484440u, x & 0x7777u);
  // spread the four nibbles to bytes, then each nibble's sign bit (bit 3) to its byte's bit 7
  uint32_t s = (x | (x << 8)) & 0x00ff00ffu;
  s = (s | (s << 4)) & 0x0f0f0f0fu;
  return mag | ((s << 4) & 0x80808080u);
}

}  // namespace lx::mma
