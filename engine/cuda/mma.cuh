// Tensor-core building blocks for the GEMM families (gemm_w.cu, moe_grouped.cu): asynchronous copies,
// ldmatrix, the mma.sync forms, and the exact conversions that expand a quantized weight to the
// MMA's operand type.
//
// Each instruction exists from some architecture on. A kernel family is a template on the arch it
// targets (80, 89, 120), and dispatch launches only the family the device (or lx_arch's override)
// supports, so the sm_80 family never executes an sm_89 instruction even on an sm_120 card. The
// wrappers below are additionally compiled out for targets that lack the instruction (the body traps),
// so a multi-arch build compiles every family for every target.
//
// MMA operand fragments, m16n8kK with 32 bytes of K per row (every form used here has that shape:
// k16 bf16, k32 int8 / e4m3, k64 e2m1):
//   A (16 x 32 bytes): a0 rows 0-7 bytes 4q..4q+3, a1 rows 8-15, a2 rows 0-7 bytes 16+4q.., a3 rows 8-15
//   B (8 x 32 bytes):  b0 column lane/4 bytes 4q..4q+3, b1 bytes 16+4q..
//   C (16 x 8 f32):    c0, c1 row lane/4 columns 2q, 2q+1; c2, c3 row lane/4 + 8
// with q = lane % 4. Block-scaled forms take one 32-bit scale register per thread: for B, threads with
// q == tid_b hold column lane/4's scales; for A, threads with q in {2 tid_a, 2 tid_a + 1} hold row
// lane/4 + 8 (q % 2). Byte j of the register scales K block j of the instruction.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// Instruction availability for the target being compiled (device code only).
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
#define LX_HAS_FP8_MMA 1
#endif
// Block-scaled mma.sync (kind::mxf8f6f4, kind::mxf4nvf4) and the FP4 conversions: the sm_120 family's
// arch-specific targets (sm_120a, sm_121a and the f variants).
#if (defined(__CUDA_ARCH_FAMILY_SPECIFIC__) && (__CUDA_ARCH_FAMILY_SPECIFIC__ == 1200 || __CUDA_ARCH_FAMILY_SPECIFIC__ == 1210)) || \
    (defined(__CUDA_ARCH_SPECIFIC__) && (__CUDA_ARCH_SPECIFIC__ == 1200 || __CUDA_ARCH_SPECIFIC__ == 1210))
#define LX_HAS_BLOCK_SCALE_MMA 1
#endif

namespace lx::mma {

__device__ __forceinline__ uint32_t smem_addr(const void* p) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// ---- asynchronous copies ------------------------------------------------------------------------

// 16 bytes global -> shared; when !pred the destination is zero-filled and nothing is read.
__device__ __forceinline__ void cp_async16(uint32_t dst, const void* src, bool pred) {
  asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;" ::"r"(dst), "l"(src), "r"(pred ? 16 : 0) : "memory");
}
// 4 or 8 bytes (scale vectors), the same zero-fill rule.
template <int N>
__device__ __forceinline__ void cp_async_small(uint32_t dst, const void* src, bool pred) {
  static_assert(N == 4 || N == 8);
  asm volatile("cp.async.ca.shared.global [%0], [%1], %2, %3;" ::"r"(dst), "l"(src), "n"(N), "r"(pred ? N : 0)
               : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;" ::: "memory"); }
template <int N>
__device__ __forceinline__ void cp_async_wait() {
  asm volatile("cp.async.wait_group %0;" ::"n"(N) : "memory");
}

// ---- shared memory loads ------------------------------------------------------------------------

__device__ __forceinline__ void ldmatrix_x4(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
// Transposed: thread t gets rows 2 (t % 4), + 1 of column t / 4 of each 8x8 matrix (an mma B operand from a
// row-major [k][n] tile).
__device__ __forceinline__ void ldmatrix_x4_trans(uint32_t (&r)[4], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];"
               : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
               : "r"(addr));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], uint32_t addr) {
  asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];" : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}
__device__ __forceinline__ uint4 lds128(uint32_t addr) {
  uint4 v;
  asm volatile("ld.shared.v4.b32 {%0,%1,%2,%3}, [%4];" : "=r"(v.x), "=r"(v.y), "=r"(v.z), "=r"(v.w) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint2 lds64(uint32_t addr) {
  uint2 v;
  asm volatile("ld.shared.v2.b32 {%0,%1}, [%2];" : "=r"(v.x), "=r"(v.y) : "r"(addr));
  return v;
}
__device__ __forceinline__ uint32_t lds32(uint32_t addr) {
  uint32_t v;
  asm volatile("ld.shared.b32 %0, [%1];" : "=r"(v) : "r"(addr));
  return v;
}

// ---- mma.sync -----------------------------------------------------------------------------------

__device__ __forceinline__ void mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ void mma_s8(int (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
  asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// e4m3 x e4m3, f32 accumulation (sm_89+).
__device__ __forceinline__ void mma_e4m3(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef LX_HAS_FP8_MMA
  asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
               : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
               : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
  __trap();
#endif
}

// e4m3 x e4m3 through the block-scaled form with unit (2^0) scales. Same math as mma_e4m3, measured at
// twice its rate on an RTX 5070 Ti (387 vs 195 TFLOPS): the block-scaled kinds run at the full FP8 rate.
__device__ __forceinline__ void mma_e4m3_mx(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef LX_HAS_BLOCK_SCALE_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11, %11}, {%10}, {%11, %11};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7f7f7f7fu), "h"((uint16_t)0));
#else
  __trap();
#endif
}

// The same into a fresh result (d = a * b), for the per-block products of W4A8.
__device__ __forceinline__ void mma_e4m3_mx_zero(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1) {
#ifdef LX_HAS_BLOCK_SCALE_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%10,%10,%10}, {%11}, {%12, %12}, {%11}, {%12, %12};"
      : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "f"(0.0f), "r"(0x7f7f7f7fu), "h"((uint16_t)0));
#else
  __trap();
#endif
}

// e2m1 x e2m1 with one ue4m3 scale per 16 K on each side (NVFP4), m16n8k64. TA / TB select which
// threads' scale registers apply (see the header).
template <int TA, int TB>
__device__ __forceinline__ void mma_nvf4(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfa,
                                         uint32_t sfb) {
#ifdef LX_HAS_BLOCK_SCALE_MMA
  asm volatile(
      "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
      "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12, %13}, {%11}, {%12, %14};"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
      : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sfa), "r"(sfb), "h"((uint16_t)0),
        "h"((uint16_t)TA), "h"((uint16_t)TB));
#else
  __trap();
#endif
}

// ---- conversions --------------------------------------------------------------------------------
// All exact: every value of the source format is a bf16 (or e4m3) value.

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi) {
  uint32_t r;
  asm("cvt.rn.bf16x2.f32 %0, %1, %2;" : "=r"(r) : "f"(hi), "f"(lo));
  return r;
}

// Four int8 (one word, element 0 in the low byte) to two bf16x2: (e0, e1), (e2, e3).
__device__ __forceinline__ void i8x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
  lo = pack_bf16(static_cast<float>(static_cast<int8_t>(w)), static_cast<float>(static_cast<int8_t>(w >> 8)));
  hi = pack_bf16(static_cast<float>(static_cast<int8_t>(w >> 16)), static_cast<float>(static_cast<int8_t>(w >> 24)));
}

// e4m3 bits as f32: the 7 magnitude bits placed under an f32 exponent give value * 2^-120 for normals and
// subnormals alike; one multiply by 2^120 restores it (f32 keeps the subnormal intermediates).
__device__ __forceinline__ float e4m3_to_f32(uint32_t b) {
  const float m = __uint_as_float(((b & 0x7fu) << 20) | ((b & 0x80u) << 24));
  return m * 0x1p120f;
}

// Four e4m3 (one word) to two bf16x2. sm_120's cvt does a pair per instruction; earlier families go
// through f32.
template <int kArch>
__device__ __forceinline__ void e4m3x4_to_bf16(uint32_t w, uint32_t& lo, uint32_t& hi) {
#ifdef LX_HAS_BLOCK_SCALE_MMA
  if constexpr (kArch >= 120) {
    asm("{\n .reg .b16 l, h;\n mov.b32 {l, h}, %2;\n cvt.rn.bf16x2.e4m3x2 %0, l;\n cvt.rn.bf16x2.e4m3x2 %1, h;\n}"
        : "=r"(lo), "=r"(hi)
        : "r"(w));
    return;
  }
#endif
  lo = pack_bf16(e4m3_to_f32(w), e4m3_to_f32(w >> 8));
  hi = pack_bf16(e4m3_to_f32(w >> 16), e4m3_to_f32(w >> 24));
}

// Eight e2m1 (one word, element 0 in the low nibble) times a bf16 block scale, as four bf16x2. The 3-bit
// magnitudes index two byte tables (high and low halves of the bf16 values 0, 0.5, 1, 1.5, 2, 3, 4, 6)
// through prmt, the sign bits move to bit 15 of each half, and one bf16x2 multiply applies the scale,
// exactly: an e2m1 times an e4m3 has at most 6 significant bits.
__device__ __forceinline__ uint32_t prmt(uint32_t a, uint32_t b, uint32_t sel) {
  uint32_t r;
  asm("prmt.b32 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(sel));
  return r;
}
__device__ __forceinline__ uint32_t bf16x2_mul(uint32_t a, uint32_t b) {
  uint32_t r;
  // fma with -0 so it exists on sm_80 (mul.bf16x2 is sm_90+); -0 keeps a zero product's sign.
  asm("fma.rn.bf16x2 %0, %1, %2, %3;" : "=r"(r) : "r"(a), "r"(b), "r"(0x80008000u));
  return r;
}
__device__ __forceinline__ void e2m1x8_to_bf16(uint32_t w, uint32_t scale2, uint32_t (&out)[4]) {
  // bf16 of the magnitudes 0..7: high bytes 00 3f 3f 3f 40 40 40 40, low bytes 00 00 80 c0 00 40 80 c0.
  constexpr uint32_t kHi0 = 0x3f3f3f00u, kHi1 = 0x40404040u, kLo0 = 0xc0800000u, kLo1 = 0xc0804000u;
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    const uint32_t nib = w >> (16 * h);
    const uint32_t sel = nib & 0x7777u;
    const uint32_t hi = prmt(kHi0, kHi1, sel);  // bytes: hi(v0) hi(v1) hi(v2) hi(v3)
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
