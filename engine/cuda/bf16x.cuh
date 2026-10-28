// Lossless compressed BF16 (LX_W_BF16X, kernels.h "Weight formats"): the layout's constants and its
// device-side decoding, shared by the encoder, the decode GEMV and the expert kernels (bf16x.cu), and
// by any later kernel that reads the format (a tensor-core mainloop expands a block the same way).
//
// A bf16 is sign (1) | exponent (8) | mantissa (7). Trained weights use few exponents: within 128
// consecutive k of a row, all but ~5e-5 of Praxis's weights sit within 16 binades of the block's
// largest. So each value keeps its sign and mantissa in one byte and its exponent as a 4-bit code below
// the block's largest exponent (the base), 12 bits a value. The few that fall further below are
// patched: the row header lists them, (position, exponent), in k order. Every row has a fixed size, so
// any row is addressed directly, and every load a kernel issues is independent of every other (no
// pointer is read to find data), which keeps single-wave kernels free of extra latency.
//
// Row record (row r at data + r * row_bytes(K)):
//   u32 slot[P]       patches, ascending position: (k << 8) | exponent; unused slots kNoPatch. A row
//                     with more than P patches keeps them in an overflow record after the rows: slot 0
//                     is then 0x80000000 | (record offset from the row's start / 8), and the record
//                     is u32 count, count patches, kNoPatch.
//   u8 base[B]        each block's largest exponent
//   pad to 8 bytes    (fewer than 4 bytes: P takes up any whole word of padding)
//   B blocks of 192 bytes:
//     [0, 128)        byte i: sign << 7 | mantissa of value i (value i is k = 128 * block + i)
//     [128, 192)      codes, 4 bits a value: u32 word g covers values 8g .. 8g + 7, byte b holds value
//                     8g + b in its low nibble and value 8g + 4 + b in its high nibble.
//                     exponent = base - code; code 0 for patched values and for k >= K (padding).
// with B = ceil(K / 128) and P = (H - B) / 4 slots, H = roundup(4 * (2 + K / 2048) + B, 8) the header
// bytes: P is 3 at K = 512, 2 at 1024, 4 at 2048 and 4096. Praxis rows have at most 3 patches, so only
// a few K = 1024 expert rows use overflow records. Every row is a multiple of 8 bytes, so u64 loads
// stay aligned.
#pragma once

#include <cuda_runtime.h>
#include <stdint.h>

namespace lx::bf16x {

constexpr int kBlock = 128;           // values per block (one base exponent)
constexpr int kBlockBytes = 192;      // sign|mantissa bytes, then codes
constexpr int kCodes = 128;           // offset of the codes in a block
constexpr uint32_t kNoPatch = 0xffffffffu;
constexpr int kMaxK = 1 << 23;        // patch positions are 23 bits
constexpr int kOverflowAlign = 8;     // overflow records are addressed in 8-byte units from their row

__host__ __device__ constexpr int blocks(int K) { return (K + kBlock - 1) / kBlock; }
// At least 2 + K / 2048 patch slots, and as many more as fit in the header's padding to 8 bytes.
__host__ __device__ constexpr int header_bytes(int K) { return (4 * (2 + K / 2048) + blocks(K) + 7) / 8 * 8; }
__host__ __device__ constexpr int patch_slots(int K) { return (header_bytes(K) - blocks(K)) / 4; }
__host__ __device__ constexpr int base_offset(int K) { return 4 * patch_slots(K); }
__host__ __device__ constexpr int64_t row_bytes(int K) {
  return header_bytes(K) + static_cast<int64_t>(kBlockBytes) * blocks(K);
}

// The four exponents of values 4q .. 4q + 3 of a block, one per byte: base - code. `codes` holds the
// quad's codes in the low nibbles of its bytes (high nibbles are masked off). Codes never exceed the
// base, so no byte borrows from its neighbour.
__device__ __forceinline__ uint32_t exponents(uint32_t base, uint32_t codes) {
  return base * 0x01010101u - (codes & 0x0f0f0f0fu);
}

// Applies one patch to the exponents of the four values k0 .. k0 + 3, if it falls among them.
__device__ __forceinline__ uint32_t patch(uint32_t e, uint32_t slot, uint32_t k0) {
  const uint32_t d = (slot >> 8) - k0;
  if (d < 4) {
    const uint32_t sh = 8 * d;
    e = (e & ~(0xffu << sh)) | ((slot & 0xffu) << sh);
  }
  return e;
}

// The bf16 bits of four values from their sign|mantissa bytes and exponents: values 0, 1 in the
// first word, 2, 3 in the second (low half first), exactly the checkpoint's bits. A bf16's high byte
// is sign | exponent >> 1, its low byte exponent & 1 | mantissa; the byte permutes interleave them.
__device__ __forceinline__ uint2 assemble(uint32_t sm, uint32_t e) {
  const uint32_t hi = (sm & 0x80808080u) | ((e >> 1) & 0x7f7f7f7fu);
  const uint32_t lo = (sm & 0x7f7f7f7fu) | ((e << 7) & 0x80808080u);
  return make_uint2(__byte_perm(lo, hi, 0x5140), __byte_perm(lo, hi, 0x7362));
}

// Four f32 from assemble()'s two words. A bf16 is the top half of an f32.
__device__ __forceinline__ void to_f32(uint2 b, float* f) {
  f[0] = __uint_as_float(b.x << 16);
  f[1] = __uint_as_float(b.x & 0xffff0000u);
  f[2] = __uint_as_float(b.y << 16);
  f[3] = __uint_as_float(b.y & 0xffff0000u);
}

// A row's patches in k order. `next` is the next one not yet passed (kNoPatch, position 0xffffff,
// once there are none), so a reader moving forward through k applies every patch whose position is
// below where it is going and advances. Rows without patches start at kNoPatch and cost one compare
// per step.
//
// Two steps, so that reading the header never holds back the weight stream: prefetch() issues the
// load of the row's first slot, and resolve() (called after the row's first weight loads are in
// flight) is the first use of it. Only a row with an overflow record makes resolve() load again.
struct Patches {
  const uint32_t* p;  // the slot after `next`
  int left;           // slots readable after `next`
  uint32_t next;

  __device__ __forceinline__ void prefetch(const uint8_t* row) {
    p = reinterpret_cast<const uint32_t*>(row);
    next = p[0];
  }
  __device__ __forceinline__ void resolve(int slots) {
    if (next != kNoPatch && (next & 0x80000000u)) {  // the patches live in the row's overflow record
      const uint32_t* rec =
          reinterpret_cast<const uint32_t*>(reinterpret_cast<const uint8_t*>(p) +
                                            static_cast<int64_t>(next & 0x7fffffffu) * kOverflowAlign);
      slots = static_cast<int>(rec[0]) + 1;  // the patches and their terminator
      p = rec + 1;
      next = p[0];
    }
    ++p;
    left = slots - 1;
  }
  __device__ __forceinline__ void start(const uint8_t* row, int slots) {
    prefetch(row);
    resolve(slots);
  }
  // Passes every patch below k (for a reader starting partway into a row).
  __device__ __forceinline__ void skip_to(uint32_t k) {
    while (pos() < k) advance();
  }
  __device__ __forceinline__ void advance() {
    next = left > 0 ? *p : kNoPatch;
    ++p;
    --left;
  }
  __device__ __forceinline__ uint32_t pos() const { return next >> 8; }
};

// The row's exponent bytes of values k .. k + 3 with its patches applied: the header's slots (ascending, unused
// kNoPatch), or, for a row whose slot 0 holds an overflow record's offset, the record (u32 count, count patches).
__device__ __forceinline__ uint32_t patched(const uint8_t* row, int K, uint32_t k, uint32_t e) {
  const uint32_t* hdr = reinterpret_cast<const uint32_t*>(row);
  const uint32_t s0 = hdr[0];
  // kNoPatch (an unused slot, and what a row without patches holds in slot 0) has the overflow bit too
  if (s0 != kNoPatch && (s0 & 0x80000000u)) {
    const uint32_t* rec =
        reinterpret_cast<const uint32_t*>(row + static_cast<int64_t>(s0 & 0x7fffffffu) * kOverflowAlign);
    const uint32_t n = rec[0];
    for (uint32_t i = 0; i < n; ++i) e = patch(e, rec[1 + i], k);
    return e;
  }
  const int slots = patch_slots(K);
  for (int i = 0; i < slots; ++i) {
    const uint32_t slot = hdr[i];
    if (slot == kNoPatch) break;
    e = patch(e, slot, k);
  }
  return e;
}

// Four bf16 values from value k (a multiple of 4, below K) of the row record at row: their bits as a uint2 (values
// 0 and 1 in x, 2 and 3 in y). Everything is read from the record: the block's sign|mantissa bytes, its codes (a
// quad at k % 8 == 4 takes the codes' high nibbles), its base and the row's patches.
__device__ __forceinline__ uint2 quad(const uint8_t* row, int K, int k) {
  const int blk = k / kBlock, off = k - blk * kBlock;
  const uint8_t* b = row + header_bytes(K) + static_cast<int64_t>(blk) * kBlockBytes;
  const uint32_t sm = *reinterpret_cast<const uint32_t*>(b + off);
  const uint32_t codes = *reinterpret_cast<const uint32_t*>(b + kCodes + (off & ~7) / 2);
  const uint32_t base = row[base_offset(K) + blk];
  const uint32_t e = exponents(base, (off & 4) ? codes >> 4 : codes);
  return assemble(sm, patched(row, K, static_cast<uint32_t>(k), e));
}

// Eight bf16 values from value k (a multiple of 8): the 16-byte chunk the tensor-core reader takes.
__device__ __forceinline__ uint4 chunk8(const uint8_t* row, int K, int k) {
  const uint2 lo = quad(row, K, k), hi = quad(row, K, k + 4);
  return make_uint4(lo.x, lo.y, hi.x, hi.y);
}

}  // namespace lx::bf16x
