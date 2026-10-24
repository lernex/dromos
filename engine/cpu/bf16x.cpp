// Lossless BF16 (kernels.h, LX_W_BF16X): the encoder, its size query and the decoder. The layout is
// kern/table.h's bf16x (cuda/bf16x.cuh in full); the products on it are the bf16 products with another row
// reader (Row<kW_BF16X> in kern/isa.cpp, reached through lx_gemv_bf16x and lx_moe_experts_bf16x).
//
// A row's encoding depends only on that row, so rows are encoded in parallel. The one thing shared is the
// overflow area after the rows, where a row with more patches than its slots keeps them: a first pass
// counts every row's patches, and the records are laid out in row order from those counts, so the bytes
// are the same however the rows are split between threads.
#include <algorithm>
#include <vector>

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

using bf16x::kBlock;
using bf16x::kBlockBytes;
using bf16x::kCodes;
using bf16x::kMaxK;
using bf16x::kNoPatch;

inline uint32_t exponent(uint16_t v) { return (v >> 7) & 0xffu; }

// The largest exponent of each block of a row (padding past K counts as 0).
void block_bases(const uint16_t* w, int K, uint8_t* bases) {
  for (int b = 0; b < bf16x::blocks(K); ++b) {
    uint32_t e0 = 0;
    for (int k = b * kBlock, end = std::min(K, (b + 1) * kBlock); k < end; ++k) e0 = std::max(e0, exponent(w[k]));
    bases[b] = static_cast<uint8_t>(e0);
  }
}

// Values more than 15 binades below their block's largest exponent: the row's patches.
int64_t count_patches(const uint16_t* w, int K, uint8_t* bases) {
  block_bases(w, K, bases);
  int64_t n = 0;
  for (int k = 0; k < K; ++k) n += bases[k / kBlock] - exponent(w[k]) > 15u;
  return n;
}

// An overflow record: u32 count, the patches, kNoPatch, padded to 8 bytes.
int64_t record_bytes(int64_t patches) { return (4 * (patches + 2) + 7) / 8 * 8; }

inline void put32(uint8_t* p, uint32_t v) { memcpy(p, &v, 4); }
inline uint32_t get32(const uint8_t* p) {
  uint32_t v;
  memcpy(&v, p, 4);
  return v;
}

// One row record at rec, and its overflow record at rec + overflow when it has more patches than slots.
void encode_row(const uint16_t* w, int K, uint8_t* rec, int64_t patches, int64_t overflow) {
  const int P = bf16x::patch_slots(K), B = bf16x::blocks(K), H = bf16x::header_bytes(K);
  uint8_t* bases = rec + bf16x::base_offset(K);
  memset(rec, 0, static_cast<size_t>(H));
  block_bases(w, K, bases);
  uint8_t* slots = rec;
  for (int s = 0; s < P; ++s) put32(slots + 4 * s, kNoPatch);
  uint8_t* list = slots;  // where the patches go: the slots, or the overflow record's entries
  if (patches > P) {
    put32(slots, 0x80000000u | static_cast<uint32_t>(overflow / 8));
    uint8_t* r = rec + overflow;
    memset(r, 0, static_cast<size_t>(record_bytes(patches)));
    put32(r, static_cast<uint32_t>(patches));
    put32(r + 4 * (patches + 1), kNoPatch);
    list = r + 4;
  }
  int64_t at = 0;
  for (int b = 0; b < B; ++b) {
    uint8_t* blk = rec + H + static_cast<int64_t>(b) * kBlockBytes;
    memset(blk, 0, kBlockBytes);
    const uint32_t e0 = bases[b];
    for (int i = 0; i < kBlock; ++i) {
      const int k = b * kBlock + i;
      if (k >= K) break;
      const uint16_t v = w[k];
      const uint32_t e = exponent(v);
      blk[i] = static_cast<uint8_t>(((v >> 8) & 0x80u) | (v & 0x7fu));
      uint32_t code = e0 - e;
      if (code > 15u) {
        put32(list + 4 * at++, (static_cast<uint32_t>(k) << 8) | e);
        code = 0;
      }
      // Code word g covers values 8g .. 8g + 7: byte b of it holds value 8g + b low, 8g + 4 + b high.
      uint8_t& cb = blk[kCodes + (i >> 3) * 4 + (i & 3)];
      cb = static_cast<uint8_t>(cb | ((i & 4) ? code << 4 : code));
    }
  }
}

// Patch counts of every row and the overflow layout they give: rec_at[r] is row r's record offset from the
// start of the overflow area (-1 for rows whose patches fit their slots). Returns the area's size.
int64_t layout(const uint16_t* w, int64_t rows, int K, std::vector<int64_t>& counts, std::vector<int64_t>& rec_at) {
  counts.assign(static_cast<size_t>(rows), 0);
  rec_at.assign(static_cast<size_t>(rows), -1);
  const int B = bf16x::blocks(K);
  parallel_for(rows, grain_for(rows, std::max<int64_t>(1, 65536 / K)), [&](int64_t r0, int64_t r1) {
    uint8_t* bases = static_cast<uint8_t*>(scratch(static_cast<size_t>(B), 0));
    for (int64_t r = r0; r < r1; ++r) counts[r] = count_patches(w + r * K, K, bases);
  });
  const int P = bf16x::patch_slots(K);
  int64_t area = 0;
  for (int64_t r = 0; r < rows; ++r) {
    if (counts[r] <= P) continue;
    rec_at[r] = area;
    area += record_bytes(counts[r]);
  }
  return area;
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int64_t lx_bf16x_row_bytes(int K) { return K > 0 && K < kMaxK ? bf16x::row_bytes(K) : -1; }

int64_t lx_bf16x_bytes(const void* w, int64_t rows, int K, void* stream) {
  LX_PROFILE("bf16x_bytes");
  (void)stream;
  if (rows <= 0 || K <= 0 || K >= kMaxK || !w) return -static_cast<int64_t>(fail(kInvalid));
  std::vector<int64_t> counts, rec_at;
  const int64_t area = layout(static_cast<const uint16_t*>(w), rows, K, counts, rec_at);
  return rows * bf16x::row_bytes(K) + area;
}

int lx_bf16x_encode(const void* w, int64_t rows, int K, void* out, int64_t out_bytes, void* stream) {
  LX_PROFILE("bf16x_encode");
  (void)stream;
  if (rows <= 0) return kOk;
  const int64_t rb = K > 0 && K < kMaxK ? bf16x::row_bytes(K) : 0;
  if (!rb || !w || !out || (reinterpret_cast<uintptr_t>(out) & 7) != 0 || out_bytes < rows * rb) return fail(kInvalid);
  const uint16_t* src = static_cast<const uint16_t*>(w);
  std::vector<int64_t> counts, rec_at;
  const int64_t area = layout(src, rows, K, counts, rec_at);
  if (rows * rb + area > out_bytes) return fail(kInvalid);
  uint8_t* o = static_cast<uint8_t*>(out);
  parallel_for(rows, grain_for(rows, std::max<int64_t>(1, 65536 / K)), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      // A record's address is kept relative to its own row, in 8-byte units.
      const int64_t overflow = rec_at[r] < 0 ? 0 : rows * rb + rec_at[r] - r * rb;
      encode_row(src + r * K, K, o + r * rb, counts[r], overflow);
    }
  });
  return kOk;
}

int lx_bf16x_decode(const void* data, int64_t rows, int K, void* w, void* stream) {
  LX_PROFILE("bf16x_decode");
  (void)stream;
  if (rows <= 0) return kOk;
  if (K <= 0 || K >= kMaxK || !data || !w) return fail(kInvalid);
  const uint8_t* d = static_cast<const uint8_t*>(data);
  uint16_t* dst = static_cast<uint16_t*>(w);
  const int64_t rb = bf16x::row_bytes(K);
  const int H = bf16x::header_bytes(K), P = bf16x::patch_slots(K);
  parallel_for(rows, grain_for(rows, std::max<int64_t>(1, 65536 / K)), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const uint8_t* rec = d + r * rb;
      const uint8_t* bases = rec + bf16x::base_offset(K);
      uint16_t* row = dst + r * K;
      for (int k = 0; k < K; ++k) {
        const uint8_t* blk = rec + H + static_cast<int64_t>(k / kBlock) * kBlockBytes;
        const int i = k % kBlock;
        const uint32_t c = blk[kCodes + (i >> 3) * 4 + (i & 3)];
        const uint32_t e = (bases[k / kBlock] - ((i & 4) ? c >> 4 : c & 15u)) & 0xffu;
        const uint32_t v = blk[i];
        row[k] = static_cast<uint16_t>(((v & 0x80u) << 8) | (e << 7) | (v & 0x7fu));
      }
      // The patches: the slots, or the overflow record they point to.
      const uint8_t* list = rec;
      int64_t n = P;
      const uint32_t s0 = get32(rec);
      if (s0 != kNoPatch && (s0 & 0x80000000u)) {
        const uint8_t* rr = rec + static_cast<int64_t>(s0 & 0x7fffffffu) * 8;
        n = get32(rr);
        list = rr + 4;
      }
      for (int64_t q = 0; q < n; ++q) {
        const uint32_t p = get32(list + 4 * q);
        if (p == kNoPatch) break;
        const uint32_t k = p >> 8;
        if (k < static_cast<uint32_t>(K)) row[k] = static_cast<uint16_t>((row[k] & 0x807fu) | ((p & 0xffu) << 7));
      }
    }
  });
  return kOk;
}

}  // extern "C"
