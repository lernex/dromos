// Lossless bf16 (LX_W_BF16X) on Metal: the encoder and decoder (kernels/bf16x.metal). The products on
// the format are lx_gemv_bf16x (ops_gemv.mm) and lx_moe_experts_bf16x (ops_moe.mm), the bf16 kernels
// reading the encoding.
#include <vector>

#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kRowsPerGroup = 4;
constexpr int kMaxK = 1 << 23;  // patch positions are 23 bits

int blocks(int K) { return (K + 127) / 128; }
int header_bytes(int K) { return (4 * (2 + K / 2048) + blocks(K) + 7) / 8 * 8; }
int patch_slots(int K) { return (header_bytes(K) - blocks(K)) / 4; }
int64_t row_bytes(int K) { return header_bytes(K) + int64_t(192) * blocks(K); }

// Bytes of a row's overflow record (count, the patches, the terminator; 8-byte aligned), or 0 when its
// patches fit its slots.
int64_t overflow_bytes(uint32_t count, int slots) {
  return count > uint32_t(slots) ? (4 * (int64_t(count) + 2) + 7) / 8 * 8 : 0;
}

// Device scratch for the encoder, freed when it goes (which waits for the device).
struct Scratch {
  void* ptr = nullptr;
  explicit Scratch(size_t bytes) {
    if (Device::get().alloc(&ptr, bytes) != kOk) ptr = nullptr;
  }
  ~Scratch() { Device::get().free(ptr); }
};

// Pass 1 on w, then the overflow layout on the host: each row's patch count into counts, each
// overflowing row's record offset into offsets (from the end of the rows), and their total.
int count_patches(const void* w, int64_t rows, int K, void* data, uint32_t* counts, int64_t* offsets,
                  int64_t* overflow, void* stream) {
  static Kernel bases("bf16x_bases");
  Bf16xArgs a{};
  a.w = addr(stream, w, size_t(rows) * K * 2);
  a.data = data ? addr_w(stream, data, size_t(rows * row_bytes(K))) : 0;
  a.counts = addr_w(stream, counts, size_t(rows) * 4);
  a.rows = rows;
  a.K = K;
  if (int e = bases.launch(stream, a, size1(NSUInteger((rows + kRowsPerGroup - 1) / kRowsPerGroup)),
                           size1(32 * kRowsPerGroup)))
    return e;
  if (int e = lx_rt_stream_sync(stream)) return e;
  const int slots = patch_slots(K);
  int64_t at = 0;
  for (int64_t r = 0; r < rows; ++r) {
    if (offsets) offsets[r] = at;
    at += overflow_bytes(counts[r], slots);
  }
  *overflow = at;
  return kOk;
}

}  // namespace

extern "C" {

int64_t lx_bf16x_row_bytes(int K) { return K > 0 && K < kMaxK ? row_bytes(K) : -1; }

int64_t lx_bf16x_bytes(const void* w, int64_t rows, int K, void* stream) {
  if (rows <= 0 || K <= 0 || K >= kMaxK || !w) return -int64_t(kInvalidValue);
  Scratch counts(size_t(rows) * 4);
  if (!counts.ptr) return -int64_t(kOutOfMemory);
  int64_t overflow = 0;
  if (int e = count_patches(w, rows, K, nullptr, static_cast<uint32_t*>(counts.ptr), nullptr, &overflow, stream))
    return -int64_t(e);
  return rows * row_bytes(K) + overflow;
}

int lx_bf16x_encode(const void* w, int64_t rows, int K, void* out, int64_t out_bytes, void* stream) {
  static Kernel encode("bf16x_rows");
  if (rows <= 0) return kOk;
  if (K <= 0 || K >= kMaxK || !w || !out || (reinterpret_cast<uintptr_t>(out) & 7) != 0 ||
      out_bytes < rows * row_bytes(K))
    return kInvalidValue;
  Scratch counts(size_t(rows) * 4), offsets(size_t(rows) * 8);
  if (!counts.ptr || !offsets.ptr) return kOutOfMemory;
  int64_t overflow = 0;
  if (int e = count_patches(w, rows, K, out, static_cast<uint32_t*>(counts.ptr), static_cast<int64_t*>(offsets.ptr),
                            &overflow, stream))
    return e;
  if (rows * row_bytes(K) + overflow > out_bytes) return kInvalidValue;
  Bf16xArgs a{};
  a.w = addr(stream, w, size_t(rows) * K * 2);
  a.data = addr_w(stream, out, size_t(rows * row_bytes(K) + overflow));
  a.counts = addr(stream, counts.ptr, size_t(rows) * 4);
  a.offsets = addr(stream, offsets.ptr, size_t(rows) * 8);
  a.rows = rows;
  a.K = K;
  if (int e = encode.launch(stream, a, size1(NSUInteger((rows + kRowsPerGroup - 1) / kRowsPerGroup)),
                            size1(32 * kRowsPerGroup)))
    return e;
  return lx_rt_stream_sync(stream);  // before the scratch goes
}

int lx_bf16x_decode(const void* data, int64_t rows, int K, void* w, void* stream) {
  static Kernel decode("bf16x_decode");
  if (rows <= 0) return kOk;
  if (K <= 0 || K >= kMaxK || !data || !w) return kInvalidValue;
  Bf16xArgs a{};
  a.data = addr(stream, data, kUnknownExtent);  // overflow records follow the rows
  a.out = addr_w(stream, w, size_t(rows) * K * 2);
  a.rows = rows;
  a.K = K;
  return decode.launch(stream, a, size1(NSUInteger((rows + kRowsPerGroup - 1) / kRowsPerGroup)),
                       size1(32 * kRowsPerGroup));
}

}  // extern "C"
