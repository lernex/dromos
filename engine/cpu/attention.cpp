// Causal GQA with no positional encoding over varlen sequences (kernels.h, lx_attention_varlen).
//
// The step's keys and values are rounded to bf16 into the cache first, so prefill and decode read the
// same values; then each (sequence, kv head, block of new tokens) is one work item for kern/isa.cpp's
// attention, which serves all of a kv head's query heads from one pass over the visible cache.
#include <vector>

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {
constexpr int kRowsPerItem = 8;
constexpr int kMaxDh = 128, kMaxG = 16;  // kern/isa.cpp's attention keeps a token's heads on the stack
}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" int lx_attention_varlen(const float* qkv, int ld_qkv, void* k_cache, void* v_cache, const int32_t* slot_ids,
                                   const int32_t* start_pos, const int32_t* cu_seqlens, int n_seq, int capacity,
                                   void* out, int H, int KVH, int Dh, float scale, void* stream) {
  LX_PROFILE("attention_varlen");
  (void)stream;
  if (n_seq <= 0 || capacity <= 0 || KVH <= 0 || H <= 0 || H % KVH != 0 || Dh <= 0 || Dh > kMaxDh ||
      H / KVH > kMaxG || ld_qkv < (H + 2 * KVH) * Dh)
    return fail(kInvalid);
  if (!qkv || !k_cache || !v_cache || !slot_ids || !start_pos || !cu_seqlens || !out) return fail(kInvalid);
  for (int s = 0; s < n_seq; ++s) {
    const int n = cu_seqlens[s + 1] - cu_seqlens[s];
    if (n < 0 || start_pos[s] < 0 || start_pos[s] + n > capacity) return fail(kInvalid);
  }
  uint16_t* kc = static_cast<uint16_t*>(k_cache);
  uint16_t* vc = static_cast<uint16_t*>(v_cache);
  uint16_t* o = static_cast<uint16_t*>(out);
  const int64_t width = static_cast<int64_t>(KVH) * Dh;
  const int T = cu_seqlens[n_seq];

  // This step's k and v into cache rows start_pos[s] + j of each sequence's slot.
  std::vector<int> seq_of(T);
  for (int s = 0; s < n_seq; ++s)
    for (int t = cu_seqlens[s]; t < cu_seqlens[s + 1]; ++t) seq_of[t] = s;
  parallel_for(T, grain_for(T, 16), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int s = seq_of[t];
      const int64_t pos = start_pos[s] + (t - cu_seqlens[s]);
      const int64_t row = (static_cast<int64_t>(slot_ids[s]) * capacity + pos) * width;
      const float* k = qkv + t * ld_qkv + static_cast<int64_t>(H) * Dh;
      const float* v = k + width;
      for (int64_t i = 0; i < width; ++i) {
        kc[row + i] = f32_to_bf16(k[i]);
        vc[row + i] = f32_to_bf16(v[i]);
      }
    }
  });

  struct Item {
    int s, kvh, q0, q1;
  };
  std::vector<Item> items;
  for (int s = 0; s < n_seq; ++s) {
    const int n = cu_seqlens[s + 1] - cu_seqlens[s];
    for (int q0 = 0; q0 < n; q0 += kRowsPerItem)
      for (int kvh = 0; kvh < KVH; ++kvh) items.push_back({s, kvh, q0, q0 + kRowsPerItem < n ? q0 + kRowsPerItem : n});
  }
  const Kern& k = kern();
  parallel_for(static_cast<int64_t>(items.size()), 1, [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const Item& it = items[i];
      const int first = cu_seqlens[it.s];
      AttnItem a;
      a.qkv = qkv + static_cast<int64_t>(first) * ld_qkv;
      a.ld_qkv = ld_qkv;
      a.k = kc + static_cast<int64_t>(slot_ids[it.s]) * capacity * width;
      a.v = vc + static_cast<int64_t>(slot_ids[it.s]) * capacity * width;
      a.out = o + static_cast<int64_t>(first) * H * Dh;
      a.H = H;
      a.KVH = KVH;
      a.Dh = Dh;
      a.kvh = it.kvh;
      a.start = start_pos[it.s];
      a.q0 = it.q0;
      a.q1 = it.q1;
      a.scale = scale;
      k.attention(a);
    }
  });
  return kOk;
}
