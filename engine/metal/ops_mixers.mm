// The mixers: Mamba-2 (kernels/mamba.metal) and attention (kernels/attention.metal).
#include <algorithm>

#include "cuda/tunable.h"
#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kScanChunk = 8;
constexpr int kAttnGroups = 512;  // fixed grid for the attention kernels (they read T on the device)
constexpr int kFlashRows = 64;    // query rows per flash threadgroup (nax/attention.metal)
constexpr int kAttnSplits = 16;   // threadgroups per item in the decode form
constexpr int kSplitBelow = 64;   // items (T * KVH) below which the decode form runs; as in the kernel

// The split decode form's partial softmax states live in a per-stream scratch buffer. Old buffers are
// kept when it grows: a recorded step may still point into them.
struct AttnScratch {
  void* ptr = nullptr;
  size_t bytes = 0;
  std::vector<void*> old;
};

void* attn_scratch(void* stream, size_t bytes) {
  static std::mutex mu;
  static std::unordered_map<Stream*, AttnScratch> scratch;
  std::lock_guard<std::mutex> g(mu);
  AttnScratch& s = scratch[stream_of(stream)];
  if (s.bytes < bytes) {
    if (s.ptr) s.old.push_back(s.ptr);
    const size_t want = std::max<size_t>(bytes, size_t{2} << 20);
    void* p = nullptr;
    if (Device::get().alloc(&p, want) != kOk) return nullptr;
    s.ptr = p;
    s.bytes = want;
  }
  return s.ptr;
}

}  // namespace

extern "C" {

int lx_conv1d_varlen(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                     const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* out,
                     int C, int W, void* stream) {
  static Kernel out_k("conv1d_out"), state_k("conv1d_state");
  if (n_seq <= 0) return kOk;
  if (C <= 0 || W < 1 || W > 8 || ld_x < C) return kInvalidValue;
  // The token count and the slot count are device data: unknown extents for x, out and the state.
  ConvArgs a{addr(stream, x, kUnknownExtent), addr(stream, w, size_t(C) * W * 2), addr(stream, bias, size_t(C) * 2),
             addr_w(stream, conv_state, kUnknownExtent), addr(stream, slot_ids, size_t(n_seq) * 4),
             addr(stream, has_state, size_t(n_seq) * 4), addr(stream, cu_seqlens, size_t(n_seq + 1) * 4),
             addr_w(stream, out, kUnknownExtent), ld_x, C, W};
  // The outputs read the old state, so they go first, and the state kernel waits for them.
  if (int e = out_k.launch(stream, a, size2(ceil_div(C, 32), n_seq), size2(32, 8))) return e;
  stream_of(stream)->again();
  return state_k.launch(stream, a, size2(ceil_div(C, 128), n_seq), size2(128, 1));
}

int lx_ssm_scan_varlen(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B, const float* C, int ld_bc,
                       const float* A_log, const float* D, const float* dt_bias, float* ssm_state, const int32_t* slot_ids,
                       const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* y, int H, int P, int G, int N,
                       void* stream) {
  static Kernel k1("ssm_scan_1"), k2("ssm_scan_2"), k4("ssm_scan_4"), k8("ssm_scan_8"), k16("ssm_scan_16");
  if (n_seq <= 0) return kOk;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || P > 64 || N < 1 || N > 256) return kInvalidValue;
  if (ld_x < H * P || ld_dt < H || ld_bc < G * N) return kInvalidValue;
  ScanArgs a{addr(stream, x, kUnknownExtent), addr(stream, dt_raw, kUnknownExtent), addr(stream, B, kUnknownExtent),
             addr(stream, C, kUnknownExtent), addr(stream, A_log, size_t(H) * 4), addr(stream, D, size_t(H) * 4),
             addr(stream, dt_bias, size_t(H) * 4), addr_w(stream, ssm_state, kUnknownExtent),
             addr(stream, slot_ids, size_t(n_seq) * 4), addr(stream, has_state, size_t(n_seq) * 4),
             addr(stream, cu_seqlens, size_t(n_seq + 1) * 4), addr_w(stream, y, kUnknownExtent), ld_x, ld_dt, ld_bc, H, P,
             G, N};
  const int threads = (4 * P + 31) / 32 * 32;  // four lanes per state row
  const int ni = ((N + 3) / 4 + 3) / 4;        // column chunks of four per lane
  Kernel& k = ni <= 1 ? k1 : ni <= 2 ? k2 : ni <= 4 ? k4 : ni <= 8 ? k8 : k16;
  const int nip = ni <= 1 ? 1 : ni <= 2 ? 2 : ni <= 4 ? 4 : ni <= 8 ? 8 : 16;
  const int row = ((P + 3) & ~3) + 2 * 16 * nip + 4;  // floats per staged token
  return k.launch(stream, a, size2(H, n_seq), size1(threads), size_t(kScanChunk) * row * 4);
}

int lx_gated_rmsnorm(const float* y, const float* z, int ld_z, const void* w, void* out, int T, int I, int groups,
                     float eps, void* stream) {
  static Kernel k("gated_rmsnorm");
  if (T <= 0) return kOk;
  if (I <= 0 || groups <= 0 || I % groups != 0 || ld_z < I) return kInvalidValue;
  const int K = I / groups;
  GatedNormArgs a{addr(stream, y, size_t(T) * I * 4), addr(stream, z, span(T, I, ld_z, 4)), addr(stream, w, size_t(I) * 2),
                  addr_w(stream, out, size_t(T) * I * 2), ld_z, I, groups, eps};
  const NSUInteger threads = static_cast<NSUInteger>(std::min(256, std::max(32, (K + 31) / 32 * 32)));
  return k.launch(stream, a, size1(size_t(T) * groups), size1(threads));
}

int lx_attention_varlen(const float* qkv, int ld_qkv, void* k_cache, void* v_cache, const int32_t* slot_ids,
                        const int32_t* start_pos, const int32_t* cu_seqlens, int n_seq, int capacity, void* out, int H,
                        int KVH, int Dh, float scale, void* stream) {
  static Kernel write_kv("attn_write_kv"), attn("attention"), combine("attn_combine");
  static KernelSet<1> flash([](int) { return std::string("attention_flash"); }, true);
  if (n_seq <= 0 || capacity <= 0 || KVH <= 0 || H <= 0 || H % KVH != 0 || Dh <= 0 || Dh % 8 != 0 || Dh > 128 ||
      ld_qkv < (H + 2 * KVH) * Dh)
    return kInvalidValue;
  const int G = H / KVH;
  // The decode form (split items) runs on Apple GPUs at the current architecture; an override below
  // 1000 takes whole items everywhere, as CUDA's pre-cluster path does.
  const int splits = lx_arch() >= 1000 ? kAttnSplits : 1;
  void* partial = nullptr;
  if (splits > 1) {
    partial = attn_scratch(stream, size_t(kSplitBelow) * splits * G * (Dh + 2) * 4);
    if (!partial) return kOutOfMemory;
  }
  AttnArgs a{addr(stream, qkv, kUnknownExtent), addr_w(stream, k_cache, kUnknownExtent),
             addr_w(stream, v_cache, kUnknownExtent), addr(stream, slot_ids, size_t(n_seq) * 4),
             addr(stream, start_pos, size_t(n_seq) * 4), addr(stream, cu_seqlens, size_t(n_seq + 1) * 4),
             addr_w(stream, out, kUnknownExtent), addr_w(stream, partial, size_t(kSplitBelow) * splits * G * (Dh + 2) * 4),
             ld_qkv, n_seq, capacity, H, KVH, Dh, scale, splits, 32, 0};
  // Prefill items go to the flash kernel on the neural accelerators (nax/attention.metal): head size
  // 64, and the G query heads of a kv head dividing its 64 query rows.
  a.flash = Device::get().nax() && Dh == 64 && kFlashRows % G == 0 ? 1 : 0;
  if (int e = write_kv.launch(stream, a, size1(256), size1(128))) return e;
  const size_t smem = 2 * 32 * size_t(Dh + 2) * 2 + 4 * size_t(Dh) * 4;
  // In the split form the grid is a multiple of the split count, so every item gets all its parts.
  const NSUInteger groups = splits > 1 ? NSUInteger(kAttnGroups / splits * splits) : NSUInteger(kAttnGroups);
  stream_of(stream)->again();
  if (int e = attn.launch(stream, a, size1(groups), size1(128), smem)) return e;
  if (a.flash) {
    // Q (64 x 64) and the K, V tiles (32 x 64) in bf16, the scores (64 x 32) f32 and P bf16, two rows of 64.
    constexpr NSUInteger kFlashSmem = (64 * 64 + 2 * 32 * 64) * 2 + 64 * 32 * 4 + 64 * 32 * 2 + 2 * 64 * 4;
    stream_of(stream)->again();
    if (int e = flash.launch(0, stream, a, size1(kAttnGroups), size1(128), kFlashSmem)) return e;
  }
  stream_of(stream)->again();
  if (splits > 1) return combine.launch(stream, a, size1(64), size1(256));
  stream_of(stream)->drop_pending();
  return kOk;
}

}  // extern "C"
