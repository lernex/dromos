// The offload kernels (kernels.h, "Offload") on the CPU. Host memory is the device here, so the VRAM pool and
// the host pool are both addresses the kernels read, and a pair's expert is wherever its lx_expert_ref points.
//
// The cache decisions are the CUDA warp version's (engine/cuda/offload.cu), made by one thread in pair order:
// hits, misses, which miss fills which slot (least recently used, lowest slot on a tie, pinned slots never),
// the CPU lane's share and job, and the counters. The expert math is lx_moe_experts_decode's: the dot kernels
// on f32 activations, weight only, the SwiGLU intermediate rounded to bf16 before the down rows, and each
// output summed over its token's pairs in k order. Work is split over (pair or group, block of rows) like
// moe.cpp, and every output row is written by one item, so no two threads write the same value.
//
// Scratch: a call's own buffers stay in slots kSlotActs and kSlotInter for the whole call, and the work items
// use kSlotTile and kSlotPanel, so an item never reuses a buffer its call still needs.
#include <algorithm>
#include <chrono>
#include <cstring>
#include <thread>

#include "cpu/cpu.h"

#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

namespace lx::cpu {
namespace {

constexpr int kMaxTopK = 8;           // the GPU kernels' bound, and the bound the expert ops hold to
constexpr int64_t kMaxBlock = 64;     // rows per work item at most: the stack tiles below hold this many
constexpr int64_t kParallelBytes = int64_t{4} << 20;  // copies this size and up are split over the pool
constexpr int64_t kCopyGrain = int64_t{1} << 20;
constexpr int kSlotActs = 1, kSlotTile = 2, kSlotPanel = 3, kSlotInter = 4;
constexpr uint32_t kProbeMagic = 0x9e3779b9u;

// Pauses a spinning thread for a moment.
inline void relax() {
#if defined(__x86_64__) || defined(__i386__)
  _mm_pause();
#elif defined(__aarch64__)
  __asm__ __volatile__("yield");
#endif
}

// ---- formats ---------------------------------------------------------------------------------------------

// The 8-bit formats carry an f32 row factor per row of gate_up [2I] and down [L]; BF16 carries none.
bool scaled_type(int type) { return type == LX_W_INT8 || type == LX_W_FP8; }
bool supported_type(int type) { return type == LX_W_BF16 || scaled_type(type); }
int elem_bytes(int type) { return type == LX_W_BF16 ? 2 : 1; }
int dot_type(int type) { return type == LX_W_BF16 ? kW_BF16 : type == LX_W_INT8 ? kW_INT8 : kW_FP8; }

// One expert as the dot kernels read it: gate_up [2I, L] and down [L, I] in the stack's format, with the
// row factors (null for BF16). The factors are applied by the caller once per output, as the kernels expect.
struct Expert {
  WeightRef gu, dn;
  const float* gu_scale;  // f32 [2I], or null
  const float* dn_scale;  // f32 [L], or null
};

Expert expert_of(int type, const lx_expert_ref& r) {
  Expert e{};
  e.gu = WeightRef{dot_type(type), r.gate_up, r.gate_up_scale};
  e.dn = WeightRef{dot_type(type), r.down, r.down_scale};
  e.gu_scale = scaled_type(type) ? static_cast<const float*>(r.gate_up_scale) : nullptr;
  e.dn_scale = scaled_type(type) ? static_cast<const float*>(r.down_scale) : nullptr;
  return e;
}

// Every expert a ref array names is complete: a gate_up has its down, and the 8-bit formats their factors.
// Null gate_up is "no expert here" and is not checked further.
int check_refs(int type, const lx_expert_ref* refs, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    const lx_expert_ref& r = refs[i];
    if (!r.gate_up) continue;
    if (!r.down || (scaled_type(type) && (!r.gate_up_scale || !r.down_scale))) return fail(kInvalid);
  }
  return kOk;
}

// Rows per work item: `units` (pairs or groups) times the blocks of `rows` should give every thread several
// items, and a block is a multiple of 8 rows (the dot kernels' tile), at most kMaxBlock.
int64_t block_rows(int64_t rows, int64_t units) {
  const int64_t want = static_cast<int64_t>(threads()) * 4;
  int64_t b = (rows * units + want - 1) / want;
  b = (b + 7) / 8 * 8;
  return std::max<int64_t>(8, std::min<int64_t>({kMaxBlock, b, rows}));
}

// out[m * ld_out + (n - n0)] = sum over k of x[m, k] * w[n, k], for rows n in [n0, n1) and M rows of x.
void dot_rows(const Kern& kn, const float* x, int64_t ld_x, int64_t M, const WeightRef& w, int64_t K, int64_t n0,
              int64_t n1, float* out, int64_t ld_out) {
  DotF32 d{x, ld_x, w, K, out, ld_out, nullptr, 0};
  if (M > 8) {
    // A panel decodes each weight row once for the whole tile, sized as moe.cpp sizes it.
    d.panel_floats = std::min<int64_t>((n1 - n0) * K, std::max<int64_t>(128 * 1024, 4 * K));
    d.panel = static_cast<float*>(scratch(sizeof(float) * d.panel_floats, kSlotPanel));
  }
  kn.dot_f32(d, 0, M, n0, n1);
}

// The gate and up rows [i0, i1) of an expert on M activation rows (x, row stride L). Writes, for each row m
// and intermediate i, bf16(silu(s_g * gate_i . x_m) * (s_u * up_i . x_m)) as f32 into out[m * I + i]. tile
// holds 2 * M * (i1 - i0) floats.
void swiglu_rows(const Kern& kn, const Expert& e, const float* x, int64_t M, int64_t L, int64_t I, int64_t i0,
                 int64_t i1, float* tile, float* out) {
  const int64_t n = i1 - i0;
  float* g = tile;
  float* u = tile + M * n;
  dot_rows(kn, x, L, M, e.gu, L, i0, i1, g, n);
  dot_rows(kn, x, L, M, e.gu, L, I + i0, I + i1, u, n);
  for (int64_t m = 0; m < M; ++m) {
    for (int64_t i = 0; i < n; ++i) {
      const float sg = e.gu_scale ? e.gu_scale[i0 + i] : 1.0f;
      const float su = e.gu_scale ? e.gu_scale[I + i0 + i] : 1.0f;
      // The intermediate is rounded to bf16 before the down projection, as lx_moe_experts does.
      out[m * I + i0 + i] = round_bf16(silu(sg * g[m * n + i]) * (su * u[m * n + i]));
    }
  }
}

// The down rows [l0, l1) of an expert on M intermediate rows (x, row stride I). Writes s_l * (down_l . x_m)
// into out[m * (l1 - l0) + (l - l0)].
void down_rows(const Kern& kn, const Expert& e, const float* x, int64_t M, int64_t I, int64_t l0, int64_t l1,
               float* out) {
  const int64_t nl = l1 - l0;
  dot_rows(kn, x, I, M, e.dn, I, l0, l1, out, nl);
  if (e.dn_scale) {
    for (int64_t m = 0; m < M; ++m)
      for (int64_t l = 0; l < nl; ++l) out[m * nl + l] *= e.dn_scale[l0 + l];
  }
}

// A pair that reads an expert across the bus also writes the bytes it read into its VRAM slot. These copy
// the rows a work item read: gate and up rows [i0, i1) with their factors, or down rows [l0, l1).
void fill_gate_up(int type, const lx_expert_ref& from, const lx_expert_ref& to, int64_t L, int64_t I, int64_t i0,
                  int64_t i1) {
  const int64_t rb = L * elem_bytes(type);
  char* dg = static_cast<char*>(const_cast<void*>(to.gate_up));
  const char* sg = static_cast<const char*>(from.gate_up);
  std::memcpy(dg + i0 * rb, sg + i0 * rb, (i1 - i0) * rb);
  std::memcpy(dg + (I + i0) * rb, sg + (I + i0) * rb, (i1 - i0) * rb);
  if (scaled_type(type)) {
    char* ds = static_cast<char*>(const_cast<void*>(to.gate_up_scale));
    const char* ss = static_cast<const char*>(from.gate_up_scale);
    std::memcpy(ds + i0 * 4, ss + i0 * 4, (i1 - i0) * 4);
    std::memcpy(ds + (I + i0) * 4, ss + (I + i0) * 4, (i1 - i0) * 4);
  }
}

void fill_down(int type, const lx_expert_ref& from, const lx_expert_ref& to, int64_t I, int64_t l0, int64_t l1) {
  const int64_t rb = I * elem_bytes(type);
  char* dd = static_cast<char*>(const_cast<void*>(to.down));
  std::memcpy(dd + l0 * rb, static_cast<const char*>(from.down) + l0 * rb, (l1 - l0) * rb);
  if (scaled_type(type)) {
    char* ds = static_cast<char*>(const_cast<void*>(to.down_scale));
    std::memcpy(ds + l0 * 4, static_cast<const char*>(from.down_scale) + l0 * 4, (l1 - l0) * 4);
  }
}

// Copies whole ranges of memory, split over the pool when they are big (as lx_rt_copy splits them).
void copy_bytes(void* dst, const void* src, int64_t bytes) {
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  if (bytes < kParallelBytes) {
    std::memcpy(d, s, static_cast<size_t>(bytes));
    return;
  }
  parallel_for(bytes, kCopyGrain, [&](int64_t b, int64_t e) { std::memcpy(d + b, s + b, static_cast<size_t>(e - b)); });
}

// ---- the cache -------------------------------------------------------------------------------------------

// The expert at `slot` of a pool, whose slots are the state's strides apart.
lx_expert_ref pool_ref(const lx_offload_state& st, const lx_expert_ref& pool, int64_t slot) {
  lx_expert_ref r{};
  r.gate_up = static_cast<const char*>(pool.gate_up) + slot * st.gate_up_stride;
  r.down = static_cast<const char*>(pool.down) + slot * st.down_stride;
  if (pool.gate_up_scale)
    r.gate_up_scale = static_cast<const char*>(pool.gate_up_scale) + slot * st.gate_up_scale_stride;
  if (pool.down_scale) r.down_scale = static_cast<const char*>(pool.down_scale) + slot * st.down_scale_stride;
  return r;
}

// The state's tables and mailbox, as the runtime allocates them with the cache.
bool state_ok(const lx_offload_state& st) {
  return st.n_layers > 0 && st.n_experts > 0 && st.top_k > 0 && st.L > 0 && st.I > 0 && st.vram_slot &&
         st.host_slot && st.cpu_first && st.layer_first && st.layer_pinned && st.slot_expert && st.slot_stamp &&
         st.slot_prefetched && st.clock && st.counters && st.job_seq && st.job_pending && st.mailbox &&
         st.cpu_latent;
}

// The least recently used dynamic slot of `layer` that was not used at clock `now`, the lowest slot on a
// tie; -1 when every dynamic slot was used at `now`. Pinned slots are never candidates.
int pick_victim(const lx_offload_state& st, int layer, uint32_t now) {
  const int first = st.layer_first[layer] + st.layer_pinned[layer], last = st.layer_first[layer + 1];
  int best = -1;
  uint32_t best_stamp = 0;
  for (int s = first; s < last; ++s) {
    const uint32_t v = st.slot_stamp[s];
    if (v < now && (best < 0 || v < best_stamp)) {
      best = s;
      best_stamp = v;
    }
  }
  return best;
}

// Moves `expert` of `layer` into slot v, evicting the slot's previous expert.
void claim_slot(const lx_offload_state& st, int layer, int expert, int v, uint32_t now, bool prefetched) {
  const int base = layer * st.n_experts;
  const int old = st.slot_expert[v];
  if (old >= 0) st.vram_slot[base + old] = -1;
  st.vram_slot[base + expert] = v;
  st.slot_expert[v] = expert;
  st.slot_stamp[v] = now;
  st.slot_prefetched[v] = prefetched ? 1 : 0;
}

// The CUDA plan kernel's decisions for one MoE layer, in pair order. Fills src and fill for every pair and,
// when pairs go to the CPU lane, posts the job: its pairs, the latent rows, then the doorbell.
int plan_layer(const int32_t* ids, const float* weights, const void* latent, int layer, const lx_offload_state& st,
               lx_expert_ref* src, lx_expert_ref* fill, int T) {
  const int K = st.top_k, P = T * K, E = st.n_experts, base = layer * E;
  const uint32_t now = *st.clock + 1u;
  *st.clock = now;

  // Hits first, so their slots carry `now` before any victim is chosen. A pair served by a prefetched slot
  // counts as a prefetch hit (the counter's meaning: pairs served from a slot a prefetch filled); the flags
  // clear once the whole step has counted them.
  int pair_miss[LX_OFFLOAD_MAX_PAIRS];  // the index of the pair's miss, -1 for a hit
  int hits = 0, prefetch_hits = 0;
  for (int p = 0; p < P; ++p) {
    const int s = st.vram_slot[base + ids[p]];
    pair_miss[p] = s >= 0 ? -1 : 0;
    if (s < 0) continue;
    st.slot_stamp[s] = now;
    prefetch_hits += st.slot_prefetched[s] ? 1 : 0;
    src[p] = pool_ref(st, st.vram_pool, s);
    fill[p] = lx_expert_ref{};
    ++hits;
  }
  for (int p = 0; p < P; ++p)
    if (pair_miss[p] < 0) st.slot_prefetched[st.vram_slot[base + ids[p]]] = 0;

  // The distinct missed experts, in pair order.
  int n_miss = 0;
  int miss_expert[LX_OFFLOAD_MAX_PAIRS], miss_slot[LX_OFFLOAD_MAX_PAIRS];
  bool miss_cpu[LX_OFFLOAD_MAX_PAIRS] = {};
  for (int p = 0; p < P; ++p) {
    if (pair_miss[p] < 0) continue;
    const int e = ids[p];
    int m = 0;
    while (m < n_miss && miss_expert[m] != e) ++m;
    if (m == n_miss) {
      miss_expert[m] = e;
      miss_slot[m] = -1;
      ++n_miss;
    }
    pair_miss[p] = m;
  }

  // The CPU lane: experts only on disk always; then its share of the rest, cold experts first (a hot one the
  // GPU reads lands in VRAM and is a hit next time). The share is counted in experts, as the CUDA plan counts.
  int cpu = 0;
  for (int m = 0; m < n_miss; ++m) {
    miss_cpu[m] = st.host_slot[base + miss_expert[m]] < 0;
    cpu += miss_cpu[m] ? 1 : 0;
  }
  int target = static_cast<int>(st.cpu_fraction * static_cast<float>(n_miss) + 0.5f);
  target = std::min(target, st.max_cpu_pairs);
  for (int pass = 0; pass < 2 && cpu < target; ++pass) {
    for (int m = n_miss - 1; m >= 0 && cpu < target; --m) {
      if (miss_cpu[m]) continue;
      if (pass == 0 && !st.cpu_first[base + miss_expert[m]]) continue;
      miss_cpu[m] = true;
      ++cpu;
    }
  }

  // Victims for the GPU's misses. Hits are stamped with `now`, so none of them is a victim; a victim claimed
  // here is stamped `now` too, so it is not chosen twice. Past the last free slot the rest read without a fill.
  int fills = 0;
  if (st.fill) {
    for (int m = 0; m < n_miss; ++m) {
      if (miss_cpu[m]) continue;
      const int v = pick_victim(st, layer, now);
      if (v < 0) break;
      claim_slot(st, layer, miss_expert[m], v, now, false);
      miss_slot[m] = v;
      ++fills;
    }
  }

  // Refs for the misses. The first pair of a filled expert writes its slot; any other pair of the same expert
  // reads host memory without writing. The CPU's pairs get null refs, which the GPU skips.
  int gpu_pairs = 0, cpu_pairs = 0;
  bool first_seen[LX_OFFLOAD_MAX_PAIRS] = {};
  for (int p = 0; p < P; ++p) {
    const int m = pair_miss[p];
    if (m < 0) continue;
    if (miss_cpu[m]) {
      src[p] = lx_expert_ref{};
      fill[p] = lx_expert_ref{};
      ++cpu_pairs;
      continue;
    }
    src[p] = pool_ref(st, st.host_pool, st.host_slot[base + miss_expert[m]]);
    const bool lead = !first_seen[m];
    first_seen[m] = true;
    fill[p] = (lead && miss_slot[m] >= 0) ? pool_ref(st, st.vram_pool, miss_slot[m]) : lx_expert_ref{};
    ++gpu_pairs;
  }

  // The CPU's job. The latent rows and the pairs are written before the doorbell's release store, so a worker
  // that sees the new sequence number sees the whole job.
  if (cpu_pairs > 0) {
    lx_offload_mailbox* mb = st.mailbox;
    int n = 0;
    for (int p = 0; p < P; ++p) {
      const int m = pair_miss[p];
      if (m < 0 || !miss_cpu[m]) continue;
      mb->token[n] = p / K;
      mb->expert[n] = ids[p];
      mb->weight[n] = weights[p];
      ++n;
    }
    mb->layer = layer;
    mb->T = T;
    mb->n = n;
    std::memcpy(st.cpu_latent, latent, sizeof(uint16_t) * static_cast<size_t>(T) * st.L);
    const uint32_t seq = *st.job_seq + 1;
    *st.job_seq = seq;
    *st.job_pending = 1;
    __atomic_store_n(&mb->doorbell, seq, __ATOMIC_RELEASE);
  } else {
    *st.job_pending = 0;
  }

  unsigned long long* c = st.counters + int64_t{layer} * LX_OFFLOAD_COUNTERS;
  c[LX_OC_HITS] += static_cast<unsigned long long>(hits);
  c[LX_OC_GPU_MISSES] += static_cast<unsigned long long>(gpu_pairs);
  c[LX_OC_CPU_MISSES] += static_cast<unsigned long long>(cpu_pairs);
  c[LX_OC_FILLS] += static_cast<unsigned long long>(fills);
  c[LX_OC_PREFETCH_HITS] += static_cast<unsigned long long>(prefetch_hits);
  c[LX_OC_CALLS] += 1;
  return kOk;
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int64_t lx_offload_workspace_bytes(int T, int L, int I, int top_k) {
  LX_PROFILE("offload_workspace_bytes");
  if (T <= 0 || L <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t P = int64_t{T} * top_k;
  return (P * I * 2 + 15) / 16 * 16 + P * L * 4;
}

int lx_offload_experts(const void* latent, const float* weights, const lx_expert_ref* src, const lx_expert_ref* fill,
                       lx_wtype type, float* out, void* workspace, int T, int L, int I, int top_k, void* stream) {
  LX_PROFILE("offload_experts");
  (void)stream;
  (void)workspace;  // the intermediates live in scratch, as lx_moe_experts_w8's do
  if (T <= 0) return kOk;
  if (!latent || !weights || !src || !out) return fail(kInvalid);
  if (L % 16 || I % 16 || top_k < 1 || top_k > kMaxTopK) return fail(kInvalid);
  if (!supported_type(type)) return fail(kNotSupported);
  const int64_t P = int64_t{T} * top_k;
  if (int s = check_refs(type, src, P)) return s;
  if (fill) {
    if (int s = check_refs(type, fill, P)) return s;
  }
  const Kern& kn = kern();
  const uint16_t* lat = static_cast<const uint16_t*>(latent);

  // Each pair's activations, f32: the latent row of its token.
  float* acts = static_cast<float*>(scratch(sizeof(float) * P * L, kSlotActs));
  parallel_for(P, 8, [&](int64_t p0, int64_t p1) {
    for (int64_t p = p0; p < p1; ++p) {
      const uint16_t* row = lat + (p / top_k) * L;
      for (int64_t l = 0; l < L; ++l) acts[p * L + l] = bf16_to_f32(row[l]);
    }
  });

  // Gate and up: items are (pair, block of intermediate rows).
  float* inter = static_cast<float*>(scratch(sizeof(float) * P * I, kSlotInter));
  const int64_t rb1 = block_rows(I, P), blocks1 = (I + rb1 - 1) / rb1;
  parallel_for(P * blocks1, 1, [&](int64_t a0, int64_t a1) {
    float* tile = static_cast<float*>(scratch(sizeof(float) * 2 * kMaxBlock, kSlotTile));
    for (int64_t it = a0; it < a1; ++it) {
      const int64_t p = it / blocks1, i0 = (it % blocks1) * rb1, i1 = std::min<int64_t>(I, i0 + rb1);
      const lx_expert_ref& r = src[p];
      if (!r.gate_up) continue;
      if (fill && fill[p].gate_up) fill_gate_up(type, r, fill[p], L, I, i0, i1);
      swiglu_rows(kn, expert_of(type, r), acts + p * L, 1, L, I, i0, i1, tile, inter + p * I);
    }
  });

  // Down: items are (token, block of output rows). The token's pairs are summed in k order, so each output
  // is written once and its value does not depend on the thread that computed it.
  const int64_t rb2 = block_rows(L, T), blocks2 = (L + rb2 - 1) / rb2;
  parallel_for(int64_t{T} * blocks2, 1, [&](int64_t a0, int64_t a1) {
    float* tile = static_cast<float*>(scratch(sizeof(float) * kMaxBlock, kSlotTile));
    for (int64_t it = a0; it < a1; ++it) {
      const int64_t t = it / blocks2, l0 = (it % blocks2) * rb2, l1 = std::min<int64_t>(L, l0 + rb2);
      const int64_t nl = l1 - l0;
      float acc[kMaxBlock] = {};
      for (int64_t j = 0; j < top_k; ++j) {
        const int64_t p = t * top_k + j;
        const lx_expert_ref& r = src[p];
        if (!r.gate_up) continue;
        if (fill && fill[p].gate_up) fill_down(type, r, fill[p], I, l0, l1);
        down_rows(kn, expert_of(type, r), inter + p * I, 1, I, l0, l1, tile);
        for (int64_t l = 0; l < nl; ++l) acc[l] += weights[p] * tile[l];
      }
      for (int64_t l = 0; l < nl; ++l) out[t * L + l0 + l] = acc[l];
    }
  });
  return kOk;
}

int lx_offload_experts_grouped(const void* latent, const float* weights, const lx_expert_ref* groups,
                               const int32_t* offsets, const int32_t* pairs, int G, lx_wtype type, float* out,
                               void* workspace, int T, int L, int I, int top_k, int flags, void* stream) {
  LX_PROFILE("offload_experts_grouped");
  (void)stream;
  if (T <= 0) return kOk;
  if (L % 16 || I % 16 || top_k < 1 || top_k > kMaxTopK) return fail(kInvalid);
  if (!supported_type(type)) return fail(kNotSupported);
  if (!latent || !weights || !out || !workspace || G < 0 || (G > 0 && (!groups || !offsets || !pairs)) ||
      (flags & ~(LX_OG_BEGIN | LX_OG_END)))
    return fail(kInvalid);
  const int64_t P = int64_t{T} * top_k;
  // Each pair's partial sums follow the intermediates in the workspace (lx_offload_workspace_bytes).
  float* partial = reinterpret_cast<float*>(static_cast<char*>(workspace) + (P * I * 2 + 15) / 16 * 16);
  if (flags & LX_OG_BEGIN) std::memset(partial, 0, sizeof(float) * static_cast<size_t>(P * L));

  if (G > 0) {
    if (int s = check_refs(type, groups, G)) return s;
    if (offsets[0] < 0) return fail(kInvalid);
    for (int g = 0; g < G; ++g)
      if (offsets[g + 1] < offsets[g]) return fail(kInvalid);
    const int64_t first = offsets[0], N = offsets[G] - first;
    for (int64_t j = first; j < first + N; ++j)
      if (pairs[j] < 0 || pairs[j] >= P) return fail(kInvalid);
    const Kern& kn = kern();
    const uint16_t* lat = static_cast<const uint16_t*>(latent);

    // The positions' activations, f32: row q is the latent row of pair pairs[first + q].
    float* acts = static_cast<float*>(scratch(sizeof(float) * N * L, kSlotActs));
    parallel_for(N, 8, [&](int64_t q0, int64_t q1) {
      for (int64_t q = q0; q < q1; ++q) {
        const uint16_t* row = lat + (pairs[first + q] / top_k) * L;
        for (int64_t l = 0; l < L; ++l) acts[q * L + l] = bf16_to_f32(row[l]);
      }
    });

    // Gate and up: items are (group, block of intermediate rows). A group's pairs are one tile of M rows.
    float* inter = static_cast<float*>(scratch(sizeof(float) * N * I, kSlotInter));
    const int64_t rb1 = block_rows(I, G), blocks1 = (I + rb1 - 1) / rb1;
    parallel_for(int64_t{G} * blocks1, 1, [&](int64_t a0, int64_t a1) {
      for (int64_t it = a0; it < a1; ++it) {
        const int64_t g = it / blocks1, i0 = (it % blocks1) * rb1, i1 = std::min<int64_t>(I, i0 + rb1);
        const int64_t cnt = offsets[g + 1] - offsets[g];
        if (cnt == 0 || !groups[g].gate_up) continue;
        const int64_t q0 = offsets[g] - first;
        float* tile = static_cast<float*>(scratch(sizeof(float) * 2 * cnt * (i1 - i0), kSlotTile));
        swiglu_rows(kn, expert_of(type, groups[g]), acts + q0 * L, cnt, L, I, i0, i1, tile, inter + q0 * I);
      }
    });

    // Down: items are (group, block of output rows). Each pair's row goes to its partial sums.
    const int64_t rb2 = block_rows(L, G), blocks2 = (L + rb2 - 1) / rb2;
    parallel_for(int64_t{G} * blocks2, 1, [&](int64_t a0, int64_t a1) {
      for (int64_t it = a0; it < a1; ++it) {
        const int64_t g = it / blocks2, l0 = (it % blocks2) * rb2, l1 = std::min<int64_t>(L, l0 + rb2);
        const int64_t cnt = offsets[g + 1] - offsets[g], nl = l1 - l0;
        if (cnt == 0 || !groups[g].gate_up) continue;
        const int64_t q0 = offsets[g] - first;
        float* tile = static_cast<float*>(scratch(sizeof(float) * cnt * nl, kSlotTile));
        down_rows(kn, expert_of(type, groups[g]), inter + q0 * I, cnt, I, l0, l1, tile);
        for (int64_t m = 0; m < cnt; ++m) {
          const int64_t p = pairs[offsets[g] + m];
          const float w = weights[p];
          float* dst = partial + p * L + l0;
          for (int64_t l = 0; l < nl; ++l) dst[l] = w * tile[m * nl + l];
        }
      }
    });
  }

  if (flags & LX_OG_END) {
    // out[t, l] = sum over k of partial[t * top_k + k, l], in k order.
    parallel_for(T, 8, [&](int64_t t0, int64_t t1) {
      for (int64_t t = t0; t < t1; ++t) {
        float* o = out + t * L;
        const float* pt = partial + t * top_k * L;
        for (int64_t l = 0; l < L; ++l) o[l] = 0.0f;
        for (int64_t j = 0; j < top_k; ++j) {
          const float* pj = pt + j * L;
          for (int64_t l = 0; l < L; ++l) o[l] += pj[l];
        }
      }
    });
  }
  return kOk;
}

int lx_offload_copy(const lx_expert_ref* src, const lx_expert_ref* dst, const int32_t* count, int max_count,
                    lx_wtype type, int L, int I, void* stream) {
  LX_PROFILE("offload_copy");
  (void)stream;
  if (max_count <= 0) return kOk;
  if (!src || !dst || !count || L <= 0 || I <= 0) return fail(kInvalid);
  if (!supported_type(type)) return fail(kNotSupported);
  const int n = std::clamp(*count, 0, max_count);
  const int64_t gu = 2ll * I * L * elem_bytes(type), dn = static_cast<int64_t>(L) * I * elem_bytes(type);
  for (int j = 0; j < n; ++j) {
    const lx_expert_ref& s = src[j];
    const lx_expert_ref& d = dst[j];
    if (!s.gate_up || !s.down || !d.gate_up || !d.down) return fail(kInvalid);
    if (scaled_type(type) && (!s.gate_up_scale || !s.down_scale || !d.gate_up_scale || !d.down_scale))
      return fail(kInvalid);
  }
  for (int j = 0; j < n; ++j) {
    const lx_expert_ref& s = src[j];
    const lx_expert_ref& d = dst[j];
    copy_bytes(const_cast<void*>(d.gate_up), s.gate_up, gu);
    copy_bytes(const_cast<void*>(d.down), s.down, dn);
    if (scaled_type(type)) {
      copy_bytes(const_cast<void*>(d.gate_up_scale), s.gate_up_scale, 2ll * I * 4);
      copy_bytes(const_cast<void*>(d.down_scale), s.down_scale, static_cast<int64_t>(L) * 4);
    }
  }
  return kOk;
}

int lx_offload_plan(const int32_t* ids, const float* weights, const void* latent, int layer, lx_offload_state* state,
                    lx_expert_ref* src, lx_expert_ref* fill, int T, void* stream) {
  LX_PROFILE("offload_plan");
  (void)stream;
  if (T <= 0) return kOk;
  if (!ids || !weights || !latent || !state || !src || !fill) return fail(kInvalid);
  const lx_offload_state& st = *state;
  if (!state_ok(st) || layer < 0 || layer >= st.n_layers || T * st.top_k > LX_OFFLOAD_MAX_PAIRS) return fail(kInvalid);
  for (int p = 0; p < T * st.top_k; ++p)
    if (ids[p] < 0 || ids[p] >= st.n_experts) return fail(kInvalid);
  return plan_layer(ids, weights, latent, layer, st, src, fill, T);
}

int lx_offload_join(lx_offload_state* state, float* out, int T, int L, void* stream) {
  LX_PROFILE("offload_join");
  (void)stream;
  if (T <= 0) return kOk;
  if (!state || !out || L <= 0 || !state->job_pending || !state->job_seq || !state->error || !state->mailbox ||
      !state->cpu_result)
    return fail(kInvalid);
  if (*state->job_pending == 0) return kOk;
  // Wait for the worker's done flag. Spin briefly, then nap, and give up after 10 s as the CUDA join does: the
  // step's output is then wrong, and the error flag says so.
  const uint32_t want = *state->job_seq;
  const auto start = std::chrono::steady_clock::now();
  bool ready = false;
  for (uint32_t spins = 0;; ++spins) {
    if (__atomic_load_n(&state->mailbox->done, __ATOMIC_ACQUIRE) == want) {
      ready = true;
      break;
    }
    if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
      *state->error = 1;
      break;
    }
    if (spins < 4096) relax();
    else std::this_thread::sleep_for(std::chrono::microseconds(20));
  }
  *state->job_pending = 0;
  if (!ready) return kOk;
  const float* r = state->cpu_result;
  for (int64_t i = 0; i < int64_t{T} * L; ++i) out[i] += r[i];
  return kOk;
}

int lx_offload_prefetch(const int32_t* predicted, int n, int layer, lx_offload_state* state, lx_expert_ref* src,
                        lx_expert_ref* dst, int32_t* count, void* stream) {
  LX_PROFILE("offload_prefetch");
  (void)stream;
  if (n < 0 || (n > 0 && !predicted) || !state || !src || !dst || !count) return fail(kInvalid);
  const lx_offload_state& st = *state;
  if (!state_ok(st) || layer < 0 || layer >= st.n_layers) return fail(kInvalid);
  for (int j = 0; j < n; ++j)
    if (predicted[j] < 0 || predicted[j] >= st.n_experts) return fail(kInvalid);

  // One past the clock: the slots the last plan used belong to that plan's step, and the next plan of this
  // layer sees the prefetched slots as used now, so it keeps them.
  const int base = layer * st.n_experts;
  const uint32_t now = *st.clock + 1u;
  // Protect what is already resident among the predictions first.
  for (int j = 0; j < n; ++j) {
    const int s = st.vram_slot[base + predicted[j]];
    if (s >= 0 && st.slot_stamp[s] < now) st.slot_stamp[s] = now;
  }
  int c = 0;
  for (int j = 0; j < n; ++j) {
    const int e = predicted[j];
    if (st.vram_slot[base + e] >= 0) continue;  // resident, or claimed by an earlier prediction
    const int h = st.host_slot[base + e];
    if (h < 0) continue;
    const int v = pick_victim(st, layer, now);
    if (v < 0) break;
    claim_slot(st, layer, e, v, now, true);
    src[c] = pool_ref(st, st.host_pool, h);
    dst[c] = pool_ref(st, st.vram_pool, v);
    ++c;
  }
  *count = c;
  st.counters[int64_t{layer} * LX_OFFLOAD_COUNTERS + LX_OC_PREFETCHES] += static_cast<unsigned long long>(c);
  return kOk;
}

int lx_offload_read_probe(const void* src, int64_t bytes, unsigned long long* sink, int blocks, void* stream) {
  LX_PROFILE("offload_read_probe");
  (void)stream;
  (void)blocks;
  if (bytes <= 0) return kOk;
  if (!src || !sink) return fail(kInvalid);
  const char* s = static_cast<const char*>(src);
  const int64_t n = bytes / 16;
  parallel_for(n, kCopyGrain / 16, [&](int64_t b, int64_t e) {
    uint32_t acc = 0;
    for (int64_t i = b; i < e; ++i) {
      uint32_t w[4];
      std::memcpy(w, s + 16 * i, 16);
      acc ^= w[0] ^ w[1] ^ w[2] ^ w[3];
    }
    // Keeps the loads: the sink is only written when the fold lands on the magic value.
    if (acc == kProbeMagic) __atomic_fetch_add(sink, 1ull, __ATOMIC_RELAXED);
  });
  return kOk;
}

}  // extern "C"
