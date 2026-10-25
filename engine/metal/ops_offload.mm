// Offloading (kernels/offload.metal): experts read through per-pair refs, expert copies, and the
// expert cache's plan, join and prefetch.
//
// The structs these calls take hold pointers, and on Metal a pointer is a CPU address, so every
// dispatch carries an OxMap the kernels translate with: the registry's table, plus ranges wrapped
// for the dispatch when a struct lives in host memory the registry does not know (the kernel tests'
// tensors). Only then are the structs read here, after the stream drained, since that is the only
// time the host can know what they point at; registered structs must point at registered memory.
// The kernels touch memory no argument names, so each is ordered against everything around it.
//
// The CPU lane. A running Metal kernel does not see a CPU write in useful time (measured on the M5:
// a doorbell and its answer took 290 ms per round trip), so the GPU never spins on the mailbox as the
// CUDA join does. The plan ends its command buffer instead, which publishes the job to the CPU worker
// once the GPU gets there; the join ends the next one, and the command buffer after it waits on an
// event that a block on the host signals when the worker answered (or there was no job, or ten seconds
// passed). Both are recorded into graphs as well, so a captured decode step keeps the protocol.
#include <algorithm>
#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

static_assert(sizeof(OxRef) == sizeof(lx_expert_ref));
static_assert(sizeof(OxState) == sizeof(lx_offload_state));
static_assert(offsetof(OxState, vram_slot) == offsetof(lx_offload_state, vram_slot));
static_assert(offsetof(OxState, vram_pool) == offsetof(lx_offload_state, vram_pool));
static_assert(offsetof(OxState, gate_up_stride) == offsetof(lx_offload_state, gate_up_stride));
static_assert(offsetof(OxState, max_cpu_pairs) == offsetof(lx_offload_state, max_cpu_pairs));
static_assert(offsetof(OxState, mailbox) == offsetof(lx_offload_state, mailbox));
static_assert(offsetof(OxState, counters) == offsetof(lx_offload_state, counters));
// offload.metal addresses the mailbox by these offsets.
static_assert(offsetof(lx_offload_mailbox, done) == 64 && offsetof(lx_offload_mailbox, layer) == 128);
static_assert(offsetof(lx_offload_mailbox, token) == 140);
static_assert(offsetof(lx_offload_mailbox, expert) == 140 + 4 * LX_OFFLOAD_MAX_PAIRS);
static_assert(offsetof(lx_offload_mailbox, weight) == 140 + 8 * LX_OFFLOAD_MAX_PAIRS);
static_assert(LX_OFFLOAD_COUNTERS == 8 && LX_OC_PREFETCHES == 4 && LX_OC_CALLS == 6);

namespace {

constexpr int kThreads = 256;  // expert kernels: 8 simdgroups, a row each
constexpr int kSimds = kThreads / 32;
constexpr int kMaxTopK = 8;
constexpr int kCopyParts = 24;  // threadgroups per copied expert

std::string gate_up_name(int f) { return "ox_gate_up_" + std::to_string(f); }
std::string down_name(int f) { return "ox_down_" + std::to_string(f); }
std::string gate_up_grouped_name(int f) { return "ox_gate_up_grouped_" + std::to_string(f); }
std::string down_grouped_name(int f) { return "ox_down_grouped_" + std::to_string(f); }
KernelSet<3> g_gate_up(gate_up_name), g_down(down_name), g_gate_up_grouped(gate_up_grouped_name),
    g_down_grouped(down_grouped_name);

bool shape_ok(int L, int I, int top_k) { return L % 16 == 0 && I % 16 == 0 && top_k >= 1 && top_k <= kMaxTopK; }
bool format_ok(lx_wtype t) { return t == LX_W_BF16 || t == LX_W_INT8 || t == LX_W_FP8; }

// Bytes of one expert's parts in format t.
struct ExpertBytes {
  int64_t gu, gus, dn, dns;
};
ExpertBytes expert_bytes(int type, int64_t L, int64_t I) {
  const int64_t elem = type == LX_W_BF16 ? 2 : 1;
  const bool scaled = type == LX_W_INT8 || type == LX_W_FP8;
  return ExpertBytes{2 * I * L * elem, scaled ? 2 * I * 4 : 0, L * I * elem, scaled ? L * 4 : 0};
}

// Host memory the registry does not know: the structs in it are read here (see the top).
bool unregistered(const void* p) {
  uintptr_t base = 0;
  return p && !Device::get().lookup(p, 1, &base);
}

uint64_t as_u64(const void* p) { return reinterpret_cast<uint64_t>(p); }

// Builds a dispatch's OxMap: the registry's table, and the unregistered ranges the kernel may reach.
class Mapper {
 public:
  explicit Mapper(void* stream) : stream_(stream_of(stream)) {}

  // Drains the stream so the structs can be read, which a capture cannot do.
  int drain() {
    if (stream_->capturing()) return kNotSupported;
    return stream_->wait();
  }

  void add(uint64_t p, int64_t bytes) {
    if (!p || bytes <= 0 || !unregistered(reinterpret_cast<const void*>(p))) return;
    ranges_.emplace_back(p, p + static_cast<uint64_t>(bytes));
  }
  void add_ref(const lx_expert_ref& r, const ExpertBytes& b) {
    add(as_u64(r.gate_up), b.gu);
    add(as_u64(r.gate_up_scale), b.gus);
    add(as_u64(r.down), b.dn);
    add(as_u64(r.down_scale), b.dns);
  }
  void add_refs(const lx_expert_ref* refs, int n, const ExpertBytes& b) {
    for (int i = 0; refs && i < n; ++i)
      if (refs[i].gate_up) add_ref(refs[i], b);
  }
  // Everything a state reaches, for a step of T tokens (the state and its arrays are readable now).
  void add_state(const lx_offload_state* st, int T) {
    const int64_t n = static_cast<int64_t>(st->n_layers) * st->n_experts;
    add(as_u64(st->vram_slot), n * 4);
    add(as_u64(st->host_slot), n * 4);
    add(as_u64(st->cpu_first), n);
    add(as_u64(st->layer_first), (st->n_layers + 1) * 4ll);
    add(as_u64(st->layer_pinned), st->n_layers * 4ll);
    const int64_t slots = st->layer_first && st->n_layers > 0 ? st->layer_first[st->n_layers] : 0;
    add(as_u64(st->slot_expert), slots * 4);
    add(as_u64(st->slot_stamp), slots * 4);
    add(as_u64(st->slot_prefetched), slots);
    add(as_u64(st->clock), 4);
    int64_t host_slots = 0;
    for (int64_t i = 0; st->host_slot && i < n; ++i) host_slots = std::max<int64_t>(host_slots, st->host_slot[i] + 1);
    const ExpertBytes b = expert_bytes(st->type, st->L, st->I);
    add_pool(st->vram_pool, slots, st, b);
    add_pool(st->host_pool, host_slots, st, b);
    add(as_u64(st->mailbox), sizeof(lx_offload_mailbox));
    add(as_u64(st->cpu_latent), static_cast<int64_t>(T) * st->L * 2);
    add(as_u64(st->cpu_result), static_cast<int64_t>(T) * st->L * 4);
    add(as_u64(st->job_seq), 4);
    add(as_u64(st->job_pending), 4);
    add(as_u64(st->error), 4);
    add(as_u64(st->counters), st->n_layers * static_cast<int64_t>(LX_OFFLOAD_COUNTERS) * 8);
  }

  // The map, with the ranges wrapped for the next dispatch (call it right before that dispatch).
  int build(OxMap* m) {
    *m = OxMap{};
    int n = 0;
    id<MTLBuffer> table = Device::get().table(&n);
    if (!table) return kOutOfMemory;
    m->table = stream_->retain(table);
    m->table_n = n;
    // Ranges in one mapped region share a wrap.
    std::sort(ranges_.begin(), ranges_.end());
    std::vector<std::pair<uint64_t, uint64_t>> merged;
    for (const auto& r : ranges_) {
      if (!merged.empty()) {
        auto& last = merged.back();
        if (r.first <= last.second || region_end(last.first, r.second - last.first) >= r.second) {
          last.second = std::max(last.second, r.second);
          continue;
        }
      }
      merged.push_back(r);
    }
    if (merged.size() > LX_OX_EXTRAS) return kNotSupported;
    for (const auto& [lo, hi] : merged) {
      const uint64_t gpu = stream_->address(reinterpret_cast<const void*>(lo), hi - lo, true);
      if (!gpu) return stream_->error() ? stream_->error() : kInvalidValue;
      m->extra[m->extra_n++] = OxRange{lo, hi, gpu};
    }
    stream_->touch_all();
    return kOk;
  }

  Stream* stream() const { return stream_; }

 private:
  void add_pool(const lx_expert_ref& pool, int64_t slots, const lx_offload_state* st, const ExpertBytes& b) {
    if (slots <= 0) return;
    add(as_u64(pool.gate_up), (slots - 1) * st->gate_up_stride + b.gu);
    add(as_u64(pool.gate_up_scale), (slots - 1) * st->gate_up_scale_stride + b.gus);
    add(as_u64(pool.down), (slots - 1) * st->down_stride + b.dn);
    add(as_u64(pool.down_scale), (slots - 1) * st->down_scale_stride + b.dns);
  }

  Stream* stream_;
  std::vector<std::pair<uint64_t, uint64_t>> ranges_;
};

// Stages 1 and 2 of the grouped form for G groups into the workspace's intermediates and partials.
int launch_grouped(Mapper& map, const void* latent, const float* weights, const lx_expert_ref* groups,
                   const int32_t* offsets, const int32_t* pairs, int G, int type, void* workspace, int T, int L, int I,
                   int top_k, void* stream) {
  const int64_t P = static_cast<int64_t>(T) * top_k;
  auto* ws = static_cast<uint8_t*>(workspace);
  const int64_t partial_off = (P * I * 2 + 15) / 16 * 16;
  const int rows1 = (I + kSimds - 1) / kSimds, rows2 = (L + kSimds - 1) / kSimds;
  for (int stage = 0; stage < 2; ++stage) {
    OxExpertsArgs a{};
    a.latent = addr(stream, latent, static_cast<size_t>(T) * L * 2);
    a.weights = addr(stream, weights, static_cast<size_t>(P) * 4);
    a.refs = addr(stream, groups, static_cast<size_t>(G) * sizeof(lx_expert_ref));
    a.offsets = addr(stream, offsets, static_cast<size_t>(G + 1) * 4);
    a.pairs = addr(stream, pairs, kUnknownExtent);
    a.inter = addr_w(stream, ws, static_cast<size_t>(P) * I * 2);
    a.partial = addr_w(stream, ws + partial_off, static_cast<size_t>(P) * L * 4);
    a.T = T;
    a.L = L;
    a.I = I;
    a.top_k = top_k;
    a.n = G;
    a.rows = stage == 0 ? rows1 : rows2;
    if (int e = map.build(&a.map)) {
      map.stream()->drop_pending();
      return e;
    }
    KernelSet<3>& k = stage == 0 ? g_gate_up_grouped : g_down_grouped;
    if (int e = k.launch(type, stream, a, size1(static_cast<NSUInteger>(G) * a.rows), size1(kThreads))) return e;
  }
  return kOk;
}

}  // namespace

extern "C" {

int64_t lx_offload_workspace_bytes(int T, int L, int I, int top_k) {
  const int64_t P = static_cast<int64_t>(T) * top_k;
  return (P * I * 2 + 15) / 16 * 16 + P * L * 4;
}

int lx_offload_experts(const void* latent, const float* weights, const lx_expert_ref* src, const lx_expert_ref* fill,
                       lx_wtype type, float* out, void* workspace, int T, int L, int I, int top_k, void* stream) {
  if (T <= 0) return kOk;
  if (!shape_ok(L, I, top_k)) return kInvalidValue;
  if (!format_ok(type)) return kNotSupported;
  const int P = T * top_k;
  Mapper map(stream);
  if (unregistered(src) || unregistered(fill)) {
    if (int e = map.drain()) return e;
    const ExpertBytes b = expert_bytes(type, L, I);
    map.add_refs(src, P, b);
    map.add_refs(fill, P, b);
  }
  const int rows1 = (I + kSimds - 1) / kSimds, rows2 = (L + kSimds - 1) / kSimds;
  for (int stage = 0; stage < 2; ++stage) {
    OxExpertsArgs a{};
    a.latent = addr(stream, latent, static_cast<size_t>(T) * L * 2);
    a.weights = addr(stream, weights, static_cast<size_t>(P) * 4);
    a.refs = addr(stream, src, static_cast<size_t>(P) * sizeof(lx_expert_ref));
    a.fill = addr(stream, fill, fill ? static_cast<size_t>(P) * sizeof(lx_expert_ref) : 0);
    a.inter = addr_w(stream, workspace, static_cast<size_t>(P) * I * 2);
    a.out = addr_w(stream, out, static_cast<size_t>(T) * L * 4);
    a.T = T;
    a.L = L;
    a.I = I;
    a.top_k = top_k;
    a.n = P;
    a.rows = stage == 0 ? rows1 : rows2;
    if (int e = map.build(&a.map)) {
      map.stream()->drop_pending();
      return e;
    }
    const NSUInteger groups = static_cast<NSUInteger>(stage == 0 ? P : T) * a.rows;
    KernelSet<3>& k = stage == 0 ? g_gate_up : g_down;
    if (int e = k.launch(type, stream, a, size1(groups), size1(kThreads))) return e;
  }
  return kOk;
}

int lx_offload_experts_grouped(const void* latent, const float* weights, const lx_expert_ref* groups,
                               const int32_t* offsets, const int32_t* pairs, int G, lx_wtype type, float* out,
                               void* workspace, int T, int L, int I, int top_k, int flags, void* stream) {
  static Kernel sum_pairs("ox_sum_pairs");
  if (T <= 0) return kOk;
  if (!shape_ok(L, I, top_k)) return kInvalidValue;
  if (!format_ok(type)) return kNotSupported;
  const int64_t P = static_cast<int64_t>(T) * top_k;
  auto* partial = static_cast<uint8_t*>(workspace) + (P * I * 2 + 15) / 16 * 16;
  if (flags & LX_OG_BEGIN) {
    if (int e = lx_rt_memset(partial, 0, P * L * 4, stream)) return e;
  }
  if (G > 0) {
    Mapper map(stream);
    if (unregistered(groups)) {
      if (int e = map.drain()) return e;
      map.add_refs(groups, G, expert_bytes(type, L, I));
    }
    if (int e = launch_grouped(map, latent, weights, groups, offsets, pairs, G, type, workspace, T, L, I, top_k, stream))
      return e;
  }
  if (flags & LX_OG_END) {
    OxExpertsArgs a{};
    a.partial = addr(stream, partial, static_cast<size_t>(P) * L * 4);
    a.out = addr_w(stream, out, static_cast<size_t>(T) * L * 4);
    a.T = T;
    a.L = L;
    a.top_k = top_k;
    return sum_pairs.launch(stream, a, size1(ceil_div(static_cast<int64_t>(T) * L, 256)), size1(256));
  }
  return kOk;
}

int lx_offload_copy(const lx_expert_ref* src, const lx_expert_ref* dst, const int32_t* count, int max_count,
                    lx_wtype type, int L, int I, void* stream) {
  static Kernel copy("ox_copy");
  if (max_count <= 0) return kOk;
  if (!format_ok(type)) return kNotSupported;
  const ExpertBytes b = expert_bytes(type, L, I);
  if (b.gu % 16 || b.dn % 16) return kInvalidValue;
  Mapper map(stream);
  if (unregistered(src) || unregistered(dst) || unregistered(count)) {
    if (int e = map.drain()) return e;
    const int n = std::clamp(*count, 0, max_count);
    map.add_refs(src, n, b);
    map.add_refs(dst, n, b);
  }
  OxCopyArgs a{};
  a.src = addr(stream, src, static_cast<size_t>(max_count) * sizeof(lx_expert_ref));
  a.dst = addr(stream, dst, static_cast<size_t>(max_count) * sizeof(lx_expert_ref));
  a.count = addr(stream, count, 4);
  a.gu_bytes = b.gu;
  a.gus_bytes = b.gus;
  a.dn_bytes = b.dn;
  a.dns_bytes = b.dns;
  if (int e = map.build(&a.map)) {
    map.stream()->drop_pending();
    return e;
  }
  return copy.launch(stream, a, size2(kCopyParts, static_cast<NSUInteger>(max_count)), size1(kThreads));
}

int lx_offload_plan(const int32_t* ids, const float* weights, const void* latent, int layer, lx_offload_state* state,
                    lx_expert_ref* src, lx_expert_ref* fill, int T, void* stream) {
  static Kernel plan("ox_plan");
  if (T <= 0) return kOk;
  if (!state || !src || !fill) return kInvalidValue;
  Mapper map(stream);
  if (unregistered(state)) {
    if (int e = map.drain()) return e;
    map.add_state(state, T);
  }
  OxCacheArgs a{};
  a.ids = addr(stream, ids, kUnknownExtent);  // T * top_k, and top_k is in the state
  a.weights = addr(stream, weights, kUnknownExtent);
  a.latent = addr(stream, latent, kUnknownExtent);
  a.state = addr(stream, state, sizeof(lx_offload_state));
  a.src = addr_w(stream, src, kUnknownExtent);
  a.fill = addr_w(stream, fill, kUnknownExtent);
  a.layer = layer;
  a.T = T;
  if (int e = map.build(&a.map)) {
    map.stream()->drop_pending();
    return e;
  }
  if (int e = plan.launch(stream, a, size1(1), size1(32))) return e;
  // The job reaches the CPU worker when this command buffer finishes: end it here, so the worker
  // starts while the GPU goes on with the experts.
  stream_of(stream)->cut();
  return stream_of(stream)->error();
}

int lx_offload_join(lx_offload_state* state, float* out, int T, int L, void* stream) {
  static Kernel join("ox_join");
  if (T <= 0) return kOk;
  if (!state) return kInvalidValue;
  Mapper map(stream);
  if (unregistered(state)) {
    if (int e = map.drain()) return e;
    map.add_state(state, T);
  }
  stream_of(stream)->mailbox_wait(state);
  OxCacheArgs a{};
  a.state = addr(stream, state, sizeof(lx_offload_state));
  a.out = addr_w(stream, out, static_cast<size_t>(T) * L * 4);
  a.T = T;
  a.L = L;
  if (int e = map.build(&a.map)) {
    map.stream()->drop_pending();
    return e;
  }
  return join.launch(stream, a, size1(1), size1(1024));
}

int lx_offload_prefetch(const int32_t* predicted, int n, int layer, lx_offload_state* state, lx_expert_ref* src,
                        lx_expert_ref* dst, int32_t* count, void* stream) {
  static Kernel prefetch("ox_prefetch");
  if (!state || !count || (n > 0 && (!predicted || !src || !dst))) return kInvalidValue;
  n = std::max(n, 0);
  Mapper map(stream);
  if (unregistered(state)) {
    if (int e = map.drain()) return e;
    map.add_state(state, 0);
  }
  OxCacheArgs a{};
  a.ids = addr(stream, predicted, static_cast<size_t>(n) * 4);
  a.state = addr(stream, state, sizeof(lx_offload_state));
  a.src = addr_w(stream, src, static_cast<size_t>(n) * sizeof(lx_expert_ref));
  a.fill = addr_w(stream, dst, static_cast<size_t>(n) * sizeof(lx_expert_ref));
  a.count = addr_w(stream, count, 4);
  a.layer = layer;
  a.n = n;
  if (int e = map.build(&a.map)) {
    map.stream()->drop_pending();
    return e;
  }
  return prefetch.launch(stream, a, size1(1), size1(32));
}

int lx_offload_read_probe(const void* src, int64_t bytes, unsigned long long* sink, int blocks, void* stream) {
  static Kernel probe("ox_read_probe");
  if (bytes < 16 || blocks <= 0) return bytes < 16 ? kOk : kInvalidValue;
  OxProbeArgs a{};
  a.src = addr(stream, src, static_cast<size_t>(bytes / 16 * 16));
  a.sink = addr_w(stream, sink, 8);
  a.n = bytes / 16;
  return probe.launch(stream, a, size1(static_cast<NSUInteger>(blocks)), size1(kThreads));
}

}  // extern "C"
