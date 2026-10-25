// Offloading (kernels.h "Offload"): the expert kernels that read each pair's expert through a ref,
// the copies, and the expert cache (plan, join, prefetch), as engine/cuda/offload.cu has them.
//
// Unified memory changes where the bytes are, not what the kernels do: a "host" expert and a "VRAM"
// slot are both shared memory here, and the ABI's pointers are CPU addresses. So every pointer read
// out of a ref or out of the state goes through ox_ptr (args.h, OxMap) to its GPU address first, and
// the refs a plan writes stay CPU addresses.
//
// Expert math follows the CUDA kernels: stage 1 gives each pair's bf16 SwiGLU intermediate, one
// simdgroup per intermediate row; stage 2 one simdgroup per output row of a token, summing its pairs
// in k order. A pair with a fill destination writes the bytes it reads there. The CPU lane's mailbox
// is posted by the plan and joined by lx_offload_join; on Metal the host splits the command buffer
// around the CPU's part (ops_offload.mm), since a running kernel cannot see a CPU write in time.

constant constexpr int kOxThreads = 256;
constant constexpr int kOxSimds = kOxThreads / 32;
constant constexpr int kOxMaxPairs = 64;  // LX_OFFLOAD_MAX_PAIRS
constant constexpr int kOxCounters = 8;   // LX_OFFLOAD_COUNTERS: hits, gpu, cpu, fills, prefetches, prefetch hits, calls

// The GPU address of CPU address `cpu`, or null when it is null or no range holds it.
inline device uchar* ox_ptr(constant OxMap& m, ulong cpu) {
  if (cpu == 0) return nullptr;
  for (int i = 0; i < m.extra_n; ++i)
    if (cpu >= m.extra[i].lo && cpu < m.extra[i].hi) return m.extra[i].gpu + (cpu - m.extra[i].lo);
  int lo = 0, hi = m.table_n;  // the first entry starting above cpu
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (m.table[mid].lo <= cpu) lo = mid + 1;
    else hi = mid;
  }
  // Registrations may overlap (two tensors on one page): of the few that start at or below cpu, the
  // one reaching furthest, so a read from cpu on stays inside it.
  int best = -1;
  for (int i = lo - 1, steps = 0; i >= 0 && steps < 8; --i, ++steps)
    if (cpu < m.table[i].hi && (best < 0 || m.table[i].hi > m.table[best].hi)) best = i;
  return best < 0 ? nullptr : m.table[best].gpu + (cpu - m.table[best].lo);
}

template <int F> constexpr int ox_elem() { return F == W_BF16 ? 2 : 1; }
template <int F> constexpr bool ox_scaled() { return F == W_INT8 || F == W_FP8; }

// acc + one 16-byte chunk of weights . its activations (chunk idx of a bf16 row x).
template <int F>
inline float ox_dot(uint4 w, const device uchar* x, int idx, float acc) {
  constexpr int V = chunk_values<F>();
  float xv[V];
  load_x<V, true, true>(x, idx * V, xv);
  return dot_chunk<F>(w, 1.0f, 1.0f, xv, acc);
}

// ---- decode form --------------------------------------------------------------------------------

// Stage 1. Threadgroup b covers rows 8b'..8b'+7 of pair p = b / rows; simdgroup s takes row i.
template <int F>
kernel void ox_gate_up(constant OxExpertsArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int p = int(tg) / a.rows;
  const OxRef s = a.refs[p];
  if (!s.gate_up) return;
  const int I = a.I, L = a.L;
  const int i = (int(tg) % a.rows) * kOxSimds + int(sg);
  if (i >= I) return;  // no barriers in this kernel
  device ushort* inter = a.inter + long(p) * I + i;
  const device uchar* gw = ox_ptr(a.map, s.gate_up);
  const device uchar* gs = ox_ptr(a.map, s.gate_up_scale);
  if (!gw || (ox_scaled<F>() && !gs)) {  // a pointer outside every range: loud, not silent
    if (lane == 0) *inter = 0x7fc0;
    return;
  }
  const long rb = long(L) * ox_elem<F>();
  const int nch = int(rb / 16);
  const device uchar* g = gw + i * rb;
  const device uchar* u = g + long(I) * rb;
  OxRef f = {0, 0, 0, 0};
  if (a.fill) f = a.fill[p];
  device uchar* fg = f.gate_up ? ox_ptr(a.map, f.gate_up) + i * rb : nullptr;
  const device uchar* x = (const device uchar*)(a.latent + long(p / a.top_k) * L);
  float ga = 0.0f, ua = 0.0f;
  int idx = int(lane);
  for (; idx + 32 < nch; idx += 64) {  // two chunks of each row in flight per lane
    const uint4 g0 = *(const device uint4*)(g + 16 * idx), g1 = *(const device uint4*)(g + 16 * (idx + 32));
    const uint4 u0 = *(const device uint4*)(u + 16 * idx), u1 = *(const device uint4*)(u + 16 * (idx + 32));
    if (fg) {
      *(device uint4*)(fg + 16 * idx) = g0;
      *(device uint4*)(fg + 16 * (idx + 32)) = g1;
      *(device uint4*)(fg + long(I) * rb + 16 * idx) = u0;
      *(device uint4*)(fg + long(I) * rb + 16 * (idx + 32)) = u1;
    }
    ga = ox_dot<F>(g1, x, idx + 32, ox_dot<F>(g0, x, idx, ga));
    ua = ox_dot<F>(u1, x, idx + 32, ox_dot<F>(u0, x, idx, ua));
  }
  for (; idx < nch; idx += 32) {
    const uint4 g0 = *(const device uint4*)(g + 16 * idx), u0 = *(const device uint4*)(u + 16 * idx);
    if (fg) {
      *(device uint4*)(fg + 16 * idx) = g0;
      *(device uint4*)(fg + long(I) * rb + 16 * idx) = u0;
    }
    ga = ox_dot<F>(g0, x, idx, ga);
    ua = ox_dot<F>(u0, x, idx, ua);
  }
  ga = simd_sum(ga);
  ua = simd_sum(ua);
  if (lane == 0) {
    float sg_ = 1.0f, su = 1.0f;
    if (ox_scaled<F>()) {
      sg_ = ((const device float*)gs)[i];
      su = ((const device float*)gs)[I + i];
      if (fg) {
        device float* fs = (device float*)ox_ptr(a.map, f.gate_up_scale);
        fs[i] = sg_;
        fs[I + i] = su;
      }
    }
    *inter = to_bf(silu(sg_ * ga) * (su * ua));
  }
}

// Stage 2. Threadgroup b covers output rows of token t = b / rows; simdgroup s takes row l.
template <int F>
kernel void ox_down(constant OxExpertsArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                    uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int t = int(tg) / a.rows;
  const int l = (int(tg) % a.rows) * kOxSimds + int(sg);
  const int I = a.I, L = a.L;
  if (l >= L) return;
  const long rb = long(I) * ox_elem<F>();
  const int nch = int(rb / 16);
  float acc = 0.0f;
  for (int k = 0; k < a.top_k; ++k) {
    const int p = t * a.top_k + k;
    const OxRef s = a.refs[p];
    if (!s.gate_up) continue;
    const device uchar* dw = ox_ptr(a.map, s.down);
    const device uchar* ds = ox_ptr(a.map, s.down_scale);
    if (!dw || (ox_scaled<F>() && !ds)) {
      acc = NAN;
      continue;
    }
    const device uchar* row = dw + l * rb;
    OxRef f = {0, 0, 0, 0};
    if (a.fill) f = a.fill[p];
    device uchar* frow = f.gate_up ? ox_ptr(a.map, f.down) + l * rb : nullptr;
    const device uchar* x = (const device uchar*)(a.inter + long(p) * I);
    float d = 0.0f;
    int idx = int(lane);
    for (; idx + 32 < nch; idx += 64) {
      const uint4 w0 = *(const device uint4*)(row + 16 * idx), w1 = *(const device uint4*)(row + 16 * (idx + 32));
      if (frow) {
        *(device uint4*)(frow + 16 * idx) = w0;
        *(device uint4*)(frow + 16 * (idx + 32)) = w1;
      }
      d = ox_dot<F>(w1, x, idx + 32, ox_dot<F>(w0, x, idx, d));
    }
    for (; idx < nch; idx += 32) {
      const uint4 w0 = *(const device uint4*)(row + 16 * idx);
      if (frow) *(device uint4*)(frow + 16 * idx) = w0;
      d = ox_dot<F>(w0, x, idx, d);
    }
    d = simd_sum(d);
    float sc = 1.0f;
    if (ox_scaled<F>()) {
      sc = ((const device float*)ds)[l];
      if (frow && lane == 0) ((device float*)ox_ptr(a.map, f.down_scale))[l] = sc;
    }
    acc = fma(a.weights[p], sc * d, acc);
  }
  if (lane == 0) a.out[long(t) * L + l] = acc;
}

// ---- grouped form (prefill) ---------------------------------------------------------------------

// Stage 1: threadgroup b covers rows of group g = b / rows; the simdgroup applies gate and up row i
// to every pair of the group.
template <int F>
kernel void ox_gate_up_grouped(constant OxExpertsArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                               uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int g = int(tg) / a.rows;
  const int begin = a.offsets[g], end = a.offsets[g + 1];
  if (begin == end) return;
  const int I = a.I, L = a.L;
  const int i = (int(tg) % a.rows) * kOxSimds + int(sg);
  if (i >= I) return;
  const OxRef s = a.refs[g];
  const device uchar* gw = ox_ptr(a.map, s.gate_up);
  const device uchar* gs = ox_ptr(a.map, s.gate_up_scale);
  const bool bad = !gw || (ox_scaled<F>() && !gs);
  const long rb = long(L) * ox_elem<F>();
  const int nch = int(rb / 16);
  const device uchar* gr = gw + i * rb;
  const device uchar* ur = gr + long(I) * rb;
  float sg_ = 1.0f, su = 1.0f;
  if (ox_scaled<F>() && !bad) {
    sg_ = ((const device float*)gs)[i];
    su = ((const device float*)gs)[I + i];
  }
  for (int j = begin; j < end; ++j) {
    const int p = a.pairs[j];
    if (bad) {
      if (lane == 0) a.inter[long(p) * I + i] = 0x7fc0;
      continue;
    }
    const device uchar* x = (const device uchar*)(a.latent + long(p / a.top_k) * L);
    float ga = 0.0f, ua = 0.0f;
    for (int idx = int(lane); idx < nch; idx += 32) {
      const uint4 g0 = *(const device uint4*)(gr + 16 * idx), u0 = *(const device uint4*)(ur + 16 * idx);
      ga = ox_dot<F>(g0, x, idx, ga);
      ua = ox_dot<F>(u0, x, idx, ua);
    }
    ga = simd_sum(ga);
    ua = simd_sum(ua);
    if (lane == 0) a.inter[long(p) * I + i] = to_bf(silu(sg_ * ga) * (su * ua));
  }
}

// Stage 2: the simdgroup takes down row l of group g and writes weight * (row . inter[p]) for every
// pair of the group into partial [P, L].
template <int F>
kernel void ox_down_grouped(constant OxExpertsArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                            uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int g = int(tg) / a.rows;
  const int begin = a.offsets[g], end = a.offsets[g + 1];
  if (begin == end) return;
  const int I = a.I, L = a.L;
  const int l = (int(tg) % a.rows) * kOxSimds + int(sg);
  if (l >= L) return;
  const OxRef s = a.refs[g];
  const device uchar* dw = ox_ptr(a.map, s.down);
  const device uchar* ds = ox_ptr(a.map, s.down_scale);
  const bool bad = !dw || (ox_scaled<F>() && !ds);
  const long rb = long(I) * ox_elem<F>();
  const int nch = int(rb / 16);
  const device uchar* row = dw + l * rb;
  const float sc = ox_scaled<F>() && !bad ? ((const device float*)ds)[l] : 1.0f;
  for (int j = begin; j < end; ++j) {
    const int p = a.pairs[j];
    const device uchar* x = (const device uchar*)(a.inter + long(p) * I);
    float d = 0.0f;
    if (!bad)
      for (int idx = int(lane); idx < nch; idx += 32) d = ox_dot<F>(*(const device uint4*)(row + 16 * idx), x, idx, d);
    d = simd_sum(d);
    if (lane == 0) a.partial[long(p) * L + l] = bad ? NAN : a.weights[p] * (sc * d);
  }
}

// Stage 3: out[t, l] = sum over k of partial[t * top_k + k, l], in k order.
kernel void ox_sum_pairs(constant OxExpertsArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  const long n = long(a.T) * a.L;
  if (long(i) >= n) return;
  const long t = long(i) / a.L, l = long(i) % a.L;
  float acc = 0.0f;
  for (int k = 0; k < a.top_k; ++k) acc += a.partial[(t * a.top_k + k) * a.L + l];
  a.out[i] = acc;
}

#define LX_OX_EXPERTS(F)                                                                              \
  template [[host_name("ox_gate_up_" #F)]] kernel void ox_gate_up<F>(constant OxExpertsArgs&, uint, uint, uint); \
  template [[host_name("ox_down_" #F)]] kernel void ox_down<F>(constant OxExpertsArgs&, uint, uint, uint);       \
  template [[host_name("ox_gate_up_grouped_" #F)]] kernel void ox_gate_up_grouped<F>(constant OxExpertsArgs&,     \
                                                                                     uint, uint, uint);           \
  template [[host_name("ox_down_grouped_" #F)]] kernel void ox_down_grouped<F>(constant OxExpertsArgs&, uint,     \
                                                                               uint, uint);
LX_OX_EXPERTS(0)
LX_OX_EXPERTS(1)
LX_OX_EXPERTS(2)

// ---- copies -------------------------------------------------------------------------------------

inline void ox_copy_bytes(const device uchar* src, device uchar* dst, long bytes, int part, int parts, uint tid) {
  const device uint4* s = (const device uint4*)src;
  device uint4* d = (device uint4*)dst;
  const long n = bytes / 16, stride = long(parts) * kOxThreads;
  long i = long(part) * kOxThreads + tid;
  for (; i + 3 * stride < n; i += 4 * stride) {  // four loads in flight per thread before the stores
    const uint4 x0 = s[i], x1 = s[i + stride], x2 = s[i + 2 * stride], x3 = s[i + 3 * stride];
    d[i] = x0;
    d[i + stride] = x1;
    d[i + 2 * stride] = x2;
    d[i + 3 * stride] = x3;
  }
  for (; i < n; i += stride) d[i] = s[i];
}

// Grid (parts, max_count): expert j = tg.y copies src[j] -> dst[j] when j < *count.
kernel void ox_copy(constant OxCopyArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                    uint2 groups [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]]) {
  const int j = int(tg.y);
  if (j >= *a.count) return;
  const OxRef s = a.src[j], d = a.dst[j];
  const device uchar* sg = ox_ptr(a.map, s.gate_up);
  const device uchar* sd = ox_ptr(a.map, s.down);
  device uchar* dg = ox_ptr(a.map, d.gate_up);
  device uchar* dd = ox_ptr(a.map, d.down);
  if (!sg || !sd || !dg || !dd) return;
  ox_copy_bytes(sg, dg, a.gu_bytes, int(tg.x), int(groups.x), tid);
  ox_copy_bytes(sd, dd, a.dn_bytes, int(tg.x), int(groups.x), tid);
  if (a.gus_bytes && tg.x == 0) {
    const device float* sgs = (const device float*)ox_ptr(a.map, s.gate_up_scale);
    const device float* sds = (const device float*)ox_ptr(a.map, s.down_scale);
    device float* dgs = (device float*)ox_ptr(a.map, d.gate_up_scale);
    device float* dds = (device float*)ox_ptr(a.map, d.down_scale);
    if (!sgs || !sds || !dgs || !dds) return;
    for (long i = tid; i < a.gus_bytes / 4; i += kOxThreads) dgs[i] = sgs[i];
    for (long i = tid; i < a.dns_bytes / 4; i += kOxThreads) dds[i] = sds[i];
  }
}

// ---- the cache ----------------------------------------------------------------------------------

// The state's arrays, at their GPU addresses.
struct OxCache {
  device int* vram_slot;
  const device int* host_slot;
  const device uchar* cpu_first;
  const device int* layer_first;
  const device int* layer_pinned;
  device int* slot_expert;
  device uint* slot_stamp;
  device uchar* slot_prefetched;
  device atomic_uint* clock;
  device ulong* counters;
};

inline OxCache ox_cache(constant OxMap& m, const device OxState& st) {
  OxCache c;
  c.vram_slot = (device int*)ox_ptr(m, st.vram_slot);
  c.host_slot = (const device int*)ox_ptr(m, st.host_slot);
  c.cpu_first = ox_ptr(m, st.cpu_first);
  c.layer_first = (const device int*)ox_ptr(m, st.layer_first);
  c.layer_pinned = (const device int*)ox_ptr(m, st.layer_pinned);
  c.slot_expert = (device int*)ox_ptr(m, st.slot_expert);
  c.slot_stamp = (device uint*)ox_ptr(m, st.slot_stamp);
  c.slot_prefetched = ox_ptr(m, st.slot_prefetched);
  c.clock = (device atomic_uint*)ox_ptr(m, st.clock);
  c.counters = (device ulong*)ox_ptr(m, st.counters);
  return c;
}

inline bool ox_cache_ok(thread const OxCache& c) {
  return c.vram_slot && c.host_slot && c.cpu_first && c.layer_first && c.layer_pinned && c.slot_expert &&
         c.slot_stamp && c.slot_prefetched && c.clock && c.counters;
}

// Slot `slot` of a pool, as CPU addresses (the refs a plan writes are the ABI's).
inline OxRef ox_pool_ref(const device OxState& st, OxRef pool, long slot) {
  OxRef r;
  r.gate_up = pool.gate_up + ulong(slot * st.gate_up_stride);
  r.down = pool.down + ulong(slot * st.down_stride);
  r.gate_up_scale = pool.gate_up_scale ? pool.gate_up_scale + ulong(slot * st.gate_up_scale_stride) : 0;
  r.down_scale = pool.down_scale ? pool.down_scale + ulong(slot * st.down_scale_stride) : 0;
  return r;
}

// The least recently used dynamic slot of a layer not yet used at clock `now`, chosen by the whole
// simdgroup; -1 when every dynamic slot was used at `now`. Ties go to the lower slot.
inline int ox_pick_victim(thread const OxCache& c, int layer, uint now, uint lane) {
  const int first = c.layer_first[layer] + c.layer_pinned[layer], last = c.layer_first[layer + 1];
  uint best = 0xffffffffu;
  int best_s = -1;
  for (int s = first + int(lane); s < last; s += 32) {
    const uint v = c.slot_stamp[s];
    if (v < now && (v < best || (v == best && s < best_s))) {
      best = v;
      best_s = s;
    }
  }
  for (ushort o = 16; o > 0; o >>= 1) {
    const uint ov = simd_shuffle_xor(best, o);
    const int os = simd_shuffle_xor(best_s, o);
    if (os >= 0 && (best_s < 0 || ov < best || (ov == best && os < best_s))) {
      best = ov;
      best_s = os;
    }
  }
  return best_s;
}

// Moves `expert` of `layer` into slot v (one lane), evicting the slot's previous expert.
inline void ox_claim_slot(thread const OxCache& c, int n_experts, int layer, int expert, int v, uint now,
                          bool prefetched) {
  const int base = layer * n_experts;
  const int old = c.slot_expert[v];
  if (old >= 0) c.vram_slot[base + old] = -1;
  c.vram_slot[base + expert] = v;
  c.slot_expert[v] = expert;
  c.slot_stamp[v] = now;
  c.slot_prefetched[v] = prefetched ? 1 : 0;
}

// One simdgroup. Where every pair of one MoE layer's decode step is computed (kernels.h).
kernel void ox_plan(constant OxCacheArgs& a [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  threadgroup int miss_expert[kOxMaxPairs], miss_lane[kOxMaxPairs], miss_slot[kOxMaxPairs];
  threadgroup int pair_miss[kOxMaxPairs];  // index into the miss list, -1 for a hit
  threadgroup int shared_n[1];
  const device OxState& st = *a.state;
  const OxCache c = ox_cache(a.map, st);
  const int K = st.top_k, P = a.T * K, E = st.n_experts, layer = a.layer, base = layer * E;
  device int* error = (device int*)ox_ptr(a.map, st.error);
  if (!ox_cache_ok(c) || P > kOxMaxPairs || layer < 0 || layer >= st.n_layers) {
    if (lane == 0 && error) *error = 2;  // a state the kernels cannot use, or too many pairs
    return;
  }
  uint now = 0;
  if (lane == 0) now = atomic_fetch_add_explicit(c.clock, 1u, memory_order_relaxed) + 1u;
  now = simd_broadcast(now, 0);

  // Hits first, so their slots carry `now` before any victim is chosen.
  int hits = 0, prefetch_hits = 0;
  for (int p = int(lane); p < P; p += 32) {
    const int e = a.ids[p];
    const int s = c.vram_slot[base + e];
    pair_miss[p] = s >= 0 ? -1 : 0;
    if (s >= 0) {
      c.slot_stamp[s] = now;
      if (c.slot_prefetched[s]) {
        c.slot_prefetched[s] = 0;
        ++prefetch_hits;
      }
      a.src[p] = ox_pool_ref(st, st.vram_pool, s);
      a.fill[p] = OxRef{0, 0, 0, 0};
      ++hits;
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  // The distinct missed experts, in pair order (lane 0; at most 64 pairs).
  if (lane == 0) {
    int n_miss = 0;
    for (int p = 0; p < P; ++p) {
      if (pair_miss[p] < 0) continue;
      const int e = a.ids[p];
      int m = 0;
      while (m < n_miss && miss_expert[m] != e) ++m;
      if (m == n_miss) {
        miss_expert[m] = e;
        miss_slot[m] = -1;
        ++n_miss;
      }
      pair_miss[p] = m;
    }
    // Lanes: experts only on disk go to the CPU; then the CPU takes its share of the rest, cold
    // experts first (a hot one read by the GPU lands in a slot and is a hit next time).
    int cpu = 0;
    for (int m = 0; m < n_miss; ++m) {
      const bool disk = c.host_slot[base + miss_expert[m]] < 0;
      miss_lane[m] = disk ? 1 : 0;
      cpu += disk;
    }
    int target = int(st.cpu_fraction * n_miss + 0.5f);
    target = min(target, st.max_cpu_pairs);
    for (int pass = 0; pass < 2 && cpu < target; ++pass) {
      for (int m = n_miss - 1; m >= 0 && cpu < target; --m) {
        if (miss_lane[m]) continue;
        if (pass == 0 && !c.cpu_first[base + miss_expert[m]]) continue;
        miss_lane[m] = 1;
        ++cpu;
      }
    }
    shared_n[0] = n_miss;
  }
  simdgroup_barrier(mem_flags::mem_threadgroup);
  const int n_miss = shared_n[0];

  // Victims for the GPU's misses, one simdgroup-wide choice each.
  int fills = 0;
  if (st.fill) {
    for (int m = 0; m < n_miss; ++m) {
      if (miss_lane[m]) continue;
      const int v = ox_pick_victim(c, layer, now, lane);
      if (v < 0) break;
      if (lane == 0) {
        ox_claim_slot(c, E, layer, miss_expert[m], v, now, false);
        miss_slot[m] = v;
      }
      ++fills;
      simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }
  }
  simdgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);

  // Refs for the misses: the first pair of a filled expert writes its slot; any other pair of the
  // same expert reads host memory without writing.
  int gpu_pairs = 0, cpu_pairs = 0;
  for (int p = int(lane); p < P; p += 32) {
    const int m = pair_miss[p];
    if (m < 0) continue;
    if (miss_lane[m]) {
      a.src[p] = OxRef{0, 0, 0, 0};
      a.fill[p] = OxRef{0, 0, 0, 0};
      ++cpu_pairs;
      continue;
    }
    const int e = miss_expert[m];
    a.src[p] = ox_pool_ref(st, st.host_pool, c.host_slot[base + e]);
    bool lead = true;
    for (int q = 0; q < p; ++q) lead &= pair_miss[q] != m;
    a.fill[p] = (lead && miss_slot[m] >= 0) ? ox_pool_ref(st, st.vram_pool, miss_slot[m]) : OxRef{0, 0, 0, 0};
    ++gpu_pairs;
  }
  hits = simd_sum(hits);
  prefetch_hits = simd_sum(prefetch_hits);
  gpu_pairs = simd_sum(gpu_pairs);
  cpu_pairs = simd_sum(cpu_pairs);

  // Post the CPU's job: its pairs, the latent rows, then the doorbell. The host ends the command
  // buffer after this kernel, which is what makes all of it visible to the CPU.
  device int* job_pending = (device int*)ox_ptr(a.map, st.job_pending);
  device uint* job_seq = (device uint*)ox_ptr(a.map, st.job_seq);
  if (cpu_pairs > 0) {
    device uchar* mb = ox_ptr(a.map, st.mailbox);
    device uint4* ldst = (device uint4*)ox_ptr(a.map, st.cpu_latent);
    if (lane == 0) {
      device int* token = (device int*)(mb + 140);
      device int* expert = (device int*)(mb + 140 + 4 * kOxMaxPairs);
      device float* weight = (device float*)(mb + 140 + 8 * kOxMaxPairs);
      int n = 0;
      for (int p = 0; p < P; ++p) {
        const int m = pair_miss[p];
        if (m < 0 || !miss_lane[m]) continue;
        token[n] = p / K;
        expert[n] = a.ids[p];
        weight[n] = a.weights[p];
        ++n;
      }
      device int* head = (device int*)(mb + 128);
      head[0] = layer;
      head[1] = a.T;
      head[2] = n;
    }
    const int chunks = a.T * st.L / 8;
    const device uint4* lsrc = (const device uint4*)a.latent;
    for (int ch = int(lane); ch < chunks; ch += 32) ldst[ch] = lsrc[ch];
    simdgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
      const uint seq = *job_seq + 1;
      *job_seq = seq;
      *job_pending = 1;
      *(device uint*)mb = seq;  // the doorbell
    }
  } else if (lane == 0) {
    *job_pending = 0;
  }

  if (lane == 0) {
    device ulong* cnt = c.counters + long(layer) * kOxCounters;
    cnt[0] += ulong(hits);
    cnt[1] += ulong(gpu_pairs);
    cnt[2] += ulong(cpu_pairs);
    cnt[3] += ulong(fills);
    cnt[5] += ulong(prefetch_hits);
    cnt[6] += 1;
  }
}

// Adds the CPU lane's result once it is done. The host has already waited for the CPU (or for the
// timeout) before this kernel runs: done == job_seq is the CPU's answer, anything else a timeout.
kernel void ox_join(constant OxCacheArgs& a [[buffer(0)]], uint tid [[thread_index_in_threadgroup]],
                    uint threads [[threads_per_threadgroup]]) {
  threadgroup int ready[1];
  const device OxState& st = *a.state;
  if (tid == 0) {
    ready[0] = 0;
    device int* job_pending = (device int*)ox_ptr(a.map, st.job_pending);
    const device uint* job_seq = (const device uint*)ox_ptr(a.map, st.job_seq);
    device uchar* mb = ox_ptr(a.map, st.mailbox);
    device int* error = (device int*)ox_ptr(a.map, st.error);
    if (job_pending && job_seq && mb && error && *job_pending) {
      const uint done = atomic_load_explicit((device atomic_uint*)(mb + 64), memory_order_relaxed);
      if (done == *job_seq) ready[0] = 1;
      else *error = 1;
      *job_pending = 0;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (!ready[0]) return;
  const device float* r = (const device float*)ox_ptr(a.map, st.cpu_result);
  if (!r) return;
  const int n = a.T * a.L;
  for (int i = int(tid); i < n; i += int(threads)) a.out[i] += r[i];
}

// One simdgroup. Claims slots for the predicted experts of a layer and writes the copy jobs.
kernel void ox_prefetch(constant OxCacheArgs& a [[buffer(0)]], uint lane [[thread_index_in_simdgroup]]) {
  const device OxState& st = *a.state;
  const OxCache c = ox_cache(a.map, st);
  const int layer = a.layer, base = layer * st.n_experts;
  if (!ox_cache_ok(c) || layer < 0 || layer >= st.n_layers) {
    if (lane == 0) *a.count = 0;
    return;
  }
  // One past the clock: the slots the last plan used (another layer's) are no business of this one,
  // and the next plan of this layer sees the prefetched slots as used now, so it keeps them.
  const uint now = atomic_load_explicit(c.clock, memory_order_relaxed) + 1u;
  for (int j = int(lane); j < a.n; j += 32) {  // protect what is already resident among the predictions
    const int s = c.vram_slot[base + a.ids[j]];
    if (s >= 0 && c.slot_stamp[s] < now) c.slot_stamp[s] = now;
  }
  simdgroup_barrier(mem_flags::mem_device);
  int count = 0;
  for (int j = 0; j < a.n; ++j) {
    const int e = a.ids[j];
    if (c.vram_slot[base + e] >= 0) continue;  // resident, or claimed by an earlier prediction
    const int h = c.host_slot[base + e];
    if (h < 0) continue;
    const int v = ox_pick_victim(c, layer, now, lane);
    if (v < 0) break;
    if (lane == 0) {
      ox_claim_slot(c, st.n_experts, layer, e, v, now, true);
      a.src[count] = ox_pool_ref(st, st.host_pool, h);
      a.fill[count] = ox_pool_ref(st, st.vram_pool, v);
    }
    ++count;
    simdgroup_barrier(mem_flags::mem_device);
  }
  if (lane == 0) {
    *a.count = count;
    c.counters[long(layer) * kOxCounters + 4] += ulong(count);
  }
}

// Reads n 16-byte words with every thread of the grid, xor-folding them; the sink is written only on
// a fold no real data gives, so the loads stay.
kernel void ox_read_probe(constant OxProbeArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]],
                          uint threads [[threads_per_grid]]) {
  uint acc = 0;
  const long stride = long(threads);
  long j = long(i);
  for (; j + 3 * stride < a.n; j += 4 * stride) {
    const uint4 x0 = a.src[j], x1 = a.src[j + stride], x2 = a.src[j + 2 * stride], x3 = a.src[j + 3 * stride];
    const uint4 v = x0 ^ x1 ^ x2 ^ x3;
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  for (; j < a.n; j += stride) {
    const uint4 v = a.src[j];
    acc ^= v.x ^ v.y ^ v.z ^ v.w;
  }
  if (acc == 0x9e3779b9u) a.sink[0] = ulong(acc);
}
