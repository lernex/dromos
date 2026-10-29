// Routed experts for prefill and batches on the tensor cores, in every weight format and math
// (kernels.h, lx_moe_experts_grouped). One call, all on the GPU, nothing read back to the host:
//
//   1. route    a stable counting sort of the T * top_k (token, slot) pairs by expert: per-chunk counts,
//               one CTA turning them into offsets and the tile tables, then a scatter that ranks each
//               chunk's pairs (warp match + per-warp counts) so every expert's rows stay in token order
//   2. gate_up  the latent quantized once per token for the math (or read as bf16), then a grouped GEMM
//               whose loader gathers each pair's token row and whose epilogue writes
//               bf16(silu(gate) * up) per pair, in sorted order
//   3. down     the intermediate quantized per pair row (for the low-precision maths), then a grouped GEMM
//               whose epilogue multiplies by the routing weight and adds into the token's output row
//
// The routing is deterministic, and so is the down epilogue's sum over a token's top_k experts: each pair
// writes its own f32 row and a final pass sums the slots in order, so a token's bits do not depend on the batch
// it runs in, nor on the run (a speculative verify step must match the one-token steps bit for bit, and the
// prefill of the same prompt must give the same cache each time). LX_MOE_ORDERED=0 sums with concurrent float
// adds instead, which land in any order and cost less at large T.
//
// Tile variants per kind are registered with the tuner as "moe_grouped.<kind>.sm<arch>"; each stage looks
// up its own (pairs, N, K).
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdlib>

#include "cuda/gemm_w.cuh"
#include "cuda/moe_grouped.cuh"
#include "cuda/tunable.h"

#include "cuda/common.cuh"
namespace {

using namespace lx::tc;
using lx::gemmw::act_layout;
using lx::gemmw::act_math;
using lx::gemmw::ActLayout;
using lx::gemmw::k_ok;
using lx::gemmw::kind_for;
using lx::gemmw::round_up;
using lx::gemmw::weight_bits;
using lx::moeg::DownEpi;
using lx::moeg::Family;
using lx::moeg::GateUpEpi;
using lx::moeg::GroupedParams;
using lx::moeg::kTileBMs;
using lx::moeg::table_capacity;
using lx::moeg::TileTables;
using lx::moeg::Variant;

constexpr int kMaxE = 1024;  // one routing CTA thread per expert; the tile tables hold the expert in 16 bits
constexpr int kRouteThreads = 256;
constexpr int kMaxChunks = 128;
constexpr int kTileBM[kTileBMs] = {32, 64, 128, 256};

// Pairs per routing chunk: 1024, or more when that keeps the chunk count at kMaxChunks.
int chunk_pairs(int P) { return std::max(1, (P + kMaxChunks * 1024 - 1) / (kMaxChunks * 1024)) * 1024; }

// LX_MOE_ORDERED=0 turns the ordered sum off (concurrent float adds). Read once.
bool ordered_for(int) {
  static const bool on = [] {
    const char* v = std::getenv("LX_MOE_ORDERED");
    return !(v && v[0] == '0');
  }();
  return on;
}

// ---- workspace --------------------------------------------------------------------------------------

struct Layout {
  int P = 0, chunk = 0, chunks = 0, capacity[kTileBMs] = {};
  int64_t counts = 0, offsets = 0, table[kTileBMs] = {}, pairs = 0;
  ActLayout lat, inter_q;  // quantized (or bf16-copied) latent; quantized intermediate
  int64_t lat_off = 0, inter = 0, ld_inter = 0, inter_q_off = 0, sums = 0, total = 0;
};

Layout layout(int T, int L, int I, int E, int top_k, lx_math math, bool ordered) {
  Layout l;
  l.P = T * top_k;
  l.chunk = chunk_pairs(l.P);
  l.chunks = (l.P + l.chunk - 1) / l.chunk;
  int64_t o = 0;
  auto take = [&](int64_t bytes) {
    const int64_t at = o;
    o = round_up(o + bytes, 256);
    return at;
  };
  l.counts = take(4LL * l.chunks * E);
  l.offsets = take(4LL * (E + 1));
  for (int b = 0; b < kTileBMs; ++b) {
    l.capacity[b] = table_capacity(l.P, E, kTileBM[b]);
    l.table[b] = take(8LL * l.capacity[b]);
  }
  l.pairs = take(4LL * l.P);
  l.lat = act_layout(T, L, math);
  l.lat_off = take(l.lat.total);
  l.ld_inter = round_up(I, 8);
  l.inter = take(2 * l.ld_inter * l.P);
  if (math != LX_MATH_BF16) {
    l.inter_q = act_layout(l.P, I, math);
    l.inter_q_off = take(l.inter_q.total);
  }
  if (ordered) l.sums = take(4LL * l.P * L);
  l.total = o;
  return l;
}

// ---- routing ----------------------------------------------------------------------------------------

__global__ void __launch_bounds__(kRouteThreads) route_count(const int32_t* __restrict__ ids, int P, int E, int chunk,
                                                             int32_t* __restrict__ counts) {
  lx::pdl_trigger();
  lx::pdl_wait();
  extern __shared__ int hist[];
  for (int e = threadIdx.x; e < E; e += blockDim.x) hist[e] = 0;
  __syncthreads();
  const int p0 = blockIdx.x * chunk, p1 = min(P, p0 + chunk);
  for (int p = p0 + threadIdx.x; p < p1; p += blockDim.x) atomicAdd(&hist[ids[p]], 1);
  __syncthreads();
  for (int e = threadIdx.x; e < E; e += blockDim.x) counts[static_cast<int64_t>(blockIdx.x) * E + e] = hist[e];
}

// Exclusive prefix sum over the block (blockDim.x a multiple of 32, at most 1024); total gets the sum.
__device__ int block_exclusive_scan(int v, int* warp_sums, int& total) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
  int x = v;
#pragma unroll
  for (int d = 1; d < 32; d <<= 1) {
    const int y = __shfl_up_sync(0xffffffffu, x, d);
    if (lane >= d) x += y;
  }
  if (lane == 31) warp_sums[warp] = x;
  __syncthreads();
  if (warp == 0) {
    int s = lane < nw ? warp_sums[lane] : 0;
#pragma unroll
    for (int d = 1; d < 32; d <<= 1) {
      const int y = __shfl_up_sync(0xffffffffu, s, d);
      if (lane >= d) s += y;
    }
    warp_sums[lane] = s;
  }
  __syncthreads();
  const int base = warp > 0 ? warp_sums[warp - 1] : 0;
  total = warp_sums[nw - 1];
  __syncthreads();
  return base + x - v;
}

struct TablesOut {
  int2* table[kTileBMs];
  int capacity[kTileBMs];
};

// One CTA, a thread per expert: each chunk's count becomes the chunk's start within the expert, the
// totals become the expert offsets, and each tile table is written out in full (empty entries past the
// end, so a stale workspace never leaks into the GEMMs).
__global__ void __launch_bounds__(1024) route_scan(int32_t* __restrict__ counts, int chunks, int E,
                                                   int32_t* __restrict__ offsets, TablesOut out) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ int warp_sums[32];
  __shared__ int off[kMaxE + 1];
  __shared__ int ts[kMaxE + 1];
  const int e = threadIdx.x;
  int total = 0;
  if (e < E) {
    for (int c = 0; c < chunks; ++c) {
      const int64_t at = static_cast<int64_t>(c) * E + e;
      const int v = counts[at];
      counts[at] = total;
      total += v;
    }
  }
  int sum = 0;
  const int o = block_exclusive_scan(total, warp_sums, sum);
  if (e < E) off[e] = o;
  if (e == 0) off[E] = sum;
  __syncthreads();
  for (int i = e; i <= E; i += blockDim.x) offsets[i] = off[i];
  for (int b = 0; b < kTileBMs; ++b) {
    const int bm = 32 << b;
    int n_tiles = 0;
    const int t0 = block_exclusive_scan(e < E ? (total + bm - 1) / bm : 0, warp_sums, n_tiles);
    if (e < E) ts[e] = t0;
    if (e == 0) ts[E] = n_tiles;
    __syncthreads();
    int2* table = out.table[b];
    for (int i = e; i < out.capacity[b]; i += blockDim.x) {
      if (i >= n_tiles) {
        table[i] = make_int2(0, 0);
        continue;
      }
      int lo = 0, hi = E;  // the last expert whose first tile is at or before i
      while (hi - lo > 1) {
        const int mid = (lo + hi) >> 1;
        if (ts[mid] <= i) lo = mid;
        else hi = mid;
      }
      const int m0 = off[lo] + (i - ts[lo]) * bm;
      const int rows = min(bm, off[lo + 1] - m0);
      table[i] = make_int2(m0, (lo << 16) | rows);
    }
    __syncthreads();
  }
}

// Each chunk's pairs to their sorted positions, in pair order within an expert: a round takes one pair per
// thread; lanes with the same expert rank themselves with a warp match, warps after them by the per-warp
// counts, and earlier rounds and chunks by the running bases.
__global__ void __launch_bounds__(kRouteThreads) route_scatter(const int32_t* __restrict__ ids, int P, int E,
                                                               int chunk, const int32_t* __restrict__ counts,
                                                               const int32_t* __restrict__ offsets,
                                                               int32_t* __restrict__ sorted) {
  lx::pdl_trigger();
  lx::pdl_wait();
  constexpr int kWarps = kRouteThreads / 32;
  extern __shared__ int sh[];
  int* base = sh;                // [E]
  int* wcnt = sh + E;            // [kWarps][E]
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  for (int e = tid; e < E; e += kRouteThreads) {
    base[e] = offsets[e] + counts[static_cast<int64_t>(blockIdx.x) * E + e];
#pragma unroll
    for (int w = 0; w < kWarps; ++w) wcnt[w * E + e] = 0;
  }
  __syncthreads();
  const int p0 = blockIdx.x * chunk, p1 = min(P, p0 + chunk);
  for (int r0 = p0; r0 < p1; r0 += kRouteThreads) {
    const int p = r0 + tid;
    const bool valid = p < p1;
    const int e = valid ? ids[p] : -1;
    const unsigned same = __match_any_sync(0xffffffffu, e);
    const int rank = __popc(same & ((1u << lane) - 1u));
    if (valid && rank == 0) wcnt[warp * E + e] = __popc(same);
    __syncthreads();
    if (valid) {
      int before = 0;
      for (int w = 0; w < warp; ++w) before += wcnt[w * E + e];
      sorted[base[e] + before + rank] = p;
    }
    __syncthreads();
    for (int x = tid; x < E; x += kRouteThreads) {
      int s = 0;
#pragma unroll
      for (int w = 0; w < kWarps; ++w) {
        s += wcnt[w * E + x];
        wcnt[w * E + x] = 0;
      }
      base[x] += s;
    }
    __syncthreads();
  }
}

// out[t, :] = sum over slots in order of sums[t * top_k + k, :] (the ordered mode's final pass), in
// vectors of V (float4 when the rows allow, else float).
template <class V>
__device__ __forceinline__ void add_to(V& s, const V& v) {
  if constexpr (sizeof(V) == 16) {
    s.x += v.x;
    s.y += v.y;
    s.z += v.z;
    s.w += v.w;
  } else {
    s += v;
  }
}
template <class V>
__global__ void ordered_sum_kernel(const V* __restrict__ sums, int top_k, int Lv, int64_t n, V* __restrict__ out) {
  lx::pdl_trigger();
  lx::pdl_wait();
  for (int64_t i = blockIdx.x * static_cast<int64_t>(blockDim.x) + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    const int64_t t = i / Lv, c = i % Lv;
    const V* row = sums + t * top_k * Lv + c;
    V s = row[0];
    for (int k = 1; k < top_k; ++k) add_to(s, row[k * Lv]);
    out[i] = s;
  }
}

// ---- the variant families ---------------------------------------------------------------------------

template <int kKind, int kArch>
struct Registered {
  Registered() {
    int n = 0;
    const Variant* v = Family<kKind, kArch>::variants(n);
    const char* m[6] = {};
    for (int i = 0; i < n && i < 6; ++i) m[i] = v[i].name;
    switch (n) {
      case 1: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0]}); break;
      case 2: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1]}); break;
      case 3: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2]}); break;
      case 4: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3]}); break;
      case 5: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4]}); break;
      default: lx::TunableRegistration(Family<kKind, kArch>::name(), {m[0], m[1], m[2], m[3], m[4], m[5]});
    }
  }
};
const Registered<kBF16, 80> reg_bf16;
const Registered<kI8, 80> reg_i8;
const Registered<kF8, 89> reg_f8_89;
const Registered<kF8, 120> reg_f8_120;
const Registered<kF4, 120> reg_f4;
const Registered<kW8I8, 80> reg_w8i8;
const Registered<kW8F8, 80> reg_w8f8_80;
const Registered<kW8F8, 120> reg_w8f8_120;
const Registered<kW4A16, 80> reg_w4a16;
const Registered<kW4A8, 120> reg_w4a8;

// The tuned variant for (pairs, N, K), else the tallest tile the average expert fills.
template <int kKind, int kArch>
const Variant& pick(int P, int E, int N, int K) {
  int n = 0;
  const Variant* v = Family<kKind, kArch>::variants(n);
  int i = lx_tune_get(Family<kKind, kArch>::name(), P, N, K);
  if (i < 0 || i >= n) {
    const int rows = std::max(1, P / E);
    i = 0;
    for (int j = 0; j < n; ++j) {
      if (v[j].max_pairs > 0) {
        // A variant measured for a band of batch sizes (bit-identical to the others, see its registration).
        if (P >= v[j].min_pairs && P <= v[j].max_pairs) i = j;
      } else if (v[j].bm <= rows) {
        i = j;
      }
    }
  }
  return v[i];
}

template <int kKind, int kArch>
int stage(bool gate_up, const GroupedParams& pp, const void* ep, const TileTables& tt, int P, int E, cudaStream_t s) {
  const Variant& v = pick<kKind, kArch>(P, E, pp.N, pp.K);
  return gate_up ? v.gate_up(pp, *static_cast<const GateUpEpi::Params*>(ep), tt, s)
                 : v.down(pp, *static_cast<const DownEpi::Params*>(ep), tt, s);
}

int run_stage(int kind, int arch, bool gate_up, const GroupedParams& pp, const void* ep, const TileTables& tt, int P,
              int E, cudaStream_t s) {
  const bool sm120 = arch >= 120 && arch < 130;
  switch (kind) {
    case kBF16: return stage<kBF16, 80>(gate_up, pp, ep, tt, P, E, s);
    case kI8: return stage<kI8, 80>(gate_up, pp, ep, tt, P, E, s);
    case kF8: return sm120 ? stage<kF8, 120>(gate_up, pp, ep, tt, P, E, s) : stage<kF8, 89>(gate_up, pp, ep, tt, P, E, s);
    case kF4: return stage<kF4, 120>(gate_up, pp, ep, tt, P, E, s);
    case kW8I8: return stage<kW8I8, 80>(gate_up, pp, ep, tt, P, E, s);
    case kW8F8:
      return sm120 ? stage<kW8F8, 120>(gate_up, pp, ep, tt, P, E, s) : stage<kW8F8, 80>(gate_up, pp, ep, tt, P, E, s);
    case kW4A16: return stage<kW4A16, 80>(gate_up, pp, ep, tt, P, E, s);
    default: return stage<kW4A8, 120>(gate_up, pp, ep, tt, P, E, s);
  }
}

bool aligned16(const void* p) { return (reinterpret_cast<uintptr_t>(p) & 15) == 0; }

// The B side of a stage: an expert stack [E, rows, K] in its format.
void set_weight(GroupedParams& pp, const lx_weight& w, int rows, int K) {
  const int64_t ldb = static_cast<int64_t>(K) * weight_bits(w.type) / 8;
  pp.b = static_cast<const uint8_t*>(w.data);
  pp.ldb = ldb;
  pp.b_stride = ldb * rows;
  pp.b_global = w.type == LX_W_NVFP4 ? w.global : 1.0f;
  pp.b_expert_global = w.type == LX_W_NVFP4 ? w.expert_global : nullptr;
  pp.b_scale = nullptr;
  pp.b_sf = nullptr;
  if (w.type == LX_W_INT8 || w.type == LX_W_FP8) {
    pp.b_scale = static_cast<const float*>(w.scale);
    pp.b_scale_stride = rows;
  } else if (w.type == LX_W_NVFP4) {
    pp.b_sf = static_cast<const uint8_t*>(w.scale);
    pp.ld_bsf = K / 16;
    pp.b_sf_stride = static_cast<int64_t>(rows) * (K / 16);
  }
}

}  // namespace

extern "C" int64_t lx_moe_grouped_workspace_bytes(int T, int L, int I, int E, int top_k, lx_math math) {
  if (T <= 0 || L <= 0 || I <= 0 || E <= 0 || top_k <= 0) return 0;
  // Sized for the ordered sum, which a call with few rows always uses; the sums are the last region, so the
  // others sit where they do without it.
  return layout(T, L, I, E, top_k, math, true).total;
}

extern "C" int lx_moe_experts_grouped(const void* latent, const int32_t* ids, const float* weights,
                                      const lx_weight* gate_up, const lx_weight* down, lx_math math, float* out,
                                      void* workspace, int T, int L, int I, int E, int top_k, void* stream) {
  if (T < 0 || L <= 0 || I <= 0 || E <= 0 || top_k <= 0 || top_k > E || !gate_up || !down)
    return static_cast<int>(cudaErrorInvalidValue);
  if (math < LX_MATH_BF16 || math > LX_MATH_FP4) return static_cast<int>(cudaErrorInvalidValue);
  if (T == 0) return 0;
  if (!latent || !ids || !weights || !out || !workspace || !gate_up->data || !down->data)
    return static_cast<int>(cudaErrorInvalidValue);
  if ((gate_up->type != LX_W_BF16 && !gate_up->scale) || (down->type != LX_W_BF16 && !down->scale))
    return static_cast<int>(cudaErrorInvalidValue);
  if (E > kMaxE || static_cast<int64_t>(T) * top_k > (1LL << 30)) return static_cast<int>(cudaErrorNotSupported);
  const int arch = lx_arch();
  const int kind_gu = kind_for(math, gate_up->type, arch), kind_dn = kind_for(math, down->type, arch);
  if (kind_gu < 0 || kind_dn < 0) return static_cast<int>(cudaErrorNotSupported);
  // gate / up pairs share 16-column blocks (I % 8); the bf16 rows of the latent and the intermediate
  // are whole 16-byte chunks (L % 8, I % 8); the weights as lx_gemm_w needs them.
  if (I % 8 != 0 || L % 8 != 0 || !k_ok(kind_gu, gate_up->type, L) || !k_ok(kind_dn, down->type, I) ||
      !aligned16(gate_up->data) || !aligned16(down->data))
    return static_cast<int>(cudaErrorNotSupported);

  auto s = static_cast<cudaStream_t>(stream);
  const bool ordered = ordered_for(T);
  const Layout l = layout(T, L, I, E, top_k, math, ordered);
  uint8_t* ws = static_cast<uint8_t*>(workspace);
  const int P = l.P;
  auto* counts = reinterpret_cast<int32_t*>(ws + l.counts);
  auto* offsets = reinterpret_cast<int32_t*>(ws + l.offsets);
  auto* pairs = reinterpret_cast<int32_t*>(ws + l.pairs);

  // 1. route
  lx::launch_pdl(route_count, l.chunks, kRouteThreads, sizeof(int) * E, s, ids, P, E, l.chunk, counts);
  TablesOut to{};
  TileTables tt{};
  for (int b = 0; b < kTileBMs; ++b) {
    to.table[b] = reinterpret_cast<int2*>(ws + l.table[b]);
    to.capacity[b] = l.capacity[b];
    tt.table[b] = to.table[b];
    tt.capacity[b] = l.capacity[b];
  }
  lx::launch_pdl(route_scan, 1, 1024, 0, s, counts, l.chunks, E, offsets, to);
  lx::launch_pdl(route_scatter, l.chunks, kRouteThreads, sizeof(int) * E * (1 + kRouteThreads / 32), s, ids, P, E, l.chunk, counts,
                                                                                          offsets, pairs);
  if (int e = lx::launch_status()) return e;

  // 2. gate_up: A is the latent, per token
  const lx_math qmath = act_math(kind_gu);  // the same for both stages: one math
  GroupedParams gp{};
  gp.pairs = pairs;
  gp.top_k = top_k;
  gp.N = 2 * I;
  gp.K = L;
  gp.half_n = I;
  set_weight(gp, *gate_up, 2 * I, L);
  uint8_t* lat = ws + l.lat_off;
  if (qmath == LX_MATH_BF16) {
    if (aligned16(latent)) {
      gp.a = static_cast<const uint8_t*>(latent);
      gp.lda = 2LL * L;
    } else {
      if (int e = lx_copy2d(latent, LX_BF16, L, lat, LX_BF16, static_cast<int>(l.lat.ld_q / 2), T, L, stream)) return e;
      gp.a = lat;
      gp.lda = l.lat.ld_q;
    }
  } else {
    if (int e = lx_quantize_act(latent, LX_BF16, L, qmath, lat + l.lat.q_off, static_cast<int>(l.lat.ld_q),
                                reinterpret_cast<float*>(lat + l.lat.scale_off), lat + l.lat.bs_off,
                                static_cast<int>(l.lat.ld_bs), T, L, stream))
      return e;
    gp.a = lat + l.lat.q_off;
    gp.lda = l.lat.ld_q;
    gp.a_scale = reinterpret_cast<const float*>(lat + l.lat.scale_off);
    gp.a_sf = lat + l.lat.bs_off;
    gp.ld_asf = l.lat.ld_bs;
  }
  auto* inter = reinterpret_cast<__nv_bfloat16*>(ws + l.inter);
  GateUpEpi::Params ge{inter, l.ld_inter};
  if (int e = run_stage(kind_gu, arch, true, gp, &ge, tt, P, E, s)) return e;

  // 3. down: A is the intermediate, per pair
  GroupedParams dp{};
  dp.pairs = pairs;
  dp.top_k = top_k;
  dp.N = L;
  dp.K = I;
  set_weight(dp, *down, L, I);
  if (qmath == LX_MATH_BF16) {
    dp.a = reinterpret_cast<const uint8_t*>(inter);
    dp.lda = 2 * l.ld_inter;
  } else {
    uint8_t* iq = ws + l.inter_q_off;
    if (int e = lx_quantize_act(inter, LX_BF16, static_cast<int>(l.ld_inter), qmath, iq + l.inter_q.q_off,
                                static_cast<int>(l.inter_q.ld_q), reinterpret_cast<float*>(iq + l.inter_q.scale_off),
                                iq + l.inter_q.bs_off, static_cast<int>(l.inter_q.ld_bs), P, I, stream))
      return e;
    dp.a = iq + l.inter_q.q_off;
    dp.lda = l.inter_q.ld_q;
    dp.a_scale = reinterpret_cast<const float*>(iq + l.inter_q.scale_off);
    dp.a_sf = iq + l.inter_q.bs_off;
    dp.ld_asf = l.inter_q.ld_bs;
  }
  DownEpi::Params de{};
  de.pairs = pairs;
  de.top_k = top_k;
  de.weights = weights;
  de.ld = L;
  de.atomic = !ordered;
  if (ordered) {
    de.out = reinterpret_cast<float*>(ws + l.sums);
    de.pair_ok = true;
  } else {
    if (cudaError_t e = cudaMemsetAsync(out, 0, sizeof(float) * T * static_cast<size_t>(L), s)) return static_cast<int>(e);
    de.out = out;
    de.pair_ok = (reinterpret_cast<uintptr_t>(out) & 7) == 0;
  }
  if (int e = run_stage(kind_dn, arch, false, dp, &de, tt, P, E, s)) return e;
  if (ordered) {
    const bool v4 = aligned16(out);
    const int Lv = v4 ? L / 4 : L;
    const int64_t n = static_cast<int64_t>(T) * Lv;
    const int blocks = static_cast<int>(std::min<int64_t>((n + 255) / 256, 4096));
    if (v4)
      lx::launch_pdl(ordered_sum_kernel<float4>, blocks, 256, 0, s, reinterpret_cast<const float4*>(ws + l.sums), top_k, Lv, n,
                                                        reinterpret_cast<float4*>(out));
    else
      lx::launch_pdl(ordered_sum_kernel<float>, blocks, 256, 0, s, reinterpret_cast<const float*>(ws + l.sums), top_k, Lv, n, out);
    if (int e = lx::launch_status()) return e;
  }
  return 0;
}
