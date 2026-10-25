// The decode program builder (kernels.h, "Decode programs"): ops in step order, their tasks, the
// dependencies between them (inferred from the buffers each op reads and writes), and the layout of
// the tasks over the SMs. The kernel that runs a program is decode_kernel.cu.
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "cuda/decode_ops.h"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

using namespace lx::decode;

namespace {

constexpr int kMaxSeg = 4;

struct Region {
  uintptr_t lo, hi;
  bool write;
};

int ceil_div(int64_t a, int64_t b) { return static_cast<int>((a + b - 1) / b); }
int round_up(int a, int b) { return (a + b - 1) / b * b; }

bool row_scaled_type(int wtype) { return wtype == LX_W_INT8 || wtype == LX_W_FP8; }

int bytes_per_value(int wtype) {
  switch (wtype) {
    case LX_W_BF16: return 2;
    case LX_W_F32: return 4;
    case LX_W_INT8: return 1;
    default: return 0;
  }
}

}  // namespace

struct lx_program {
  int M;
  const int32_t *tokens, *slots, *has_state, *start_pos;
  std::vector<OpDesc> ops;
  std::vector<std::vector<Region>> regions;
  std::vector<double> cost;                   // per task, in bytes moved (estimated for non-products)
  std::vector<std::vector<uint64_t>> ancestors;
  std::vector<void*> scratch;
  int blocks = 0;
  bool failed = false;
  // built
  bool built = false;
  OpDesc* d_ops = nullptr;
  uint32_t* d_lists = nullptr;
  int32_t* d_off = nullptr;
  bool bulk = false;
  uint32_t* d_done = nullptr;
  int64_t* d_trace = nullptr;
  uint32_t* d_ahead = nullptr;
  PfSpan* d_spans = nullptr;
  int32_t* d_span_off = nullptr;
  int total_tasks = 0, x_bytes = 0;
  size_t smem = 0;
  int x_need = 0;
  // The x buffer: whatever shared memory the fixed slots leave. A product whose x does not fit is
  // split along K into partial products.
  int x_cap = 0;
  int prefetch_bytes = 0;

  void* alloc(size_t bytes) {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      failed = true;
      return nullptr;
    }
    cudaMemset(p, 0, bytes);
    scratch.push_back(p);
    return p;
  }

  // Adds an op: infers what it waits on from its regions, reduced to the ops it needs directly.
  // task_cost balances the layout.
  int add(OpDesc d, std::vector<Region> regs, double task_cost) {
    if (built || failed) return -1;
    const int id = static_cast<int>(ops.size());
    const int words = (id + 64) / 64;
    std::vector<int> direct;
    for (int i = id - 1; i >= 0; --i) {
      bool hit = false;
      for (const Region& a : regions[i]) {
        for (const Region& b : regs) {
          if ((a.write || b.write) && a.lo < b.hi && b.lo < a.hi) {
            hit = true;
            break;
          }
        }
        if (hit) break;
      }
      if (hit) direct.push_back(i);
    }
    std::vector<uint64_t> anc(words, 0);
    for (int i : direct) {
      anc[i / 64] |= 1ull << (i % 64);
      for (size_t w = 0; w < ancestors[i].size(); ++w) anc[w] |= ancestors[i][w];
    }
    d.ndeps = 0;
    for (int i : direct) {
      // implied when a later direct dependency already waits on it (an op's ancestors are all
      // earlier ops, and its bit set holds only as many words as its own index needs)
      bool implied = false;
      for (int k : direct)
        if (k > i && (ancestors[k][i / 64] >> (i % 64) & 1)) {
          implied = true;
          break;
        }
      if (implied) continue;
      if (d.ndeps == kMaxDeps) {
        failed = true;
        break;
      }
      d.deps[d.ndeps++] = i;
    }
    ops.push_back(d);
    regions.push_back(std::move(regs));
    cost.push_back(task_cost);
    ancestors.push_back(std::move(anc));
    return id;
  }

  // The last op that wrote any byte of [p, p + bytes).
  int last_writer(const void* p, size_t bytes) const {
    const uintptr_t lo = reinterpret_cast<uintptr_t>(p), hi = lo + bytes;
    for (int i = static_cast<int>(ops.size()) - 1; i >= 0; --i)
      for (const Region& a : regions[i])
        if (a.write && a.lo < hi && lo < a.hi) return i;
    return -1;
  }
};

namespace {

Region rd(const void* p, size_t bytes) {
  const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
  return {lo, lo + bytes, false};
}
Region wr(const void* p, size_t bytes) {
  const uintptr_t lo = reinterpret_cast<uintptr_t>(p);
  return {lo, lo + bytes, true};
}
// Rows m < M of a row-major buffer with row stride ld (elements), first width elements of each.
size_t span(int M, int ld, int width, int elem) { return (static_cast<size_t>(M - 1) * ld + width) * elem; }

OpDesc blank(int kind, int ntasks) {
  OpDesc d;
  std::memset(&d, 0, sizeof(d));
  d.kind = kind;
  d.ntasks = ntasks;
  d.dyn_dep = -1;
  return d;
}

// Rows per product task: up to kTileBytes of weights, but spread over every SM when the product is
// small (its latency is then one task's).
int tile_rows(int N, int row_bytes, int blocks, int factor = 1) {
  int rows = std::max(1, std::min(kMaxTileRows, kTileBytes / (row_bytes * factor)));
  return std::max(1, std::min(rows, ceil_div(N, blocks)));
}

}  // namespace

extern "C" {

lx_program* lx_program_create(int M, const int32_t* tokens, const int32_t* slots, const int32_t* has_state,
                              const int32_t* start_pos) {
  if (M < 1 || M > kMaxM) return nullptr;
  auto* p = new lx_program();
  p->M = M;
  int dev = 0, smem_max = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&smem_max, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev);
  p->x_cap = (smem_max - kXOff) / 128 * 128;
  p->tokens = tokens;
  p->slots = slots;
  p->has_state = has_state;
  p->start_pos = start_pos;
  cudaDeviceGetAttribute(&p->blocks, cudaDevAttrMultiProcessorCount, dev);
  return p;
}

void lx_program_destroy(lx_program* p) {
  if (!p) return;
  for (void* s : p->scratch) cudaFree(s);
  cudaFree(p->d_ops);
  cudaFree(p->d_lists);
  cudaFree(p->d_off);
  cudaFree(p->d_ahead);
  cudaFree(p->d_spans);
  cudaFree(p->d_span_off);
  cudaFree(p->d_done);
  cudaFree(p->d_trace);
  delete p;
}

int lx_program_tasks(const lx_program* p, int op) {
  if (!p || op < 0 || op >= static_cast<int>(p->ops.size())) return 0;
  return p->ops[op].ntasks;
}

// ---- Products -----------------------------------------------------------------------------------

// How many parts (2^sh) each row's k is read in, from the shape alone: the tensor-core reader takes
// the fewest parts for the least time per warp (units per warp times steps per unit, plus about a
// step of overhead per unit), the FMA reader splits rows across warps when a block has fewer rows
// than warps. A block of the run's rows by every part by kMaxM tokens must fit the scratch.
static int k_split(const Gemv& g, int ntasks, int blocks, bool parts) {
  const int R = ceil_div(ntasks, blocks) * g.rows, Kc = g.k1 - g.k0;
  int sh = 0;
  if (g.mma) {
    const int spr = Kc / (g.mma == 1 || g.mma == 3 ? 32 : 64), groups = ceil_div(R, 16);
    int best = 1 << 30;
    for (int k = 0; (1 << k) <= spr && k <= 6; ++k) {
      const int cost = ceil_div(groups << k, kComputeWarps) * (ceil_div(spr, 1 << k) + 1);
      if (cost < best) {
        best = cost;
        sh = k;
      }
    }
  } else {
    while ((1 << sh) < kComputeWarps && (2 << sh) * R <= kComputeWarps) ++sh;
  }
  while (sh > 0 && g.rows * ((1 << sh) + (parts ? 1 : 0)) * kMaxM > kScratchFloats) --sh;
  return sh;
}

int lx_program_gemv(lx_program* p, const lx_fx* x, const lx_weight* w, const lx_fy* y, int N, int K) {
  if (!p || p->built || !x || !w || !y || N < 1 || K < 8 || K % 8) return -1;
  const int M = p->M, wtype = static_cast<int>(w->type), bpv = bytes_per_value(wtype);
  if (!bpv) return -1;
  // How x is staged: bf16 when its values are bf16 already (or rounded to bf16 on the way in).
  bool xs_bf16 = x->kind == LX_FX_RMS || x->kind == LX_FX_GATED || x->kind == LX_FX_SWIGLU ||
                 (x->kind == LX_FX_PLAIN && x->dtype == LX_BF16) ||
                 (x->kind == LX_FX_CONCAT && x->round_bf16);
  if (wtype == LX_W_F32) xs_bf16 = false;
  // The tensor-core reader takes bf16 rows, and int8 rows (which convert to f16 exactly), in whole
  // steps of four 16-byte chunks; x is staged as bf16 or f16, its rows padded against bank conflicts.
  const int cv = wtype == LX_W_BF16 ? 8 : wtype == LX_W_INT8 ? 16 : 0;  // values per 16-byte chunk
  int mma = 0;
  if (cv && K % (4 * cv) == 0 && reinterpret_cast<uintptr_t>(w->data) % 16 == 0 && !std::getenv("LX_PROGRAM_FMA"))
    mma = (wtype == LX_W_BF16 ? 1 : 2) + (xs_bf16 ? 0 : 2);
  const int xe = mma >= 3 ? 4 : mma || xs_bf16 ? 2 : 4;  // bytes of x staged per value
  const int xpad = mma == 1 || mma == 3 ? 32 : mma ? 8 : 0;
  const int kstep = mma ? 4 * cv : 8;
  // K chunks: the chunk's x must fit the x buffer. A norm over the whole row needs the whole row.
  // Sized for the most tokens a step takes, so a row is summed the same way whatever M is.
  int chunks = ceil_div(static_cast<int64_t>(kMaxM) * (K + xpad) * xe, p->x_cap);
  if (x->kind == LX_FX_GATED && chunks > 1) chunks = 1 << static_cast<int>(std::ceil(std::log2(chunks)));
  int Kc = round_up(ceil_div(K, chunks), kstep);
  chunks = ceil_div(K, Kc);
  if (chunks > 1 && (x->kind == LX_FX_RMS || (x->kind == LX_FX_GATED && Kc % x->group))) return -1;

  // The x regions every chunk reads.
  std::vector<Region> xr;
  Gemv base;
  std::memset(&base, 0, sizeof(base));
  base.w = *w;
  base.wtype = wtype;
  base.N = N;
  base.K = K;
  base.x_kind = x->kind;
  base.x_dtype = x->dtype;
  base.ld_x = x->ld;
  base.n_parts = x->n_parts;
  base.group = x->group;
  base.round_bf16 = x->round_bf16;
  base.xs_bf16 = xs_bf16;
  base.mma = mma;
  base.x = x->x;
  base.parts = x->parts;
  base.norm_w = x->norm_w;
  base.eps = x->eps;
  switch (x->kind) {
    case LX_FX_PLAIN: xr.push_back(rd(x->x, span(M, x->ld, K, x->dtype == LX_BF16 ? 2 : 4))); break;
    case LX_FX_RMS:
    case LX_FX_GATED:
      xr.push_back(rd(x->x, span(M, x->ld, K, 4)));
      xr.push_back(rd(x->parts, static_cast<size_t>(M) * x->n_parts * 4));
      if (x->kind == LX_FX_GATED && (x->group < 4 || K % x->group || x->n_parts % (K / x->group))) return -1;
      break;
    case LX_FX_SWIGLU:
      if (x->ld < 2 * K) return -1;
      xr.push_back(rd(x->x, span(M, x->ld, 2 * K, 4)));
      break;
    case LX_FX_CONCAT: {
      int end = 0;
      for (int i = 0; i < kMaxSeg && x->width[i] > 0; ++i) {
        if (x->width[i] % 4) return -1;
        base.seg[i] = x->seg[i];
        end += x->width[i];
        base.seg_end[i] = end;
        base.nseg = i + 1;
        xr.push_back(rd(x->seg[i], static_cast<size_t>(M) * x->width[i] * 4));
      }
      for (int i = base.nseg; i < kMaxSeg; ++i) base.seg_end[i] = 1 << 30;
      if (end != K) return -1;
      break;
    }
    default: return -1;
  }

  // Partial products of all but the last chunk go to scratch the last one adds in.
  float* partial = nullptr;
  if (chunks > 1) partial = static_cast<float*>(p->alloc(static_cast<size_t>(chunks - 1) * M * N * 4));
  int last = -1;
  for (int c = 0; c < chunks; ++c) {
    Gemv g = base;
    g.k0 = c * Kc;
    g.k1 = std::min(K, g.k0 + Kc);
    g.row_bytes = (g.k1 - g.k0) * bpv;
    g.rows = tile_rows(N, g.row_bytes, p->blocks);
    const int ntasks = ceil_div(N, g.rows);
    g.sh = k_split(g, ntasks, p->blocks, y->kind == LX_FY_WRITE && y->next_parts && c == chunks - 1);
    std::vector<Region> regs = xr;
    const bool final_chunk = c == chunks - 1;
    if (!final_chunk) {
      g.y_kind = LX_FY_STORE;
      g.out = partial + static_cast<size_t>(c) * M * N;
      g.out_dtype = LX_F32;
      g.ld_out = N;
      g.alpha = 1.0f;
      regs.push_back(wr(g.out, static_cast<size_t>(M) * N * 4));
    } else {
      g.y_kind = y->kind;
      g.out = y->out;
      g.out_dtype = y->dtype;
      g.ld_out = y->ld;
      g.bias = y->bias;
      g.bias_dtype = y->bias_dtype;
      g.alpha = y->alpha;
      g.beta = y->beta;
      g.act = y->act;
      g.out_bf = y->out_bf;
      g.ld_bf = y->ld_bf;
      g.lora_up = y->lora_up;
      g.lora = y->lora;
      g.rank = y->rank;
      g.lora_gate = y->lora_gate;
      g.streams = y->streams;
      g.matrix = y->matrix;
      g.write = y->write;
      g.S = y->S;
      g.state = y->state;
      g.next_read = y->next_read;
      g.next_src = y->next_src;
      g.next_parts = y->next_parts;
      if (chunks > 1) {
        g.y_add = partial;
        g.n_add = chunks - 1;
        regs.push_back(rd(partial, static_cast<size_t>(chunks - 1) * M * N * 4));
      }
      if (y->kind == LX_FY_STORE) {
        const int oe = y->dtype == LX_BF16 ? 2 : 4;
        regs.push_back(wr(y->out, span(M, y->ld, N, oe)));
        if (y->beta != 0.0f) regs.push_back(rd(y->out, span(M, y->ld, N, oe)));
        if (y->out_bf) regs.push_back(wr(y->out_bf, span(M, y->ld_bf, N, 2)));
        if (y->lora_up) {
          if (y->rank < 1) return -1;
          regs.push_back(rd(y->lora, static_cast<size_t>(M) * y->rank * 4));
        }
      } else if (y->kind == LX_FY_WRITE) {
        if (y->S < 1 || y->S > kMaxStreams) return -1;
        regs.push_back(wr(y->streams, static_cast<size_t>(M) * y->S * N * 4));
        regs.push_back(rd(y->streams, static_cast<size_t>(M) * y->S * N * 4));
        if (y->state) regs.push_back(wr(y->state, static_cast<size_t>(M) * N * 4));
        if (y->next_read) {
          regs.push_back(wr(y->next_src, static_cast<size_t>(M) * N * 4));
          if (y->next_parts) regs.push_back(wr(y->next_parts, static_cast<size_t>(M) * ntasks * 4));
        }
      } else {
        return -1;
      }
    }
    OpDesc d = blank(kGemv, ntasks);
    d.pool = kStatic;
    d.gemv = g;
    p->x_need = std::max(p->x_need, M * (g.k1 - g.k0 + xpad) * xe);
    last = p->add(d, std::move(regs), static_cast<double>(g.rows) * g.row_bytes);
  }
  return p->failed ? -1 : last;
}

// ---- The rest of a step -------------------------------------------------------------------------

int lx_program_embed(lx_program* p, const void* table, const void* stream_emb, float* streams, float* previous,
                     float* history, float* continuation, int S, int D, int R) {
  if (!p || S < 1 || S > kMaxStreams || D < 1) return -1;
  const int M = p->M, chunks = std::max(1, std::min(8, p->blocks / M));
  OpDesc d = blank(kEmbed, M * chunks);
  d.embed = {table, stream_emb, streams, previous, history, continuation, S, D, R, chunks};
  std::vector<Region> regs = {wr(streams, static_cast<size_t>(M) * S * D * 4),
                              wr(previous, static_cast<size_t>(M) * D * 4), wr(history, static_cast<size_t>(M) * R * 4),
                              wr(continuation, static_cast<size_t>(M) * 4)};
  return p->add(d, std::move(regs), 4096.0);
}

int lx_program_memory_read(lx_program* p, const lx_memread_args* a) {
  if (!p || !a || a->S < 1 || a->S > kMaxStreams || a->D % 4 || a->D > 4 * 2 * kComputeThreads ||
      a->n_entries < 0 || a->n_entries > kMaxEntries || (a->lora_down && (a->rank < 1 || a->rank > kMaxEntries)) ||
      a->entry_stride % 4 || a->token_stride % 4 || reinterpret_cast<uintptr_t>(a->bank) % 16)
    return -1;
  // the token's entries are staged in the x buffer, one half (keys or values) at a time
  if (a->n_entries * a->D * 4 > p->x_cap) return -1;
  p->x_need = std::max(p->x_need, a->n_entries * a->D * 4);
  const int M = p->M, S = a->S, D = a->D;
  OpDesc d = blank(kMemRead, M);
  d.pool = kStatic;  // its weights, and the token's bank entries and previous mean (rewritten rarely)
  MemRead& m = d.mem;
  m.streams = a->streams;
  m.bank = a->bank;
  m.gate = a->gate;
  m.gate_bias = a->gate_bias;
  m.state = a->state;
  m.summary = a->summary;
  m.read = a->read;
  m.norm_w = a->norm_w;
  m.h = a->h;
  m.lora_down = a->lora_down;
  m.lora = a->lora;
  m.previous = a->previous;
  m.difference = a->difference;
  m.rms = a->rms;
  m.eps = a->eps;
  m.n_entries = a->n_entries;
  m.entry_stride = a->entry_stride;
  m.token_stride = a->token_stride;
  m.mem_dim = a->mem_dim;
  m.rank = a->rank;
  m.S = S;
  m.D = D;
  const size_t md = static_cast<size_t>(M) * D * 4;
  std::vector<Region> regs = {rd(a->streams, md * S), wr(a->streams, md * S), wr(a->state, md), wr(a->summary, md)};
  if (a->n_entries > 0)
    regs.push_back(rd(a->bank, (static_cast<size_t>(M - 1) * a->token_stride +
                                static_cast<size_t>(a->n_entries) * a->entry_stride) * 4));
  if (a->rms) regs.push_back(wr(a->rms, static_cast<size_t>(2) * M * 4));
  if (a->previous) {
    regs.push_back(rd(a->previous, md));
    regs.push_back(wr(a->previous, md));
    regs.push_back(wr(a->difference, md));
  }
  if (a->h) {
    regs.push_back(wr(a->h, static_cast<size_t>(M) * D * 2));
    if (a->lora_down) regs.push_back(wr(a->lora, static_cast<size_t>(M) * a->rank * 4));
  }
  return p->add(d, std::move(regs), 8192.0 * (1 + a->n_entries));
}

int lx_program_ssm(lx_program* p, const lx_ssm_args* a) {
  if (!p || !a || a->P < 1 || kComputeThreads % a->P || a->W < 2 || a->W > 8 || a->G < 1 || a->H % a->G) return -1;
  const int lpr = kComputeThreads / a->P;
  if (lpr > 32 || a->N % (4 * lpr)) return -1;
  const int ni = a->N / (4 * lpr);
  if (ni != 1 && ni != 2 && ni != 4 && ni != 8) return -1;
  const int state_bytes = a->P * a->N * 4;
  if (a->P + 2 * a->N + a->P > kScratchFloats) return -1;
  const int M = p->M, conv_dim = a->I + 2 * a->G * a->N;
  OpDesc d = blank(kSsm, M * a->H);
  d.pool = kStatic;
  d.ssm = {a->zxbcdt, a->conv_w, a->conv_b, a->conv_state, a->ssm_state, a->A_log, a->D, a->dt_bias, a->v, a->head_ss,
           a->ld, a->I, a->H, a->P, a->G, a->N, a->W};
  (void)conv_dim;
  // The persistent state is touched only by its own layer's ops: its first byte stands for all of it.
  std::vector<Region> regs = {rd(a->zxbcdt, span(M, a->ld, a->ld, 4)), rd(a->conv_state, 1), wr(a->conv_state, 1),
                              rd(a->ssm_state, 1), wr(a->ssm_state, 1), wr(a->v, static_cast<size_t>(M) * a->I * 4),
                              wr(a->head_ss, static_cast<size_t>(M) * a->H * 4)};
  const int step = p->add(d, std::move(regs), state_bytes * 2.0);
  if (step < 0) return -1;
  OpDesc c = blank(kConvBC, M);
  c.ssm = d.ssm;
  std::vector<Region> cregs = {rd(a->zxbcdt, span(M, a->ld, a->ld, 4)), rd(a->conv_state, 1), wr(a->conv_state, 1)};
  if (p->add(c, std::move(cregs), 4096.0) < 0) return -1;
  return step;
}

int lx_program_select(lx_program* p, const lx_select_args* a) {
  if (!p || !a || a->top_k < 1 || a->top_k > 32 || a->top_k > a->E || a->E > 256) return -1;
  const int M = p->M;
  OpDesc d = blank(kSelect, 1);
  d.pool = kStatic;  // the expert embeddings and the selection bias
  d.sel = {a->logits, a->selection_bias, a->ids, a->weights, a->entropy, a->confidence, a->expert_emb, a->summary,
           a->history, a->E, a->top_k, a->R};
  const size_t mk = static_cast<size_t>(M) * a->top_k * 4, mr = static_cast<size_t>(M) * a->R * 4;
  std::vector<Region> regs = {rd(a->logits, static_cast<size_t>(M) * a->E * 4), wr(a->ids, mk), wr(a->weights, mk),
                              wr(a->entropy, static_cast<size_t>(M) * 4), wr(a->confidence, static_cast<size_t>(M) * 4),
                              wr(a->summary, mr), rd(a->history, mr), wr(a->history, mr)};
  return p->add(d, std::move(regs), 16384.0);
}

int lx_program_experts(lx_program* p, const lx_experts_args* a) {
  if (!p || !a || a->L % 8 || a->I % 8 || a->Is % 8 || a->top_k < 1 || a->top_k > 8) return -1;
  const int M = p->M, L = a->L, I = a->I, Is = a->Is, K = a->top_k;
  const int select = p->last_writer(a->ids, static_cast<size_t>(M) * K * 4);
  if (select < 0) return -1;
  const int gt = static_cast<int>(a->gate_up.type), dt = static_cast<int>(a->down.type);
  const int sgt = static_cast<int>(a->shared_gate_up.type);
  if (gt != LX_W_BF16 && gt != LX_W_INT8) return -1;
  if (dt != LX_W_BF16 && dt != LX_W_INT8) return -1;
  if (sgt != LX_W_BF16 && sgt != LX_W_INT8) return -1;
  if (K * I * 2 > p->x_cap) return -1;
  float* shared_gu = static_cast<float*>(p->alloc(static_cast<size_t>(M) * 2 * Is * 4));
  void* inter_r = p->alloc(static_cast<size_t>(M) * K * I * 2);
  float* shared_out = static_cast<float*>(p->alloc(static_cast<size_t>(M) * L * 4));
  if (p->failed) return -1;
  const size_t lat = static_cast<size_t>(M) * L * 2;

  // the shared expert: its gate|up product, then the down product of bf16(silu(gate) * up)
  {
    lx_fx fx = {};
    fx.kind = LX_FX_PLAIN;
    fx.x = a->latent;
    fx.dtype = LX_BF16;
    fx.ld = L;
    lx_fy fy = {};
    fy.kind = LX_FY_STORE;
    fy.out = shared_gu;
    fy.dtype = LX_F32;
    fy.ld = 2 * Is;
    fy.alpha = 1.0f;
    if (lx_program_gemv(p, &fx, &a->shared_gate_up, &fy, 2 * Is, L) < 0) return -1;
    fx.kind = LX_FX_SWIGLU;
    fx.x = shared_gu;
    fx.dtype = LX_F32;
    fx.ld = 2 * Is;
    fy.out = shared_out;
    fy.ld = L;
    if (lx_program_gemv(p, &fx, &a->shared_down, &fy, L, Is) < 0) return -1;
  }
  // the routed experts: gate|up rows per (token, pick), then the down rows per token. On the tensor
  // cores when the format and shape allow (16-byte chunks in whole steps), else on the FMA reader.
  // The k split comes from the shape (and one token), never from M, so a row's sums are the same
  // whatever the step's batch.
  const bool fma = std::getenv("LX_PROGRAM_FMA") != nullptr;
  auto tc_ok = [&](const lx_weight& w, int k) {
    const int cv = w.type == LX_W_BF16 ? 8 : w.type == LX_W_INT8 ? 16 : 0;
    return !fma && cv && k % (4 * cv) == 0 && reinterpret_cast<uintptr_t>(w.data) % 16 == 0;
  };
  // 2^sh parts per unit for the least time per warp (units per warp times steps per unit, plus about
  // a step per unit), with a block's share of one token's units.
  auto unit_split = [&](int units, int spr, int per_unit_floats) {
    int sh = 0, best = 1 << 30;
    for (int k = 0; (1 << k) <= spr && k <= 4; ++k) {
      const int cost = ceil_div(static_cast<int64_t>(units) << k, kComputeWarps) * (ceil_div(spr, 1 << k) + 1);
      if (cost < best && per_unit_floats << k <= kScratchFloats - 2 * kMaxM * 8) {
        best = cost;
        sh = k;
      }
    }
    return sh;
  };
  {
    const int rb = L * bytes_per_value(gt);
    const bool tc = tc_ok(a->gate_up, L) && I % 8 == 0 && M * (L + 32) * 2 <= p->x_cap;
    const int rows = tc ? 8 : tile_rows(I, rb, p->blocks, 2);
    if (I % rows) return -1;
    const int bpp = I / rows;
    int sh = 0;
    if (tc) {
      const int spr = L / (gt == LX_W_BF16 ? 32 : 64);
      sh = unit_split(ceil_div(K * bpp, p->blocks), spr, 16);
    }
    OpDesc d = blank(kExpUp, M * K * bpp);
    d.pool = kDynamic;
    d.dyn_dep = select;
    d.up = {a->latent, a->ids, a->gate_up, inter_r, gt, L, I, K, tc ? 1 : 0, sh, rows, rb, bpp};
    p->x_need = std::max(p->x_need, tc ? M * (L + 32) * 2 : L * 2);
    std::vector<Region> regs = {rd(a->latent, lat), rd(a->ids, static_cast<size_t>(M) * K * 4),
                                wr(inter_r, static_cast<size_t>(M) * K * I * 2)};
    if (p->add(d, std::move(regs), 2.0 * rows * rb) < 0) return -1;
  }
  {
    const int rb = I * bytes_per_value(dt);
    const bool tc = tc_ok(a->down, I) && L % 16 == 0 && M * K * (I + 32) * 2 <= p->x_cap;
    const int rows = tc ? 16 : tile_rows(L, rb, p->blocks, K);
    if (L % rows) return -1;
    const int bpt = L / rows;
    int sh = 0;
    if (tc) {
      const int spr = I / (dt == LX_W_BF16 ? 32 : 64);
      sh = unit_split(ceil_div(bpt, p->blocks) * K, spr, 16 * K);
    }
    OpDesc d = blank(kExpDown, M * bpt);
    d.pool = kDynamic;
    d.dyn_dep = select;
    d.down = {inter_r, a->ids, a->weights, a->down, a->out, shared_out, dt, L, I, K, tc ? 1 : 0, sh, rows, rb, bpt};
    p->x_need = std::max(p->x_need, tc ? M * K * (I + 32) * 2 : K * I * 2);
    std::vector<Region> regs = {rd(inter_r, static_cast<size_t>(M) * K * I * 2),
                                rd(a->ids, static_cast<size_t>(M) * K * 4),
                                rd(a->weights, static_cast<size_t>(M) * K * 4),
                                rd(shared_out, static_cast<size_t>(M) * L * 4),
                                wr(a->out, static_cast<size_t>(M) * L * 4)};
    if (p->add(d, std::move(regs), static_cast<double>(K) * rows * rb) < 0) return -1;
  }
  return static_cast<int>(p->ops.size()) - 1;
}

int lx_program_attention(lx_program* p, const lx_attention_args* a) {
  if (!p || !a || a->KVH < 1 || a->H % a->KVH || a->H / a->KVH > 8 || a->Dh % 8 || kComputeThreads % a->Dh) return -1;
  const int M = p->M, G = a->H / a->KVH;
  OpDesc d = blank(kAttention, M * a->KVH);
  d.attn = {a->qkv, a->k_cache, a->v_cache, a->out, a->scale, a->ld_qkv, a->capacity, a->H, a->KVH, a->Dh};
  p->x_need = std::max(p->x_need, (G * a->Dh + kComputeThreads * G) * 4);
  std::vector<Region> regs = {rd(a->qkv, span(M, a->ld_qkv, (a->H + 2 * a->KVH) * a->Dh, 4)), rd(a->k_cache, 1),
                              wr(a->k_cache, 1), rd(a->v_cache, 1), wr(a->v_cache, 1),
                              wr(a->out, static_cast<size_t>(M) * a->H * a->Dh * 2)};
  return p->add(d, std::move(regs), 16384.0);
}

int lx_program_memory_entry(lx_program* p, const lx_entry_args* a) {
  if (!p || !a || a->D % 8 || (a->R + 4) % 4) return -1;
  const int M = p->M, D = a->D, Km = a->R + 4, wt = static_cast<int>(a->w_state.type);
  if (wt != LX_W_BF16 && wt != LX_W_INT8) return -1;
  OpDesc d = blank(kEntry, 0);
  Entry& e = d.entry;
  e.state = a->state;
  e.w_state = a->w_state;
  e.b_state = a->b_state;
  e.summary = a->summary;
  e.entropy = a->entropy;
  e.confidence = a->confidence;
  e.continuation = a->continuation;
  e.w_meta = a->w_meta;
  e.b_meta = a->b_meta;
  e.pass_emb = a->pass_emb;
  e.anchor_emb = a->anchor_emb;
  e.entry = a->entry;
  e.mean_k = a->mean_k;
  e.inv_log_e = a->inv_log_e;
  e.wtype = wt;
  e.D = D;
  e.R = a->R;
  e.mem_dim = a->mem_dim;
  e.state_bytes = D * bytes_per_value(wt);
  const int mb = Km * 2;
  const int rows = tile_rows(a->mem_dim, e.state_bytes + mb, p->blocks);
  e.rows = rows;
  int kc = D;
  while (kMaxM * (Km + kc) * 4 > p->x_cap && kc % 16 == 0) kc /= 2;  // whatever M is
  if (kMaxM * (Km + kc) * 4 > p->x_cap || D % kc) return -1;
  e.kc = kc;
  d.ntasks = ceil_div(a->mem_dim, rows);
  d.pool = kStatic;
  p->x_need = std::max(p->x_need, M * (Km + kc) * 4);
  std::vector<Region> regs = {rd(a->state, static_cast<size_t>(M) * D * 4),
                              rd(a->summary, static_cast<size_t>(M) * a->R * 4),
                              rd(a->entropy, static_cast<size_t>(M) * 4), rd(a->confidence, static_cast<size_t>(M) * 4),
                              rd(a->continuation, static_cast<size_t>(M) * 4),
                              wr(a->entry, static_cast<size_t>(M) * a->mem_dim * 4)};
  return p->add(d, std::move(regs), static_cast<double>(rows) * (e.state_bytes + mb));
}

int lx_program_continuation(lx_program* p, const float* hidden, const float* w_out, const float* b_out,
                            const float* rms, float limit, float* out, int R, int D) {
  if (!p) return -1;
  const int M = p->M;
  OpDesc d = blank(kContinuation, 1);
  d.cont = {hidden, w_out, b_out, rms, out, limit, R, D};
  std::vector<Region> regs = {rd(hidden, static_cast<size_t>(M) * R * 4), rd(rms, static_cast<size_t>(2) * M * 4),
                              wr(out, static_cast<size_t>(M) * 4)};
  return p->add(d, std::move(regs), 4096.0);
}

// ---- Layout and launch --------------------------------------------------------------------------

// What a static task reads that the step does not write, for its block to have L2 fetch ahead of
// time: its weight rows (and their scales), or the Mamba state of its head.
static void task_spans(const OpDesc& d, int task, std::vector<PfSpan>& out) {
  auto span = [&](const void* a, int64_t bytes, int m = 0, int64_t stride = 0) {
    if (bytes <= 0) return;
    PfSpan s{};
    s.addr = reinterpret_cast<uint64_t>(a);
    s.bytes = static_cast<uint32_t>(bytes);
    s.m = m;
    s.stride = static_cast<uint64_t>(stride);
    out.push_back(s);
  };
  switch (d.kind) {
    case kGemv: {
      const Gemv& g = d.gemv;
      const int n0 = task * g.rows, rows = std::min(g.rows, g.N - n0);
      const int64_t bpv = g.row_bytes / (g.k1 - g.k0);
      const char* w = static_cast<const char*>(g.w.data);
      // a K chunk's rows are not contiguous: the span runs from the first row's chunk to the last
      // row's (the other chunks' bytes in between are read by their own ops right after)
      const int64_t a = (static_cast<int64_t>(n0) * g.K + g.k0) * bpv;
      const int64_t b = (static_cast<int64_t>(n0 + rows - 1) * g.K + g.k1) * bpv;
      span(w + a, b - a);
      if (row_scaled_type(g.wtype)) span(static_cast<const float*>(g.w.scale) + n0, rows * 4);
      break;
    }
    case kSsm: {
      const Ssm& a = d.ssm;
      const int64_t head = static_cast<int64_t>(a.P) * a.N * 4;
      span(reinterpret_cast<const char*>(a.ssm_state) + (task % a.H) * head, head, task / a.H + 1, a.H * head);
      break;
    }
    case kMemRead: {
      // One token, one block: everything it reads from far away, so the block's single pass over it
      // finds it in L2 (a prefetch is only a hint, so an entry written later in the step is fine).
      const MemRead& a = d.mem;
      const int64_t D = a.D;
      if (a.n_entries > 0) {
        span(static_cast<const float*>(a.bank) + static_cast<int64_t>(task) * a.token_stride,
             static_cast<int64_t>(a.n_entries) * a.entry_stride * 4);
        span(a.gate, a.S * D * 2);
      }
      if (a.previous) span(a.previous + task * D, D * 4);
      if (a.h) span(a.norm_w, D * 2);
      if (a.lora_down) span(a.lora_down, a.rank * D * 2);
      break;
    }
    case kSelect: {
      const Select& a = d.sel;
      span(a.expert_emb, static_cast<int64_t>(a.E) * a.R * 2);
      if (a.selection_bias) span(a.selection_bias, static_cast<int64_t>(a.E) * 4);
      break;
    }
    case kEntry: {
      const Entry& a = d.entry;
      const int r0 = task * a.rows, rows = std::min(a.rows, a.mem_dim - r0);
      span(static_cast<const char*>(a.w_state.data) + static_cast<int64_t>(r0) * a.state_bytes,
           static_cast<int64_t>(rows) * a.state_bytes);
      span(static_cast<const char*>(a.w_meta) + static_cast<int64_t>(r0) * (a.R + 4) * 2,
           static_cast<int64_t>(rows) * (a.R + 4) * 2);
      if (row_scaled_type(a.wtype)) span(static_cast<const float*>(a.w_state.scale) + r0, rows * 4);
      break;
    }
    default: break;
  }
}

int lx_program_build(lx_program* p) {
  if (!p || p->built || p->failed || p->ops.empty()) return static_cast<int>(cudaErrorInvalidValue);
  const int nops = static_cast<int>(p->ops.size()), B = p->blocks;
  if (nops >= (1 << 15)) return static_cast<int>(cudaErrorInvalidValue);
  for (const OpDesc& d : p->ops)
    if (d.ntasks < 1 || d.ntasks > 0xffff) return static_cast<int>(cudaErrorInvalidValue);

  // Shared memory: the fixed slots, then the x buffer the ops need.
  p->x_bytes = round_up(std::max(p->x_need, 1024), 128);
  if (p->x_bytes > p->x_cap) return static_cast<int>(cudaErrorInvalidConfiguration);
  p->smem = step_smem_bytes(p->x_bytes);
  if (int e = step_configure(p->smem)) return e;

  // Tasks to blocks. The ops of a step mostly wait on each other, so what matters is that each op
  // is even over the blocks: every block gets the same number of its tasks, the remainder going to
  // the blocks with the least work so far (ties in turn), each block a contiguous range of tasks.
  std::vector<double> load(B, 0.0);
  std::vector<std::vector<uint32_t>> lists(B);
  std::vector<int> order(B), count(B);
  int turn = 0;
  for (int op = 0; op < nops; ++op) {
    OpDesc& d = p->ops[op];
    for (int k = 0; k < B; ++k) order[k] = (turn + k) % B;
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return load[a] < load[b]; });
    const int q = d.ntasks / B, r = d.ntasks % B;
    for (int k = 0; k < B; ++k) count[order[k]] = q + (k < r ? 1 : 0);
    int t = 0;
    d.nblocks = 0;
    for (int b = 0; b < B; ++b) {
      if (!count[b]) continue;
      ++d.nblocks;
      for (int i = 0; i < count[b]; ++i, ++t)
        lists[b].push_back(static_cast<uint32_t>(op) << 16 | static_cast<uint32_t>(t));
      load[b] += count[b] * p->cost[op];
    }
    if (r) turn = (order[r - 1] + 1) % B;
  }
  // Each dependency carries its op's block count, so a block waits for it from its descriptor alone.
  for (OpDesc& d : p->ops)
    for (int k = 0; k < d.ndeps; ++k) {
      const int dep = d.deps[k];
      d.deps[k] = dep | p->ops[dep].nblocks << 16;
    }
  // The flattened lists; per block its prefetch spans (adjacent ones merged up to 32 KB) and per
  // entry the block's prefetch bytes before it.
  std::vector<uint32_t> flat, ahead;
  std::vector<int32_t> off(B + 1, 0), soff(B + 1, 0);
  std::vector<PfSpan> spans, tmp;
  for (int b = 0; b < B; ++b) {
    off[b] = static_cast<int32_t>(flat.size());
    soff[b] = static_cast<int32_t>(spans.size());
    uint32_t sum = 0;
    for (uint32_t e : lists[b]) {
      flat.push_back(e);
      ahead.push_back(sum);
      const OpDesc& d = p->ops[e >> 16];
      if (d.pool != kStatic) continue;
      tmp.clear();
      task_spans(d, static_cast<int>(e & 0xffff), tmp);
      for (PfSpan s : tmp) {
        PfSpan* last = spans.size() > static_cast<size_t>(soff[b]) ? &spans.back() : nullptr;
        if (last && !s.m && !last->m && last->addr + last->bytes == s.addr && last->bytes + s.bytes <= 32768) {
          last->bytes += s.bytes;
        } else {
          s.cum = sum;
          spans.push_back(s);
        }
        sum += s.bytes;
      }
    }
  }
  soff[B] = static_cast<int32_t>(spans.size());
  if (spans.empty()) spans.push_back(PfSpan{});  // so the upload has something to copy
  off[B] = static_cast<int32_t>(flat.size());
  p->total_tasks = off[B];
  p->bulk = lx_arch() >= 90;
  // How far ahead of the compute warps the loader keeps a block's weights in L2 (LX_PROGRAM_AHEAD_KB)
  p->prefetch_bytes = 192 * 1024;
  if (const char* e = std::getenv("LX_PROGRAM_AHEAD_KB")) p->prefetch_bytes = std::max(0, std::atoi(e)) * 1024;

  auto upload = [](void** dst, const void* src, size_t bytes) {
    if (cudaMalloc(dst, bytes) != cudaSuccess) return false;
    return cudaMemcpy(*dst, src, bytes, cudaMemcpyHostToDevice) == cudaSuccess;
  };
  if (!upload(reinterpret_cast<void**>(&p->d_ops), p->ops.data(), sizeof(OpDesc) * nops) ||
      !upload(reinterpret_cast<void**>(&p->d_lists), flat.data(), flat.size() * 4) ||
      !upload(reinterpret_cast<void**>(&p->d_off), off.data(), off.size() * 4) ||
      !upload(reinterpret_cast<void**>(&p->d_ahead), ahead.data(), ahead.size() * 4) ||
      !upload(reinterpret_cast<void**>(&p->d_spans), spans.data(), spans.size() * sizeof(PfSpan)) ||
      !upload(reinterpret_cast<void**>(&p->d_span_off), soff.data(), soff.size() * 4) ||
      cudaMalloc(&p->d_done, (nops + 1) * 4) != cudaSuccess)
    return static_cast<int>(cudaErrorMemoryAllocation);
  cudaMemset(p->d_done, 0, (nops + 1) * 4);
  if (std::getenv("LX_PROGRAM_TRACE")) {
    if (cudaMalloc(&p->d_trace, static_cast<size_t>(p->total_tasks) * 2 * 4 * 8) != cudaSuccess)
      return static_cast<int>(cudaErrorMemoryAllocation);
    cudaMemset(p->d_trace, 0, static_cast<size_t>(p->total_tasks) * 2 * 4 * 8);
  }
  p->built = true;
  return static_cast<int>(cudaGetLastError());
}

int lx_program_run(const lx_program* p, void* stream) {
  if (!p || !p->built) return static_cast<int>(cudaErrorInvalidValue);
  Program P;
  P.ops = p->d_ops;
  P.lists = p->d_lists;
  P.list_off = p->d_off;
  P.ahead = p->d_ahead;
  P.spans = p->d_spans;
  P.span_off = p->d_span_off;
  P.done = p->d_done;
  P.trace = p->d_trace;
  P.tokens = p->tokens;
  P.slots = p->slots;
  P.has_state = p->has_state;
  P.start_pos = p->start_pos;
  P.nops = static_cast<int>(p->ops.size());
  P.M = p->M;
  P.x_bytes = p->x_bytes;
  P.bulk = p->bulk ? 1 : 0;
  P.prefetch_bytes = p->prefetch_bytes;
  return step_launch(P, p->blocks, p->smem, static_cast<cudaStream_t>(stream));
}


int lx_program_trace(const lx_program* p, int64_t* out, int capacity) {
  if (!p || !p->built || !p->d_trace || !out) return 0;
  const int n = std::min(capacity, 2 * p->total_tasks);
  if (cudaMemcpy(out, p->d_trace, static_cast<size_t>(n) * 4 * 8, cudaMemcpyDeviceToHost) != cudaSuccess) return 0;
  return n;
}

}  // extern "C"
