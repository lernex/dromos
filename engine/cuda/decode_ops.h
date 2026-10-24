// A decode program as the persistent kernel reads it (decode_kernel.cu) and the builder writes it
// (decode_program.cu): one descriptor per op, and per block the list of tasks it runs in order.
// Plain data only; kernels.h ("Decode programs") is the contract these implement.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "cuda/kernels.h"

namespace lx::decode {

// One block per SM: 16 compute warps, one epilogue warp (the products' outputs), one loader warp.
constexpr int kComputeWarps = 16;
constexpr int kComputeThreads = 32 * kComputeWarps;
constexpr int kEpilogueWarp = kComputeWarps;
constexpr int kLoaderWarp = kComputeWarps + 1;
constexpr int kBlockThreads = kComputeThreads + 64;
constexpr int kMaxM = 8;                     // tokens per step
constexpr int kMaxDeps = 32;                 // ops a task waits on, after the transitive reduction
constexpr int kMaxStreams = 4;               // hyper-connection streams (Metis runs 4)
constexpr int kMaxEntries = 8;               // depth-memory entries per token
constexpr int kPageData = 16384;             // a page holds up to 16 KB of weight rows...
constexpr int kPageBytes = kPageData + 256;  // ...and their per-row scales
constexpr int kScratchFloats = 1024;         // per-block reduction space
constexpr int kMaxTileRows = 64;             // rows of a product tile; rows * M <= kComputeThreads too
constexpr int kDescBytes = 512;              // shared-memory copy of the current op's descriptor
constexpr int kMaxPages = 16;                // pages in both rings together

// The kernel's shared memory: fixed slots, then the x buffer, then the pages.
constexpr int kCtxBytes = 128;
constexpr int kCtxOff = 0;
constexpr int kDescOff = kCtxOff + kCtxBytes;
constexpr int kLoaderDescOff = kDescOff + kDescBytes;
constexpr int kScratchOff = kLoaderDescOff + 3 * kDescBytes;
constexpr int kBarOff = kScratchOff + kScratchFloats * 4;
constexpr int kBarCount = 2 * kMaxPages + 4;  // full, empty per page; the products' reduction slots: ready, free
constexpr int kMiscOff = kBarOff + kBarCount * 8;
constexpr int kXOff = (kMiscOff + 16 + 127) / 128 * 128;
__host__ __device__ constexpr int pages_offset(int x_bytes) { return kXOff + (x_bytes + 127) / 128 * 128; }

enum Kind : int32_t {
  kEmbed,
  kMemRead,
  kGemv,
  kSsm,
  kConvBC,
  kSelect,
  kExpUp,
  kExpDown,
  kAttention,
  kEntry,
  kContinuation,
};

// Where a task's weight tile goes: the static ring (addresses known when the program is built, so the
// loader streams them as far ahead as pages allow) or the dynamic ring (expert rows, whose addresses
// come from the routing; the loader waits for the op named in dyn_dep before it can issue them).
enum Pool : int32_t { kNoPool = 0, kStatic = 1, kDynamic = 2 };

struct Gemv {
  lx_weight w;
  int wtype;            // w.type (LX_W_F32 included)
  int N, K, k0, k1;     // this op multiplies columns [k0, k1) of the [N, K] weight
  int rows;             // rows per tile (the last tile may have fewer)
  int row_bytes;        // bytes of one row's [k0, k1) in the tile
  int scale_off;        // byte offset of the per-row scales in the page (INT8 / FP8)
  // x
  int x_kind, x_dtype, ld_x, n_parts, group, nseg, round_bf16, xs_bf16;
  const void* x;
  const float* parts;
  const void* norm_w;
  float eps;
  const float* seg[4];
  int seg_end[4];       // cumulative widths
  // y
  int y_kind, out_dtype, ld_out, bias_dtype, act, ld_bf, rank, S;
  float alpha, beta, lora_gate;
  void* out;
  const void* bias;
  void* out_bf;
  const void* lora_up;
  const float* lora;
  float* streams;
  const float* matrix;
  const float* write;
  float* state;
  const float* next_read;
  float* next_src;
  float* next_parts;
  // K-split partials: y += sum_c y_add[c * M * N + m * N + n] before the epilogue (null for none)
  const float* y_add;
  int n_add;
};

struct Embed {
  const void* table;
  const void* stream_emb;
  float* streams;
  float* previous;
  float* history;
  float* continuation;
  int S, D, R, chunks;
};

struct MemRead {
  float* streams;
  const float* bank;
  const void* gate;
  const void* gate_bias;
  float* state;
  float* summary;
  const float* read;
  const void* norm_w;
  void* h;
  const void* lora_down;
  float* lora;
  float* previous;
  float* difference;
  float* rms;
  float eps;
  int n_entries, entry_stride, token_stride, mem_dim, rank, S, D;
};

struct Ssm {
  const float* zxbcdt;
  const void* conv_w;
  const void* conv_b;
  float* conv_state;
  float* ssm_state;
  const float* A_log;
  const float* D;
  const float* dt_bias;
  float* v;
  float* head_ss;
  int ld, I, H, P, G, N, W;
};

struct Select {
  const float* logits;
  const float* selection_bias;
  int32_t* ids;
  float* weights;
  float* entropy;
  float* confidence;
  const void* expert_emb;
  float* summary;
  float* history;
  int E, top_k, R;
};

// Expert gate|up rows. Routed: task = (pair, row block), the expert from ids, x = the pair's token.
// Shared: task = row block, every token.
struct ExpUp {
  const void* latent;
  const int32_t* ids;
  lx_weight w;          // [E, 2I, L] (routed) or [2I, L] (shared)
  void* inter;          // bf16 [pairs, I] (routed: pair = m * top_k + k; shared: pair = m)
  int wtype, L, I, top_k, shared, rows, row_bytes, scale_off, blocks_per_pair;
};

// Expert down rows for one token: rows [l0, l0 + rows) of each of the token's top_k experts.
struct ExpDown {
  const void* inter;
  const int32_t* ids;
  const float* route_w;
  lx_weight w;          // [E, L, I]
  float* out;           // f32 [M, L]
  const float* add;     // f32 [M, L] added after the routed sum (the shared expert), or null
  int wtype, L, I, top_k, rows, row_bytes, scale_off, blocks_per_token;
};

struct Attention {
  const float* qkv;
  void* k_cache;
  void* v_cache;
  void* out;
  float scale;
  int ld_qkv, capacity, H, KVH, Dh;
};

struct Entry {
  const float* state;
  lx_weight w_state;
  const void* b_state;
  const float* summary;
  const float* entropy;
  const float* confidence;
  const float* continuation;
  const void* w_meta;
  const void* b_meta;
  const void* pass_emb;
  const void* anchor_emb;
  float* entry;
  float mean_k, inv_log_e;
  int wtype, D, R, mem_dim, rows, state_bytes, scale_off, meta_off;
  int kc;  // columns of the state staged at a time (D when it fits the x buffer at once)
};

struct Continuation {
  const float* hidden;
  const float* w_out;
  const float* b_out;
  const float* rms;
  float* out;
  float limit;
  int R, D;
};

// nblocks: the blocks that run any of the op's tasks. Each signals the op's counter once, after its
// last task of the op, so an op is complete when its counter reaches nblocks.
struct OpDesc {
  int32_t kind, ntasks, nblocks, ndeps, pool, npages, dyn_dep;
  int32_t deps[kMaxDeps];
  union {
    Gemv gemv;
    Embed embed;
    MemRead mem;
    Ssm ssm;
    Select sel;
    ExpUp up;
    ExpDown down;
    Attention attn;
    Entry entry;
    Continuation cont;
  };
};
static_assert(sizeof(OpDesc) <= kDescBytes, "an op descriptor must fit its shared-memory copy");

// The kernel's argument.
struct Program {
  const OpDesc* ops;
  const uint32_t* lists;     // per block: (op << 16) | task, in run order
  const int32_t* list_off;   // [blocks + 1]
  // the same lists split by pool, for the loader: static tiles, and dynamic ones
  const uint32_t* static_lists;
  const int32_t* static_off;
  const uint32_t* dynamic_lists;
  const int32_t* dynamic_off;
  uint32_t* done;            // [nops + 1]: tasks finished per op, then blocks finished
  int64_t* trace;            // optional: per list entry (op << 32 | block, wait start, start, end)
  const int32_t* tokens;
  const int32_t* slots;
  const int32_t* has_state;
  const int32_t* start_pos;
  int nops, M, pages_static, pages_dynamic, x_bytes;
  int bulk;      // copy tiles with bulk copies (sm_90 and newer), else with cp.async (sm_80)
  int prefetch;  // static tiles the loader asks L2 for ahead of the ring (0: none)
};

// decode_kernel.cu: the kernel's shared memory for a layout, its attributes, and the launch.
size_t step_smem_bytes(int npages, int x_bytes);
int step_configure(size_t smem);
int step_launch(const Program& P, int blocks, size_t smem, cudaStream_t stream);

}  // namespace lx::decode
