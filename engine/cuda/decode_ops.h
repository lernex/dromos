// A decode program as the persistent kernel reads it (decode_kernel.cu) and the builder writes it
// (decode_program.cu): one descriptor per op, and per block the list of tasks it runs in order.
// Plain data only; kernels.h ("Decode programs") is the contract these implement.
#pragma once

#include <cuda_runtime.h>
#include <stddef.h>
#include <stdint.h>

#include "cuda/kernels.h"

namespace lx::decode {

// One block per SM: 16 compute warps and one loader warp, which keeps L2 ahead of them.
constexpr int kComputeWarps = 16;
constexpr int kComputeThreads = 32 * kComputeWarps;
constexpr int kBlockThreads = kComputeThreads;  // 16 warps: four per scheduler, so 128 registers each
constexpr int kMaxM = 8;                     // tokens per step
constexpr int kMaxDeps = 32;                 // ops a task waits on, after the transitive reduction
constexpr int kMaxStreams = 4;               // hyper-connection streams (Metis runs 4)
constexpr int kMaxEntries = 8;               // depth-memory entries per token
constexpr int kTileBytes = 16384;            // weight bytes of one product task, at most (one row may be more)
constexpr int kMaxTileRows = 64;             // rows of a product task
constexpr int kScratchFloats = 6144;         // per-block reduction space
constexpr int kDescBytes = 512;              // shared-memory copy of an op's descriptor
constexpr int kConvParts = 8;                // tasks per token of a B and C conv-state update

// The kernel's shared memory: fixed slots, then the x buffer (as large as the rest allows).
constexpr int kCtxBytes = 128;
constexpr int kCtxOff = 0;
constexpr int kDescOff = kCtxOff + kCtxBytes;
constexpr int kMaxRunTasks = 256;  // tasks of one run (a block's longer run of an op is cut into runs)
constexpr int kMiscOff = kDescOff + kDescBytes;
constexpr int kRunOff = kMiscOff + 128;  // the run's task numbers, uint16
constexpr int kScratchOff = kRunOff + kMaxRunTasks * 2;
constexpr int kXOff = kScratchOff + kScratchFloats * 4;

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

// Whether the loader can have L2 fetch a task's weights ahead of time: static ops know their addresses
// when the program is built; a dynamic op's (the expert rows) come from the routing op in dyn_dep.
enum Pool : int32_t { kNoPool = 0, kStatic = 1, kDynamic = 2 };

struct Gemv {
  lx_weight w;
  int wtype;            // w.type (LX_W_F32 included)
  int N, K, k0, k1;     // this op multiplies columns [k0, k1) of the [N, K] weight
  int rows;             // rows per task (the last task may have fewer)
  int row_bytes;        // bytes of one row's [k0, k1)
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
  // the reader: 0 FMA (x f32 or bf16); tensor cores on 1 bf16 (x bf16), 2 int8 as f16 (x f16), and 3, 4
  // the same for f32 x, staged as two 16-bit planes (hi + lo) read one product each
  int mma;
  int sh;   // each row's k is read in 2^sh parts, summed in order: fixed by the shape, never by M
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

// Expert gate|up rows of a routed expert: task = (pair, row block), the expert from ids, x = the
// pair's token. On the tensor cores (mma), a task is 8 gate rows and the same 8 up rows, one unit
// whose k is read in 2^sh parts.
struct ExpUp {
  const void* latent;
  const int32_t* ids;
  lx_weight w;          // [E, 2I, L]
  void* inter;          // bf16 [pairs, I], pair = m * top_k + k
  int wtype, L, I, top_k, mma, sh, rows, row_bytes, blocks_per_pair;
};

// Expert down rows for one token: rows [l0, l0 + rows) of each of the token's top_k experts. On the
// tensor cores (mma), 16 rows: a unit per expert, k in 2^sh parts.
struct ExpDown {
  const void* inter;
  const int32_t* ids;
  const float* route_w;
  lx_weight w;          // [E, L, I]
  float* out;           // f32 [M, L]
  const float* add;     // f32 [M, L] added after the routed sum (the shared expert), or null
  int wtype, L, I, top_k, mma, sh, rows, row_bytes, blocks_per_token;
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
  int wtype, D, R, mem_dim, rows, state_bytes;
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
// last task of the op, so an op is complete when its counter reaches nblocks. waited: another op
// waits on this one's counter (an op nothing waits on signals nothing). Copied to shared memory 16
// bytes at a time, so its size is a multiple of 16.
struct alignas(16) OpDesc {
  int32_t kind, ntasks, nblocks, ndeps, pool, dyn_dep, waited;
  int32_t deps[kMaxDeps];  // op | (that op's nblocks << 16): the counters a task waits for
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

// An L2 prefetch of [addr, addr + bytes); for the Mamba state of token m - 1 (m > 0), at
// addr + slots[m - 1] * stride, when the token has state. cum: the block's prefetch bytes before it.
struct PfSpan {
  uint64_t addr;
  uint64_t stride;
  uint32_t bytes;
  int32_t m;
  uint32_t cum;
  uint32_t pad;
};

// The kernel's argument.
struct Program {
  const OpDesc* ops;
  const uint32_t* lists;     // per block: (op << 16) | task, in run order
  const int32_t* list_off;   // [blocks + 1]
  const uint32_t* ahead;     // per list entry: the block's prefetch bytes for the entries before it
  const PfSpan* spans;       // per block, in list order
  const int32_t* span_off;   // [blocks + 1]
  uint32_t* done;            // [nops + 1]: blocks finished per op, then blocks finished with the step
  int64_t* trace;            // optional: the timestamps of lx_program_trace
  const int32_t* tokens;
  const int32_t* slots;
  const int32_t* has_state;
  const int32_t* start_pos;
  int nops, M, x_bytes;
  int bulk;                  // L2 prefetches as bulk prefetches (sm_90 and newer), else per line
  int prefetch_bytes;        // how far past a run's start its block keeps L2 fetched (0: off)
};

// decode_kernel.cu: the kernel's shared memory for a layout, its attributes, and the launch.
size_t step_smem_bytes(int x_bytes);
int step_configure(size_t smem);
int step_launch(const Program& P, int blocks, size_t smem, cudaStream_t stream);

}  // namespace lx::decode
