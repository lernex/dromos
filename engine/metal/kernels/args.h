// Kernel argument structs, shared by the host (C++) and the kernels (Metal Shading Language). Each
// kernel takes one of these in the constant address space at buffer(0). Device memory is passed as GPU
// addresses: a 64-bit integer on the host, a typed device pointer in the kernel, so both sides lay the
// struct out the same way (pointers first, 8-byte aligned, then 4-byte scalars).
#ifndef __METAL_VERSION__
#pragma once
#endif

#ifdef __METAL_VERSION__
#define LX_PTR(T) device T*
#define LX_CPTR(T) device const T*
typedef long lx_i64;
typedef ulong lx_u64;
#else
#include <stdint.h>
#define LX_PTR(T) uint64_t
#define LX_CPTR(T) uint64_t
typedef int64_t lx_i64;
typedef uint64_t lx_u64;
typedef uint32_t uint;
#endif

// ---- runtime --------------------------------------------------------------------------------
struct CopyArgs {
  LX_CPTR(uchar) src;
  LX_PTR(uchar) dst;
  lx_i64 bytes;
};
struct FillArgs {
  LX_PTR(uchar) dst;
  lx_i64 bytes;
  uint value;
};

// ---- elementwise, norms ---------------------------------------------------------------------
struct RmsnormArgs {
  LX_CPTR(void) x;
  LX_CPTR(ushort) w;
  LX_PTR(void) y;
  int dim;
  float eps;
};
struct Copy2dArgs {
  LX_CPTR(void) src;
  LX_PTR(void) dst;
  int ld_src, ld_dst, rows, cols;
};
struct ActivateArgs {
  LX_PTR(float) y;
  lx_i64 n;
};
struct SiluMulArgs {
  LX_CPTR(float) x;
  LX_PTR(void) out;
  int ld_x, rows, cols;
};
struct AddRowArgs {
  LX_PTR(float) y;
  LX_CPTR(ushort) row;
  int rows, cols;
};
struct AxpbyArgs {
  LX_PTR(float) y;
  LX_CPTR(float) x;
  lx_i64 n;
  float a, b;
};
struct SubArgs {
  LX_CPTR(float) a;
  LX_CPTR(float) b;
  LX_PTR(float) out;
  lx_i64 n;
};
struct MetaScalarsArgs {
  LX_CPTR(float) entropy;
  LX_CPTR(float) confidence;
  LX_CPTR(float) continuation;
  LX_PTR(ushort) meta;
  float mean_k, inv_log_experts;
  int ld, col, T;
};
struct GatherLastArgs {
  LX_CPTR(float) src;
  LX_CPTR(int) cu;
  LX_PTR(float) out;
  int D;
};

// ---- streams --------------------------------------------------------------------------------
struct EmbedArgs {
  LX_CPTR(ushort) table;
  LX_CPTR(int) ids;
  LX_CPTR(ushort) stream_emb;
  LX_PTR(float) streams;
  int S, D;
};
struct StreamMeanArgs {
  LX_CPTR(float) streams;
  LX_PTR(float) mean;
  int T, S, D;
};
struct MhcReadArgs {
  LX_CPTR(float) streams;
  LX_CPTR(float) matrix;
  LX_CPTR(float) read;
  LX_CPTR(ushort) norm_w;
  LX_PTR(float) mixed;
  LX_PTR(ushort) h;
  float eps;
  int S, D, split;  // split: threadgroups per token (the decode form), 1 for one per token
};
struct MhcWriteArgs {
  LX_CPTR(float) mixed;
  LX_CPTR(float) write;
  LX_CPTR(float) update;
  LX_PTR(float) streams;
  int T, S, D;
};

// ---- depth memory ---------------------------------------------------------------------------
struct MemoryAttendArgs {
  LX_CPTR(float) q;
  LX_CPTR(float) bank_k;
  LX_CPTR(float) bank_v;
  LX_PTR(ushort) out;
  int n_entries, max_entries, rows, S, M;
};
struct MemoryFuseArgs {
  LX_PTR(float) streams;
  LX_CPTR(float) projected;
  LX_CPTR(ushort) gate;
  LX_CPTR(ushort) gate_bias;
  LX_PTR(float) summary;
  int S, D, split;
};
struct ContinuationArgs {
  LX_CPTR(float) logit;
  LX_CPTR(float) state;
  LX_CPTR(float) difference;
  LX_PTR(float) out;
  float limit;
  int D;
};

// ---- Mamba-2 --------------------------------------------------------------------------------
struct ConvArgs {
  LX_CPTR(float) x;
  LX_CPTR(ushort) w;
  LX_CPTR(ushort) bias;
  LX_PTR(float) state;
  LX_CPTR(int) slot_ids;
  LX_CPTR(int) has_state;
  LX_CPTR(int) cu;
  LX_PTR(float) out;
  int ld_x, C, W;
};
struct ScanArgs {
  LX_CPTR(float) x;
  LX_CPTR(float) dt_raw;
  LX_CPTR(float) B;
  LX_CPTR(float) C;
  LX_CPTR(float) A_log;
  LX_CPTR(float) D;
  LX_CPTR(float) dt_bias;
  LX_PTR(float) state;
  LX_CPTR(int) slot_ids;
  LX_CPTR(int) has_state;
  LX_CPTR(int) cu;
  LX_PTR(float) y;
  int ld_x, ld_dt, ld_bc, H, P, G, N;
};
struct GatedNormArgs {
  LX_CPTR(float) y;
  LX_CPTR(float) z;
  LX_CPTR(ushort) w;
  LX_PTR(ushort) out;
  int ld_z, I, groups;
  float eps;
};

// ---- attention ------------------------------------------------------------------------------
struct AttnArgs {
  LX_CPTR(float) qkv;
  LX_PTR(ushort) k_cache;
  LX_PTR(ushort) v_cache;
  LX_CPTR(int) slot_ids;
  LX_CPTR(int) start_pos;
  LX_CPTR(int) cu;
  LX_PTR(ushort) out;
  LX_PTR(float) partial;  // split decode: [items, splits, G, Dh + 2] partial softmax states
  int ld_qkv, n_seq, capacity, H, KVH, Dh;
  float scale;
  int splits;  // threadgroups per (token, kv head); 1 for whole items
  int tile;    // cache positions per split
  int flash;   // nax/attention.metal takes the prefill items (see flash_takes)
};

// ---- experts and routing --------------------------------------------------------------------
struct RouteArgs {
  LX_CPTR(float) logits;
  LX_CPTR(float) bias;
  LX_PTR(int) ids;
  LX_PTR(float) weights;
  LX_PTR(float) entropy;
  LX_PTR(float) confidence;
  int T, E, top_k;
};
struct SummaryArgs {
  LX_CPTR(int) ids;
  LX_CPTR(float) weights;
  LX_CPTR(ushort) emb;
  LX_PTR(float) summary;
  int T, R, top_k;
};

// A weight in one of the formats of kernels.h ("Weight formats"), as kernels see it.
struct WeightRef {
  LX_CPTR(uchar) data;
  LX_CPTR(uchar) scale;          // f32 [N] (INT8, FP8), e4m3 [N, K / 16] (NVFP4), unused for BF16
  LX_CPTR(float) expert_global;  // NVFP4 stacks: f32 [E], or 0 for `global`
  float global;
  int type;
};

// Few-token experts. links [2P]: per pair its rank among the pairs on the same expert, then the next
// such pair (or -1). A pair whose rank is a multiple of 8 leads a batch of up to 8 pairs that share
// every weight read.
struct LinkArgs {
  LX_CPTR(int) ids;
  LX_PTR(int) links;
  int P;
};
// Stage 1: inter[p, i] = bf16(silu(gate . x) * (up . x)) for pair p = t * top_k + k.
struct ExpertUpArgs {
  WeightRef w;  // gate_up stack [E, 2I, L]
  LX_CPTR(ushort) latent;
  LX_CPTR(int) ids;
  LX_CPTR(int) links;
  LX_PTR(ushort) inter;
  int L, I, top_k, P;
};
// Stage 2: out[t, l] = sum_k weights[t, k] * (down[ids[t, k]][l] . inter[t * top_k + k]).
struct ExpertDownArgs {
  WeightRef w;  // down stack [E, L, I]
  LX_CPTR(ushort) inter;
  LX_CPTR(int) ids;
  LX_CPTR(float) weights;
  LX_PTR(float) out;
  int T, L, I, top_k;
};

// Grouped experts (any T): pairs bucketed by expert in the workspace, then a GEMM per expert tile.
struct GroupArgs {
  LX_CPTR(int) ids;
  LX_PTR(int) counts;  // [E + 1]: the start of each expert's pairs in `order`, then P
  LX_PTR(int) order;   // [P]: pair indices, grouped by expert, in pair order within an expert
  LX_PTR(int) tiles;   // [1 + max_tiles * 3]: the tile count, then (expert, first row in order, rows)
  int P, E, tile_rows, max_tiles;
};
struct CombineArgs {
  LX_CPTR(float) rows;  // [P, L] per-pair down outputs, already weighted
  LX_PTR(float) out;    // [T, L]
  int T, L, top_k;
};

// ---- products -------------------------------------------------------------------------------
struct GemvArgs {
  WeightRef w;
  LX_CPTR(void) x;
  LX_CPTR(void) bias;
  LX_PTR(void) y;
  int ld_x, ld_y, M, N, K;
  float alpha, beta;
  int x_bf16, bias_bf16, y_bf16, split;  // split: threadgroups sharing one row group's K range
};
// The tile GEMM (gemm.metal): y = alpha * (x @ W^T + bias) + beta * y, x in one of the activation
// formats, W in one of the weight formats. In the grouped forms the rows of a tile are pairs, gathered
// through `order`, and the weight is the tile's expert's.
struct GemmArgs {
  WeightRef w;
  LX_CPTR(uchar) x;        // activations: bf16 / f32 rows of ld_x elements, or quantized rows of ld_x bytes
  LX_CPTR(float) x_scale;  // quantized activations: the per-row scale (the global for FP4)
  LX_CPTR(uchar) x_block;  // FP4 activations: e4m3 block scales, rows of ld_xb bytes
  LX_CPTR(void) bias;
  LX_PTR(void) y;
  LX_CPTR(int) order;      // grouped: pairs by expert
  LX_CPTR(int) tiles;      // grouped: tile list (GroupArgs)
  LX_CPTR(float) pair_w;   // grouped down: the routing weight of each pair
  int ld_x, ld_xb, ld_y, M, N, K;
  float alpha, beta;
  int bias_bf16, y_bf16, x_vec;  // x_vec: activation rows allow 16-byte loads
  int top_k, stack_n;            // grouped: pairs per token, rows of one expert in the stack
};

// ---- quantization ---------------------------------------------------------------------------
struct QuantArgs {
  LX_CPTR(ushort) w;
  LX_PTR(uchar) q;
  LX_PTR(uchar) scale;
  LX_PTR(uint) amax;  // NVFP4: the matrix absmax, as f32 bits (atomic max of non-negative floats)
  lx_i64 rows;
  int cols;
  float global;
};
struct DequantArgs {
  WeightRef w;
  LX_PTR(ushort) out;
  lx_i64 rows;
  int cols;
};
// Per-token activation quantization (lx_quantize_act).
struct QuantActArgs {
  LX_CPTR(void) x;
  LX_PTR(uchar) q;
  LX_PTR(float) scale;
  LX_PTR(uchar) block_scale;
  int ld_x, ld_q, ld_bs, M, K, x_bf16;
};

// ---- speculative decoding -------------------------------------------------------------------
struct SpecGemvArgs {
  LX_CPTR(float) x;
  LX_CPTR(float) w;
  LX_CPTR(float) bias;  // 0 for none
  LX_PTR(float) y;
  int ld_x, ld_y, M, N, K;
};

// ---- sampling -------------------------------------------------------------------------------
// One row's parameters: lx_sampling, field for field (56 bytes on both sides).
struct SamplingRow {
  lx_u64 seed;
  lx_u64 position;
  int slot, row;
  float temperature, top_p;
  int top_k;
  float min_p, repetition_penalty, presence_penalty, frequency_penalty;
};
struct SampleArgs {
  LX_CPTR(float) logits;
  LX_CPTR(SamplingRow) params;
  LX_PTR(int) seen;
  LX_PTR(int) out_counts;
  LX_PTR(float) work;
  LX_PTR(int) tokens;
  LX_PTR(float) token_logprob;  // 0 for none
  LX_PTR(int) top_tokens;
  LX_PTR(float) top_logprobs;
  int V, top_n;
};
struct ObserveArgs {
  LX_PTR(int) counts;
  LX_CPTR(int) tokens;
  int n;
};
struct MaskLogitsArgs {
  LX_PTR(float) logits;
  LX_CPTR(int) rows;
  LX_CPTR(uint) bits;
  int V, words;
};
