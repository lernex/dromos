// The kernel library's C ABI. Every kernel is launched through one of these functions, which
// return a cudaError_t (0 on success). Pointers are device pointers to row-major, contiguous
// buffers; `stream` is a cudaStream_t (null for the default stream). Kept C so the tests in
// tests/kernels can call each kernel through ctypes on PyTorch tensors.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define LX_API __attribute__((visibility("default")))

typedef enum { LX_F32 = 0, LX_BF16 = 1 } lx_dtype;

// ---------------------------------------------------------------------------------------------
// Conventions
//   T        tokens in this step (all sequences concatenated); rows of every activation.
//   ld_*     row stride in elements, for reading a column range out of a wider buffer.
//   Varlen   kernels that look across positions take the step's sequences as cu_seqlens
//            [n_seq + 1] (int32, token offsets into the step, cu_seqlens[0] = 0) and slot ids
//            [n_seq] (int32) naming each sequence's row in the persistent state buffers.
//   Weights  are row-major [out, in] like the checkpoint (y = x @ W^T).
//   Unless a comment says otherwise, math is float32 and outputs are rounded once at the end.
// ---------------------------------------------------------------------------------------------

// ---- Norms and elementwise (rmsnorm.cu, elementwise.cu) --------------------------------------

// y[r, :] = x[r, :] * rsqrt(mean(x[r, :]^2) + eps) * w, computed in float32.
// x: [rows, dim] (x_dtype), w: [dim] bf16, y: [rows, dim] (y_dtype).
LX_API int lx_rmsnorm(const void* x, lx_dtype x_dtype, const void* w, void* y, lx_dtype y_dtype,
                      int rows, int dim, float eps, void* stream);

// Strided 2-D copy with a dtype cast: dst[r, c] = src[r, c] for c < cols.
LX_API int lx_copy2d(const void* src, lx_dtype src_dtype, int ld_src, void* dst, lx_dtype dst_dtype,
                     int ld_dst, int rows, int cols, void* stream);

// In place, y = act(y) over n elements of f32. act: 0 = tanh, 1 = silu, 2 = sigmoid.
LX_API int lx_activate_f32(float* y, int64_t n, int act, void* stream);

// out[r, c] = silu(x[r, c]) * x[r, cols + c] (a gate|up pair packed in one row of width ld_x).
// x: f32 [rows, ld_x], out: [rows, cols] (out_dtype).
LX_API int lx_silu_mul(const float* x, int ld_x, void* out, lx_dtype out_dtype, int rows, int cols,
                       void* stream);

// y[r, c] += row[c] (f32 y [rows, cols], row: [cols] bf16). Adds a bias or an embedding row.
LX_API int lx_add_row_bf16(float* y, const void* row, int rows, int cols, void* stream);

// ---- Dense GEMM (gemm.cu, cuBLASLt) ---------------------------------------------------------

// y[M, N] = alpha * (x[M, K] @ w[N, K]^T + bias[N]) + beta * y. x and w bf16, float32
// accumulation; bias bf16 or null; y f32 or bf16 (with beta != 0, y is read in its own dtype).
LX_API int lx_gemm_bf16(const void* x, int ld_x, const void* w, const void* bias, void* y,
                        lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
                        void* stream);

// The same in full float32 (no TF32), for the routers and the depth head, whose weights are
// float32 in the checkpoint: y[M, N] = x[M, K] @ w[N, K]^T + bias[N] (bias may be null).
LX_API int lx_gemm_f32(const float* x, int ld_x, const float* w, const float* bias, float* y,
                       int ld_y, int M, int N, int K, void* stream);

// ---- Embedding and the four residual streams (streams.cu) -----------------------------------

// streams[t, s, :] = table[ids[t], :] + stream_emb[s, :]. table bf16 [V, D], ids int32 [T],
// stream_emb bf16 [S, D], streams f32 [T, S, D].
LX_API int lx_embed_streams(const void* table, const int32_t* ids, const void* stream_emb,
                            float* streams, int T, int S, int D, void* stream);

// mean[t, :] = mean over s of streams[t, s, :]. streams f32 [T, S, D], mean f32 [T, D].
LX_API int lx_stream_mean(const float* streams, float* mean, int T, int S, int D, void* stream);

// Hyper-connection read. With matrix [S, S], read [S] (both f32, per pass, precomputed):
//   mixed[t, o, :] = sum_i matrix[o, i] * streams[t, i, :]
//   source[t, :]   = sum_i read[i] * streams[t, i, :]
//   h[t, :]        = rmsnorm(source[t, :], norm_w, eps) as bf16
// streams, mixed f32 [T, S, D]; h bf16 [T, D]. mixed may not alias streams.
LX_API int lx_mhc_read(const float* streams, const float* matrix, const float* read, const void* norm_w,
                       float eps, float* mixed, void* h, int T, int S, int D, void* stream);

// Hyper-connection write: streams[t, s, :] = mixed[t, s, :] + write[s] * update[t, :].
// mixed, streams f32 [T, S, D] (may alias); write f32 [S]; update f32 [T, D].
LX_API int lx_mhc_write(const float* mixed, const float* write, const float* update, float* streams,
                        int T, int S, int D, void* stream);

// ---- Depth memory (memory.cu) ---------------------------------------------------------------
// Each token keeps a bank of up to E entries; for each entry its key and value projections
// (f32 [T, E, M]) are stored when the entry is written, so a read only attends.

// out[t, s, :] = softmax_e(q[t, s, :] . bank_k[t, e, :] / sqrt(M)) @ bank_v[t, e, :], over the
// first n_entries entries (n_entries >= 1). q f32 [T, S, M]; out bf16 [T, S, M].
LX_API int lx_memory_attend(const float* q, const float* bank_k, const float* bank_v, int n_entries,
                            int max_entries, void* out, int T, int S, int M, void* stream);

// Gated fuse of the retrieved memory into the streams, and its mean:
//   g[t, s]         = sigmoid(sum_d streams[t, s, d] * gate[s, d] / sqrt(D) + gate_bias[s])
//   streams[t, s, :] += g[t, s] * projected[t, s, :]           (gate uses streams before the add)
//   summary[t, :]   = mean over s of projected[t, s, :]
// streams f32 [T, S, D] in place; projected f32 [T, S, D]; gate bf16 [S, D]; gate_bias bf16 [S];
// summary f32 [T, D].
LX_API int lx_memory_fuse(float* streams, const float* projected, const void* gate, const void* gate_bias,
                          float* summary, int T, int S, int D, void* stream);

// The depth head's last step, per token: given logit[t] (f32, the output projection already
// applied), state[t, :] and difference[t, :] (f32 [T, D]):
//   scale = max(sqrt(mean(state^2)), 1e-6), innovation = sqrt(mean(difference^2))
//   out[t] = sigmoid(clip(logit + tanh(innovation / scale) - 0.5, -limit, limit))
LX_API int lx_continuation_finish(const float* logit, const float* state, const float* difference,
                                  float limit, float* out, int T, int D, void* stream);

// ---- Mamba-2 (mamba.cu) ---------------------------------------------------------------------

// Causal depthwise conv (width W) then silu, per sequence, continuing from conv_state:
//   out[t, c] = silu(bias[c] + sum_{k<W} w[c, k] * x[t - (W - 1) + k, c])
// where positions before the sequence start read conv_state[slot, :, c] (its last W - 1
// inputs; zeros when has_state[seq] == 0). Afterwards conv_state[slot] holds the last W - 1
// inputs of the sequence (including older state when the sequence is shorter than W - 1).
// x f32 [T, C] (row stride ld_x); w bf16 [C, W]; bias bf16 [C]; conv_state f32
// [slots, W - 1, C]; out f32 [T, C].
LX_API int lx_conv1d_varlen(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                            const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens,
                            int n_seq, float* out, int C, int W, void* stream);

// The selective scan, per sequence, continuing from ssm_state (zeros when has_state == 0):
//   dt[t, h]  = softplus(dt_raw[t, h] + dt_bias[h]);  a[t, h] = exp(dt[t, h] * -exp(A_log[h]))
//   S[h]      = a * S[h] + dt * x[t, h, :] (outer) B[t, g(h), :]       S[h]: [P, N]
//   y[t, h, p] = sum_n S[h][p, n] * C[t, g(h), n] + D[h] * x[t, h, p]
// with g(h) = h / (H / G). x f32 rows of H*P (stride ld_x); dt_raw f32 rows of H (stride ld_dt);
// B, C f32 rows of G*N (stride ld_bc); A_log, D, dt_bias f32 [H]; ssm_state f32 [slots, H, P, N]
// (updated to the state after the last token); y f32 [T, H, P].
LX_API int lx_ssm_scan_varlen(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B,
                              const float* C, int ld_bc, const float* A_log, const float* D,
                              const float* dt_bias, float* ssm_state, const int32_t* slot_ids,
                              const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* y,
                              int H, int P, int G, int N, void* stream);

// Mamba-2's gated norm, per group of channels:
//   v = y * silu(z);  out[t, g*K + i] = v * rsqrt(mean over the group's K channels of v^2 + eps) * w
// y f32 [T, I]; z f32 rows (stride ld_z); w bf16 [I]; out bf16 [T, I]; K = I / groups.
LX_API int lx_gated_rmsnorm(const float* y, const float* z, int ld_z, const void* w, void* out, int T,
                            int I, int groups, float eps, void* stream);

// ---- Attention (attention.cu) ---------------------------------------------------------------

// Causal GQA with no positional encoding, appending this step's keys and values to the cache:
//   for seq s with n = cu_seqlens[s + 1] - cu_seqlens[s] new tokens and p0 = start_pos[s]
//   tokens already cached in slot slot_ids[s]: write k, v of the new tokens to cache positions
//   p0 .. p0 + n - 1, then token j attends to cache positions 0 .. p0 + j.
//   out[t, h, :] = softmax(q[t, h, :] . K[pos, h / (H / KVH), :] * scale) @ V[...]
// qkv f32 rows of (H + 2 * KVH) * Dh, q | k | v, stride ld_qkv; k_cache, v_cache bf16
// [slots, capacity, KVH, Dh]; out bf16 [T, H * Dh]. Requires p0 + n <= capacity.
LX_API int lx_attention_varlen(const float* qkv, int ld_qkv, void* k_cache, void* v_cache,
                               const int32_t* slot_ids, const int32_t* start_pos, const int32_t* cu_seqlens,
                               int n_seq, int capacity, void* out, int H, int KVH, int Dh, float scale,
                               void* stream);

// ---- Latent mixture of experts (moe.cu) -----------------------------------------------------

// Routing, per token, from router logits [T, E] (f32):
//   p = softmax(logits); entropy[t] = -sum p * log(max(p, 1e-9)); confidence[t] = max p
//   ids[t, :] = the top_k experts by logits + selection_bias (ties: lower index first),
//   weights[t, :] = softmax over those experts of the UNBIASED logits.
// ids int32 [T, top_k]; weights, entropy, confidence f32.
LX_API int lx_moe_route(const float* logits, const float* selection_bias, int32_t* ids, float* weights,
                        float* entropy, float* confidence, int T, int E, int top_k, void* stream);

// Routed experts on the latent:  out[t, :] = sum_k weights[t, k] * expert(ids[t, k], latent[t, :])
//   expert(e, x) = down[e] @ (silu(gate_up[e][:I] @ x) * (gate_up[e][I:] @ x))
// latent bf16 [T, L]; gate_up bf16 [E, 2I, L]; down bf16 [E, L, I]; out f32 [T, L]
// (written, not accumulated). The expert's intermediate is rounded to bf16 before down.
LX_API int lx_moe_experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                          const void* down, float* out, int T, int L, int I, int top_k, void* stream);

// summary[t, :] = sum_k weights[t, k] * expert_emb[ids[t, k], :]. expert_emb bf16 [E, R]; summary f32 [T, R].
LX_API int lx_moe_summary(const int32_t* ids, const float* weights, const void* expert_emb, float* summary,
                          int T, int R, int top_k, void* stream);

// ---- Decode GEMV (gemv.cu) ------------------------------------------------------------------
// Matrix-vector products for decode steps (M <= 8 rows), memory-bound by design: one pass over
// the weights, which dominate the traffic, at close to DRAM bandwidth. Weights are bf16, or int8
// with one f32 scale per output row (w = q * scale[n], symmetric, see lx_quantize_w8). The input
// may be f32 or bf16 and is read directly, so no cast pass is needed before the product.
//   y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y[m, n]
// x: [M, K] (x_dtype, row stride ld_x); bias: [N] (bias_dtype, or null); y: [M, N] (y_dtype,
// row stride ld_y; read only when beta != 0). Accumulation in f32. Deterministic (no atomics).
typedef enum { LX_W_BF16 = 0, LX_W_INT8 = 1, LX_W_FP8 = 2, LX_W_NVFP4 = 3 } lx_wtype;
LX_API int lx_gemv(const void* x, lx_dtype x_dtype, int ld_x, const void* w, lx_wtype w_type, const float* w_scale,
                   const void* bias, lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K,
                   float alpha, float beta, void* stream);

// Per-row symmetric int8: scale[n] = max_k |w[n, k]| / 127 (1 if the row is all zero),
// q[n, k] = round_half_even(w[n, k] / scale[n]) clamped to [-127, 127]. w bf16 [rows, cols];
// q int8 [rows, cols]; scale f32 [rows]. Rows may be the flattened leading dims of a 3-D tensor.
LX_API int lx_quantize_w8(const void* w, int8_t* q, float* scale, int64_t rows, int cols, void* stream);

// lx_moe_experts with int8 expert weights (per-row scales as above: gate_up_scale [E, 2I],
// down_scale [E, L]) and a caller-owned workspace of at least lx_moe_workspace_bytes(T, I, top_k)
// bytes, so nothing is allocated per call (and the call can be captured in a CUDA graph).
LX_API int lx_moe_experts_w8(const void* latent, const int32_t* ids, const float* weights, const int8_t* gate_up,
                             const float* gate_up_scale, const int8_t* down, const float* down_scale, float* out,
                             void* workspace, int T, int L, int I, int top_k, void* stream);
LX_API int64_t lx_moe_workspace_bytes(int T, int I, int top_k);

// ---- Weight formats ------------------------------------------------------------------------
// A projection is held once, in one format, and every kernel that multiplies by it (the decode
// GEMV, the tensor-core GEMM for prefill and batches, the expert kernels) reads that one copy.
// All formats keep the checkpoint's row-major [N, K] order, K contiguous, which is what a GEMV
// streams and what a tensor-core GEMM wants for its B operand (y = x @ W^T, "TN").
//
//   LX_W_BF16   data bf16 [N, K]. scale unused.
//   LX_W_INT8   data int8 [N, K], scale f32 [N]:      w[n, k] = q[n, k] * scale[n]
//   LX_W_FP8    data e4m3 [N, K], scale f32 [N]:      w[n, k] = fp8[n, k] * scale[n]
//               (a per-tensor checkpoint scale is expanded to [N] at load)
//   LX_W_NVFP4  data uint8 [N, K / 2], two e2m1 per byte, the low nibble holding the even k;
//               scale e4m3 [N, K / 16], one per 16 consecutive k; global f32:
//                                                     w[n, k] = e2m1[n, k] * scale[n, k / 16] * global
//               (requires K % 16 == 0; the loader folds a checkpoint's global scale so the
//               engine always multiplies)
// Stacked expert weights are [E, N, K] in the same format, every expert laid out as above, one
// after the other (scales likewise [E, N] or [E, N, K / 16]; global is one value for the stack).
typedef struct {
  lx_wtype type;
  const void* data;
  const void* scale;  // f32 [N] for INT8 / FP8, e4m3 [N, K / 16] for NVFP4, null for BF16
  float global;       // NVFP4 only (1 otherwise)
} lx_weight;

// How a tensor-core product does its math. Weights are always read in their stored format; this
// says what the activations are converted to and which instruction multiplies them.
//   LX_MATH_BF16  activations bf16; quantized weights are expanded to bf16 inside the kernel.
//                 Runs on every GPU (sm_80+).
//   LX_MATH_INT8  activations quantized per token to int8 (symmetric, absmax / 127); int8 weights
//                 only. IMMA, sm_80+.
//   LX_MATH_FP8   activations quantized per token to e4m3 (absmax / 448); FP8 weights, or NVFP4
//                 weights through the mixed e2m1 x e4m3 instruction (sm_100 / sm_120). sm_89+.
//   LX_MATH_FP4   activations quantized to NVFP4 (per 16 values along K, e4m3 scale, per-token
//                 global); NVFP4 weights only. sm_100 / sm_120.
// A combination the GPU cannot run returns cudaErrorNotSupported; the engine picks the math per
// GPU (engine/runtime/plan) and never asks for one the card lacks.
typedef enum { LX_MATH_BF16 = 0, LX_MATH_INT8 = 1, LX_MATH_FP8 = 2, LX_MATH_FP4 = 3 } lx_math;

// Decode GEMV on any weight format (generalizes lx_gemv; same contract for x, bias, y, alpha,
// beta, M <= 8). Weight-only: the weight is expanded in registers, the math is f32.
LX_API int lx_gemv_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                     lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                     float beta, void* stream);

// Tensor-core GEMM on any weight format, for prefill and batched decode:
//   y[M, N] = alpha * (x[M, K] @ W[N, K]^T + bias[N]) + beta * y
// x bf16 or f32 (quantized on the fly when math needs it, into `workspace`, at least
// lx_gemm_workspace_bytes(M, K, math) bytes). Same dtype rules for bias and y as lx_gemm_bf16.
LX_API int lx_gemm_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, lx_math math, const void* bias,
                     lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                     float beta, void* workspace, void* stream);
LX_API int64_t lx_gemm_workspace_bytes(int M, int K, lx_math math);

// Routed experts on any weight format (contract of lx_moe_experts; gate_up [E, 2I, L],
// down [E, L, I] as lx_weight stacks). Two implementations with one contract:
//   lx_moe_experts_decode  few tokens (T <= 8): one pass over the chosen experts' weights,
//                          weight-only, f32 math. Workspace: lx_moe_decode_workspace_bytes.
//   lx_moe_experts_grouped any T: tokens grouped by expert, a tensor-core grouped GEMM per
//                          stage with the given math. Workspace: lx_moe_grouped_workspace_bytes.
LX_API int lx_moe_experts_decode(const void* latent, const int32_t* ids, const float* weights,
                                 const lx_weight* gate_up, const lx_weight* down, float* out, void* workspace, int T,
                                 int L, int I, int top_k, void* stream);
LX_API int64_t lx_moe_decode_workspace_bytes(int T, int I, int top_k);
LX_API int lx_moe_experts_grouped(const void* latent, const int32_t* ids, const float* weights,
                                  const lx_weight* gate_up, const lx_weight* down, lx_math math, float* out,
                                  void* workspace, int T, int L, int I, int E, int top_k, void* stream);
LX_API int64_t lx_moe_grouped_workspace_bytes(int T, int L, int I, int E, int top_k, lx_math math);

// Round-to-nearest quantization of a bf16 [rows, cols] weight into `out`, whose data and scale
// buffers the caller allocated for out->type (sizes as in the layouts above). INT8 and FP8 take
// per-row absmax scales (/127, /448); NVFP4 takes the global as max|w| / (448 * 6) and each
// block's e4m3 scale as its absmax / (6 * global), rounded to e4m3. Used when the engine is asked
// for a format the file is not stored in; checkpoints made with GPTQ are loaded as they are.
LX_API int lx_quantize_weight(const void* w, lx_weight* out, int64_t rows, int cols, void* stream);

// =============================================================================================
// ---- Decode programs (decode_*.cu; the Decode Overhead workstream) ---------------------------
// =============================================================================================
// A decode step (M <= 8 tokens, one per sequence) runs as one persistent kernel. The step is built
// once per shape as a program of ops (products, the depth-memory read, the Mamba step, routing,
// experts, attention, the memory write, the depth head), each op is split into tasks spread evenly
// over the SMs, and every SM runs its tasks in program order. Products read their weight rows
// straight from global memory on the tensor cores (bf16, and int8 converted to f16 exactly), with the
// x staging and norms before the reader and the epilogue (bias, activations, the hyper-connection
// write) after it, in the same task. Each SM has L2 fetch the weights of its coming tasks a fixed
// number of bytes ahead, so DRAM keeps streaming while tasks wait, through counters in global
// memory, on the ops they read from. Dependencies are inferred from the buffers each op reads and
// writes, so ops are added in step order and nothing else.
//
// A token's results do not depend on how many tokens the step takes: every row is summed in an order
// fixed by its shape alone (speculative verification compares against single steps bit for bit).
//
//   lx_program_create, the lx_program_* ops in step order, lx_program_build, then lx_program_run
//   once per step. Every pointer is fixed at build time; what changes per step (token ids, slots,
//   start positions) is read on the device from the buffers named at create.
//
// Runs on sm_80 and newer (mma.sync; bulk L2 prefetches from sm_90). Products read BF16, INT8 and
// LX_W_F32 weights; a caller falls back to the per-kernel path for other formats.

typedef struct lx_program lx_program;

// f32 weights [N, K] (the routers and the depth head), for program products only.
#define LX_W_F32 ((lx_wtype)16)

// tokens [M] (ids), slots [M] (state slots), has_state [M] (0 on a sequence's first step) and
// start_pos [M] (tokens already in the KV cache) are int32 device buffers, read when the step runs.
LX_API lx_program* lx_program_create(int M, const int32_t* tokens, const int32_t* slots, const int32_t* has_state,
                                     const int32_t* start_pos);
LX_API void lx_program_destroy(lx_program* p);
// Lays the tasks out over the SMs and uploads the program; no ops can be added afterwards.
LX_API int lx_program_build(lx_program* p);
LX_API int lx_program_run(const lx_program* p, void* stream);
// Tasks of an op. For a product: its row tiles, which is also how many partial sums of squares a
// LX_FY_WRITE with next_read leaves per token (the n_parts of the LX_FX_RMS that reads them).
LX_API int lx_program_tasks(const lx_program* p, int op);
// With LX_PROGRAM_TRACE set at build, timestamps (ns, globaltimer) of the last run, two tables of
// [tasks][4] int64 in list order (block by block): per task (op << 32 | task << 16 | block, start,
// inputs staged (the start but for products), end), then per run of an op's tasks, at the run's
// first task (run start, dependencies met, 0, 0). Returns the number of records written (up to 2 *
// tasks).
LX_API int lx_program_trace(const lx_program* p, int64_t* out, int capacity);

// ---- Products: out = epilogue(W x) for a weight [N, K] ----------------------------------------
typedef enum {
  LX_FX_PLAIN = 0,   // x [M, K] (dtype f32 or bf16, row stride ld)
  LX_FX_RMS = 1,     // bf16(x[m, k] * rsqrt(sum_j parts[m, j] / K + eps) * norm_w[k]); x f32 [M, ld]
  LX_FX_GATED = 2,   // bf16(x[m, k] * rsqrt(sum_{j of g} parts[m, j] / group + eps) * norm_w[k]) with g = k / group;
                     //   x f32 [M, ld]; parts [M, n_parts] are sums of squares over equal consecutive slices of K,
                     //   n_parts / (K / group) of them per group
  LX_FX_CONCAT = 3,  // the f32 rows seg[0] | seg[1] | ... (seg[i] [M, width[i]] contiguous; a 0 width ends the
                     //   list; widths sum to K), rounded to bf16 when round_bf16
  LX_FX_SWIGLU = 4,  // bf16(silu(x[m, k]) * x[m, K + k]); x f32 [M, ld] holding gate | up (a gated MLP's middle)
} lx_fx_kind;

typedef struct {
  int kind;
  const void* x; int dtype; int ld;
  const float* parts; int n_parts;
  const void* norm_w; float eps; int group;
  const float* seg[4]; int width[4]; int round_bf16;
} lx_fx;

typedef enum {
  // v = alpha * (y + bias[n]) + beta * out[m, n], then + lora_gate * (lora_up[n, :] . lora[m, :]) when lora_up
  // is set (lora_up bf16 [N, rank]); out[m, n] = act(v) with act 0 none, 1 tanh, 2 silu; out_bf, when set,
  // gets bf16(act(v)). out f32 or bf16 (row stride ld), bias bf16 or f32 or null.
  LX_FY_STORE = 0,
  // The hyper-connection write of the update y into streams f32 [M, S, N] (S <= 4), with the mix of the
  // read it belongs to applied here: new[s] = sum_i matrix[s, i] * streams[m, i, n] + write[s] * y.
  // Optional: state[m, n] = mean_s new[s]; next_src[m, n] = sum_s next_read[s] * new[s], and
  // next_parts[m, task] = sum of next_src^2 over the task's rows (lx_program_tasks per token).
  LX_FY_WRITE = 1,
} lx_fy_kind;

typedef struct {
  int kind;
  void* out; int dtype; int ld; const void* bias; int bias_dtype; float alpha, beta; int act;
  void* out_bf; int ld_bf;
  const void* lora_up; const float* lora; int rank; float lora_gate;
  float* streams; const float* matrix; const float* write; int S;
  float* state; const float* next_read; float* next_src; float* next_parts;
} lx_fy;

// y[m, n] = sum_k W[n, k] x[m, k] with the format's scales, then the epilogue. Returns the op id.
LX_API int lx_program_gemv(lx_program* p, const lx_fx* x, const lx_weight* w, const lx_fy* y, int N, int K);

// ---- The rest of a Metis step ----------------------------------------------------------------

// streams[m, s, :] = table[tokens[m], :] + stream_emb[s, :] (f32 sums of the bf16 rows),
// previous[m, :] = mean_s streams[m, s, :], history[m, :R] = 0, continuation[m] = 0.
LX_API int lx_program_embed(lx_program* p, const void* table, const void* stream_emb, float* streams, float* previous,
                            float* history, float* continuation, int S, int D, int R);

// The depth-memory read, per token, over the first n_entries entries of the token's bank. Entry e
// holds U | P | c (bank + m * token_stride + e * entry_stride, f32): U = Wq^T k_e (D), P = Wout v_e +
// out bias (D), c = bq . k_e, so that with x[s] = streams[m, s, :] it computes exactly the read of
// lx_memory_attend + the output projection + lx_memory_fuse, without the per-read projections:
//   p[s, :] = softmax_e((x[s] . U_e + c_e) / sqrt(mem_dim)), g[s] = sigmoid(x[s] . gate[s] / sqrt(D) + gate_bias[s])
//   projected[s] = sum_e p[s, e] P_e; streams[m, s] += g[s] * projected[s]; summary[m] = mean_s projected[s]
// (summary = 0 and streams unchanged when n_entries == 0). Then, from the new streams:
//   state[m] = mean_s streams[m, s];
//   with rms (f32 [2, M]): rms[m] = sum state^2 (the final norm's LX_FX_RMS parts, n_parts 1);
//   with previous: difference[m] = state - previous, previous = state, and rms[M + m] = sum difference^2;
//   with h: src = sum_s read[s] streams[m, s], h[m] = bf16(src * rsqrt(mean(src^2) + eps) * norm_w);
//   with lora_down (bf16 [rank, D]): lora[m, j] = bf16(lora_down[j] . h[m]).
typedef struct {
  float* streams; const float* bank; int n_entries; int entry_stride; int token_stride; int mem_dim;
  const void* gate; const void* gate_bias;
  float* state; float* summary;
  const float* read; const void* norm_w; float eps; void* h;
  const void* lora_down; int rank; float* lora;
  float* previous; float* difference; float* rms;
  int S, D;
} lx_memread_args;
LX_API int lx_program_memory_read(lx_program* p, const lx_memread_args* a);

// The Mamba-2 step for one token per sequence, from the in_proj output zxbcdt [M, ld] (z | x | B | C | dt):
// the causal conv (continuing conv_state, lx_conv1d_varlen's layout), the selective scan (continuing
// ssm_state) and v = y * silu(z) [M, I] f32, with each head's sum of squares of v in head_ss [M, H] for the
// out_proj's LX_FX_GATED. Adds the step and the conv-state update of the B and C channels (shared by a
// group's heads); returns the step's op id.
typedef struct {
  const float* zxbcdt; int ld; const void* conv_w; const void* conv_b; float* conv_state; float* ssm_state;
  const float* A_log; const float* D; const float* dt_bias; float* v; float* head_ss;
  int I, H, P, G, N, W;
} lx_ssm_args;
LX_API int lx_program_ssm(lx_program* p, const lx_ssm_args* a);

// Routing on logits [M, E] as lx_moe_route, the expert summary as lx_moe_summary, and the routing
// history: history = 0.5 * history + 0.5 * summary.
typedef struct {
  const float* logits; const float* selection_bias; int32_t* ids; float* weights; float* entropy; float* confidence;
  const void* expert_emb; float* summary; float* history; int E, top_k, R;
} lx_select_args;
LX_API int lx_program_select(lx_program* p, const lx_select_args* a);

// out[m] = sum_k weights[m, k] expert(ids[m, k], latent[m]) + shared(latent[m]), each expert as in
// lx_moe_experts (intermediates rounded to bf16); the shared expert's product is added to the routed sum.
// latent bf16 [M, L]; gate_up [E, 2I, L], down [E, L, I], shared_gate_up [2Is, L], shared_down [L, Is].
// The shared expert's ops wait only for the latent. Returns the last op (the routed down product).
typedef struct {
  const void* latent; const int32_t* ids; const float* weights;
  lx_weight gate_up; lx_weight down; lx_weight shared_gate_up; lx_weight shared_down;
  float* out; int L, I, Is, top_k;
} lx_experts_args;
LX_API int lx_program_experts(lx_program* p, const lx_experts_args* a);

// Decode attention: lx_attention_varlen for one new token per sequence (k, v rounded to bf16 into the
// caches at start_pos, then attention over positions 0 .. start_pos). qkv f32 [M, ld_qkv]; out bf16 [M, H * Dh].
typedef struct {
  const float* qkv; int ld_qkv; void* k_cache; void* v_cache; int capacity; void* out; int H, KVH, Dh; float scale;
} lx_attention_args;
LX_API int lx_program_attention(lx_program* p, const lx_attention_args* a);

// A depth-memory entry: meta = bf16([summary (R), mean_k, entropy * inv_log_e, confidence, continuation]),
// entry[m] = (W_state state[m] + b_state) + (W_meta meta + b_meta) + pass_emb + anchor_emb.
// w_state [mem_dim, D] in any program format; w_meta bf16 [mem_dim, R + 4]; biases and embeddings bf16.
typedef struct {
  const float* state; lx_weight w_state; const void* b_state;
  const float* summary; const float* entropy; const float* confidence; const float* continuation;
  float mean_k, inv_log_e; const void* w_meta; const void* b_meta; const void* pass_emb; const void* anchor_emb;
  float* entry; int D, R, mem_dim;
} lx_entry_args;
LX_API int lx_program_memory_entry(lx_program* p, const lx_entry_args* a);

// The depth head's last step from its hidden layer: logit[m] = w_out . hidden[m] + b_out (f32, R), then
// lx_continuation_finish with rms [2, M] = (sum state^2, sum difference^2) over D from the memory read.
LX_API int lx_program_continuation(lx_program* p, const float* hidden, const float* w_out, const float* b_out,
                                   const float* rms, float limit, float* out, int R, int D);

#ifdef __cplusplus
}
#endif
