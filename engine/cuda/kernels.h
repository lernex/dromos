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

// ---- Runtime (every backend) ----------------------------------------------------------------
// The kernels.h ABI is the backend interface: the CUDA, HIP, Metal and CPU backends are each one
// shared library exporting every lx_* function here, and the model code calls nothing else (no
// CUDA runtime calls), so one model implementation runs on all of them. "Device memory" is what
// the backend's kernels read: VRAM on CUDA/HIP, shared MTLBuffers on Metal (pointers are their
// contents addresses), host memory on the CPU. A stream orders work (a CUDA/HIP stream, a Metal
// command queue, a CPU work queue); null is the backend's default. Errors use the CUDA codes
// (0 success, LX_ERR_NOT_SUPPORTED when a backend lacks a feature, e.g. graph capture).
#define LX_ERR_NOT_SUPPORTED 801
typedef enum { LX_BACKEND_CUDA = 0, LX_BACKEND_HIP = 1, LX_BACKEND_METAL = 2, LX_BACKEND_CPU = 3 } lx_backend;
typedef enum { LX_COPY_H2D = 0, LX_COPY_D2H = 1, LX_COPY_D2D = 2 } lx_copy_kind;

LX_API int lx_rt_backend(void);
LX_API int lx_rt_device_count(void);
LX_API int lx_rt_set_device(int ordinal);
LX_API int lx_rt_alloc(void** ptr, int64_t bytes);  // zeroed
LX_API int lx_rt_free(void* ptr);
LX_API int lx_rt_host_alloc(void** ptr, int64_t bytes);  // pinned / page-locked host memory where it exists
LX_API int lx_rt_host_free(void* ptr);
LX_API int lx_rt_memset(void* ptr, int value, int64_t bytes, void* stream);
LX_API int lx_rt_copy(void* dst, const void* src, int64_t bytes, lx_copy_kind kind, void* stream);
LX_API int lx_rt_stream_create(void** stream);
LX_API int lx_rt_stream_destroy(void* stream);
LX_API int lx_rt_stream_sync(void* stream);
// Events: record on a stream, wait for on the host.
LX_API int lx_rt_event_create(void** event);
LX_API int lx_rt_event_destroy(void* event);
LX_API int lx_rt_event_record(void* event, void* stream);
LX_API int lx_rt_event_sync(void* event);
// Capture the work issued on a stream into a replayable graph (CUDA graphs, HIP graphs; a backend
// without it returns LX_ERR_NOT_SUPPORTED from capture_begin and the model runs eagerly).
LX_API int lx_rt_capture_begin(void* stream);
LX_API int lx_rt_capture_end(void* stream, void** graph);
LX_API int lx_rt_graph_launch(void* graph, void* stream);
LX_API int lx_rt_graph_destroy(void* graph);
LX_API int lx_rt_last_error(void);  // and clears it

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

// ---- Sampling (sampler.cu) ------------------------------------------------------------------
// One launch draws the next token of every sequence in a batch. Row i reads logits row
// params[i].row ([*, V] f32, the model's output) and its sequence's penalty counts at
// params[i].slot. Per row, in order:
//   penalties on the raw logits: repetition (a token seen in the prompt or the output is divided by
//     the penalty when positive, multiplied when negative), then presence and frequency over the
//     output only: x -= frequency * count + presence when count > 0;
//   greedy when temperature <= 0: the argmax (ties to the lower index);
//   otherwise scaled by 1 / temperature, then the tokens that survive top-k, top-p (the smallest
//     set, of the top-k distribution, holding top_p of its mass) and min-p (scaled logit >=
//     max + log(min_p)) are drawn with the Gumbel-max trick: argmax of logit + Gumbel noise, the
//     noise Philox keyed by (seed, position, token id). Same seed and position, same token.
// The chosen token then counts once in seen and out_counts. token_logprob (may be null) gets the
// chosen token's log-probability under the raw softmax; top_tokens / top_logprobs ([n, top_n], may
// be null when top_n is 0) get the top_n most likely tokens of the same distribution, in order.
// work: f32 [n, V] scratch. seen, out_counts: int32 [slots, V], one row per sequence slot.
typedef struct {
  uint64_t seed;             // the request's seed (the Philox key)
  uint64_t position;         // index of the output token being drawn (the Philox counter)
  int32_t slot;              // the sequence's row in seen and out_counts
  int32_t row;               // the row of the logits to sample
  float temperature;         // <= 0 is greedy
  float top_p;               // in (0, 1]; 1 disables
  int32_t top_k;             // keeps that many (1 .. V - 1); 0, or V or more, disables
  float min_p;               // in [0, 1]; 0 disables
  float repetition_penalty;  // > 0; 1 disables
  float presence_penalty;    // 0 disables
  float frequency_penalty;   // 0 disables
} lx_sampling;

LX_API int lx_sample(const float* logits, int V, const lx_sampling* params, int n, int32_t* seen,
                     int32_t* out_counts, float* work, int32_t* tokens, float* token_logprob, int top_n,
                     int32_t* top_tokens, float* top_logprobs, void* stream);

// Forgets a slot's counts (a new sequence starts in it).
LX_API int lx_sampler_reset(int32_t* seen, int32_t* out_counts, int slot, int V, void* stream);

// Counts tokens into a slot: the prompt (is_output 0, repetition only) or generated output (is_output 1,
// which also feeds presence and frequency).
LX_API int lx_sampler_observe(int32_t* seen, int32_t* out_counts, int slot, int V, const int32_t* tokens, int n,
                              int is_output, void* stream);

// ---- Offload (offload.cu; the runtime half is engine/runtime/offload.h) ---------------------
// A model larger than VRAM runs from every memory the machine has at once: the dense weights and
// the experts the routing uses most in VRAM, every expert in pinned host memory (read by the GPU
// across PCIe, or computed by the CPU straight from RAM), the coldest on NVMe when RAM runs out.
// Decode steps stay inside one CUDA graph: the cache decisions are made on the GPU (lx_offload_plan),
// misses are read across PCIe by the expert kernel itself and written into VRAM as they are read,
// and the CPU's share is handed over through a mailbox in mapped host memory, so no step waits for
// the host. Correctness never depends on where an expert is: every location holds the same bytes.

// Pinned host memory the GPU reads and writes directly (zero-copy). `*device` is the address
// kernels use for the block at `*host`.
LX_API int lx_rt_host_alloc_mapped(void** host, void** device, int64_t bytes);
// Pins (page-locks) and maps an existing host range; `*device` as above.
LX_API int lx_rt_host_register(void* host, int64_t bytes, void** device);
LX_API int lx_rt_host_unregister(void* host);
// Work issued on `stream` after this call waits for `event` (recorded on another stream). Inside a
// capture this forks and joins graph branches.
LX_API int lx_rt_stream_wait(void* stream, void* event);
// Free and total device memory, in bytes.
LX_API int lx_rt_mem_info(int64_t* free_bytes, int64_t* total_bytes);

// One expert's weights where they are now (VRAM, or pinned host memory through its mapped address),
// in the layout of one entry of the expert stacks ("Weight formats"): gate_up [2I, L], down [L, I];
// scales f32 [2I] / [L] for INT8 and FP8 (null for BF16). A null gate_up means "no expert here":
// the pair is skipped (another device computes it). Destinations of a fill use the same struct.
typedef struct {
  const void* gate_up;
  const void* gate_up_scale;
  const void* down;
  const void* down_scale;
} lx_expert_ref;

// Routed experts of a decode step, each pair p = t * top_k + k reading its expert through src[p]:
//   out[t, :] = sum over k with src[p].gate_up != null of weights[p] * expert(src[p], latent[t, :])
//   expert(x) = down @ bf16(silu(gate @ x) * (up @ x))           (the math of lx_moe_experts)
// Weight-only, f32 accumulation, each output written once (no atomics). With fill non-null, a pair
// whose fill[p].gate_up is non-null also writes the bytes it reads (weights and scales) to fill[p]:
// an expert read across PCIe lands in its VRAM cache slot in the same pass. latent bf16 [T, L];
// weights f32 [T, top_k]; src, fill [T * top_k] in device memory; out f32 [T, L]. type: LX_W_BF16,
// LX_W_INT8 or LX_W_FP8. Requires L % 16 == 0 and I % 16 == 0, top_k <= 8.
// Workspace: lx_offload_workspace_bytes(T, L, I, top_k).
LX_API int lx_offload_experts(const void* latent, const float* weights, const lx_expert_ref* src,
                              const lx_expert_ref* fill, lx_wtype type, float* out, void* workspace, int T, int L,
                              int I, int top_k, void* stream);
// The same contract for any T, with the pairs grouped by expert so each expert is read once: group g
// applies groups[g] to the pairs pairs[offsets[g] .. offsets[g + 1]) (pair index t * top_k + k).
// groups, offsets [G + 1], pairs in device memory. A layer's groups may come in several calls (the
// experts staged into VRAM in batches): LX_OG_BEGIN on the first clears the per-pair partial sums in
// the workspace, LX_OG_END on the last writes out; a pair no call covered contributes zero.
#define LX_OG_BEGIN 1
#define LX_OG_END 2
LX_API int lx_offload_experts_grouped(const void* latent, const float* weights, const lx_expert_ref* groups,
                                      const int32_t* offsets, const int32_t* pairs, int G, lx_wtype type, float* out,
                                      void* workspace, int T, int L, int I, int top_k, int flags, void* stream);
LX_API int64_t lx_offload_workspace_bytes(int T, int L, int I, int top_k);

// Copies whole experts, src[i] -> dst[i] for i < *count (weights and scales; count in device memory,
// at most max_count, so the launch is capturable). Prefetches use it with host sources.
LX_API int lx_offload_copy(const lx_expert_ref* src, const lx_expert_ref* dst, const int32_t* count, int max_count,
                           lx_wtype type, int L, int I, void* stream);

// The CPU lane's mailbox, in mapped host memory. The GPU posts a job (its pairs, and the latent rows
// in latent_rows) and rings the doorbell with the job's sequence number; the CPU writes
// result[t, :] = sum over the job's pairs of token t of weight * expert(...) for every t < T (zeros
// where a token has none), then sets done to the same number.
#define LX_OFFLOAD_MAX_PAIRS 64
typedef struct {
  volatile uint32_t doorbell;
  uint32_t pad0[15];
  volatile uint32_t done;
  uint32_t pad1[15];
  int32_t layer, T, n;
  int32_t token[LX_OFFLOAD_MAX_PAIRS], expert[LX_OFFLOAD_MAX_PAIRS];
  float weight[LX_OFFLOAD_MAX_PAIRS];
} lx_offload_mailbox;

// Per-layer counters, uint64 each, in lx_offload_state.counters[layer * LX_OFFLOAD_COUNTERS + i].
enum {
  LX_OC_HITS = 0,        // pairs served from VRAM
  LX_OC_GPU_MISSES,      // pairs the GPU read across PCIe
  LX_OC_CPU_MISSES,      // pairs computed by the CPU
  LX_OC_FILLS,           // misses written into a VRAM slot
  LX_OC_PREFETCHES,      // experts copied ahead by lx_offload_prefetch
  LX_OC_PREFETCH_HITS,   // pairs served from a slot a prefetch filled
  LX_OC_CALLS,           // plan calls
  LX_OFFLOAD_COUNTERS = 8
};

// The expert cache of one model, in device memory. The host fills it at load (and may rewrite it
// between steps when it re-plans); during steps only the kernels below change it.
typedef struct {
  int n_layers, n_experts, top_k, L, I;
  int type;              // lx_wtype of every expert
  int32_t* vram_slot;    // [n_layers * n_experts]: VRAM pool slot, -1 when not in VRAM
  int32_t* host_slot;    // [n_layers * n_experts]: host pool slot, -1 when only on disk (CPU lane only)
  uint8_t* cpu_first;    // [n_layers * n_experts]: 1 for experts whose misses the CPU should take
  // VRAM slots, grouped by layer: layer l owns [layer_first[l], layer_first[l + 1]); the first
  // layer_pinned[l] of them hold experts placed at load that are never evicted.
  int32_t* layer_first;  // [n_layers + 1]
  int32_t* layer_pinned; // [n_layers]
  int32_t* slot_expert;  // [slots]: the expert in each slot, -1 when empty
  uint32_t* slot_stamp;  // [slots]: clock of the slot's last use (least recently used is evicted)
  uint8_t* slot_prefetched;  // [slots]: 1 while a prefetched expert has not been used yet
  uint32_t* clock;       // advanced by every plan
  lx_expert_ref vram_pool, host_pool;  // slot 0 of each; slot s at + s * the strides below
  int64_t gate_up_stride, gate_up_scale_stride, down_stride, down_scale_stride;  // bytes
  int max_cpu_pairs;     // CPU pairs per plan call, 0 disables the CPU lane
  float cpu_fraction;    // share of a step's misses the CPU takes (cold experts first)
  int fill;              // write the GPU's misses into VRAM slots as they are read
  lx_offload_mailbox* mailbox;  // mapped host memory
  void* cpu_latent;      // mapped host memory, bf16 [max T, L]
  const float* cpu_result;  // mapped host memory, f32 [max T, L]
  uint32_t* job_seq;     // the last job's sequence number
  int32_t* job_pending;  // set by plan when it posted a job, cleared by join
  int32_t* error;        // nonzero after a join timed out (the step's output is wrong)
  unsigned long long* counters;  // [n_layers * LX_OFFLOAD_COUNTERS]
} lx_offload_state;

// Decides where every pair of one MoE layer's decode step is computed, from ids [T, top_k] (the
// routing) and the cache in *state (device memory): a pair whose expert is in VRAM reads it there;
// a miss is either read across PCIe by lx_offload_experts (written into the layer's least recently
// used dynamic slot as it is read, when fill is on and a slot is free this step) or posted to the
// CPU lane (cold experts first, up to the policy's share; experts that are only on disk always).
// Writes src and fill [T * top_k] for lx_offload_experts. One warp; T * top_k <= LX_OFFLOAD_MAX_PAIRS.
LX_API int lx_offload_plan(const int32_t* ids, const float* weights, const void* latent, int layer,
                           lx_offload_state* state, lx_expert_ref* src, lx_expert_ref* fill, int T, void* stream);
// Adds the CPU lane's result to out [T, L] once it is done (waits on the mailbox; returns at once
// when the plan posted no job). After a timeout it sets state->error and adds nothing.
LX_API int lx_offload_join(lx_offload_state* state, float* out, int T, int L, void* stream);
// Prefetch for a layer about to run: for the first n experts of predicted [n] (any order) that are
// not in VRAM and can come from host memory, claims the layer's least recently used dynamic slots
// (never one the layer's pinned set or another prediction holds) and writes copy jobs (src host,
// dst VRAM, *count) for lx_offload_copy. The cache tables point at the slots at once, so the plan
// that follows must run after the copy (a graph join), which is how the engine launches it.
LX_API int lx_offload_prefetch(const int32_t* predicted, int n, int layer, lx_offload_state* state,
                               lx_expert_ref* src, lx_expert_ref* dst, int32_t* count, void* stream);

// Reads `bytes` of (possibly host-mapped) memory with `blocks` blocks of 16-byte loads, xor-folding
// them into *sink: the probe for the GPU's zero-copy read bandwidth.
LX_API int lx_offload_read_probe(const void* src, int64_t bytes, unsigned long long* sink, int blocks, void* stream);

#ifdef __cplusplus
}
#endif
