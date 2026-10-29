// The kernel library's C ABI. Every kernel is launched through one of these functions, which
// return a cudaError_t (0 on success). Pointers are device pointers to row-major, contiguous
// buffers; `stream` is a cudaStream_t (null for the default stream). Kept C so the tests in
// tests/kernels can call each kernel through ctypes on PyTorch tensors.
#pragma once

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Every backend library exports these. A Windows backend is built with LX_BUILDING_BACKEND (dllexport); the
// engine's own declarations of them, the dispatch stubs, need nothing.
#ifndef LX_API
#if defined(_WIN32)
#if defined(LX_BUILDING_BACKEND)
#define LX_API __declspec(dllexport)
#else
#define LX_API
#endif
#else
#define LX_API __attribute__((visibility("default")))
#endif
#endif

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

// The write above and the read below, back to back on the same streams (the mixer's write and the MoE's read): one
// kernel where clusters run. mixed is updated in place, as the write's input and then the read's mix; h is the read's
// output. Bit-identical to lx_mhc_write followed by lx_mhc_read.
LX_API int lx_mhc_write_read(float* mixed, const float* write, const float* update, float* streams,
                             const float* matrix, const float* read, const void* norm_w, float eps, void* h,
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

// The Mamba-2 step of a decode step (one token per sequence, row s is sequence s), as one kernel: the
// causal conv, the selective scan and the gated RMSNorm of lx_conv1d_varlen, lx_ssm_scan_varlen and
// lx_gated_rmsnorm, with the same bits. zxbcdt f32 [n_seq, ld] is in_proj's output in Mamba-2's order
// z | xBC | dt: z is columns [0, I), xBC [I, I + conv_dim) with conv_dim = I + 2 G N, dt columns
// [I + conv_dim, I + conv_dim + H), where I = H P (ld >= 2 I + 2 G N + H). Sequence s's conv state (slot
// slot_ids[s], as lx_conv1d_varlen keeps it; zeros when has_state[s] == 0) and SSM state
// ([slots, H, P, N] f32) are each read once and written once. y f32 [n_seq, I] receives the scan's output
// before the gate, and out bf16 [n_seq, I] the gated norm of y with z and norm_w (bf16 [I]), as
// lx_gated_rmsnorm with groups = G. sync: int32 [n_seq * G] arrival counters for the groups, zero on the
// first call; the kernel leaves them zero after every call. A row's bits do not depend on the other rows.
// The bits are those of the three kernels run in turn when the gated norm's K = I / G is at least 256 (the
// sums run over the same 256 threads), which holds for every Metis model. Returns LX_ERR_NOT_SUPPORTED on a
// backend without it (the model then runs the three kernels).
LX_API int lx_mamba_decode_step(const float* zxbcdt, int ld, const void* conv_w, const void* conv_b, float* conv_state,
                                float* ssm_state, const float* A_log, const float* D, const float* dt_bias,
                                const int32_t* slot_ids, const int32_t* has_state, const void* norm_w, float eps,
                                float* y, void* out, int32_t* sync, int n_seq, int H, int P, int G, int N, int W,
                                void* stream);

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
// lx_moe_summary, and history = a * history + b * summary on the same elements (one launch: the layer's history
// update, glue::axpby, with the summary it reads). history f32 [T, R], updated in place.
LX_API int lx_moe_summary_history(const int32_t* ids, const float* weights, const void* expert_emb, float* summary,
                                  float* history, float a, float b, int T, int R, int top_k, void* stream);

// ---- Decode GEMV (gemv.cu) ------------------------------------------------------------------
// Matrix-vector products for decode steps (M <= 8 rows), memory-bound by design: one pass over
// the weights, which dominate the traffic, at close to DRAM bandwidth. Weights are bf16, or int8
// with one f32 scale per output row (w = q * scale[n], symmetric, see lx_quantize_w8). The input
// may be f32 or bf16 and is read directly, so no cast pass is needed before the product.
//   y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y[m, n]
// x: [M, K] (x_dtype, row stride ld_x); bias: [N] (bias_dtype, or null); y: [M, N] (y_dtype,
// row stride ld_y; read only when beta != 0). Accumulation in f32. Deterministic (no atomics).
typedef enum { LX_W_BF16 = 0, LX_W_INT8 = 1, LX_W_FP8 = 2, LX_W_NVFP4 = 3, LX_W_BF16X = 4 } lx_wtype;
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

// ---- Model glue (glue.cu) -------------------------------------------------------------------
// Small kernels the Metis forward pass needs between the library kernels. They are specific to how
// the model wires its pieces together, so they are plain ops that every backend provides.
// y = a * y + b * x over n f32 elements.
LX_API int lx_axpby_f32(float* y, const float* x, float a, float b, int64_t n, void* stream);
// out = a - b over n f32 elements.
LX_API int lx_sub_f32(const float* a, const float* b, float* out, int64_t n, void* stream);
// meta[t, col .. col + 3] = mean_k, entropy[t] * inv_log_experts, confidence[t], continuation[t] (bf16),
// with `ld` the row stride of meta in elements.
LX_API int lx_meta_scalars(const float* entropy, const float* confidence, const float* continuation, float mean_k,
                           float inv_log_experts, void* meta, int ld, int col, int T, void* stream);
// out[i, :] = src[cu[i + 1] - 1, :] for each of n_seq sequences (cu on the device).
LX_API int lx_gather_last(const float* src, const int32_t* cu, float* out, int n_seq, int D, void* stream);
// out[t, :] = bf16 of [a[t] | b[t] | c[t] | d[t]]: a, b and c are D wide, d is R wide, each an f32 row of its
// own width, and out is one row of 3 * D + R bf16 per token (the routing features' input).
LX_API int lx_concat4_bf16(const float* a, const float* b, const float* c, const float* d, void* out, int D, int R,
                           int T, void* stream);
// The router's input once the features are ready: latent[t, :Lat] = bf16(router_in[t, :Lat]), then
// router_in[t, Lat :] = features[t, :R] (f32). router_in is one row of Lat + R f32 per token.
LX_API int lx_pack_router(float* router_in, void* latent, const float* features, int Lat, int R, int T, void* stream);
// A memory entry's metadata, one row of R + 4 bf16 per token: meta[t, :R] = bf16(summary[t, :R]), then the four
// scalars of lx_meta_scalars (mean_k, entropy * inv_log_experts, confidence, continuation).
LX_API int lx_meta_entry(const float* summary, const float* entropy, const float* confidence,
                         const float* continuation, float mean_k, float inv_log_experts, void* meta, int R, int T,
                         void* stream);

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
// LX_COPY_DEFAULT takes the direction from the pointers (unified addressing), for a copy whose ends
// may each be device or host memory.
typedef enum { LX_COPY_H2D = 0, LX_COPY_D2H = 1, LX_COPY_D2D = 2, LX_COPY_DEFAULT = 3 } lx_copy_kind;

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
// Events that can be read back as time: lx_rt_event_elapsed gives the milliseconds between two
// recorded events (both made by lx_rt_event_create_timed; lx_rt_event_create events do not time).
LX_API int lx_rt_event_create_timed(void** event);
LX_API int lx_rt_event_elapsed(void* start, void* stop, float* ms);
// Waits for all the work on the device, every stream.
LX_API int lx_rt_device_sync(void);
// The text of an error code this ABI returned, for messages.
LX_API const char* lx_rt_error_string(int status);
// 1 when `ptr` is pinned host memory that the device reads across the bus (lx_rt_host_alloc, or a range
// registered or mapped for it), else 0. A backend whose memory is all host memory answers 0.
LX_API int lx_rt_is_host_memory(const void* ptr);
// One device's properties (ordinal from 0). cc_major / cc_minor are its real compute capability,
// sms its multiprocessor count, l2_bytes its L2 cache, mem_clock_khz and bus_bits the memory clock
// and bus width (peak bandwidth is clock * 2 * bus_bits / 8). The versions use the CUDA encoding,
// 1000 * major + 10 * minor, of the driver and of the runtime; a backend without one reports 0.
// The fields after those are what a backend can add, zero or empty where it has nothing to say (the GPU
// backends leave them): l1_bytes one core's L1 data cache and l3_bytes the cache past L2; read_bandwidth
// the bytes per second the backend measured its kernels reading memory at (the host takes it over the
// peak that mem_clock_khz and bus_bits give); isa the instruction set the kernels run, in words; and
// native_math bit (1 << m) set for each lx_math m whose products the hardware multiplies with an
// instruction of its own rather than by expanding the operands.
typedef struct {
  char name[256];
  int cc_major, cc_minor;
  int sms;
  int64_t l2_bytes;
  int64_t mem_clock_khz;
  int bus_bits;
  int driver_version, runtime_version;
  int64_t l1_bytes, l3_bytes;
  int64_t read_bandwidth;
  char isa[192];
  int native_math;
} lx_device_props;
LX_API int lx_rt_device_props(int ordinal, lx_device_props* props);
// The device memory this process holds, its context included, summed over the devices, where the driver
// reports it per process (NVML on CUDA). LX_ERR_NOT_SUPPORTED where the backend cannot tell.
LX_API int lx_rt_process_memory(int64_t* bytes);

// The memory read bandwidth the device sustains, measured. A read-only kernel over every SM reads `bytes` of device
// memory on `stream` (16-byte loads, several in flight per thread, grid-stride) after one warm-up, `reps` times, and
// *read_gbps is the best run (at least 1 rep). The buffer is pseudo-random words, not zeros: uniform memory can be
// cheaper to read. Four variants are tried, the default load and the L1 no-allocate load, each unrolled 4 and 8 ways,
// and the best is kept; LX_BANDWIDTH_TRACE=1 prints each one's rate to stderr. Pass bytes well past the L2 (1 GiB, the
// roofline's probe) so the reads are DRAM's. bytes must be a multiple of 16 and at least 64 KiB.
// *spec_gbps is the device's peak: memory clock x 2 x bus width / 8, from cudaDeviceGetAttribute. A 16 GB sm_120 card
// whose attributes give another figure reports 896 GB/s instead, the RTX 5070 Ti's data rate (28 Gbps on a 256-bit
// bus), since the attributes are not what the card's datasheet says there. CPU and Metal: LX_ERR_NOT_SUPPORTED (the
// CPU's own reading is lx_device_props' read_bandwidth).
LX_API int lx_bandwidth_probe(int64_t bytes, int reps, double* read_gbps, double* spec_gbps, void* stream);

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
//   LX_W_BF16X  lossless compressed bf16: exactly the checkpoint's bf16 values in ~12.1 to 12.3 bits
//               each. data is a sequence of N fixed-size rows of lx_bf16x_row_bytes(K) bytes (then
//               rarely an overflow area); scale unused. Each row: u32 patch slots (at least 2 + K /
//               2048), one base exponent byte per 128 k, padding to 8 bytes, then per 128 k a 192-byte block:
//               128 bytes of sign << 7 | mantissa, then 64 bytes of 4-bit codes (exponent = base -
//               code). Values further than 15 binades below their block's base are listed in the
//               row's patch slots as (k << 8 | exponent). Full layout: cuda/bf16x.cuh.
// Stacked expert weights are [E, N, K] in the same format, every expert laid out as above, one
// after the other (scales likewise [E, N] or [E, N, K / 16]). An NVFP4 stack's experts each have a
// global scale in the checkpoint (weight_scale_2 is [E]); folding those into the e4m3 block scales
// would round them twice, so the stack carries them: expert_global is [E] f32 on the device, and
// null means every expert uses `global`. Consumers of a stack read the expert's own global.
// For LX_W_BF16X a stack is simply E * N rows.
typedef struct {
  lx_wtype type;
  const void* data;
  const void* scale;          // f32 [N] for INT8 / FP8, e4m3 [N, K / 16] for NVFP4, null for BF16
  float global;               // NVFP4 only (1 otherwise)
  const float* expert_global;  // NVFP4 stacks only: f32 [E] on the device, or null
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
// beta, M <= 8). Weight-only: the weight is expanded in registers, the math is f32. LX_W_BF16X
// weights are run by lx_gemv_bf16x (below), so every format enters through this one call.
LX_API int lx_gemv_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                     lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                     float beta, void* stream);

// Tensor-core GEMM on any weight format, for prefill and batched decode:
//   y[M, N] = alpha * (x[M, K] @ W[N, K]^T + bias[N]) + beta * y
// x bf16 or f32 (quantized on the fly when math needs it, into `workspace`, at least
// lx_gemm_workspace_bytes(M, K, math) bytes). Same dtype rules for bias and y as lx_gemm_bf16.
// Deterministic: the same call gives the same bits (small products on sm_90+ may split K across a
// thread block cluster; the partial sums are added in a fixed order). A row's bits do not depend on the
// other rows of the call either (every tile adds the K steps in one order), with two exceptions that
// change with M: that split-K, and BF16 x BF16, which runs on cuBLASLt and its per-shape algorithm.
LX_API int lx_gemm_w(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, lx_math math, const void* bias,
                     lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                     float beta, void* workspace, void* stream);
LX_API int64_t lx_gemm_workspace_bytes(int M, int K, lx_math math);

// ---- Skinny decode GEMM (gemm_skinny.cu) -----------------------------------------------------------
// y[m, n] = alpha * (sum_k x[m, k] * w[n, k] + bias[n]) + beta * y[m, n] for the decode products of a step: one row
// per decoding token, up to 64 rows per call. A larger M runs as chunks of 64, each row computed the same way.
//
// Reduction order (the contract, for every format of this family):
//   1. The K-tiles (16-wide k-steps, 64 k per tile) of a (K, N) are cut into S(K, N) contiguous slices, slice r taking
//      tiles [r * KT / S, (r + 1) * KT / S) of the KT = K / 64 tiles. Each slice is one f32 accumulation chain: it starts
//      from 0 and takes its k-steps in ascending order: acc = mma.m16n8k16(x[m, k0:k0+16], w[n, k0:k0+16], acc).
//      The mma sums its 16 products in the hardware's fixed order, so each chain is the same on every launch.
//   2. The S partial sums of an output are added in f32 in ascending slice order: ((p0 + p1) + p2) + ... They are
//      exchanged through the cluster's distributed shared memory: no atomics, no workspace, one launch.
//   3. The epilogue is f32: v = alpha * (sum + bias[n]) + beta * y[m, n] (the beta term only when beta != 0), rounded
//      once to y's dtype.
//   4. S and the tile shape depend only on (K, N) and the card's SM count (two CTAs per SM is the target), never on M.
//      So the bits of output row m depend only on x[m, :], the weight and the epilogue constants: not on M, on where the
//      row sits in the call, or on the other rows, present or not. A decode token's bits are the same whatever the
//      number of streams in its step, from one row up. The bits do depend on the card's SM count, so a card with a
//      different SM count may split K differently; each card is internally invariant.
//      tests/kernels/test_gemm_skinny.py checks this at M = 1, 2, 8, 16, 32, 64 and across chunks of 64.
// x is bf16; bias and y are bf16 or f32 (bias may be null). The bf16-weight kernel needs N a multiple of 16 and K a
// multiple of 64, and returns LX_ERR_NOT_SUPPORTED otherwise (also for f32 x, until its loads are pipelined). The tile
// shape (BM rows, BN columns) is chosen per call; it does not change any bits. Other formats join with the same order.
LX_API int lx_gemm_skinny(const void* x, lx_dtype x_dtype, int ld_x, const lx_weight* w, const void* bias,
                          lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                          float beta, void* stream);

// Per-token activation quantization: the first step of lx_gemm_w and lx_moe_experts_grouped under the
// low-precision maths, exposed so one quantization can serve several products of the same x.
// x: [M, K] (x_dtype, row stride ld_x). Per row, with a = max_k |x[m, k]|:
//   LX_MATH_INT8  q int8 [M, ld_q]: q = round_half_even(x * (127 / a)) clamped to [-127, 127];
//                 scale[m] = a / 127
//   LX_MATH_FP8   q e4m3 [M, ld_q]: q = e4m3(x * (448 / a)); scale[m] = a / 448
//   LX_MATH_FP4   q [M, ld_q], two e2m1 per byte (low nibble = even k); block_scale e4m3 [M, ld_bs], one
//                 per 16 k; scale[m] = g = a / (448 * 6). Block b with absmax a_b:
//                 s_b = e4m3(a_b / (6 * g)), q = e2m1(x * (1 / (s_b * g))), and q = 0 where s_b = 0.
//                 Requires K % 16 == 0.
// Every rounding is to nearest, ties to even, saturating at the format's largest value; the products
// and quotients are float32. A zero row gets scale 1 and q = 0. ld_q and ld_bs are in bytes;
// block_scale is used for FP4 only.
LX_API int lx_quantize_act(const void* x, lx_dtype x_dtype, int ld_x, lx_math math, void* q, int ld_q, float* scale,
                           void* block_scale, int ld_bs, int M, int K, void* stream);

// Routed experts on any weight format (contract of lx_moe_experts; gate_up [E, 2I, L],
// down [E, L, I] as lx_weight stacks). Two implementations with one contract:
//   lx_moe_experts_decode  few tokens (T <= 8): one pass over the chosen experts' weights,
//                          weight-only, f32 math. Workspace: lx_moe_decode_workspace_bytes.
//   lx_moe_experts_grouped any T: tokens grouped by expert, a tensor-core grouped GEMM per
//                          stage with the given math. Workspace: lx_moe_grouped_workspace_bytes.
//                          The top_k weighted rows of a token are summed in a fixed order, so a token's bits
//                          do not depend on its batch or on the run. LX_MOE_ORDERED=0 sums with f32 atomics
//                          instead, whose last bits can differ between calls.
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

// The reverse: out[r, c] = w[r, c] as bf16 for an [rows, cols] matrix in any format, with its row
// factor applied (the row scale, or the NVFP4 global). The loader uses it to hold a quantized
// embedding's bf16 gather table, and the acceptance checks compare dequantized weights with BF16.
LX_API int lx_dequantize_weight(const lx_weight* w, void* out, int64_t rows, int cols, void* stream);

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

// ---- Lossless BF16 (bf16x.cu) ---------------------------------------------------------------
// LX_W_BF16X: the checkpoint's bf16 weights in 12.1 to 12.3 bits each, reproducing every bit, so a
// decode product reads ~1.3x fewer bytes with no change at all to its result. The products below are
// bit-identical to the bf16 kernels they stand in for (same lane mapping, split and summation order):
// a model that swaps them in produces the same logits to the last bit.

// Bytes of one encoded row of K values (-1 for K out of range: 0 < K < 2^23).
LX_API int64_t lx_bf16x_row_bytes(int K);
// Bytes the encoding of w (bf16 [rows, K], row-major; a stack [E, N, K] is E * N rows) needs: rows *
// lx_bf16x_row_bytes(K), plus overflow records for rows with more patches than their slots (none in
// Praxis). Synchronizes the stream; scratch from cudaMallocAsync. Returns -(CUDA error) on failure.
LX_API int64_t lx_bf16x_bytes(const void* w, int64_t rows, int K, void* stream);
// Encodes w into out (8-byte aligned, out_bytes >= lx_bf16x_bytes). Deterministic: the same weights
// always give the same bytes. Synchronizes the stream once; for load time, not for graph capture.
LX_API int lx_bf16x_encode(const void* w, int64_t rows, int K, void* out, int64_t out_bytes, void* stream);
// Decodes an encoding back to bf16 [rows, K] (for tests and for checking a load).
LX_API int lx_bf16x_decode(const void* data, int64_t rows, int K, void* w, void* stream);

// lx_gemv for LX_W_BF16X weights w ([N, K] encoded, 8-byte aligned): the same contract (x, bias, y,
// alpha, beta, M <= 8, f32 accumulation, deterministic, programmatic dependent launch), and y equal
// bit for bit to lx_gemv on the bf16 weights the encoding came from. It runs the bf16 order class of the
// shape, so like lx_gemv a row's bits do not depend on M.
LX_API int lx_gemv_bf16x(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias,
                         lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha,
                         float beta, void* stream);
// The bf16 baseline the format is measured against: lx_gemv on LX_W_BF16 weights, kept as an entry
// point for callers built against it.
LX_API int lx_gemv_decode_bf16(const void* x, lx_dtype x_dtype, int ld_x, const void* w, const void* bias,
                               lx_dtype bias_dtype, void* y, lx_dtype y_dtype, int ld_y, int M, int N, int K,
                               float alpha, float beta, void* stream);

// Routed experts on LX_W_BF16X stacks, for decode (few tokens): the contract of lx_moe_experts with
// gate_up the encoding of [E, 2I, L] and down of [E, L, I] (16-byte aligned), a caller-owned
// workspace (16-byte aligned, at least lx_moe_bf16x_workspace_bytes(T, I, top_k)), and top_k <= 8.
// out equals lx_moe_experts on the bf16 stacks bit for bit. Nothing is allocated, so the call can be
// captured in a CUDA graph.
LX_API int lx_moe_experts_bf16x(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                                const void* down, float* out, void* workspace, int T, int L, int I, int top_k,
                                void* stream);
// The same kernels on plain bf16 stacks: the baseline the format is measured against, and a
// drop-in decode replacement for lx_moe_experts (bit-identical to it, graph-capturable).
LX_API int lx_moe_experts_decode_bf16(const void* latent, const int32_t* ids, const float* weights,
                                      const void* gate_up, const void* down, float* out, void* workspace, int T,
                                      int L, int I, int top_k, void* stream);
LX_API int64_t lx_moe_bf16x_workspace_bytes(int T, int I, int top_k);

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

// The persisting L2 window: the kernels of every graph captured while it is set get an access policy window over
// [base, base + bytes), and their accesses to it are kept in L2 (persisting) while the rest of their traffic streams
// past. The device's persisting carve-out is sized to the window, clamped to what the device grants (its persisting
// maximum and its access policy window maximum). hit_ratio is the fraction of the window's accesses given the
// persisting property (1 keeps all of them). granted_bytes (may be null) receives the size the device granted, 0 when
// nothing is set. Three uses:
//   bytes <= 0:          clears the window and the persisting lines.
//   base null, bytes>0: only asks: granted_bytes is what a window of bytes would get; nothing changes. The model sizes
//                        its arena from this before it allocates.
//   base and bytes > 0: sets the window; granted_bytes is what it got.
// Eager launches are not windowed. A backend without the carve-out (and every backend but CUDA) returns
// LX_ERR_NOT_SUPPORTED, and the model then runs unpinned.
LX_API int lx_rt_l2_persist(const void* base, int64_t bytes, float hit_ratio, int64_t* granted_bytes);

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

// Constrained decoding: sets to -inf the logits that a token mask rules out. Entry j applies its
// mask bits[j * ((V + 31) / 32) ...] (one bit per token) to logits row rows[j] ([rows, V] f32).
LX_API int lx_mask_logits(float* logits, int V, const int32_t* rows, const uint32_t* bits, int n, void* stream);

// ---- Speculative decoding (spec.cu) ---------------------------------------------------------
// A verify step feeds a sequence its pending token and k drafted tokens in one step and keeps the
// logits of every position; afterwards only the prefix the target accepted stays. Attention just
// truncates: later steps overwrite the cache rows past the kept length. The Mamba-2 state cannot
// be truncated, so in speculative mode each Mamba layer holds, per slot:
//   the base state (conv_state and ssm_state, as lx_conv1d_varlen and lx_ssm_scan_varlen keep
//   them), covering every position up to the slot's folded length, and
//   up to LX_SPEC_MAX_PENDING pending positions after it, as the inputs their state update needs:
//   conv_hist f32 [histories, LX_SPEC_MAX_PENDING, C] (the conv inputs) and ssm_hist f32
//   [histories, lx_spec_ssm_hist_floats(H, P, N)] (each head's x, B and dt_raw).
// A history is shared, not per slot: hist_ids[s] names the history sequence s uses (null: history 0 for every
// sequence). The engine keeps one history for the one stream that verifies at a time, and folds its pending positions
// before another stream takes it over (MetisModel::forward_spec).
// Each step a sequence names fold[s] (its first pending positions, folded into the base state for
// good) and keep[s] (the pending positions after those, which stay pending); pending positions past
// fold + keep are dropped (rejected drafts). The step's new tokens run from the state after all
// fold + keep positions and become pending positions after the kept ones. Requires
// fold + keep <= pending positions held and keep + new tokens <= LX_SPEC_MAX_PENDING. has_state[s]
// says whether the base state holds anything (0: it reads as zeros). The base state is written only
// when fold > 0. A fold replays each position with the float operations of the step that first ran
// it, so the state after folding j positions is bit-identical to feeding those j tokens normally,
// and so is every output of the new tokens.
#define LX_SPEC_MAX_PENDING 8

// Floats of ssm_hist per slot.
LX_API int64_t lx_spec_ssm_hist_floats(int H, int P, int N);

// lx_conv1d_varlen with pending positions (contract above): out for the new tokens; conv_state
// changes only when fold > 0; conv_hist ends as [kept rows, new rows].
LX_API int lx_spec_conv1d(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                          float* conv_hist, const int32_t* slot_ids, const int32_t* hist_ids,
                          const int32_t* has_state, const int32_t* fold, const int32_t* keep,
                          const int32_t* cu_seqlens, int n_seq, float* out, int C, int W, void* stream);

// lx_ssm_scan_varlen with pending positions (contract above): y for the new tokens; ssm_state
// changes only when fold > 0; ssm_hist ends as [kept rows, new rows] for every head.
LX_API int lx_spec_ssm_scan(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B, const float* C,
                            int ld_bc, const float* A_log, const float* D, const float* dt_bias, float* ssm_state,
                            float* ssm_hist, const int32_t* slot_ids, const int32_t* hist_ids,
                            const int32_t* has_state, const int32_t* fold, const int32_t* keep,
                            const int32_t* cu_seqlens, int n_seq, float* y, int H, int P, int G, int N, void* stream);

// lx_gemm_f32 for decode-sized steps (M <= 8): y[M, N] = x[M, K] @ w[N, K]^T + bias[N], every row
// summed in an order that does not depend on M, so a row's bits never depend on the other rows.
// The routers and the depth head of a step of up to 8 tokens go through this, plain decode and
// verify steps alike: with it a verify step's logits equal single-token steps' bit for bit.
LX_API int lx_spec_gemv_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M,
                            int N, int K, void* stream);

// Sampling parameters of one row: lx_sampling's filters (sampler.cu), without the penalties.
typedef struct {
  float temperature;  // <= 0 is greedy
  float top_p;        // in (0, 1]; 1 disables
  int32_t top_k;      // keeps that many (1 .. V - 1); 0, or V or more, disables
  float min_p;        // in [0, 1]; 0 disables
} lx_spec_filter;

// The draw's key: seed and the index of the output token being drawn, as in lx_sampling.
typedef struct {
  uint64_t seed;
  uint64_t position;
} lx_spec_rng;

// probs[r, :] = the distribution lx_sample draws row r from: greedy rows are a point mass on the
// argmax (ties to the lower index); otherwise softmax(logits / temperature) over the tokens that
// survive top-k, top-p and min-p, renormalized, zeros elsewhere. The kept set is computed exactly
// as sampler.cu computes it. logits, probs f32 [rows, V]; filters [rows].
LX_API int lx_spec_probs(const float* logits, int V, const lx_spec_filter* filters, int rows, float* probs,
                         void* stream);

// Sampling penalties of one sequence: lx_sampling's, without the filters.
typedef struct {
  float repetition_penalty;  // > 0; 1 disables
  float presence_penalty;    // 0 disables
  float frequency_penalty;   // 0 disables
} lx_spec_penalty;

// The penalties lx_sample applies to its logits (sampler.cu, step 1), for the rows of a verify step of one sequence, in
// place on logits [rows, V]. Row j is the distribution after the step's first j + 1 tokens, so it is penalised with the
// sequence's counts (seen and out_counts: its slot's rows of the sampler's tables) plus tokens[1 .. j], the drafts the
// step fed before that row, which count as output once they are accepted. Each row is then the distribution a plain
// step at that position samples from, and the verify path draws from the same one. tokens is device data.
LX_API int lx_spec_penalize(float* logits, int V, lx_spec_penalty pen, const int32_t* tokens, int rows,
                            const int32_t* seen, const int32_t* out_counts, void* stream);

// Speculative sampling over a verify step. Sequence s fed tokens[cu[s] .. cu[s + 1] - 1]: its pending
// token, then k_s drafts. Row cu[s] + j of p (from lx_spec_probs on the target's logits) is the
// target's distribution after the step's first j + 1 tokens; row cu[s] + j of q is the drafter's
// distribution that draft j + 1 (tokens[cu[s] + j + 1]) was drawn from, or q is null for drafters
// that propose a token outright (a point mass). Draft j + 1 holds with probability
// min(1, p(d) / q(d)); at the first that fails the step ends with a token drawn from
// max(0, p - q) renormalized, and when all k_s hold, with a token drawn from the last row. Then the
// emitted tokens (accepted drafts and the final one) are distributed exactly as plain sampling from
// p. accepted[s] = drafts that held (0 .. k_s), next[s] = the final token. Draws are Philox keyed by
// (rng[s].seed, rng[s].position + j), so they are reproducible and independent of the batch.
LX_API int lx_spec_verify(const float* p, const float* q, int V, const int32_t* tokens, const int32_t* cu_seqlens,
                          const lx_spec_rng* rng, int n_seq, int32_t* accepted, int32_t* next, void* stream);

// Greedy verification over the same layout, on the target's logits: argmax[r] = the argmax of row r
// (ties to the lower index, rows = cu[n_seq]); drafts hold while each equals the argmax of the row
// before it. accepted[s] = drafts that held, next[s] = the argmax of the row after the last one.
LX_API int lx_spec_verify_greedy(const float* logits, int V, const int32_t* tokens, const int32_t* cu_seqlens,
                                 int n_seq, int rows, int32_t* argmax, int32_t* accepted, int32_t* next, void* stream);

// tokens[i] = a draw from probs[rows[i], :] (rows may be null: row i), Philox keyed by
// (rng[i].seed, rng[i].position, domain). For drafters that sample their proposals.
LX_API int lx_spec_sample(const float* probs, int V, const int32_t* rows, const lx_spec_rng* rng, uint32_t domain,
                          int n, int32_t* tokens, void* stream);

// ---- Block drafter (spec.cu) ----------------------------------------------------------------
// A one-pass block drafter (runtime/block_drafter.h, format lernex.block/1) is dense layers over a
// block of up to 8 rows; these are its parts the dense kernels don't cover.

// In place on f32 rows: for each row r and head h of x[r, h * head_dim ..], with norm_w (bf16
// [head_dim]) v = v * rsqrt(mean(v^2) + eps) * norm_w (no norm when null), then RoPE in the
// rotate-halves layout: (v[i], v[i + head_dim / 2]) rotate by (pos0 + r) * theta^(-2i / head_dim),
// the angle taken in double.
LX_API int lx_spec_head_norm_rope(float* x, int ld_x, int rows, int heads, int head_dim, const void* norm_w,
                                  float eps, int64_t pos0, double theta, void* stream);

// out[b, h, :] = softmax_j(q[b, h, :] . key_j * scale) @ value_j over the n_ctx context rows
// (k_ctx, v_ctx f32 [n_ctx, KVH, head_dim], any order) and the B block rows (k_blk, v_blk f32 rows of
// KVH * head_dim at stride ld_blk), every query seeing all of them; head h reads kv head
// h / (H / KVH). q f32 [B, H * head_dim] (stride ld_q), out f32 (stride ld_out). Keys are read as
// float4: head_dim a multiple of 4 dividing 256, ld_blk a multiple of 4, k_ctx and k_blk 16-byte
// aligned. n_ctx + B + head_dim + 256 floats must fit 48 KB of shared memory (n_ctx to about 11900).
LX_API int lx_spec_block_attention(const float* q, int ld_q, const float* k_ctx, const float* v_ctx, int n_ctx,
                                   const float* k_blk, const float* v_blk, int ld_blk, float* out, int ld_out, int B,
                                   int H, int KVH, int head_dim, float scale, void* stream);

// The reverse: out[r, c] = w[r, c] as bf16 for an [rows, cols] matrix in any format, with its row
// factor applied (the row scale, or the NVFP4 global). The loader uses it to hold a quantized
// embedding's bf16 gather table, and the acceptance checks compare dequantized weights with BF16.
LX_API int lx_dequantize_weight(const lx_weight* w, void* out, int64_t rows, int cols, void* stream);

// =============================================================================================
// ---- Decode programs (decode_*.cu; the Decode Overhead workstream) ---------------------------
// =============================================================================================
// A decode step (M <= 8 tokens, one per sequence) runs as one persistent kernel. The step is built
// once per shape as a program of ops (products, the depth-memory read, the Mamba step, routing,
// experts, attention, the memory write, the depth head), each op is split into tasks spread evenly
// over the SMs, and every SM runs its tasks in program order. Products read their weight rows
// straight from global memory on the tensor cores (bf16, and the other formats as in the note below), with the
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
// Runs on sm_80 and newer (mma.sync; bulk L2 prefetches from sm_90). Products and experts read BF16, BF16X, INT8,
// FP8 and NVFP4 weights, and the routers and the depth head read LX_W_F32; the memory entry reads its w_state in any
// of these but LX_W_F32. Each format's values reach the tensor cores exactly as f16 (INT8, FP8, NVFP4: the block
// scale of an NVFP4 word is folded in exactly) or as bf16 (BF16, BF16X: decoded from the record as they are read),
// and a caller falls back to the per-kernel path for other formats.

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
// history: history = 0.5 * history + 0.5 * summary. E <= 256.
typedef struct {
  const float* logits; const float* selection_bias; int32_t* ids; float* weights; float* entropy; float* confidence;
  const void* expert_emb; float* summary; float* history; int E, top_k, R;
} lx_select_args;
LX_API int lx_program_select(lx_program* p, const lx_select_args* a);

// out[m] = sum_k weights[m, k] expert(ids[m, k], latent[m]) + shared(latent[m]), each expert as in
// lx_moe_experts (intermediates rounded to bf16); the shared expert's product is added to the routed sum.
// latent bf16 [M, L]; gate_up [E, 2I, L], down [E, L, I], shared_gate_up [2Is, L], shared_down [L, Is].
// The shared expert's ops wait only for the latent. Returns the last op (the routed down product).
// A block reads all of its routed rows at once, on the tensor cores for BF16 and INT8 stacks.
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
