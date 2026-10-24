// Depth memory: the per-token attention over its bank, the gated fuse into the streams, and the
// depth head's last step. Keys and values are projected when an entry is written, so a read only
// attends.
#include "cuda/common.cuh"

namespace {

// A token's bank holds at most 8 entries (Metis). Scores stay in registers at this cap.
constexpr int kMaxEntries = 8;

// Attend runs one warp per (token, stream) row.
constexpr int kWarpsPerBlock = 4;

inline int row_threads(int dim) { return dim >= 256 ? 256 : ((dim + 31) / 32) * 32; }

__device__ __forceinline__ float sigmoid(float v) { return 1.0f / (1.0f + expf(-v)); }

// out[row, :] = softmax over the valid entries of (q[row] . k[e]) / sqrt(M), times v[e]. Lanes split
// M for the dot and the output; each score is a warp sum, so every lane holds the whole softmax.
// Entries past n_entries are never read, so a bank can hold stale or NaN slots there.
__global__ void memory_attend_kernel(const float* q, const float* bank_k, const float* bank_v, int n_entries,
                                     int max_entries, __nv_bfloat16* out, int rows, int S, int M) {
  const int row = blockIdx.x * kWarpsPerBlock + (threadIdx.x >> 5);
  if (row >= rows) return;  // the whole warp shares the row, so it leaves together
  const int lane = threadIdx.x & 31;
  const int t = row / S;
  const float* qr = q + static_cast<int64_t>(row) * M;
  const float* kb = bank_k + static_cast<int64_t>(t) * max_entries * M;
  const float* vb = bank_v + static_cast<int64_t>(t) * max_entries * M;
  const float scale = rsqrtf(static_cast<float>(M));

  float score[kMaxEntries];
#pragma unroll
  for (int e = 0; e < kMaxEntries; ++e) {
    if (e < n_entries) {
      float dot = 0.0f;
      for (int m = lane; m < M; m += 32) dot += qr[m] * kb[e * M + m];
      score[e] = lx::warp_sum(dot) * scale;
    }
  }

  // Subtract the max before exp: scores are unbounded, and exp overflows past about 88.
  float top = score[0];
#pragma unroll
  for (int e = 1; e < kMaxEntries; ++e)
    if (e < n_entries) top = fmaxf(top, score[e]);
  float total = 0.0f;
#pragma unroll
  for (int e = 0; e < kMaxEntries; ++e) {
    if (e < n_entries) {
      score[e] = expf(score[e] - top);
      total += score[e];
    }
  }

  const float norm = 1.0f / total;
  for (int m = lane; m < M; m += 32) {
    float acc = 0.0f;
#pragma unroll
    for (int e = 0; e < kMaxEntries; ++e)
      if (e < n_entries) acc += score[e] * vb[e * M + m];
    out[static_cast<int64_t>(row) * M + m] = __float2bfloat16_rn(acc * norm);
  }
}

// One block per token. The gate for each stream is a dot over its own row, taken before that row
// is updated, so the gate sees the streams as they were. Streams are handled one at a time, each
// with its own block reduction. The summary reads only projected, so it runs last.
__global__ void memory_fuse_kernel(float* streams, const float* projected, const __nv_bfloat16* gate,
                                   const __nv_bfloat16* gate_bias, float* summary, int S, int D) {
  __shared__ float scratch[32];
  const int t = blockIdx.x;
  const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
  const int64_t base = static_cast<int64_t>(t) * S * D;

  for (int s = 0; s < S; ++s) {
    float* row = streams + base + static_cast<int64_t>(s) * D;
    const float* proj = projected + base + static_cast<int64_t>(s) * D;
    float dot = 0.0f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) dot += row[d] * lx::to_f32(gate[s * D + d]);
    // block_sum hands every thread the total, so each one gates its own columns.
    const float g = sigmoid(lx::block_sum(dot, scratch) * inv_sqrt_d + lx::to_f32(gate_bias[s]));
    for (int d = threadIdx.x; d < D; d += blockDim.x) row[d] += g * proj[d];
  }

  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += projected[base + static_cast<int64_t>(s) * D + d];
    summary[static_cast<int64_t>(t) * D + d] = acc / S;
  }
}

// The depth head's last step, one block per token. The two RMS terms are two block reductions.
// Thread 0 writes the result, so the sigmoid runs once per token.
__global__ void continuation_kernel(const float* logit, const float* state, const float* difference, float limit,
                                    float* out, int D) {
  __shared__ float scratch[32];
  const int t = blockIdx.x;
  const float* st = state + static_cast<int64_t>(t) * D;
  const float* df = difference + static_cast<int64_t>(t) * D;
  float ss = 0.0f, dd = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float a = st[d], b = df[d];
    ss += a * a;
    dd += b * b;
  }
  const float state_ms = lx::block_sum(ss, scratch);
  const float diff_ms = lx::block_sum(dd, scratch);

  // The 1e-6 floor keeps the ratio finite for an all-zero state.
  const float scale = fmaxf(sqrtf(state_ms / D), 1e-6f);
  const float innovation = sqrtf(diff_ms / D);
  const float z = fminf(fmaxf(logit[t] + tanhf(innovation / scale) - 0.5f, -limit), limit);
  if (threadIdx.x == 0) out[t] = sigmoid(z);
}

}  // namespace

extern "C" int lx_memory_attend(const float* q, const float* bank_k, const float* bank_v, int n_entries,
                                int max_entries, void* out, int T, int S, int M, void* stream) {
  if (n_entries < 1 || n_entries > max_entries || n_entries > kMaxEntries)
    return static_cast<int>(cudaErrorInvalidValue);
  if (T <= 0 || S <= 0 || M <= 0) return 0;
  const int rows = T * S;
  memory_attend_kernel<<<(rows + kWarpsPerBlock - 1) / kWarpsPerBlock, 32 * kWarpsPerBlock, 0,
                         static_cast<cudaStream_t>(stream)>>>(
      q, bank_k, bank_v, n_entries, max_entries, static_cast<__nv_bfloat16*>(out), rows, S, M);
  return lx::launch_status();
}

extern "C" int lx_memory_fuse(float* streams, const float* projected, const void* gate, const void* gate_bias,
                              float* summary, int T, int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  memory_fuse_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      streams, projected, static_cast<const __nv_bfloat16*>(gate), static_cast<const __nv_bfloat16*>(gate_bias),
      summary, S, D);
  return lx::launch_status();
}

extern "C" int lx_continuation_finish(const float* logit, const float* state, const float* difference, float limit,
                                      float* out, int T, int D, void* stream) {
  if (T <= 0 || D <= 0) return 0;
  continuation_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      logit, state, difference, limit, out, D);
  return lx::launch_status();
}
