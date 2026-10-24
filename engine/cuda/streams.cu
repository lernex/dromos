// Embedding and the residual streams. Token-major kernels run one block per token: the token's
// row is read once and all S streams of it are written by the same block.
#include <algorithm>

#include "cuda/common.cuh"

namespace {

// mhc_read keeps every stream's value for a column in registers, so the stream count is a
// compile-time cap. Metis runs 4 streams.
constexpr int kMaxStreams = 8;

// Threads for one token's D-wide row: a warp multiple, at most 256. Row loops cover any D.
inline int row_threads(int dim) { return dim >= 256 ? 256 : ((dim + 31) / 32) * 32; }

// streams[t, s, :] = table[ids[t], :] + stream_emb[s, :]
__global__ void embed_kernel(const __nv_bfloat16* table, const int32_t* ids, const __nv_bfloat16* stream_emb,
                             float* streams, int S, int D) {
  const int t = blockIdx.x;
  const __nv_bfloat16* row = table + static_cast<int64_t>(ids[t]) * D;
  float* out = streams + static_cast<int64_t>(t) * S * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float v = lx::to_f32(row[d]);
    for (int s = 0; s < S; ++s) out[s * D + d] = v + lx::to_f32(stream_emb[s * D + d]);
  }
}

// mean[t, d] = (1 / S) sum_s streams[t, s, d]. One thread per output; tokens go over blockIdx.y.
__global__ void stream_mean_kernel(const float* streams, float* mean, int T, int S, int D) {
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  for (int t = blockIdx.y; t < T; t += gridDim.y) {
    const float* in = streams + static_cast<int64_t>(t) * S * D + d;
    float acc = 0.0f;
    for (int s = 0; s < S; ++s) acc += in[s * D];
    mean[static_cast<int64_t>(t) * D + d] = acc / S;
  }
}

// Mix, weighted read and RMSNorm in one pass per token. Each thread takes a stripe of columns, loads
// that column across the streams once, writes its mixed values and keeps the source value in shared
// memory until the block has the sum of squares for the RMSNorm.
__global__ void mhc_read_kernel(const float* streams, const float* matrix, const float* read_weights,
                                const __nv_bfloat16* norm_w, float eps, float* mixed, __nv_bfloat16* h,
                                int S, int D) {
  extern __shared__ float smem[];
  float* sm_matrix = smem;             // [S, S]
  float* sm_read = sm_matrix + S * S;  // [S]
  float* source = sm_read + S;         // [D]
  __shared__ float scratch[32];

  for (int k = threadIdx.x; k < S * S; k += blockDim.x) sm_matrix[k] = matrix[k];
  for (int k = threadIdx.x; k < S; k += blockDim.x) sm_read[k] = read_weights[k];
  __syncthreads();

  const int t = blockIdx.x;
  const float* in = streams + static_cast<int64_t>(t) * S * D;
  float* mix = mixed + static_cast<int64_t>(t) * S * D;
  float sq = 0.0f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    float x[kMaxStreams];
#pragma unroll
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) x[i] = in[i * D + d];

    float src = 0.0f;
#pragma unroll
    for (int i = 0; i < kMaxStreams; ++i)
      if (i < S) src += sm_read[i] * x[i];

    for (int o = 0; o < S; ++o) {
      float acc = 0.0f;
#pragma unroll
      for (int i = 0; i < kMaxStreams; ++i)
        if (i < S) acc += sm_matrix[o * S + i] * x[i];
      mix[o * D + d] = acc;
    }

    source[d] = src;
    sq += src * src;
  }

  const float scale = rsqrtf(lx::block_sum(sq, scratch) / D + eps);
  for (int d = threadIdx.x; d < D; d += blockDim.x)
    h[static_cast<int64_t>(t) * D + d] =
        lx::from_f32<__nv_bfloat16>(source[d] * scale * __bfloat162float(norm_w[d]));
}

// streams[t, s, :] = mixed[t, s, :] + write[s] * update[t, :]. mixed and streams may be the same
// buffer: each element is read by the thread that then writes it, so no other thread sees a change.
__global__ void mhc_write_kernel(const float* mixed, const float* write_weights, const float* update,
                                 float* streams, int S, int D) {
  const int t = blockIdx.x;
  const int64_t base = static_cast<int64_t>(t) * S * D;
  const float* up = update + static_cast<int64_t>(t) * D;
  for (int d = threadIdx.x; d < D; d += blockDim.x) {
    const float u = up[d];
    for (int s = 0; s < S; ++s) streams[base + s * D + d] = mixed[base + s * D + d] + write_weights[s] * u;
  }
}

}  // namespace

extern "C" int lx_embed_streams(const void* table, const int32_t* ids, const void* stream_emb, float* streams,
                                int T, int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  embed_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      static_cast<const __nv_bfloat16*>(table), ids, static_cast<const __nv_bfloat16*>(stream_emb), streams,
      S, D);
  return lx::launch_status();
}

extern "C" int lx_stream_mean(const float* streams, float* mean, int T, int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  const dim3 grid(static_cast<unsigned>((D + 255) / 256), static_cast<unsigned>(std::min(T, 65535)));
  stream_mean_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(streams, mean, T, S, D);
  return lx::launch_status();
}

extern "C" int lx_mhc_read(const float* streams, const float* matrix, const float* read, const void* norm_w,
                           float eps, float* mixed, void* h, int T, int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  if (S > kMaxStreams) return static_cast<int>(cudaErrorInvalidValue);
  // Dynamic shared memory: the mix, the read weights and one source row. Stays under the default
  // 48 KB per block for every width this engine uses (Praxis 2048, Logos 2560).
  const size_t smem = (static_cast<size_t>(S) * S + S + D) * sizeof(float);
  if (smem > 48 * 1024) return static_cast<int>(cudaErrorInvalidValue);
  mhc_read_kernel<<<T, row_threads(D), smem, static_cast<cudaStream_t>(stream)>>>(
      streams, matrix, read, static_cast<const __nv_bfloat16*>(norm_w), eps, mixed,
      static_cast<__nv_bfloat16*>(h), S, D);
  return lx::launch_status();
}

extern "C" int lx_mhc_write(const float* mixed, const float* write, const float* update, float* streams, int T,
                            int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  mhc_write_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      mixed, write, update, streams, S, D);
  return lx::launch_status();
}
