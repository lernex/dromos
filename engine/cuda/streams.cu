// Embedding and the residual streams. Token-major kernels run one block per token: the token's
// row is read once and all S streams of it are written by the same block. That is right for
// prefill, but at decode (T <= kDecodeTokens) one block walks a whole row in serial memory round
// trips while the rest of the GPU idles. The decode kernels spread each row over the SMs instead:
// mhc_read splits a row over a cluster of blocks, and mhc_write splits it over column blocks.
#include <cooperative_groups.h>

#include <algorithm>

#include "cuda/common.cuh"

namespace cg = cooperative_groups;

namespace {

// mhc_read keeps every stream's value for a column in registers, so the stream count is a
// compile-time cap. Metis runs 4 streams.
constexpr int kMaxStreams = 8;

// Decode path: at most this many tokens per step take the spread-out kernels. memory.cu has the
// same constants for its own decode kernels.
constexpr int kDecodeTokens = 8;

// A decode token's row is split over a cluster of this many blocks, one column slice each. Eight is
// the largest portable cluster size.
constexpr int kCluster = 8;

// A decode block holds one column per thread, so a slice (row / kCluster columns) is at most this
// wide. Metis rows (2048, 2560) give slices of 256 and 320; wider rows take the one-block path.
constexpr int kClusterThreads = 512;

// Threads for one token's D-wide row: a warp multiple, at most 256. Row loops cover any D.
inline int row_threads(int dim) { return dim >= 256 ? 256 : ((dim + 31) / 32) * 32; }

// Columns of a token's row owned by one block of its cluster.
inline int slice_width(int dim) { return (dim + kCluster - 1) / kCluster; }

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

// Decode mhc_read. Block `rank` of a token's cluster owns the columns [rank * W, rank * W + W), one per
// thread. The RMSNorm needs the whole row's sum of squares: each block pushes its slice's sum into
// every peer, and after the cluster barrier every block adds the kCluster sums in rank order, so all
// blocks get the same total.
__global__ void __launch_bounds__(kClusterThreads) __cluster_dims__(kCluster, 1, 1)
mhc_read_cluster_kernel(const float* streams, const float* matrix, const float* read_weights,
                        const __nv_bfloat16* norm_w, float eps, float* mixed, __nv_bfloat16* h, int S, int D,
                        int W) {
  __shared__ float sm_matrix[kMaxStreams * kMaxStreams];
  __shared__ float sm_read[kMaxStreams];
  __shared__ float parts[kCluster];    // the cluster's sums of squares, by rank, filled by the peers
  __shared__ float scratch[32];

  cg::cluster_group cluster = cg::this_cluster();
  const int rank = static_cast<int>(cluster.block_rank());
  const int t = blockIdx.x / kCluster;
  const int d = rank * W + threadIdx.x;
  const bool own = threadIdx.x < W && d < D;
  const float* in = streams + static_cast<int64_t>(t) * S * D;
  float* mix = mixed + static_cast<int64_t>(t) * S * D;

  // The column loads go out first; the per-pass constants are staged in shared memory while they fly.
  // Padding threads read a valid column (dc) and never write it.
  const int dc = min(d, D - 1);
  float x[kMaxStreams];
#pragma unroll
  for (int i = 0; i < kMaxStreams; ++i) x[i] = (i < S) ? in[i * D + dc] : 0.0f;
  for (int k = threadIdx.x; k < S * S; k += blockDim.x) sm_matrix[k] = matrix[k];
  for (int k = threadIdx.x; k < S; k += blockDim.x) sm_read[k] = read_weights[k];
  __syncthreads();

  float src = 0.0f;
#pragma unroll
  for (int i = 0; i < kMaxStreams; ++i)
    if (i < S) src += sm_read[i] * x[i];
  if (own) {
#pragma unroll
    for (int o = 0; o < kMaxStreams; ++o) {
      if (o < S) {
        float acc = 0.0f;
#pragma unroll
        for (int i = 0; i < kMaxStreams; ++i)
          if (i < S) acc += sm_matrix[o * S + i] * x[i];
        mix[o * D + d] = acc;
      }
    }
  }

  // Push this block's sum of squares into every peer's slot for its rank. After the one cluster
  // barrier each slot is filled, and each block reads only its own shared memory, so no block can
  // exit while a peer still reads it.
  const float block_sq = lx::block_sum(own ? src * src : 0.0f, scratch);
  if (threadIdx.x < kCluster) *cluster.map_shared_rank(&parts[rank], threadIdx.x) = block_sq;
  cluster.sync();

  float total = 0.0f;
  for (int k = 0; k < kCluster; ++k) total += parts[k];
  if (own) {
    const float scale = rsqrtf(total / D + eps);
    h[static_cast<int64_t>(t) * D + d] =
        lx::from_f32<__nv_bfloat16>(src * scale * __bfloat162float(norm_w[d]));
  }
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

// Decode mhc_write: one thread per column, a grid of column blocks per token, so one token's row
// covers many SMs. Each element is still read and written by the same thread, so aliasing holds.
__global__ void mhc_write_columns_kernel(const float* mixed, const float* write_weights, const float* update,
                                         float* streams, int S, int D) {
  const int t = blockIdx.y;
  const int d = blockIdx.x * blockDim.x + threadIdx.x;
  if (d >= D) return;
  const float u = update[static_cast<int64_t>(t) * D + d];
  const int64_t base = static_cast<int64_t>(t) * S * D + d;
  for (int s = 0; s < S; ++s) streams[base + s * D] = mixed[base + s * D] + write_weights[s] * u;
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
  if (T <= kDecodeTokens && slice_width(D) <= kClusterThreads) {
    const int W = slice_width(D);
    const int threads = (W + 31) / 32 * 32;  // one thread per owned column
    mhc_read_cluster_kernel<<<T * kCluster, threads, 0, static_cast<cudaStream_t>(stream)>>>(
        streams, matrix, read, static_cast<const __nv_bfloat16*>(norm_w), eps, mixed,
        static_cast<__nv_bfloat16*>(h), S, D, W);
    return lx::launch_status();
  }
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
  if (T <= kDecodeTokens) {
    const dim3 grid(static_cast<unsigned>((D + 255) / 256), static_cast<unsigned>(T));
    mhc_write_columns_kernel<<<grid, 256, 0, static_cast<cudaStream_t>(stream)>>>(
        mixed, write, update, streams, S, D);
    return lx::launch_status();
  }
  mhc_write_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      mixed, write, update, streams, S, D);
  return lx::launch_status();
}
