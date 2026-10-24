// Depth memory: the per-token attention over its bank, the gated fuse into the streams, and the
// depth head's last step. Keys and values are projected when an entry is written, so a read only
// attends. At decode (T <= kDecodeTokens) attend loads each row's vectors up front, and fuse splits a
// token over a cluster of blocks, so one token uses several SMs rather than one.
#include <algorithm>
#include <cmath>
#include <random>
#include <vector>

#include "cuda/common.cuh"

namespace {

// A token's bank holds at most 8 entries (Metis). Scores stay in registers at this cap.
constexpr int kMaxEntries = 8;

// Attend runs one warp per (token, stream) row.
constexpr int kWarpsPerBlock = 4;

// Decode path: at most this many tokens per step take the spread-out kernels.
constexpr int kDecodeTokens = 8;

// Streams held in registers by the decode fuse, and the cluster width a token's columns split over.
constexpr int kMaxStreams = 8;
constexpr int kCluster = 8;

// A decode block holds one column per thread (see streams.cu): slices are at most this wide.
constexpr int kClusterThreads = 512;

// Decode attend: a lane holds this many float4 of a row, so the vector path covers M <= 4 * 32 * this.
constexpr int kLaneVecs = 2;

inline int row_threads(int dim) { return dim >= 256 ? 256 : ((dim + 31) / 32) * 32; }

// Columns of a token's row owned by one block of its cluster.
inline int slice_width(int dim) { return (dim + kCluster - 1) / kCluster; }

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

// Decode attend: one warp per (token, stream) row, one row per block. Every load the row needs is
// issued before any math (the query, then each entry's key and value), so the warp waits on memory
// once rather than once per entry. Lanes take float4 chunks of M, and the eight score sums are
// independent warp reductions that overlap. Needs M % 4 == 0 for the vector loads to stay aligned.
__global__ void memory_attend_vec_kernel(const float* q, const float* bank_k, const float* bank_v, int n_entries,
                                         int max_entries, __nv_bfloat16* out, int rows, int S, int M) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const int t = row / S;
  const int chunks = M / 4;  // float4 per row
  const float4* qr = reinterpret_cast<const float4*>(q + static_cast<int64_t>(row) * M);
  const float4* kb = reinterpret_cast<const float4*>(bank_k + static_cast<int64_t>(t) * max_entries * M);
  const float4* vb = reinterpret_cast<const float4*>(bank_v + static_cast<int64_t>(t) * max_entries * M);

  // Every load is unconditional, with indices clamped into the row and the bank: a guarded load
  // would split the warp's memory traffic into one round trip per entry. Clamped entries and lanes
  // past chunks re-read finite data that the arithmetic below never uses.
  const int last = n_entries - 1;
  float4 qv[kLaneVecs], kv[kMaxEntries][kLaneVecs], vv[kMaxEntries][kLaneVecs];
#pragma unroll
  for (int j = 0; j < kLaneVecs; ++j) qv[j] = qr[min(lane + 32 * j, chunks - 1)];
#pragma unroll
  for (int e = 0; e < kMaxEntries; ++e) {
    const int be = min(e, last) * chunks;
#pragma unroll
    for (int j = 0; j < kLaneVecs; ++j) {
      const int c = min(lane + 32 * j, chunks - 1);
      kv[e][j] = kb[be + c];
      vv[e][j] = vb[be + c];
    }
  }

  float score[kMaxEntries];
#pragma unroll
  for (int e = 0; e < kMaxEntries; ++e) {
    float dot = 0.0f;
#pragma unroll
    for (int j = 0; j < kLaneVecs; ++j) {
      if (lane + 32 * j < chunks)
        dot += qv[j].x * kv[e][j].x + qv[j].y * kv[e][j].y + qv[j].z * kv[e][j].z + qv[j].w * kv[e][j].w;
    }
    score[e] = dot;
  }
  const float scale = rsqrtf(static_cast<float>(M));
#pragma unroll
  for (int e = 0; e < kMaxEntries; ++e) score[e] = lx::warp_sum(score[e]) * scale;

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

#pragma unroll
  for (int j = 0; j < kLaneVecs; ++j) {
    const int c = lane + 32 * j;
    if (c < chunks) {
      float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
      for (int e = 0; e < kMaxEntries; ++e) {
        if (e < n_entries) {
          acc.x += score[e] * vv[e][j].x;
          acc.y += score[e] * vv[e][j].y;
          acc.z += score[e] * vv[e][j].z;
          acc.w += score[e] * vv[e][j].w;
        }
      }
      __nv_bfloat162* dst = reinterpret_cast<__nv_bfloat162*>(out + static_cast<int64_t>(row) * M + 4 * c);
      dst[0] = __floats2bfloat162_rn(acc.x * norm, acc.y * norm);
      dst[1] = __floats2bfloat162_rn(acc.z * norm, acc.w * norm);
    }
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

// Decode fuse. Block `rank` of a token's cluster owns the columns [rank * W, rank * W + W), one per
// thread. Each block forms its slice's share of every stream's gate dot and pushes it into every
// peer. After the cluster barrier each block sums the shares in rank order, so all blocks get the
// same totals, then gates and updates only its own columns. The dots read the streams as they were:
// each column is read and written by its one owning thread, and nothing is written before the barrier.
__global__ void __launch_bounds__(kClusterThreads) LX_CLUSTER_DIMS(kCluster)
memory_fuse_cluster_kernel(float* streams, const float* projected, const __nv_bfloat16* gate,
                           const __nv_bfloat16* gate_bias, float* summary, int S, int D, int W) {
  __shared__ float warp_dots[32][kMaxStreams];
  __shared__ float parts[kCluster][kMaxStreams];  // the cluster's shares, [rank][stream]

  const int rank = static_cast<int>(lx::cluster_block_rank());
  const int t = blockIdx.x / kCluster;
  const int d = rank * W + threadIdx.x;
  const bool own = threadIdx.x < W && d < D;
  const int64_t base = static_cast<int64_t>(t) * S * D;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;

  // Padding threads read a valid column (dc) and never write it; streams past S are never loaded.
  const int dc = min(d, D - 1);
  float x[kMaxStreams], p[kMaxStreams], g_in[kMaxStreams];
#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s) {
    const int64_t at = base + static_cast<int64_t>(s) * D + dc;
    x[s] = (s < S) ? streams[at] : 0.0f;
    p[s] = (s < S) ? projected[at] : 0.0f;
    g_in[s] = (s < S) ? lx::to_f32(gate[s * D + dc]) : 0.0f;
  }
  float dot[kMaxStreams];
#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s) dot[s] = (own && s < S) ? x[s] * g_in[s] : 0.0f;

#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s) dot[s] = lx::warp_sum(dot[s]);
  if (lane == 0) {
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s) warp_dots[warp][s] = dot[s];
  }
  __syncthreads();

  // Push each stream's share of this block into every peer's slot for this rank: one remote store
  // per (peer, stream). Every thread that pushes a stream computes the same share from the warp sums.
  const int warps = (blockDim.x + 31) / 32;
  for (int i = threadIdx.x; i < kCluster * S; i += blockDim.x) {
    const int s = i % S;
    const int k = i / S;
    float share = 0.0f;
    for (int w = 0; w < warps; ++w) share += warp_dots[w][s];
    *lx::cluster_peer(&parts[rank][s], k) = share;
  }
  lx::cluster_sync();

  // Every share is now in this block's parts. Sum them in rank order: all blocks get identical totals.
  float total[kMaxStreams];
#pragma unroll
  for (int s = 0; s < kMaxStreams; ++s) {
    total[s] = 0.0f;
#pragma unroll
    for (int k = 0; k < kCluster; ++k)
      if (s < S) total[s] += parts[k][s];
  }

  if (own) {
    const float inv_sqrt_d = rsqrtf(static_cast<float>(D));
    float acc = 0.0f;
#pragma unroll
    for (int s = 0; s < kMaxStreams; ++s) {
      if (s < S) {
        const float g = sigmoid(total[s] * inv_sqrt_d + lx::to_f32(gate_bias[s]));
        streams[base + static_cast<int64_t>(s) * D + d] = x[s] + g * p[s];
        acc += p[s];
      }
    }
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
  if (T <= kDecodeTokens && M % 4 == 0 && M <= 4 * 32 * kLaneVecs) {
    memory_attend_vec_kernel<<<rows, 32, 0, static_cast<cudaStream_t>(stream)>>>(
        q, bank_k, bank_v, n_entries, max_entries, static_cast<__nv_bfloat16*>(out), rows, S, M);
    return lx::launch_status();
  }
  memory_attend_kernel<<<(rows + kWarpsPerBlock - 1) / kWarpsPerBlock, 32 * kWarpsPerBlock, 0,
                         static_cast<cudaStream_t>(stream)>>>(
      q, bank_k, bank_v, n_entries, max_entries, static_cast<__nv_bfloat16*>(out), rows, S, M);
  return lx::launch_status();
}

namespace {

constexpr int kFuseCluster = 1;  // the variant index of the cluster path ("cluster8"); 0 is "block"

// Whether the decode fuse's cluster path can run for this step.
bool fuse_cluster_ok(int T, int S, int D) {
  return T <= kDecodeTokens && S <= kMaxStreams && slice_width(D) <= kClusterThreads && lx_arch() >= lx::kClusterArch;
}

int launch_memory_fuse(bool cluster, float* streams, const float* projected, const void* gate, const void* gate_bias,
                       float* summary, int T, int S, int D, cudaStream_t s) {
  if (cluster) {
    const int W = slice_width(D);
    const int threads = (W + 31) / 32 * 32;  // one thread per owned column
    memory_fuse_cluster_kernel<<<T * kCluster, threads, 0, s>>>(streams, projected,
                                                                static_cast<const __nv_bfloat16*>(gate),
                                                                static_cast<const __nv_bfloat16*>(gate_bias), summary,
                                                                S, D, W);
    return lx::launch_status();
  }
  memory_fuse_kernel<<<T, row_threads(D), 0, s>>>(streams, projected, static_cast<const __nv_bfloat16*>(gate),
                                                  static_cast<const __nv_bfloat16*>(gate_bias), summary, S, D);
  return lx::launch_status();
}

// The autotuner's runner: both paths on fixed random inputs. The summary is the output the tuner
// compares; it reads only `projected`, so it does not change as the streams are updated run to run.
struct FuseInputs {
  int T = 0, S = 0, D = 0;
  float *streams = nullptr, *projected = nullptr;
  void *gate = nullptr, *gate_bias = nullptr;
  void reset() {
    for (void* p : {static_cast<void*>(streams), static_cast<void*>(projected), gate, gate_bias})
      if (p) cudaFree(p);
    *this = FuseInputs{};
  }
};

int tune_memory_fuse(int variant, const lx::TuneProblem& p, float* out, void* stream) {
  const int T = p.M, D = p.N, S = p.K;
  if (S <= 0 || S > kMaxStreams) return static_cast<int>(cudaErrorNotSupported);
  if (variant == kFuseCluster && !fuse_cluster_ok(T, S, D)) return static_cast<int>(cudaErrorNotSupported);
  static FuseInputs in;
  if (in.T != T || in.S != S || in.D != D) {
    in.reset();
    in.T = T;
    in.S = S;
    in.D = D;
    std::mt19937 rng(11);
    std::normal_distribution<float> normal;
    std::vector<float> host(static_cast<size_t>(T) * S * D);
    for (float& v : host) v = normal(rng);
    cudaMalloc(reinterpret_cast<void**>(&in.streams), host.size() * sizeof(float));
    cudaMemcpy(in.streams, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    for (float& v : host) v = normal(rng);
    cudaMalloc(reinterpret_cast<void**>(&in.projected), host.size() * sizeof(float));
    cudaMemcpy(in.projected, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::vector<__nv_bfloat16> gate(static_cast<size_t>(S) * D), bias(S);
    for (auto& v : gate) v = __float2bfloat16(normal(rng) * 0.1f);
    for (auto& v : bias) v = __float2bfloat16(normal(rng) * 0.1f);
    cudaMalloc(&in.gate, gate.size() * 2);
    cudaMemcpy(in.gate, gate.data(), gate.size() * 2, cudaMemcpyHostToDevice);
    cudaMalloc(&in.gate_bias, bias.size() * 2);
    cudaMemcpy(in.gate_bias, bias.data(), bias.size() * 2, cudaMemcpyHostToDevice);
  }
  return launch_memory_fuse(variant == kFuseCluster, in.streams, in.projected, in.gate, in.gate_bias, out, T, S, D,
                            static_cast<cudaStream_t>(stream));
}

const lx::TunableRegistration reg_memory_fuse("memory_fuse", {"block", "cluster8"}, &tune_memory_fuse);

}  // namespace

extern "C" int lx_memory_fuse(float* streams, const float* projected, const void* gate, const void* gate_bias,
                              float* summary, int T, int S, int D, void* stream) {
  if (T <= 0 || S <= 0 || D <= 0) return 0;
  // The table's choice for this shape when the autotuner measured one; otherwise clusters where they run.
  const int tuned = lx_tune_get("memory_fuse", T, D, S);
  const bool cluster = fuse_cluster_ok(T, S, D) && (tuned < 0 || tuned == kFuseCluster);
  return launch_memory_fuse(cluster, streams, projected, gate, gate_bias, summary, T, S, D,
                            static_cast<cudaStream_t>(stream));
}

extern "C" int lx_continuation_finish(const float* logit, const float* state, const float* difference, float limit,
                                      float* out, int T, int D, void* stream) {
  if (T <= 0 || D <= 0) return 0;
  continuation_kernel<<<T, row_threads(D), 0, static_cast<cudaStream_t>(stream)>>>(
      logit, state, difference, limit, out, D);
  return lx::launch_status();
}
