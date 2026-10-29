// The sampler: one launch picks the next token of every sequence in a batch. Each row is one
// block. The row's logits go through the sequence's penalties, then either an argmax (greedy) or
// a Gumbel-max draw over the tokens that survive top-k, top-p and min-p. The draw's noise is
// Philox keyed by (seed, output position, token id), so the same request gives the same tokens
// whatever else shares the batch.
#include <cstdint>

#include "cuda/common.cuh"

static_assert(sizeof(lx_sampling) == 56, "lx_sampling is mirrored by ctypes in tests/kernels/test_sampler.py");

namespace {

constexpr int kThreads = 1024;
constexpr int kMaxTopN = 20;
// Candidates held in shared memory for the top-p search; more than this searches the vocabulary.
constexpr int kCandidates = 1024;

// Order-preserving map from float to uint32: larger floats get larger keys.
__device__ __forceinline__ uint32_t float_key(float v) {
  const uint32_t u = __float_as_uint(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Philox4x32-10: a counter-based generator, so draws depend only on (key, counter).
__device__ __forceinline__ void philox4x32(uint32_t c[4], uint32_t k0, uint32_t k1) {
  constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u, kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
#pragma unroll
  for (int round = 0; round < 10; ++round) {
    const uint32_t hi0 = __umulhi(kM0, c[0]), lo0 = kM0 * c[0];
    const uint32_t hi1 = __umulhi(kM1, c[2]), lo1 = kM1 * c[2];
    const uint32_t n0 = hi1 ^ c[1] ^ k0, n1 = lo1, n2 = hi0 ^ c[3] ^ k1, n3 = lo0;
    c[0] = n0, c[1] = n1, c[2] = n2, c[3] = n3;
    k0 += kW0;
    k1 += kW1;
  }
}

// Standard Gumbel noise for element `index` of draw `position` of sequence `seed`. The uniform has
// 23 bits and lies strictly inside (0, 1), so both logarithms are finite.
__device__ __forceinline__ float gumbel_noise(uint64_t seed, uint64_t position, uint32_t index) {
  uint32_t c[4] = {index, static_cast<uint32_t>(position), static_cast<uint32_t>(position >> 32), 0x4C58u};
  philox4x32(c, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const float u = ((c[0] >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return -logf(-logf(u));
}

// Block-wide max, broadcast to every thread. `scratch` holds 32 floats.
__device__ __forceinline__ float block_max(float v, float* scratch) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  v = lx::warp_max(v);
  __syncthreads();
  if (lane == 0) scratch[warp] = v;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  v = lane < warps ? scratch[lane] : -INFINITY;
  return lx::warp_max(v);
}

// Block-wide argmax of (value, index) with ties to the lower index, broadcast to every thread.
// `scratch` holds 32 values and 32 indices.
__device__ __forceinline__ void block_argmax(float v, int idx, float* sv, int* si, float& out_v, int& out_i) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
  }
  __syncthreads();
  if (lane == 0) sv[warp] = v, si[warp] = idx;
  __syncthreads();
  const int warps = (blockDim.x + 31) >> 5;
  v = lane < warps ? sv[lane] : -INFINITY;
  idx = lane < warps ? si[lane] : 0x7fffffff;
  for (int o = 16; o > 0; o >>= 1) {
    const float ov = __shfl_xor_sync(0xffffffffu, v, o);
    const int oi = __shfl_xor_sync(0xffffffffu, idx, o);
    if (ov > v || (ov == v && oi < idx)) v = ov, idx = oi;
  }
  out_v = v;
  out_i = idx;
}

struct Shared {
  float red[32];
  int redi[32];
  uint32_t hist[256];
  uint32_t cand_key[kCandidates];
  float cand_exp[kCandidates];
  int cand_count;
  int cand_offset[kThreads];
  int chosen[kMaxTopN];
  uint32_t prefix, mask;
  int need, digit;
  float bis_sum;
  float max_scaled;
};

// One sampling row. The buffers are indexed by the row's sampling index i (work) and by its slot
// (seen, out_counts); logits by its row.
__global__ void __launch_bounds__(kThreads) sample_kernel(const float* __restrict__ logits, int V,
                                                          const lx_sampling* __restrict__ params,
                                                          int32_t* __restrict__ seen,
                                                          int32_t* __restrict__ out_counts, float* __restrict__ work,
                                                          int32_t* __restrict__ tokens,
                                                          float* __restrict__ token_logprob, int top_n,
                                                          int32_t* __restrict__ top_tokens,
                                                          float* __restrict__ top_logprobs) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ Shared sh;
  const int i = blockIdx.x;
  const lx_sampling p = params[i];
  const float* l = logits + static_cast<int64_t>(p.row) * V;
  float* w = work + static_cast<int64_t>(i) * V;
  int32_t* sc = seen + static_cast<int64_t>(p.slot) * V;
  int32_t* oc = out_counts + static_cast<int64_t>(p.slot) * V;
  const int tid = threadIdx.x;

  // Greedy with neutral penalties and no log-probabilities (the common decode step): the argmax of the raw logits,
  // read once. The penalties leave w == l here and both paths keep the lowest index among equal maxima, so the token
  // is the general path's, and the counts take the same update as step 8 below.
  if (!(p.temperature > 0.0f) && p.repetition_penalty == 1.0f && p.frequency_penalty == 0.0f &&
      p.presence_penalty == 0.0f && token_logprob == nullptr && top_n == 0) {
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += blockDim.x) {
      if (l[e] > best) best = l[e], best_i = e;
    }
    float bv;
    int bi;
    block_argmax(best, best_i, sh.red, sh.redi, bv, bi);
    if (tid == 0) {
      tokens[i] = bi;
      atomicAdd(&sc[bi], 1);
      atomicAdd(&oc[bi], 1);
    }
    return;
  }

  // 1. Penalties on the raw logits: repetition over prompt and output, presence and frequency over
  // the output alone.
  for (int e = tid; e < V; e += blockDim.x) {
    float x = l[e];
    if (p.repetition_penalty != 1.0f && sc[e] > 0) x = x > 0.0f ? x / p.repetition_penalty : x * p.repetition_penalty;
    const int32_t co = oc[e];
    if (co > 0) x -= p.frequency_penalty * static_cast<float>(co) + p.presence_penalty;
    w[e] = x;
  }
  __syncthreads();

  int32_t token = 0;
  const bool greedy = !(p.temperature > 0.0f);
  if (greedy) {
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += blockDim.x) {
      if (w[e] > best) best = w[e], best_i = e;
    }
    float bv;
    int bi;
    block_argmax(best, best_i, sh.red, sh.redi, bv, bi);
    token = bi;
  } else {
    // 2. Temperature, then the largest scaled logit.
    const float inv_t = 1.0f / p.temperature;
    float m = -INFINITY;
    for (int e = tid; e < V; e += blockDim.x) {
      const float s = w[e] * inv_t;
      w[e] = s;
      m = fmaxf(m, s);
    }
    m = block_max(m, sh.red);
    if (tid == 0) sh.max_scaled = m;
    __syncthreads();
    const float top = sh.max_scaled;

    // 3. top-k: the key of the k-th largest scaled logit, by a 4-pass radix select over the keys.
    uint32_t k_key = 0;  // 0 keeps everything
    const bool use_top_k = p.top_k > 0 && p.top_k < V;
    if (use_top_k) {
      if (tid == 0) sh.prefix = 0, sh.mask = 0, sh.need = p.top_k;
      __syncthreads();
      for (int pass = 0; pass < 4; ++pass) {
        const int shift = 24 - 8 * pass;
        for (int b = tid; b < 256; b += blockDim.x) sh.hist[b] = 0;
        __syncthreads();
        const uint32_t prefix = sh.prefix, mask = sh.mask;
        for (int e = tid; e < V; e += blockDim.x) {
          const uint32_t key = float_key(w[e]);
          if ((key & mask) == prefix) atomicAdd(&sh.hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (tid == 0) {
          int cum = 0, digit = 0, need = sh.need;
          for (int b = 255; b >= 0; --b) {
            const int c = static_cast<int>(sh.hist[b]);
            if (cum + c >= need) {
              digit = b;
              break;
            }
            cum += c;
          }
          sh.need = need - cum;
          sh.digit = digit;
          sh.prefix = prefix | (static_cast<uint32_t>(sh.digit) << shift);
          sh.mask = mask | (0xFFu << shift);
        }
        __syncthreads();
      }
      k_key = sh.prefix;
      __syncthreads();
    }

    // The candidates: elements at or above the top-k key. Their order is fixed (each thread's
    // elements in index order, threads in order), so every sum below is reproducible.
    int count = 0;
    for (int e = tid; e < V; e += blockDim.x) count += float_key(w[e]) >= k_key;
    const float count_f = lx::block_sum(static_cast<float>(count), sh.red);  // exact below 2^24
    const int candidates = static_cast<int>(count_f + 0.5f);

    // 4. top-p: the largest key whose tokens still hold top_p of the top-k mass. Searched over the
    // candidates in shared memory when they fit, otherwise over the vocabulary.
    uint32_t p_key = 0;
    const bool use_top_p = p.top_p > 0.0f && p.top_p < 1.0f;
    if (use_top_p) {
      const bool small = candidates <= kCandidates;
      if (small) {
        if (tid == 0) sh.cand_count = 0;
        __syncthreads();
        // per-thread counts, then offsets in thread order
        int mine = 0;
        for (int e = tid; e < V; e += blockDim.x) mine += float_key(w[e]) >= k_key;
        sh.cand_offset[tid] = mine;
        __syncthreads();
        if (tid == 0) {
          int acc = 0;
          for (int t = 0; t < kThreads; ++t) {
            const int c = sh.cand_offset[t];
            sh.cand_offset[t] = acc;
            acc += c;
          }
          sh.cand_count = acc;
        }
        __syncthreads();
        int at = sh.cand_offset[tid];
        for (int e = tid; e < V; e += blockDim.x) {
          const uint32_t key = float_key(w[e]);
          if (key >= k_key) {
            sh.cand_key[at] = key;
            sh.cand_exp[at] = expf(w[e] - top);
            ++at;
          }
        }
        __syncthreads();
      }
      // total mass of the top-k set
      float mass_k = 0.0f;
      if (small) {
        for (int j = tid; j < candidates; j += blockDim.x) mass_k += sh.cand_exp[j];
      } else {
        for (int e = tid; e < V; e += blockDim.x)
          if (float_key(w[e]) >= k_key) mass_k += expf(w[e] - top);
      }
      const float total = lx::block_sum(mass_k, sh.red);
      const float target = p.top_p * total;

      auto mass_from = [&](uint32_t key) -> float {
        float acc = 0.0f;
        if (small) {
          for (int j = tid; j < candidates; j += blockDim.x)
            if (sh.cand_key[j] >= key) acc += sh.cand_exp[j];
        } else {
          for (int e = tid; e < V; e += blockDim.x)
            if (float_key(w[e]) >= key) acc += expf(w[e] - top);
        }
        return lx::block_sum(acc, sh.red);
      };

      // Invariant: mass_from(lo) >= target, mass_from(hi) < target.
      uint32_t lo = k_key, hi = float_key(top) + 1u;  // top is finite, so this cannot wrap
      while (hi - lo > 1u) {
        const uint32_t mid = lo + (hi - lo) / 2u;
        if (mass_from(mid) >= target)
          lo = mid;
        else
          hi = mid;
      }
      p_key = lo;
      __syncthreads();
    }

    // 5. min-p: keep tokens at least min_p times as likely as the most likely one.
    uint32_t mp_key = 0;
    if (p.min_p > 0.0f) mp_key = float_key(top + logf(fminf(p.min_p, 1.0f)));

    const uint32_t keep_key = max(max(k_key, p_key), mp_key);

    // 6. Gumbel-max draw over the kept tokens: argmax of scaled logit + Gumbel noise.
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += blockDim.x) {
      if (float_key(w[e]) >= keep_key) {
        const float v = w[e] + gumbel_noise(p.seed, p.position, static_cast<uint32_t>(e));
        if (v > best) best = v, best_i = e;
      }
    }
    float bv;
    int bi;
    block_argmax(best, best_i, sh.red, sh.redi, bv, bi);
    token = bi;
  }

  // 7. Log-probabilities of the raw logits, for the chosen token and the top_n most likely.
  if (token_logprob != nullptr || top_n > 0) {
    float raw_max = -INFINITY;
    for (int e = tid; e < V; e += blockDim.x) raw_max = fmaxf(raw_max, l[e]);
    raw_max = block_max(raw_max, sh.red);
    float raw_sum = 0.0f;
    for (int e = tid; e < V; e += blockDim.x) raw_sum += expf(l[e] - raw_max);
    const float lse = raw_max + logf(lx::block_sum(raw_sum, sh.red));
    if (token_logprob != nullptr && tid == 0) token_logprob[i] = l[token] - lse;

    for (int j = 0; j < top_n; ++j) {
      float best = -INFINITY;
      int best_i = 0x7fffffff;
      for (int e = tid; e < V; e += blockDim.x) {
        bool taken = false;
        for (int c = 0; c < j; ++c) taken |= sh.chosen[c] == e;
        if (!taken && l[e] > best) best = l[e], best_i = e;
      }
      float bv;
      int bi;
      block_argmax(best, best_i, sh.red, sh.redi, bv, bi);
      if (tid == 0) {
        sh.chosen[j] = bi;
        top_tokens[static_cast<int64_t>(i) * top_n + j] = bi;
        top_logprobs[static_cast<int64_t>(i) * top_n + j] = l[bi] - lse;
      }
      __syncthreads();
    }
  }

  // 8. The sequence's counts: the chosen token is now part of its output.
  if (tid == 0) {
    tokens[i] = token;
    atomicAdd(&sc[token], 1);
    atomicAdd(&oc[token], 1);
  }
}

__global__ void observe_kernel(int32_t* __restrict__ counts, const int32_t* __restrict__ tokens, int n) {
  lx::pdl_trigger();
  lx::pdl_wait();
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) atomicAdd(&counts[tokens[i]], 1);
}

}  // namespace

extern "C" int lx_sample(const float* logits, int V, const lx_sampling* rows, int n, int32_t* seen,
                         int32_t* out_counts, float* work, int32_t* tokens, float* token_logprob, int top_n,
                         int32_t* top_tokens, float* top_logprobs, void* stream) {
  if (n <= 0) return 0;
  if (top_n < 0 || top_n > kMaxTopN || V <= 0 || V > (1 << 24)) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  lx::launch_pdl(sample_kernel, n, kThreads, 0, s, logits, V, rows, seen, out_counts, work, tokens, token_logprob, top_n,
                                       top_tokens, top_logprobs);
  return lx::launch_status();
}

extern "C" int lx_sampler_reset(int32_t* seen, int32_t* out_counts, int slot, int V, void* stream) {
  auto s = static_cast<cudaStream_t>(stream);
  const size_t offset = static_cast<size_t>(slot) * V;
  if (cudaMemsetAsync(seen + offset, 0, sizeof(int32_t) * V, s) != cudaSuccess) return lx::launch_status();
  if (cudaMemsetAsync(out_counts + offset, 0, sizeof(int32_t) * V, s) != cudaSuccess) return lx::launch_status();
  return lx::launch_status();
}

extern "C" int lx_sampler_observe(int32_t* seen, int32_t* out_counts, int slot, int V, const int32_t* tokens, int n,
                                  int is_output, void* stream) {
  if (n <= 0) return 0;
  auto s = static_cast<cudaStream_t>(stream);
  const size_t offset = static_cast<size_t>(slot) * V;
  lx::launch_pdl(observe_kernel, (n + 255) / 256, 256, 0, s, seen + offset, tokens, n);
  if (is_output) lx::launch_pdl(observe_kernel, (n + 255) / 256, 256, 0, s, out_counts + offset, tokens, n);
  return lx::launch_status();
}

// Constrained decoding: the logits of each listed row that the token mask rules out become -inf.
// bits[j] is the mask of entry j (one bit per token, row-major, (V + 31) / 32 words).
__global__ void mask_logits_kernel(float* logits, int V, const int32_t* rows, const uint32_t* bits, int words) {
  lx::pdl_trigger();
  lx::pdl_wait();
  const int j = blockIdx.y;
  const int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= V) return;
  const uint32_t word = bits[static_cast<size_t>(j) * words + (v >> 5)];
  if (!((word >> (v & 31)) & 1u)) logits[static_cast<size_t>(rows[j]) * V + v] = -INFINITY;
}

extern "C" int lx_mask_logits(float* logits, int V, const int32_t* rows, const uint32_t* bits, int n, void* stream) {
  if (n <= 0) return 0;
  auto s = static_cast<cudaStream_t>(stream);
  const int words = (V + 31) / 32;
  lx::launch_pdl(mask_logits_kernel, dim3((V + 255) / 256, n), 256, 0, s, logits, V, rows, bits, words);
  return lx::launch_status();
}
