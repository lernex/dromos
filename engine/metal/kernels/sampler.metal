// The sampler (kernels.h, "Sampling"; host side in ops_sampler.mm): one threadgroup per row picks the next
// token of a sequence. The row's logits go through the sequence's penalties, then either an argmax
// (greedy) or a Gumbel-max draw over the tokens that survive top-k, top-p and min-p. The draw's noise is
// Philox keyed by (seed, output position, token id), so the same request gives the same token whatever
// else shares the batch. The steps follow sampler.cu one for one; lx_spec_probs (spec.metal) computes
// the kept set with the same helpers, so its distribution is the one drawn from here.

constant constexpr int kSampleThreads = 1024;
constant constexpr int kSampleTopN = 20;
// Candidates held in threadgroup memory for the top-p search; more than this searches the vocabulary.
constant constexpr int kSampleCandidates = 1024;

// Order-preserving map from float to uint: larger floats get larger keys.
inline uint float_key(float v) {
  const uint u = as_type<uint>(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Philox4x32-10: a counter-based generator, so draws depend only on (key, counter).
inline void philox4x32(thread uint* c, uint k0, uint k1) {
  constexpr uint kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u, kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    const uint hi0 = mulhi(kM0, c[0]), lo0 = kM0 * c[0];
    const uint hi1 = mulhi(kM1, c[2]), lo1 = kM1 * c[2];
    const uint n0 = hi1 ^ c[1] ^ k0, n1 = lo1, n2 = hi0 ^ c[3] ^ k1, n3 = lo0;
    c[0] = n0;
    c[1] = n1;
    c[2] = n2;
    c[3] = n3;
    k0 += kW0;
    k1 += kW1;
  }
}

// A uniform in (0, 1) with 23 bits, from one Philox draw keyed by (seed, position) at counter
// (index, domain): both logarithms of it are finite.
inline float philox_uniform(ulong seed, ulong position, uint index, uint domain) {
  uint c[4] = {index, uint(position), uint(position >> 32), domain};
  philox4x32(c, uint(seed), uint(seed >> 32));
  return (float(c[0] >> 9) + 0.5f) * (1.0f / 8388608.0f);
}

// Standard Gumbel noise for element `index` of draw `position` of sequence `seed`.
inline float gumbel_noise(ulong seed, ulong position, uint index) {
  return -log(-log(philox_uniform(seed, position, index, 0x4C58u)));
}

// Threadgroup-wide argmax of (value, index), ties to the lower index, on every thread. sv and si hold
// 32 entries each.
inline void group_argmax(float v, int idx, threadgroup float* sv, threadgroup int* si, uint tid, uint threads,
                         thread float& out_v, thread int& out_i) {
  const uint lane = tid & 31u, sg = tid >> 5;
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(v, o);
    const int oi = simd_shuffle_xor(idx, o);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lane == 0) {
    sv[sg] = v;
    si[sg] = idx;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint groups = (threads + 31u) >> 5;
  v = lane < groups ? sv[lane] : -INFINITY;
  idx = lane < groups ? si[lane] : 0x7fffffff;
  for (ushort o = 16; o > 0; o >>= 1) {
    const float ov = simd_shuffle_xor(v, o);
    const int oi = simd_shuffle_xor(idx, o);
    if (ov > v || (ov == v && oi < idx)) {
      v = ov;
      idx = oi;
    }
  }
  out_v = v;
  out_i = idx;
}

// What a row keeps: the scaled logits w (temperature applied) at or above the returned key survive top-k,
// top-p and min-p. `top` is the largest scaled logit. Shared by the sampler and lx_spec_probs.
struct KeepScratch {
  float red[32];
  uint hist[256];
  uint cand_key[kSampleCandidates];
  float cand_exp[kSampleCandidates];
  int cand_offset[kSampleThreads];
  uint prefix, mask;
  int need, digit;
};

inline uint kept_key(device const float* w, int V, int top_k, float top_p, float min_p, float top,
                     threadgroup KeepScratch& sh, uint tid, uint threads) {
  // top-k: the key of the k-th largest scaled logit, by a 4-pass radix select over the keys.
  uint k_key = 0;  // 0 keeps everything
  if (top_k > 0 && top_k < V) {
    if (tid == 0) {
      sh.prefix = 0;
      sh.mask = 0;
      sh.need = top_k;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      for (uint b = tid; b < 256; b += threads) sh.hist[b] = 0;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      const uint prefix = sh.prefix, mask = sh.mask;
      for (int e = tid; e < V; e += threads) {
        const uint key = float_key(w[e]);
        if ((key & mask) == prefix)
          atomic_fetch_add_explicit((threadgroup atomic_uint*)&sh.hist[(key >> shift) & 255u], 1u,
                                    memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        int cum = 0, digit = 0;
        const int need = sh.need;
        for (int b = 255; b >= 0; --b) {
          const int c = int(sh.hist[b]);
          if (cum + c >= need) {
            digit = b;
            break;
          }
          cum += c;
        }
        sh.need = need - cum;
        sh.digit = digit;
        sh.prefix = prefix | (uint(digit) << shift);
        sh.mask = mask | (0xFFu << shift);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    k_key = sh.prefix;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The candidates: elements at or above the top-k key, in a fixed order (each thread's elements in index
  // order, threads in order), so every sum below is reproducible.
  int count = 0;
  for (int e = tid; e < V; e += threads) count += float_key(w[e]) >= k_key;
  const int candidates = int(group_sum(float(count), sh.red, tid, threads) + 0.5f);  // exact below 2^24

  // top-p: the largest key whose tokens still hold top_p of the top-k mass, searched over the candidates
  // in threadgroup memory when they fit, otherwise over the vocabulary.
  uint p_key = 0;
  if (top_p > 0.0f && top_p < 1.0f) {
    const bool small = candidates <= kSampleCandidates;
    if (small) {
      int mine = 0;
      for (int e = tid; e < V; e += threads) mine += float_key(w[e]) >= k_key;
      sh.cand_offset[tid] = mine;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0) {
        int acc = 0;
        for (uint t = 0; t < threads; ++t) {
          const int c = sh.cand_offset[t];
          sh.cand_offset[t] = acc;
          acc += c;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      int at = sh.cand_offset[tid];
      for (int e = tid; e < V; e += threads) {
        const uint key = float_key(w[e]);
        if (key >= k_key) {
          sh.cand_key[at] = key;
          sh.cand_exp[at] = exp(w[e] - top);
          ++at;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    float mass_k = 0.0f;
    if (small) {
      for (int j = tid; j < candidates; j += threads) mass_k += sh.cand_exp[j];
    } else {
      for (int e = tid; e < V; e += threads)
        if (float_key(w[e]) >= k_key) mass_k += exp(w[e] - top);
    }
    const float target = top_p * group_sum(mass_k, sh.red, tid, threads);

    // Invariant: the mass from lo holds the target, the mass from hi does not.
    uint lo = k_key, hi = float_key(top) + 1u;  // top is finite, so this cannot wrap
    while (hi - lo > 1u) {
      const uint mid = lo + (hi - lo) / 2u;
      float acc = 0.0f;
      if (small) {
        for (int j = tid; j < candidates; j += threads)
          if (sh.cand_key[j] >= mid) acc += sh.cand_exp[j];
      } else {
        for (int e = tid; e < V; e += threads)
          if (float_key(w[e]) >= mid) acc += exp(w[e] - top);
      }
      if (group_sum(acc, sh.red, tid, threads) >= target)
        lo = mid;
      else
        hi = mid;
    }
    p_key = lo;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // min-p: keep tokens at least min_p times as likely as the most likely one.
  const uint mp_key = min_p > 0.0f ? float_key(top + log(min(min_p, 1.0f))) : 0u;
  return max(max(k_key, p_key), mp_key);
}

struct SampleShared {
  KeepScratch keep;
  int redi[32];
  int chosen[kSampleTopN];
};

kernel void sample(constant SampleArgs& a [[buffer(0)]], uint i [[threadgroup_position_in_grid]],
                   uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup SampleShared sh;
  const device SamplingRow& p = a.params[i];
  const int V = a.V;
  device const float* l = a.logits + long(p.row) * V;
  device float* w = a.work + long(i) * V;
  device int* sc = a.seen + long(p.slot) * V;
  device int* oc = a.out_counts + long(p.slot) * V;

  // 1. Penalties on the raw logits: repetition over prompt and output, presence and frequency over the
  // output alone.
  for (int e = tid; e < V; e += threads) {
    float x = l[e];
    if (p.repetition_penalty != 1.0f && sc[e] > 0) x = x > 0.0f ? x / p.repetition_penalty : x * p.repetition_penalty;
    const int co = oc[e];
    if (co > 0) x -= p.frequency_penalty * float(co) + p.presence_penalty;
    w[e] = x;
  }
  threadgroup_barrier(mem_flags::mem_device);

  int token = 0;
  if (!(p.temperature > 0.0f)) {  // greedy
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += threads)
      if (w[e] > best) {
        best = w[e];
        best_i = e;
      }
    float bv;
    group_argmax(best, best_i, sh.keep.red, sh.redi, tid, threads, bv, token);
  } else {
    // 2. Temperature, then the largest scaled logit.
    const float inv_t = 1.0f / p.temperature;
    float m = -INFINITY;
    for (int e = tid; e < V; e += threads) {
      const float s = w[e] * inv_t;
      w[e] = s;
      m = max(m, s);
    }
    threadgroup_barrier(mem_flags::mem_device);
    const float top = group_max(m, sh.keep.red, tid, threads);
    // 3 to 5. top-k, top-p, min-p.
    const uint keep_key = kept_key(w, V, p.top_k, p.top_p, p.min_p, top, sh.keep, tid, threads);
    // 6. Gumbel-max draw over the kept tokens: argmax of scaled logit + Gumbel noise.
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += threads) {
      if (float_key(w[e]) >= keep_key) {
        const float v = w[e] + gumbel_noise(p.seed, p.position, uint(e));
        if (v > best) {
          best = v;
          best_i = e;
        }
      }
    }
    float bv;
    group_argmax(best, best_i, sh.keep.red, sh.redi, tid, threads, bv, token);
  }

  // 7. Log-probabilities of the raw logits, for the chosen token and the top_n most likely.
  if (a.token_logprob || a.top_n > 0) {
    float raw_max = -INFINITY;
    for (int e = tid; e < V; e += threads) raw_max = max(raw_max, l[e]);
    raw_max = group_max(raw_max, sh.keep.red, tid, threads);
    float raw_sum = 0.0f;
    for (int e = tid; e < V; e += threads) raw_sum += exp(l[e] - raw_max);
    const float lse = raw_max + log(group_sum(raw_sum, sh.keep.red, tid, threads));
    if (a.token_logprob && tid == 0) a.token_logprob[i] = l[token] - lse;
    for (int j = 0; j < a.top_n; ++j) {
      float best = -INFINITY;
      int best_i = 0x7fffffff;
      for (int e = tid; e < V; e += threads) {
        bool taken = false;
        for (int c = 0; c < j; ++c) taken |= sh.chosen[c] == e;
        if (!taken && l[e] > best) {
          best = l[e];
          best_i = e;
        }
      }
      float bv;
      int bi;
      group_argmax(best, best_i, sh.keep.red, sh.redi, tid, threads, bv, bi);
      if (tid == 0) {
        sh.chosen[j] = bi;
        a.top_tokens[long(i) * a.top_n + j] = bi;
        a.top_logprobs[long(i) * a.top_n + j] = l[bi] - lse;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // 8. The sequence's counts: the chosen token is now part of its output.
  if (tid == 0) {
    a.tokens[i] = token;
    atomic_fetch_add_explicit((device atomic_int*)&sc[token], 1, memory_order_relaxed);
    atomic_fetch_add_explicit((device atomic_int*)&oc[token], 1, memory_order_relaxed);
  }
}

// counts[tokens[i]] += 1 for each of n tokens.
kernel void sampler_observe(constant ObserveArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
  if (int(i) >= a.n) return;
  atomic_fetch_add_explicit((device atomic_int*)&a.counts[a.tokens[i]], 1, memory_order_relaxed);
}

// Constrained decoding: the logits of each listed row that the token mask rules out become -inf. bits[j]
// is the mask of entry j (one bit per token, row-major, (V + 31) / 32 words).
kernel void mask_logits(constant MaskLogitsArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int v = int(gid.x), j = int(gid.y);
  if (v >= a.V) return;
  const uint word = a.bits[long(j) * a.words + (v >> 5)];
  if (!((word >> (v & 31)) & 1u)) a.logits[long(a.rows[j]) * a.V + v] = -INFINITY;
}
