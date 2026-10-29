// The sampler and speculative sampling (kernels.h, "Sampling" and the sampling half of "Speculative
// decoding"). cuda/sampler.cu and cuda/spec.cu are the reference, and the draws are theirs: the same
// Philox4x32-10 with the same counter words and key, the same 23-bit uniform behind the Gumbel noise and
// the same 53-bit uniform of the acceptance test, so a seed and a position draw the same token on either
// backend. The kept sets come from the same float32 keys, selected the same way. Two things can still
// differ in the last bit: the sums (the GPU adds in a block tree, here in a pairwise tree) and logf and
// expf (the device's math library against the host's), so a probability within rounding of a top-p
// boundary can land on the other side of it.
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr int kMaxTopN = 20;                        // sampler.cu's cap on top_n
constexpr int64_t kChunkElems = 32 * 1024;          // work per parallel chunk, in elements
constexpr int64_t kMaxVocab = int64_t{1} << 24;     // sampler.cu's bound on V (counts stay exact in f32)
constexpr int64_t kColumnGrain = 4096;              // columns per chunk of the logit mask
constexpr uint32_t kSampleDomain = 0x4C58u;         // lx_sample's Philox domain
constexpr uint32_t kDomainAccept = 0x53504131u;     // "SPA1": the acceptance uniforms
constexpr uint32_t kDomainResidual = 0x53505231u;   // "SPR1": residual and bonus draws
constexpr float kInf = std::numeric_limits<float>::infinity();

// Rows per parallel chunk: about kChunkElems elements of work each, and at least one row.
int64_t row_grain(int64_t rows, int64_t cols) {
  const int64_t per = (kChunkElems + cols - 1) / cols;
  return grain_for(rows, per < 1 ? 1 : per);
}

// Order-preserving map from float to uint32: a larger float gets a larger key (sampler.cu's float_key).
// Ties in the keys are exact ties of the floats, so the kept sets below break ties the way the GPU does.
inline uint32_t float_key(float v) {
  const uint32_t u = f32_bits(v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Philox4x32-10, the generator the GPU kernels run: the same multipliers, key bumps and word order.
inline void philox4x32(uint32_t c[4], uint32_t k0, uint32_t k1) {
  constexpr uint32_t kM0 = 0xD2511F53u, kM1 = 0xCD9E8D57u, kW0 = 0x9E3779B9u, kW1 = 0xBB67AE85u;
  for (int round = 0; round < 10; ++round) {
    const uint64_t p0 = static_cast<uint64_t>(kM0) * c[0];
    const uint64_t p1 = static_cast<uint64_t>(kM1) * c[2];
    const uint32_t n0 = static_cast<uint32_t>(p1 >> 32) ^ c[1] ^ k0;
    const uint32_t n2 = static_cast<uint32_t>(p0 >> 32) ^ c[3] ^ k1;
    c[0] = n0;
    c[1] = static_cast<uint32_t>(p1);
    c[2] = n2;
    c[3] = static_cast<uint32_t>(p0);
    k0 += kW0;
    k1 += kW1;
  }
}

// Standard Gumbel noise for element `index` of draw `position` of `seed`. The uniform has 23 bits and
// lies strictly inside (0, 1), so both logarithms are finite.
inline float gumbel(uint64_t seed, uint64_t position, uint32_t index, uint32_t domain) {
  uint32_t c[4] = {index, static_cast<uint32_t>(position), static_cast<uint32_t>(position >> 32), domain};
  philox4x32(c, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const float u = (static_cast<float>(c[0] >> 9) + 0.5f) * (1.0f / 8388608.0f);
  return -std::log(-std::log(u));
}

// A uniform in (0, 1) from 53 random bits, for the acceptance test u * q < p (spec.cu's uniform53).
inline double uniform53(uint64_t seed, uint64_t position, uint32_t domain) {
  uint32_t c[4] = {0, static_cast<uint32_t>(position), static_cast<uint32_t>(position >> 32), domain};
  philox4x32(c, static_cast<uint32_t>(seed), static_cast<uint32_t>(seed >> 32));
  const uint64_t bits = (static_cast<uint64_t>(c[0]) << 21) ^ (c[1] >> 11);
  return (static_cast<double>(bits & ((uint64_t{1} << 53) - 1)) + 0.5) * (1.0 / 9007199254740992.0);
}

// The first index holding the largest value (ties go to the lower index, as the GPU's block_argmax).
// A row of -inf alone gives index 0.
int32_t argmax(const float* x, int64_t n) {
  float best = -kInf;
  int32_t best_i = 0;
  for (int64_t i = 0; i < n; ++i)
    if (x[i] > best) best = x[i], best_i = static_cast<int32_t>(i);
  return best_i;
}

// Sum of n f32 values in a tree: the error grows with log n, as the GPU's block sums do, not with n.
float pairwise_sum(const float* x, int64_t n) {
  if (n <= 16) {
    float s = 0.0f;
    for (int64_t i = 0; i < n; ++i) s += x[i];
    return s;
  }
  const int64_t half = n / 2;
  return pairwise_sum(x, half) + pairwise_sum(x + half, n - half);
}

// The largest key K such that the weight of the elements whose key is at least K reaches target. With
// counts as the weights this is the k-th largest key (top-k); with the masses of the kept candidates it is
// the top-p key. A radix select over the keys of w, four passes of eight bits: each pass keeps the digit
// where the weight from the top first reaches the target, as sampler.cu's top-k select does.
template <class Weight>
uint32_t key_reaching(const float* w, int64_t V, float target, Weight weight) {
  uint32_t prefix = 0, mask = 0;
  float above = 0.0f;  // the weight of the keys above the digit being chosen
  for (int pass = 0; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    float hist[256] = {};
    for (int64_t e = 0; e < V; ++e) {
      const uint32_t key = float_key(w[e]);
      if ((key & mask) == prefix) hist[(key >> shift) & 255u] += weight(e);
    }
    int digit = -1;
    float cum = above;
    for (int b = 255; b >= 0; --b) {
      if (cum + hist[b] >= target) {
        digit = b;
        break;
      }
      cum += hist[b];
    }
    if (digit < 0) {
      // Rounding left the target above all the weight: take the lowest key that holds any.
      digit = 0;
      while (digit < 255 && hist[digit] == 0.0f) ++digit;
      cum -= hist[digit];
    }
    above = cum;
    prefix |= static_cast<uint32_t>(digit) << shift;
    mask |= 0xFFu << shift;
  }
  return prefix;
}

// The kept set of one row under temperature > 0 (sampler.cu's steps 2 to 5, spec.cu's filter_row): w
// receives x * (1 / T), as the GPU forms it. A token is kept when float_key(w[e]) >= key. top is the
// largest scaled logit. mass is V floats of scratch. x and w may be the same buffer.
struct Kept {
  uint32_t key;
  float top;
};

Kept kept_set(const float* x, float* w, int64_t V, const lx_spec_filter& f, float* mass) {
  const float inv_t = 1.0f / f.temperature;
  float top = -kInf;
  for (int64_t e = 0; e < V; ++e) {
    w[e] = x[e] * inv_t;
    top = std::fmax(top, w[e]);
  }

  uint32_t k_key = 0;  // 0 keeps everything
  if (f.top_k > 0 && f.top_k < V)
    k_key = key_reaching(w, V, static_cast<float>(f.top_k), [](int64_t) { return 1.0f; });

  // The masses are taken over the top-k candidates only, so the top-p key never falls below k_key.
  uint32_t p_key = 0;
  if (f.top_p > 0.0f && f.top_p < 1.0f) {
    for (int64_t e = 0; e < V; ++e) mass[e] = float_key(w[e]) >= k_key ? std::exp(w[e] - top) : 0.0f;
    const float target = f.top_p * pairwise_sum(mass, V);
    p_key = key_reaching(w, V, target, [mass](int64_t e) { return mass[e]; });
  }

  uint32_t mp_key = 0;
  if (f.min_p > 0.0f) mp_key = float_key(top + std::log(std::fmin(f.min_p, 1.0f)));

  return Kept{std::max({k_key, p_key, mp_key}), top};
}

// The penalized logits of a row (sampler.cu's step 1): repetition over the prompt and the output, then
// presence and frequency over the output alone.
void penalize(const float* l, const int32_t* seen, const int32_t* counts, int64_t V, const lx_sampling& p, float* w) {
  for (int64_t e = 0; e < V; ++e) {
    float x = l[e];
    if (p.repetition_penalty != 1.0f && seen[e] > 0)
      x = x > 0.0f ? x / p.repetition_penalty : x * p.repetition_penalty;
    const int32_t co = counts[e];
    if (co > 0) x -= p.frequency_penalty * static_cast<float>(co) + p.presence_penalty;
    w[e] = x;
  }
}

// The Gumbel-max draw over the kept tokens (sampler.cu's step 6): argmax of the scaled logit plus noise.
int32_t gumbel_draw(const float* w, int64_t V, uint32_t keep_key, uint64_t seed, uint64_t position) {
  float best = -kInf;
  int32_t best_i = 0;
  for (int64_t e = 0; e < V; ++e) {
    if (float_key(w[e]) < keep_key) continue;
    const float v = w[e] + gumbel(seed, position, static_cast<uint32_t>(e), kSampleDomain);
    if (v > best) best = v, best_i = static_cast<int32_t>(e);
  }
  return best_i;
}

// The log-probabilities of the raw logits (sampler.cu's step 7): the chosen token's, and the top_n most
// likely tokens, largest first, ties to the lower index. e is V floats of scratch.
void raw_logprobs(const float* l, int64_t V, int32_t token, int top_n, float* e, float* token_lp, int32_t* top_ids,
                  float* top_lp) {
  float raw_max = -kInf;
  for (int64_t i = 0; i < V; ++i) raw_max = std::fmax(raw_max, l[i]);
  for (int64_t i = 0; i < V; ++i) e[i] = std::exp(l[i] - raw_max);
  const float lse = raw_max + std::log(pairwise_sum(e, V));
  if (token_lp) *token_lp = l[token] - lse;
  if (top_n == 0) return;

  // A short list kept in order. Scanning in index order, a later equal value never displaces an earlier one.
  float vals[kMaxTopN];
  int32_t ids[kMaxTopN];
  int count = 0;
  for (int64_t i = 0; i < V; ++i) {
    const float v = l[i];
    if (count == top_n && !(v > vals[top_n - 1])) continue;
    int at = count < top_n ? count++ : top_n - 1;
    for (; at > 0 && v > vals[at - 1]; --at) vals[at] = vals[at - 1], ids[at] = ids[at - 1];
    vals[at] = v;
    ids[at] = static_cast<int32_t>(i);
  }
  for (int j = 0; j < top_n; ++j) {
    top_ids[j] = ids[j];
    top_lp[j] = vals[j] - lse;
  }
}

bool valid_filter(float top_p, int32_t top_k, float min_p) {
  return top_k >= 0 && top_p > 0.0f && top_p <= 1.0f && min_p >= 0.0f && min_p <= 1.0f;
}

bool valid_sampling(const lx_sampling& p) {
  return p.slot >= 0 && p.row >= 0 && p.repetition_penalty > 0.0f &&
         valid_filter(p.top_p, p.top_k, p.min_p);
}

// cu_seqlens: n_seq + 1 offsets, each sequence owning at least one row.
bool valid_layout(const int32_t* cu, int n_seq) {
  if (cu[0] < 0) return false;
  for (int s = 0; s < n_seq; ++s)
    if (cu[s + 1] - cu[s] < 1) return false;
  return true;
}

// The mass buffer of one chunk: V floats of this thread's scratch (null when the allocation failed).
float* mass_scratch(int64_t V) { return static_cast<float*>(scratch(sizeof(float) * V, 0)); }

// One sequence of lx_spec_verify (spec.cu's verify_kernel). The drafts hold while u * q < p. The token that
// ends the step is drawn from the residual max(0, p - q) at the first draft that fails, or from the last row
// when every draft holds. Without q (a point-mass draft) the residual zeroes the draft's own token.
void verify_sequence(const float* p, const float* q, int64_t V, const int32_t* tokens, int32_t start, int32_t k,
                     lx_spec_rng r, int32_t* accepted, int32_t* next) {
  int32_t j = 0;
  for (; j < k; ++j) {
    const int32_t d = tokens[start + j + 1];
    const int64_t at = static_cast<int64_t>(start + j) * V + d;
    const double pd = p[at];
    const double qd = q ? static_cast<double>(q[at]) : 1.0;
    const double u = uniform53(r.seed, r.position + static_cast<uint64_t>(j), kDomainAccept);
    if (!(u * qd < pd)) break;
  }

  const bool residual = j < k;
  const int64_t row = static_cast<int64_t>(start + j) * V;
  const int32_t d = residual ? tokens[start + j + 1] : -1;
  const float* pr = p + row;
  const float* qr = q ? q + row : nullptr;
  const uint64_t position = r.position + static_cast<uint64_t>(j);

  float best = -kInf;
  int32_t best_i = 0;
  for (int64_t e = 0; e < V; ++e) {
    float m = pr[e];
    if (residual) m = qr ? std::fmax(m - qr[e], 0.0f) : (e == d ? 0.0f : m);
    if (!(m > 0.0f)) continue;
    const float v = std::log(m) + gumbel(r.seed, position, static_cast<uint32_t>(e), kDomainResidual);
    if (v > best) best = v, best_i = static_cast<int32_t>(e);
  }
  if (best == -kInf) {
    // No residual mass is left (p equals q up to rounding): draw from the row's own distribution.
    for (int64_t e = 0; e < V; ++e) {
      if (!(pr[e] > 0.0f)) continue;
      const float v = std::log(pr[e]) + gumbel(r.seed, position, static_cast<uint32_t>(e), kDomainResidual);
      if (v > best) best = v, best_i = static_cast<int32_t>(e);
    }
  }
  *accepted = j;
  *next = best_i;
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_sample(const float* logits, int V, const lx_sampling* params, int n, int32_t* seen, int32_t* out_counts,
              float* work, int32_t* tokens, float* token_logprob, int top_n, int32_t* top_tokens,
              float* top_logprobs, void* stream) {
  LX_PROFILE("sample");
  (void)stream;
  if (n < 0) return fail(kInvalid);
  if (n == 0) return kOk;
  if (!logits || !params || !seen || !out_counts || !work || !tokens) return fail(kInvalid);
  if (V <= 0 || V > kMaxVocab || top_n < 0 || top_n > kMaxTopN || top_n > V) return fail(kInvalid);
  if (top_n > 0 && (!top_tokens || !top_logprobs)) return fail(kInvalid);
  for (int i = 0; i < n; ++i)
    if (!valid_sampling(params[i])) return fail(kInvalid);

  // Each row reads the counts of its slot and writes only its own work row and outputs, so rows run in
  // parallel. The counts move after the batch, so a slot named twice sees the same counts for both rows.
  std::atomic<bool> no_memory{false};
  parallel_for(n, row_grain(n, V), [&](int64_t i0, int64_t i1) {
    float* mass = mass_scratch(V);
    if (!mass) {
      no_memory = true;
      return;
    }
    for (int64_t i = i0; i < i1; ++i) {
      const lx_sampling& p = params[i];
      const float* l = logits + static_cast<int64_t>(p.row) * V;
      float* w = work + i * V;
      const int64_t at = static_cast<int64_t>(p.slot) * V;
      penalize(l, seen + at, out_counts + at, V, p, w);

      int32_t token;
      if (!(p.temperature > 0.0f)) {
        token = argmax(w, V);
      } else {
        const lx_spec_filter f{p.temperature, p.top_p, p.top_k, p.min_p};
        const Kept kept = kept_set(w, w, V, f, mass);
        token = gumbel_draw(w, V, kept.key, p.seed, p.position);
      }
      tokens[i] = token;

      if (token_logprob || top_n > 0)
        raw_logprobs(l, V, token, top_n, w, token_logprob ? token_logprob + i : nullptr,
                     top_n > 0 ? top_tokens + i * top_n : nullptr, top_n > 0 ? top_logprobs + i * top_n : nullptr);
    }
  });
  if (no_memory) return fail(kNoMemory);

  for (int64_t i = 0; i < n; ++i) {
    const int64_t at = static_cast<int64_t>(params[i].slot) * V + tokens[i];
    ++seen[at];
    ++out_counts[at];
  }
  return kOk;
}

int lx_sampler_reset(int32_t* seen, int32_t* out_counts, int slot, int V, void* stream) {
  LX_PROFILE("sampler_reset");
  (void)stream;
  if (!seen || !out_counts || slot < 0 || V <= 0) return fail(kInvalid);
  const int64_t at = static_cast<int64_t>(slot) * V;
  const size_t bytes = sizeof(int32_t) * static_cast<size_t>(V);
  memset(seen + at, 0, bytes);
  memset(out_counts + at, 0, bytes);
  return kOk;
}

int lx_sampler_observe(int32_t* seen, int32_t* out_counts, int slot, int V, const int32_t* tokens, int n,
                       int is_output, void* stream) {
  LX_PROFILE("sampler_observe");
  (void)stream;
  if (n < 0) return fail(kInvalid);
  if (n == 0) return kOk;
  if (!seen || !out_counts || !tokens || slot < 0 || V <= 0) return fail(kInvalid);
  // Check every token before counting any, so a bad call leaves the counts as they were.
  for (int i = 0; i < n; ++i)
    if (tokens[i] < 0 || tokens[i] >= V) return fail(kInvalid);
  const int64_t at = static_cast<int64_t>(slot) * V;
  for (int i = 0; i < n; ++i) {
    ++seen[at + tokens[i]];
    if (is_output) ++out_counts[at + tokens[i]];
  }
  return kOk;
}

int lx_mask_logits(float* logits, int V, const int32_t* rows, const uint32_t* bits, int n, void* stream) {
  LX_PROFILE("mask_logits");
  (void)stream;
  if (n < 0) return fail(kInvalid);
  if (n == 0) return kOk;
  if (!logits || !rows || !bits || V <= 0) return fail(kInvalid);
  for (int j = 0; j < n; ++j)
    if (rows[j] < 0) return fail(kInvalid);

  // Chunks split the columns, not the entries: two entries that name one row then never write one
  // element from two threads, and each element is written in entry order.
  const int64_t words = (static_cast<int64_t>(V) + 31) / 32;
  parallel_for(V, kColumnGrain, [&](int64_t v0, int64_t v1) {
    for (int j = 0; j < n; ++j) {
      float* row = logits + static_cast<int64_t>(rows[j]) * V;
      const uint32_t* mask = bits + j * words;
      for (int64_t v = v0; v < v1; ++v)
        if (!((mask[v >> 5] >> (v & 31)) & 1u)) row[v] = -kInf;
    }
  });
  return kOk;
}

int lx_spec_logprobs(const float* logits, int V, const int32_t* tokens, int rows, int top_n, float* token_lp,
                     int32_t* top_tokens, float* top_logprobs, void* stream) {
  LX_PROFILE("spec_logprobs");
  (void)stream;
  if (rows < 0 || top_n < 0 || top_n > kMaxTopN || top_n > V) return fail(kInvalid);
  if (rows == 0) return kOk;
  if (V <= 0 || V > kMaxVocab || !logits || !tokens) return fail(kInvalid);
  if (top_n > 0 && (!top_tokens || !top_logprobs)) return fail(kInvalid);
  // The same scan lx_sample runs for its report (raw_logprobs), one row after another.
  std::vector<float> scratch(static_cast<size_t>(V));
  for (int r = 0; r < rows; ++r)
    raw_logprobs(logits + static_cast<int64_t>(r) * V, V, tokens[r], top_n, scratch.data(),
                 token_lp ? token_lp + r : nullptr, top_n > 0 ? top_tokens + static_cast<int64_t>(r) * top_n : nullptr,
                 top_n > 0 ? top_logprobs + static_cast<int64_t>(r) * top_n : nullptr);
  return kOk;
}

int lx_spec_penalize(float* logits, int V, lx_spec_penalty pen, const int32_t* tokens, int rows, const int32_t* seen,
                     const int32_t* out_counts, void* stream) {
  LX_PROFILE("spec_penalize");
  (void)stream;
  if (rows < 0) return fail(kInvalid);
  if (rows == 0) return kOk;
  if (V <= 0 || !logits || !tokens || !seen || !out_counts) return fail(kInvalid);
  lx_sampling p{};
  p.repetition_penalty = pen.repetition_penalty;
  p.presence_penalty = pen.presence_penalty;
  p.frequency_penalty = pen.frequency_penalty;
  std::vector<int32_t> sc(static_cast<size_t>(V)), co(static_cast<size_t>(V));
  for (int r = 0; r < rows; ++r) {
    // Row r sees the counts plus the step's drafts before it, tokens[1 .. r].
    for (int64_t e = 0; e < V; ++e) {
      int32_t extra = 0;
      for (int t = 1; t <= r; ++t) extra += tokens[t] == e;
      sc[static_cast<size_t>(e)] = seen[e] + extra;
      co[static_cast<size_t>(e)] = out_counts[e] + extra;
    }
    float* row = logits + static_cast<int64_t>(r) * V;
    penalize(row, sc.data(), co.data(), V, p, row);
  }
  return kOk;
}

int lx_spec_probs(const float* logits, int V, const lx_spec_filter* filters, int rows, float* probs, void* stream) {
  LX_PROFILE("spec_probs");
  (void)stream;
  if (rows < 0) return fail(kInvalid);
  if (rows == 0) return kOk;
  if (V <= 0 || V > kMaxVocab || !logits || !filters || !probs) return fail(kInvalid);
  for (int r = 0; r < rows; ++r)
    if (!valid_filter(filters[r].top_p, filters[r].top_k, filters[r].min_p)) return fail(kInvalid);

  std::atomic<bool> no_memory{false};
  parallel_for(rows, row_grain(rows, V), [&](int64_t r0, int64_t r1) {
    float* mass = mass_scratch(V);
    if (!mass) {
      no_memory = true;
      return;
    }
    for (int64_t r = r0; r < r1; ++r) {
      const lx_spec_filter& f = filters[r];
      const float* l = logits + r * V;
      float* out = probs + r * V;
      if (!(f.temperature > 0.0f)) {
        std::fill(out, out + V, 0.0f);
        out[argmax(l, V)] = 1.0f;
        continue;
      }
      // out holds the scaled logits until the kept ones are exponentiated in place.
      const Kept kept = kept_set(l, out, V, f, mass);
      for (int64_t e = 0; e < V; ++e) out[e] = float_key(out[e]) >= kept.key ? std::exp(out[e] - kept.top) : 0.0f;
      const float inv_z = 1.0f / pairwise_sum(out, V);
      for (int64_t e = 0; e < V; ++e) out[e] *= inv_z;
    }
  });
  return no_memory ? fail(kNoMemory) : kOk;
}

int lx_spec_verify(const float* p, const float* q, int V, const int32_t* tokens, const int32_t* cu_seqlens,
                   const lx_spec_rng* rng, int n_seq, int32_t* accepted, int32_t* next, void* stream) {
  LX_PROFILE("spec_verify");
  (void)stream;
  if (n_seq < 0) return fail(kInvalid);
  if (n_seq == 0) return kOk;
  if (V <= 0 || !p || !tokens || !cu_seqlens || !rng || !accepted || !next) return fail(kInvalid);
  if (!valid_layout(cu_seqlens, n_seq)) return fail(kInvalid);

  parallel_for(n_seq, row_grain(n_seq, V), [&](int64_t s0, int64_t s1) {
    for (int64_t s = s0; s < s1; ++s) {
      const int32_t start = cu_seqlens[s];
      const int32_t k = cu_seqlens[s + 1] - start - 1;
      verify_sequence(p, q, V, tokens, start, k, rng[s], accepted + s, next + s);
    }
  });
  return kOk;
}

int lx_spec_verify_greedy(const float* logits, int V, const int32_t* tokens, const int32_t* cu_seqlens, int n_seq,
                          int rows, int32_t* argmax_out, int32_t* accepted, int32_t* next, void* stream) {
  LX_PROFILE("spec_verify_greedy");
  (void)stream;
  if (n_seq < 0 || rows < 0) return fail(kInvalid);
  if (n_seq == 0) return kOk;
  if (V <= 0 || !logits || !tokens || !cu_seqlens || !argmax_out || !accepted || !next) return fail(kInvalid);
  if (!valid_layout(cu_seqlens, n_seq) || cu_seqlens[n_seq] > rows) return fail(kInvalid);

  parallel_for(rows, row_grain(rows, V), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) argmax_out[r] = argmax(logits + r * V, V);
  });
  // Drafts hold while each equals the argmax of the row before it; the sequences are cheap, so one thread walks them.
  for (int s = 0; s < n_seq; ++s) {
    const int32_t start = cu_seqlens[s];
    const int32_t k = cu_seqlens[s + 1] - start - 1;
    int32_t j = 0;
    while (j < k && argmax_out[start + j] == tokens[start + j + 1]) ++j;
    accepted[s] = j;
    next[s] = argmax_out[start + j];
  }
  return kOk;
}

int lx_spec_sample(const float* probs, int V, const int32_t* rows, const lx_spec_rng* rng, uint32_t domain, int n,
                   int32_t* tokens, void* stream) {
  LX_PROFILE("spec_sample");
  (void)stream;
  if (n < 0) return fail(kInvalid);
  if (n == 0) return kOk;
  if (V <= 0 || V > kMaxVocab || !probs || !rng || !tokens) return fail(kInvalid);
  if (rows) {
    for (int i = 0; i < n; ++i)
      if (rows[i] < 0) return fail(kInvalid);
  }

  parallel_for(n, row_grain(n, V), [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i) {
      const float* pr = probs + static_cast<int64_t>(rows ? rows[i] : i) * V;
      const lx_spec_rng r = rng[i];
      float best = -kInf;
      int32_t best_i = 0;
      for (int64_t e = 0; e < V; ++e) {
        if (!(pr[e] > 0.0f)) continue;
        const float v = std::log(pr[e]) + gumbel(r.seed, r.position, static_cast<uint32_t>(e), domain);
        if (v > best) best = v, best_i = static_cast<int32_t>(e);
      }
      tokens[i] = best_i;
    }
  });
  return kOk;
}

}  // extern "C"
