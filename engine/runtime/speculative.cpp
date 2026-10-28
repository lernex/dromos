#include "runtime/speculative.h"

#include <algorithm>
#include <chrono>
#include <cstring>
#include <stdexcept>
#include <string>

namespace lx::runtime {
namespace {

using Clock = std::chrono::steady_clock;

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed: " + lx_rt_error_string(status));
}
#define LX_CHECK(expr) check(static_cast<int>(expr), #expr)

template <typename T>
T* device_alloc(size_t count) {
  void* p = nullptr;
  check(lx_rt_alloc(&p, static_cast<int64_t>(count * sizeof(T))),
        "device allocation of " + std::to_string(count * sizeof(T)) + " bytes");  // zeroed
  return static_cast<T*>(p);
}

template <typename T>
T* host_alloc(size_t count) {
  void* p = nullptr;
  check(lx_rt_host_alloc(&p, static_cast<int64_t>(count * sizeof(T))),
        "host allocation of " + std::to_string(count * sizeof(T)) + " bytes");
  return static_cast<T*>(p);
}

// Philox domain of a model drafter's proposals ("DRFT"), apart from the acceptance draws.
constexpr uint32_t kDomainDraft = 0x44524654u;

uint64_t hash_ngram(const int32_t* t, int n) {
  uint64_t h = 0x9E3779B97F4A7C15ull ^ static_cast<uint64_t>(n);
  for (int i = 0; i < n; ++i) {
    h ^= static_cast<uint32_t>(t[i]);
    h *= 0xBF58476D1CE4E5B9ull;
    h ^= h >> 31;
  }
  return h;
}

// Runs `tokens` through `model` on `slot` in steps of at most `chunk` tokens; the last row's
// logits end in `logits`.
void prefill(MetisModel& model, int slot, const std::vector<int32_t>& tokens, int chunk, float* logits,
             void* s) {
  for (size_t off = 0; off < tokens.size(); off += chunk) {
    const int n = static_cast<int>(std::min<size_t>(chunk, tokens.size() - off));
    StepBatch b{{tokens.begin() + off, tokens.begin() + off + n}, {0, n}, {slot}};
    model.forward(b, logits, false, s);
  }
}

}  // namespace

// ---- Draft length ----------------------------------------------------------------------------

DraftCost DraftCost::rtx5070ti_praxis() { return DraftCost{{0.0, 1.04, 1.16, 1.29, 1.36, 1.49, 1.57, 1.70}}; }

int DraftCost::choose(const std::vector<double>& p_hold, int max_drafts, double lambda) const {
  const int limit = std::min<int>({max_drafts, static_cast<int>(p_hold.size()), static_cast<int>(cost.size()) - 2});
  // the expected tokens minus lambda times the cost, for k = 0 .. limit; the best k wins
  double best = 1.0 - lambda * cost[1], gain = 1.0, reach = 1.0;
  int k_best = 0;
  for (int k = 1; k <= limit; ++k) {
    reach *= p_hold[k - 1];
    gain += reach;
    const double value = gain - lambda * cost[k + 1];
    if (value > best) best = value, k_best = k;
  }
  return k_best;
}

// ---- Prompt lookup ----------------------------------------------------------------------------

namespace {
// Priors: the chance a proposal from a match of n tokens holds (measured on the benchmark prompts),
// worth kPriorWeight observations.
constexpr double kPriorHold[] = {0.0, 0.15, 0.3, 0.5, 0.75, 0.8, 0.85, 0.85, 0.85};
constexpr double kPriorWeight = 4.0;
}  // namespace

NgramDrafter::NgramDrafter(int max_ngram, int min_ngram, DraftCost cost)
    : drafted_by_n(max_ngram + 1, 0),
      accepted_by_n(max_ngram + 1, 0),
      max_n_(std::min(max_ngram, 8)),
      min_n_(std::max(1, min_ngram)),
      cost_(std::move(cost)),
      last_end_(max_ngram + 1) {}

void NgramDrafter::begin(const std::vector<int32_t>&, void*) {
  for (auto& m : last_end_) m.clear();
  indexed_ = 0;
  last_n_ = last_k_ = 0;
  held_.assign(max_n_ + 1, std::vector<double>(LX_SPEC_MAX_PENDING + 1));
  tried_.assign(max_n_ + 1, std::vector<double>(LX_SPEC_MAX_PENDING + 1, kPriorWeight));
  for (int n = 0; n <= max_n_; ++n)
    for (auto& h : held_[n]) h = kPriorWeight * kPriorHold[n];
  lambda_ = 1.0;
}

void NgramDrafter::index_through(const std::vector<int32_t>& ctx, size_t end) {
  for (size_t e = indexed_; e < end; ++e)
    for (int n = min_n_; n <= max_n_; ++n)
      if (e + 1 >= static_cast<size_t>(n)) last_end_[n][hash_ngram(ctx.data() + e + 1 - n, n)] = static_cast<int32_t>(e);
  indexed_ = std::max(indexed_, end);
}

Draft NgramDrafter::propose(const DraftRequest& r) {
  const std::vector<int32_t>& ctx = *r.context;
  const size_t L = ctx.size();
  Draft d;
  last_n_ = last_k_ = 0;
  if (L < 2 || r.max_drafts <= 0) return d;
  // every n-gram that ends before the last token: the suffix itself must not match itself
  index_through(ctx, L - 1);
  for (int n = std::min<int>(max_n_, static_cast<int>(L) - 1); n >= min_n_; --n) {
    const int32_t* suffix = ctx.data() + L - n;
    auto it = last_end_[n].find(hash_ngram(suffix, n));
    if (it == last_end_[n].end()) continue;
    const size_t e = it->second;
    if (!std::equal(suffix, suffix + n, ctx.data() + e + 1 - n)) continue;  // a hash collision
    int k = static_cast<int>(std::min<size_t>(r.max_drafts, L - 1 - e));
    if (adaptive_) {
      std::vector<double> p(k);
      for (int i = 0; i < k; ++i) p[i] = held_[n][i + 1] / tried_[n][i + 1];
      k = cost_.choose(p, k, lambda_);
    }
    last_n_ = n;
    last_k_ = k;
    d.tokens.assign(ctx.begin() + e + 1, ctx.begin() + e + 1 + k);
    break;
  }
  return d;
}

void NgramDrafter::observe(int accepted, int32_t, void*) {
  const int k = last_k_;
  const double c = cost_.cost[std::min<size_t>(k + 1, cost_.cost.size() - 1)];
  lambda_ = 0.9 * lambda_ + 0.1 * (accepted + 1) / c;
  if (k == 0) return;
  drafted_by_n[last_n_] += k;
  accepted_by_n[last_n_] += accepted;
  // positions 1 .. accepted held, position accepted + 1 failed; later ones were never judged
  for (int i = 1; i <= std::min(k, accepted + 1); ++i) {
    tried_[last_n_][i] += 1.0;
    if (i <= accepted) held_[last_n_][i] += 1.0;
  }
}

// ---- A smaller model drafting ---------------------------------------------------------------

ModelDrafter::ModelDrafter(MetisModel& model, int slot, lx_spec_filter filter, uint64_t seed, float min_confidence)
    : model_(model), slot_(slot), filter_(filter), seed_(seed), min_confidence_(min_confidence) {
  const size_t V = model.vocab_size();
  logits_ = device_alloc<float>(LX_SPEC_MAX_PENDING * V);
  q_ = device_alloc<float>(LX_SPEC_MAX_PENDING * V);
  conf_probs_ = device_alloc<float>(V);
  d_tokens_ = device_alloc<int32_t>(1);
  d_cu_ = device_alloc<int32_t>(2);
  d_out_ = device_alloc<int32_t>(4);
  d_filter_ = device_alloc<lx_spec_filter>(2);
  d_rng_ = device_alloc<lx_spec_rng>(1);
  h_out_ = host_alloc<int32_t>(4);
  h_conf_ = host_alloc<float>(1);
  const int32_t cu[2] = {0, 1};
  const lx_spec_filter filters[2] = {filter_, {1.0f, 1.0f, 0, 0.0f}};  // the proposal's, and the plain softmax
  LX_CHECK(lx_rt_copy(d_cu_, cu, static_cast<int64_t>(sizeof(cu)), LX_COPY_H2D, nullptr));
  LX_CHECK(lx_rt_stream_sync(nullptr));
  LX_CHECK(lx_rt_copy(d_filter_, filters, static_cast<int64_t>(sizeof(filters)), LX_COPY_H2D, nullptr));
  LX_CHECK(lx_rt_stream_sync(nullptr));
}

ModelDrafter::~ModelDrafter() {
  for (void* p : {static_cast<void*>(logits_), static_cast<void*>(q_), static_cast<void*>(conf_probs_),
                  static_cast<void*>(d_tokens_), static_cast<void*>(d_cu_), static_cast<void*>(d_out_),
                  static_cast<void*>(d_filter_), static_cast<void*>(d_rng_)})
    lx_rt_free(p);
  lx_rt_host_free(h_out_);
  lx_rt_host_free(h_conf_);
}

void ModelDrafter::begin(const std::vector<int32_t>& prompt, void* s) {
  model_.reset_slot(slot_);
  prefill(model_, slot_, prompt, 512, logits_, s);
  fed_ = static_cast<int>(prompt.size());
  drafted_ = 0;
}

Draft ModelDrafter::propose(const DraftRequest& r) {
  const std::vector<int32_t>& ctx = *r.context;
  const int V = model_.vocab_size();
  const bool greedy = !(filter_.temperature > 0.0f);
  ctx_size_ = static_cast<int>(ctx.size());
  Draft d;
  drafted_ = 0;
  std::vector<int32_t> feed(ctx.begin() + fed_, ctx.end());
  // The proposals of this round stay pending until the target has judged them, so they and the
  // tokens fed to make them must fit the pending positions.
  const int k = std::min<int>(r.max_drafts, LX_SPEC_MAX_PENDING + 1 - static_cast<int>(feed.size()));
  for (int i = 0; i < k; ++i) {
    StepBatch b{feed, {0, static_cast<int>(feed.size())}, {slot_}};
    model_.forward_spec(b, logits_, r.stream);
    fed_ += static_cast<int>(feed.size());
    const float* row = logits_ + static_cast<size_t>(feed.size() - 1) * V;
    if (greedy) {
      LX_CHECK(lx_spec_verify_greedy(row, V, d_tokens_, d_cu_, 1, 1, d_out_ + 2, d_out_, d_out_ + 1, r.stream));
    } else {
      const lx_spec_rng rng{seed_ ^ 0x5D4A5D4Aull, r.position + i};
      LX_CHECK(lx_rt_copy(d_rng_, &rng, static_cast<int64_t>(sizeof(rng)), LX_COPY_H2D, r.stream));
      LX_CHECK(lx_spec_probs(row, V, d_filter_, 1, q_ + static_cast<size_t>(i) * V, r.stream));
      LX_CHECK(lx_spec_sample(q_ + static_cast<size_t>(i) * V, V, nullptr, d_rng_, kDomainDraft, 1, d_out_ + 1,
                              r.stream));
    }
    LX_CHECK(lx_rt_copy(h_out_, d_out_, static_cast<int64_t>(2 * sizeof(int32_t)), LX_COPY_D2H, r.stream));
    LX_CHECK(lx_rt_stream_sync(r.stream));
    const int32_t token = h_out_[1];
    d.tokens.push_back(token);
    if (min_confidence_ > 0.0f && i + 1 < k) {
      const float* probs = q_ + static_cast<size_t>(i) * V;
      if (greedy) {
        LX_CHECK(lx_spec_probs(row, V, d_filter_ + 1, 1, conf_probs_, r.stream));
        probs = conf_probs_;
      }
      LX_CHECK(lx_rt_copy(h_conf_, probs + token, static_cast<int64_t>(sizeof(float)), LX_COPY_D2H, r.stream));
      LX_CHECK(lx_rt_stream_sync(r.stream));
      if (*h_conf_ < min_confidence_) break;
    }
    feed.assign(1, token);
  }
  drafted_ = static_cast<int>(d.tokens.size());
  if (!greedy) d.q = q_;
  return d;
}

void ModelDrafter::observe(int accepted, int32_t, void*) {
  // It ran the context and every proposal but the last; it keeps what the target kept. When every
  // proposal held, the last one is not run yet and goes in with the next round's feed.
  if (accepted < drafted_) {
    model_.truncate(slot_, ctx_size_ + accepted);
    fed_ = ctx_size_ + accepted;
  }
}

// ---- The decode loop --------------------------------------------------------------------------

SpecDecoder::SpecDecoder(MetisModel& target) : target_(target), max_rows_(target.max_verify_tokens()) {
  const size_t V = target.vocab_size();
  logits_ = device_alloc<float>(max_rows_ * V);
  probs_ = device_alloc<float>(max_rows_ * V);
  d_tokens_ = device_alloc<int32_t>(max_rows_);
  d_cu_ = device_alloc<int32_t>(2);
  d_argmax_ = device_alloc<int32_t>(max_rows_);
  d_out_ = device_alloc<int32_t>(2);
  d_filters_ = device_alloc<lx_spec_filter>(max_rows_);
  d_rng_ = device_alloc<lx_spec_rng>(1);
  h_out_ = host_alloc<int32_t>(2);
}

SpecDecoder::~SpecDecoder() {
  for (void* p : {static_cast<void*>(logits_), static_cast<void*>(probs_), static_cast<void*>(d_tokens_),
                  static_cast<void*>(d_cu_), static_cast<void*>(d_argmax_), static_cast<void*>(d_out_),
                  static_cast<void*>(d_filters_), static_cast<void*>(d_rng_)})
    lx_rt_free(p);
  lx_rt_host_free(h_out_);
}

void SpecDecoder::accept(const SpecConfig& config, const std::vector<int32_t>& step, const float* q,
                         uint64_t position, int& accepted, int32_t& next, void* s) {
  const int rows = static_cast<int>(step.size());
  const int V = target_.vocab_size();
  const int32_t cu[2] = {0, rows};
  LX_CHECK(lx_rt_copy(d_tokens_, step.data(), static_cast<int64_t>(sizeof(int32_t) * rows), LX_COPY_H2D, s));
  LX_CHECK(lx_rt_copy(d_cu_, cu, static_cast<int64_t>(sizeof(cu)), LX_COPY_H2D, s));
  if (!(config.filter.temperature > 0.0f)) {
    LX_CHECK(lx_spec_verify_greedy(logits_, V, d_tokens_, d_cu_, 1, rows, d_argmax_, d_out_, d_out_ + 1, s));
  } else {
    const lx_spec_rng rng{config.seed, position};
    LX_CHECK(lx_rt_copy(d_rng_, &rng, static_cast<int64_t>(sizeof(rng)), LX_COPY_H2D, s));
    LX_CHECK(lx_spec_probs(logits_, V, d_filters_, rows, probs_, s));
    LX_CHECK(lx_spec_verify(probs_, q, V, d_tokens_, d_cu_, d_rng_, 1, d_out_, d_out_ + 1, s));
  }
  LX_CHECK(lx_rt_copy(h_out_, d_out_, static_cast<int64_t>(2 * sizeof(int32_t)), LX_COPY_D2H, s));
  LX_CHECK(lx_rt_stream_sync(s));
  accepted = h_out_[0];
  next = h_out_[1];
}

void SpecDecoder::verify(int slot, const SpecConfig& config, const std::vector<int32_t>& step, const float* q,
                         uint64_t position, int& accepted, int32_t& next, void* s) {
  const int rows = static_cast<int>(step.size());
  if (rows < 1 || rows > max_rows_)
    throw std::runtime_error("a verify step takes 1 to " + std::to_string(max_rows_) + " rows");
  const int before = target_.slot_length(slot);
  const std::vector<lx_spec_filter> filters(rows, config.filter);
  LX_CHECK(lx_rt_copy(d_filters_, filters.data(), static_cast<int64_t>(sizeof(lx_spec_filter) * rows), LX_COPY_H2D, s));
  target_.forward_spec({step, {0, rows}, {slot}}, logits_, s);
  accept(config, step, q, position, accepted, next, s);
  target_.truncate(slot, before + accepted + 1);
}

std::vector<int32_t> SpecDecoder::generate(Drafter* drafter, const std::vector<int32_t>& prompt,
                                           const SpecConfig& config, SpecStats& stats, void* s) {
  if (prompt.empty()) throw std::runtime_error("empty prompt");
  stats = SpecStats{};
  const int k_cap = drafter ? std::max(0, std::min(config.max_drafts, max_rows_ - 1)) : 0;
  stats.accept_hist.assign(k_cap + 1, 0);
  target_.set_feature_taps(drafter ? drafter->feature_taps() : std::vector<FeatureTap>{});
  const int F = target_.feature_dim();
  std::vector<lx_spec_filter> filters(max_rows_, config.filter);
  LX_CHECK(lx_rt_copy(d_filters_, filters.data(), static_cast<int64_t>(sizeof(lx_spec_filter) * max_rows_),
                      LX_COPY_H2D, s));
  auto stops = [&](int32_t t) { return std::find(config.stop_ids.begin(), config.stop_ids.end(), t) != config.stop_ids.end(); };

  // the prompt, and the first token from its last row
  const auto t0 = Clock::now();
  target_.reset_slot(0);
  if (drafter) drafter->begin(prompt, s);
  const float* features = nullptr;
  for (size_t off = 0; off < prompt.size(); off += config.prefill_chunk) {
    const int n = static_cast<int>(std::min<size_t>(config.prefill_chunk, prompt.size() - off));
    target_.forward({{prompt.begin() + off, prompt.begin() + off + n}, {0, n}, {0}}, logits_, false, s);
    if (F) {
      drafter->observe_features(target_.features(), n, F, s);
      features = target_.features() + static_cast<size_t>(n - 1) * F;
    }
  }
  int accepted = 0;
  int32_t next = 0;
  accept(config, {prompt.back()}, nullptr, 0, accepted, next, s);
  stats.prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();

  std::vector<int32_t> context = prompt, out;
  context.push_back(next);
  out.push_back(next);
  bool done = stops(next) || config.max_new_tokens <= 1;

  const auto t1 = Clock::now();
  while (!done) {
    const int room = config.max_new_tokens - static_cast<int>(out.size());
    std::vector<int32_t> step{context.back()};
    Draft draft;
    if (drafter && k_cap > 0 && room > 1) {
      DraftRequest r;
      r.context = &context;
      r.features = features;
      r.feature_dim = F;
      r.max_drafts = std::min(k_cap, room - 1);
      r.position = out.size();
      r.stream = s;
      draft = drafter->propose(r);
      if (static_cast<int>(draft.tokens.size()) > r.max_drafts) draft.tokens.resize(r.max_drafts);
      step.insert(step.end(), draft.tokens.begin(), draft.tokens.end());
    }
    const int k = static_cast<int>(step.size()) - 1;
    const int before = target_.slot_length(0);
    StepBatch b{step, {0, k + 1}, {0}};
    // A step with no drafts is a plain one-token step: a verify step of one token costs about 7% more on Praxis
    // (spec-cost), so a drafter that sits out must not pay it on every step.
    if (drafter && k > 0)
      target_.forward_spec(b, logits_, s);
    else
      target_.forward(b, logits_, false, s);
    accept(config, step, draft.q, out.size(), accepted, next, s);
    if (drafter && k > 0) target_.truncate(0, before + accepted + 1);
    if (F) {
      drafter->observe_features(target_.features(), accepted + 1, F, s);
      features = target_.features() + static_cast<size_t>(accepted) * F;
    }

    ++stats.steps;
    stats.drafted += k;
    stats.accepted += accepted;
    ++stats.accept_hist[accepted];
    for (int j = 1; j <= accepted + 1 && !done; ++j) {
      const int32_t t = j <= accepted ? step[j] : next;
      context.push_back(t);
      out.push_back(t);
      done = stops(t) || static_cast<int>(out.size()) >= config.max_new_tokens;
    }
    if (drafter) drafter->observe(accepted, next, s);
  }
  LX_CHECK(lx_rt_stream_sync(s));
  stats.decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
  stats.emitted = static_cast<int>(out.size());
  return out;
}

}  // namespace lx::runtime
