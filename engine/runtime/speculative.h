// Speculative single-stream decoding. Each step a drafter proposes up to k tokens, the target runs
// the pending token and the proposals in one verify step (MetisModel::forward_spec: about the
// weight bytes of one token), exact acceptance keeps a prefix of them (kernels.h, lx_spec_verify
// and lx_spec_verify_greedy), and the target truncates the rest. The emitted tokens are those plain
// decoding would emit: identical under greedy decoding, identically distributed under sampling.
//
// The drafter interface is deliberately small, so every kind of drafter fits behind it: given the
// sequence so far and the target's hidden features at the last accepted position, propose up to k
// tokens (and, when they were drawn from a distribution, that distribution). A per-token
// autoregressive head (EAGLE style), a one-pass block drafter (DFlash style), a smaller model of
// the same family, or an n-gram lookup are all a Drafter.
#pragma once

#include <stdint.h>

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda/kernels.h"
#include "model/metis.h"

namespace lx::runtime {

struct DraftRequest {
  // The sequence so far: the prompt, then every emitted token. The last one is the pending token
  // (emitted, not yet run through the target).
  const std::vector<int32_t>* context = nullptr;
  // The target's feature taps (Drafter::feature_taps) at the last position it accepted, f32
  // [feature_dim] on the device; null when the drafter asked for none.
  const float* features = nullptr;
  int feature_dim = 0;
  int max_drafts = 0;     // propose at most this many
  uint64_t position = 0;  // index, among the emitted tokens, of the first proposal (keys any draws)
  void* stream = nullptr;
};

struct Draft {
  std::vector<int32_t> tokens;
  // Row j: the distribution tokens[j] was drawn from, f32 [V] on the device, rows contiguous. Null
  // when every proposal is a point mass (lookups, greedy drafters).
  const float* q = nullptr;
};

class Drafter {
 public:
  virtual ~Drafter() = default;
  virtual const char* name() const = 0;
  // The target hidden states this drafter reads; empty when it needs none.
  virtual std::vector<FeatureTap> feature_taps() const { return {}; }
  // A new sequence; the target is about to prefill `prompt`.
  virtual void begin(const std::vector<int32_t>& prompt, void* stream) = 0;
  // The target's feature rows (f32 [n, feature_dim], device, valid during the call) for the next n
  // positions it committed, in order: every prefill chunk, then after each step the positions it
  // kept (the pending token and the accepted drafts). Drafters that keep their own context state
  // over the target's features (an EAGLE head's attention, a block drafter's injected keys and
  // values) build it from these. Called only when feature_taps() is not empty.
  virtual void observe_features(const float* rows, int n, int feature_dim, void* stream) {}
  virtual Draft propose(const DraftRequest& request) = 0;
  // The target kept the first `accepted` proposals of the last draft, then emitted `next`.
  virtual void observe(int accepted, int32_t next, void* stream) = 0;
};

// How long a draft is worth verifying. cost[T] is a verify step of T tokens in units of a plain
// one-token step (measured: `dromos spec-cost`). A drafter that knows the chance p_i that its i-th
// proposal holds once the ones before it held drafts the i-th only if p_1 ... p_i (the tokens it
// adds, on average) beats lambda * (cost[i + 1] - cost[i]), where lambda is the tokens per unit of
// cost the sequence is getting: the rule that maximizes long-run throughput.
struct DraftCost {
  std::vector<double> cost;  // [T] for T = 0 .. max, cost[0] unused
  // Measured on the RTX 5070 Ti, Praxis BF16 (int8 decode), 1k context.
  static DraftCost rtx5070ti_praxis();
  int choose(const std::vector<double>& p_hold, int max_drafts, double lambda) const;
};

// Prompt lookup: the longest suffix of the sequence (up to max_ngram tokens, at least min_ngram)
// that occurred earlier proposes the tokens that followed its latest occurrence. No model, no
// device work; it shines wherever the output repeats its input (editing code, quoting a document).
// How many it proposes is decided per step: it learns, per sequence and per match length, how often
// each draft position holds (starting from priors), and drafts as far as DraftCost says pays.
class NgramDrafter final : public Drafter {
 public:
  explicit NgramDrafter(int max_ngram = 4, int min_ngram = 1, DraftCost cost = DraftCost::rtx5070ti_praxis());
  const char* name() const override { return "ngram"; }
  void begin(const std::vector<int32_t>& prompt, void* stream) override;
  Draft propose(const DraftRequest& request) override;
  void observe(int accepted, int32_t next, void* stream) override;
  void set_adaptive(bool on) { adaptive_ = on; }

  // Per match length n: drafted and accepted tokens, for tuning.
  std::vector<int64_t> drafted_by_n, accepted_by_n;

 private:
  void index_through(const std::vector<int32_t>& ctx, size_t end);
  int max_n_, min_n_;
  DraftCost cost_;
  bool adaptive_ = true;
  size_t indexed_ = 0;  // n-grams ending before this position are in the maps
  std::vector<std::unordered_map<uint64_t, int32_t>> last_end_;  // per n: n-gram hash -> latest end
  int last_n_ = 0, last_k_ = 0;
  // per n and draft position i (1-based): times it held, times it was tried (priors included)
  std::vector<std::vector<double>> held_, tried_;
  double lambda_ = 1.0;  // tokens per unit of cost, a running average
};

// A smaller Metis model drafting for a larger one with the same tokenizer (Praxis for Logos). It
// runs k one-token verify steps on its own slot, so it rolls back to whatever the target accepted.
class ModelDrafter final : public Drafter {
 public:
  // `model` must be built with speculative = true. filter: greedy proposals (temperature <= 0) or
  // proposals drawn from the drafter's own filtered distribution (then Draft::q is set). Drafting
  // stops early once the drafter's probability of its own proposal falls below min_confidence.
  ModelDrafter(MetisModel& model, int slot, lx_spec_filter filter, uint64_t seed, float min_confidence = 0.0f);
  ~ModelDrafter() override;
  const char* name() const override { return "model"; }
  void begin(const std::vector<int32_t>& prompt, void* stream) override;
  Draft propose(const DraftRequest& request) override;
  void observe(int accepted, int32_t next, void* stream) override;

 private:
  MetisModel& model_;
  int slot_;
  lx_spec_filter filter_;
  uint64_t seed_;
  float min_confidence_;
  int fed_ = 0;       // tokens of the sequence the drafter has run
  int ctx_size_ = 0;  // context size at the last proposal
  int drafted_ = 0;
  float *logits_ = nullptr, *q_ = nullptr, *conf_probs_ = nullptr;
  int32_t *d_tokens_ = nullptr, *d_cu_ = nullptr, *d_out_ = nullptr;
  lx_spec_filter* d_filter_ = nullptr;
  lx_spec_rng* d_rng_ = nullptr;
  int32_t* h_out_ = nullptr;
  float* h_conf_ = nullptr;
};

struct SpecConfig {
  lx_spec_filter filter{0.0f, 1.0f, 0, 0.0f};  // temperature <= 0: greedy
  uint64_t seed = 0;
  int max_drafts = 6;  // clamped to the target's max_verify_tokens() - 1
  int max_new_tokens = 256;
  std::vector<int32_t> stop_ids;
  int prefill_chunk = 512;
};

struct SpecStats {
  int steps = 0, emitted = 0, drafted = 0, accepted = 0;
  double prefill_s = 0, decode_s = 0;
  std::vector<int> accept_hist;  // steps by number of accepted drafts
};

// Owns the target's step buffers. generate() runs one sequence on the target's slot 0: plainly
// (drafter null: one-token forward steps, the engine's normal decode path) or speculatively.
class SpecDecoder {
 public:
  explicit SpecDecoder(MetisModel& target);
  ~SpecDecoder();
  SpecDecoder(const SpecDecoder&) = delete;
  SpecDecoder& operator=(const SpecDecoder&) = delete;

  std::vector<int32_t> generate(Drafter* drafter, const std::vector<int32_t>& prompt, const SpecConfig& config,
                                SpecStats& stats, void* stream);

  // One verify step of a decoding sequence on `slot`, for the engine's serving loop (engine.cpp, spec_step). `step` is
  // its pending token, then its drafts (1 to max rows). The rows are judged as generate judges them: greedy by the
  // argmax, sampled by the standard acceptance under config's filter, drawn by (config.seed, position). `q` is the
  // drafts' distribution, or null when they are point masses (prompt lookup). Afterwards the slot holds the pending
  // token and the accepted drafts; the rest is truncated. `next` is the token after the accepted drafts.
  void verify(int slot, const SpecConfig& config, const std::vector<int32_t>& step, const float* q, uint64_t position,
              int& accepted, int32_t& next, void* stream);

 private:
  // Picks the step's tokens: accepted drafts and the final token. rows = 1 + drafts.
  void accept(const SpecConfig& config, const std::vector<int32_t>& step, const float* q, uint64_t position,
              int& accepted, int32_t& next, void* stream);

  MetisModel& target_;
  int max_rows_;
  float *logits_ = nullptr, *probs_ = nullptr;
  int32_t *d_tokens_ = nullptr, *d_cu_ = nullptr, *d_argmax_ = nullptr, *d_out_ = nullptr;
  lx_spec_filter* d_filters_ = nullptr;
  lx_spec_rng* d_rng_ = nullptr;
  int32_t* h_out_ = nullptr;
};

}  // namespace lx::runtime
