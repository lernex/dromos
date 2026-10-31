#include "runtime/engine.h"

#include <bit>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#include "core/checkpoint.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"
#include "model/metis.h"
#include "runtime/offload_model.h"
#include "runtime/prefix_cache.h"
#include "runtime/scheduler.h"
#include "runtime/speculative.h"
#include "runtime/json_grammar.h"
#include "runtime/tool_grammar.h"

namespace lx::runtime {
namespace {

using Clock = std::chrono::steady_clock;

// Every backend call returns 0 on success; anything else throws, with the backend's text for the code.
void check(int status, const char* what) {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed: " + lx_rt_error_string(status));
}

// Pinned host memory, so the step's copies are asynchronous.
template <typename T>
class PinnedArray {
 public:
  explicit PinnedArray(size_t count) : count_(count) {
    const auto bytes = static_cast<int64_t>(sizeof(T) * std::max<size_t>(count, 1));
    check(lx_rt_host_alloc(reinterpret_cast<void**>(&data_), bytes), "pinned host allocation");
  }
  ~PinnedArray() {
    if (data_) lx_rt_host_free(data_);
  }
  PinnedArray(const PinnedArray&) = delete;
  PinnedArray& operator=(const PinnedArray&) = delete;
  T* data() const { return data_; }
  size_t size() const { return count_; }

 private:
  T* data_ = nullptr;
  size_t count_ = 0;
};

// The bytes at the end of `text` that could be the start of a stop string: they wait until the
// next token says whether the stop string completes.
size_t held_suffix(const std::string& text, const std::vector<std::string>& strings) {
  size_t hold = 0;
  for (const std::string& s : strings) {
    const size_t longest = std::min(s.size() - 1, text.size());
    for (size_t k = longest; k > hold; --k) {
      if (text.compare(text.size() - k, k, s, 0, k) == 0) {
        hold = k;
        break;
      }
    }
  }
  return hold;
}

}  // namespace

// The offload runtime's options for a plan: its formats and budgets, the CPU lane's share as the plan placed it,
// and the measurement knobs `dromos offload` reads (LX_ROUTING_STATS, LX_OFFLOAD_DYNAMIC, LX_OFFLOAD_CPU_FRACTION,
// LX_OFFLOAD_CPU_PAIRS, LX_OFFLOAD_FILL), which win when set.
lx::offload::OffloadLoadOptions offload_options(const Plan& plan) {
  auto env = [](const char* name, double fallback) {
    const char* v = std::getenv(name);
    return v && *v ? std::atof(v) : fallback;
  };
  lx::offload::OffloadLoadOptions o;
  o.weights = plan.weights;
  o.experts = plan.recipe.experts;
  o.vram_budget = static_cast<int64_t>(plan.offload.vram_budget);
  o.ram_budget = plan.offload.ram_budget > 0 ? static_cast<int64_t>(plan.offload.ram_budget) : -1;
  if (const char* stats = std::getenv("LX_ROUTING_STATS")) o.routing_stats = stats;
  o.policy.dynamic_fraction = env("LX_OFFLOAD_DYNAMIC", 1.0);
  o.policy.cpu_fraction = env("LX_OFFLOAD_CPU_FRACTION", plan.offload.cpu_fraction);
  o.policy.max_cpu_pairs = static_cast<int>(env("LX_OFFLOAD_CPU_PAIRS", plan.offload.max_cpu_pairs));
  o.policy.fill = env("LX_OFFLOAD_FILL", 1) != 0;
  return o;
}

struct Sequence {
  uint64_t id = 0;
  GenerateRequest request;
  std::unique_ptr<ToolGrammar> grammar;               // when the request must call a tool
  std::unique_ptr<JsonGrammar> json;                  // when the reply must be JSON of a given shape
  RequestCallbacks callbacks;
  std::vector<int32_t> stop_ids;                       // the request's, plus the tokenizer's end ids
  std::unique_ptr<lx::text::Tokenizer::Stream> detok;  // null without a tokenizer
  std::unique_ptr<lx::text::OutputParser> parser;      // null without a tokenizer
  std::string held;                                    // text that may begin a stop string
  int slot = -1;
  int prompt_fed = 0;
  int cached = 0;                       // prompt tokens restored from the cache, not fed
  int generated = 0;
  int spec_steps = 0, spec_drafted = 0, spec_accepted = 0;  // prompt-lookup verify steps, drafts, accepted drafts
  bool has_pending = false;  // a sampled token waits to be fed
  int32_t pending = 0;
  std::vector<int32_t> history;                   // the prompt, then every token shown: the drafter's context
  std::unique_ptr<NgramDrafter> drafter;          // prompt lookup for this sequence, made when it first speculates
  bool cancelled = false;
  bool ended = false;
  double submitted_at = 0.0, first_token_at = 0.0;
};

struct Engine::Impl {
  Impl(const std::string& dir, const Plan& plan, const EngineOptions& options)
      : config(lx::load_config(dir)),
        V(config.vocab_size),
        max_slots(plan.max_slots),
        kv_capacity(plan.kv_capacity),
        max_step_tokens(plan.max_step_tokens),
        prefill_chunk(plan.prefill_chunk > 0 ? plan.prefill_chunk : plan.max_step_tokens),
        grid(options.prefill_grid > 0 ? std::min(options.prefill_grid, plan.prefill_chunk > 0 ? plan.prefill_chunk : plan.max_step_tokens) : 0),
        max_queue(options.max_queue) {
    if (max_slots < 1 || kv_capacity < 2 || max_step_tokens < 1)
      throw std::invalid_argument("the plan leaves no room for a sequence");
    if (options.load_tokenizer) tokenizer = lx::text::Tokenizer::load(dir);

    check(lx_rt_stream_create(&stream), "stream creation");
    // The weights in the plan's format, and the model in the plan's formats and maths. An offloaded plan loads
    // the dense weights alone (ModelWeights with skip_experts, inside OffloadedModel) and holds the routed
    // experts in the offload runtime, in the plan's expert format.
    const lx::ModelFormats formats = plan.formats();
    if (plan.offload.enabled) {
      offloaded = lx::offload::OffloadedModel::load(dir, config, offload_options(plan));
      model = std::make_unique<lx::MetisModel>(config, offloaded->weights(), formats, max_slots, kv_capacity,
                                               max_step_tokens, &offloaded->experts());
    } else {
      lx::LoadOptions load;
      load.target = plan.weights;
      weights = std::make_unique<lx::ModelWeights>(lx::ModelWeights::load(dir, config, load));
      model = std::make_unique<lx::MetisModel>(config, *weights, formats, max_slots, kv_capacity, max_step_tokens,
                                               nullptr, options.speculate);
    }
    if (options.speculate && !plan.offload.enabled && model->max_verify_tokens() > 1)
      spec = std::make_unique<SpecDecoder>(*model);
    if (!options.speculate) speculation_note = "speculation off: --no-spec";
    else if (plan.offload.enabled) speculation_note = "speculation off: experts are offloaded";
    else if (!spec) speculation_note = "speculation off: this model has no verify step";
    else speculation_note = "speculation on: a lone stream verifies drafts; with several streams decoding, steps are plain";
#ifdef LX_SLOT_STATE
    if (options.prefix_cache && plan.prefix_device_bytes + plan.prefix_host_bytes > 0)
      cache = std::make_unique<PrefixCache>(plan.prefix_device_bytes, plan.prefix_host_bytes, stream);
#else
    (void)options.prefix_cache;
#endif

    const size_t slots = static_cast<size_t>(max_slots);
    logits = DeviceBuffer(sizeof(float) * slots * V);
    seen = DeviceBuffer(sizeof(int32_t) * slots * V);
    out_counts = DeviceBuffer(sizeof(int32_t) * slots * V);
    work = DeviceBuffer(sizeof(float) * slots * V);
    params = DeviceBuffer(sizeof(lx_sampling) * slots);
    tokens = DeviceBuffer(sizeof(int32_t) * slots);
    token_lp = DeviceBuffer(sizeof(float) * slots);
    top_ids = DeviceBuffer(sizeof(int32_t) * slots * kMaxTopLogprobs);
    top_lp = DeviceBuffer(sizeof(float) * slots * kMaxTopLogprobs);
    prompt_scratch = DeviceBuffer(sizeof(int32_t) * static_cast<size_t>(kv_capacity));
    check(lx_rt_memset(seen.data(), 0, static_cast<int64_t>(seen.size()), nullptr), "clearing the seen counts");
    check(lx_rt_stream_sync(nullptr), "clearing the seen counts");
    check(lx_rt_memset(out_counts.data(), 0, static_cast<int64_t>(out_counts.size()), nullptr),
          "clearing the output counts");
    check(lx_rt_stream_sync(nullptr), "clearing the output counts");

    if (tokenizer) {
      token_texts.resize(static_cast<size_t>(V));
      for (int i = 0; i < V; ++i) token_texts[static_cast<size_t>(i)] = tokenizer->decode({i}, false);
      std::vector<std::string> bytes(static_cast<size_t>(V));
      for (int i = 0; i < V; ++i) bytes[static_cast<size_t>(i)] = tokenizer->token_bytes(i);
      vocab_index = std::make_unique<VocabIndex>(bytes);
    }
    mask_words = (static_cast<size_t>(V) + 31) / 32;
    mask_rows = DeviceBuffer(sizeof(int32_t) * slots);
    mask_bits = DeviceBuffer(sizeof(uint32_t) * slots * mask_words);
    host_mask_rows = std::make_unique<PinnedArray<int32_t>>(slots);
    host_mask_bits = std::make_unique<PinnedArray<uint32_t>>(slots * mask_words);

    host_params = std::make_unique<PinnedArray<lx_sampling>>(slots);
    host_tokens = std::make_unique<PinnedArray<int32_t>>(slots);
    host_lp = std::make_unique<PinnedArray<float>>(slots);
    host_top_ids = std::make_unique<PinnedArray<int32_t>>(slots * kMaxTopLogprobs);
    host_top_lp = std::make_unique<PinnedArray<float>>(slots * kMaxTopLogprobs);

    for (int s = max_slots - 1; s >= 0; --s) free_slots.push_back(s);
    stats.slots = max_slots;
    // Last, with everything else allocated: the expert cache takes the VRAM the budget leaves.
    if (offloaded) {
      offloaded->finalize();
      placement = offloaded->describe();
    }
    epoch = Clock::now();
  }

  ~Impl() {
    if (stream) lx_rt_stream_destroy(stream);
  }

  double now() const { return std::chrono::duration<double>(Clock::now() - epoch).count(); }

  // scheduler thread
  void loop();
  void admit();
  void step();
  void sample(const std::vector<int>& rows, const std::vector<std::shared_ptr<Sequence>>& seqs);
  bool accept(const std::shared_ptr<Sequence>& s, int32_t token, float logprob, const int32_t* top_ids_row,
              const float* top_lp_row, int top_n, bool feed = true);
  bool spec_eligible(const std::shared_ptr<Sequence>& s) const;
  bool spec_step(const std::shared_ptr<Sequence>& s);
  void end(const std::shared_ptr<Sequence>& s, FinishReason reason, const std::string& error, bool show_held);
  void deliver(const std::shared_ptr<Sequence>& s, const TokenDelta& delta);
  void retire(const std::shared_ptr<Sequence>& s);
  void fail_all(const std::string& reason);
  void publish_stats();
  void restore_prefix(const std::shared_ptr<Sequence>& s);
  void snapshot(const std::shared_ptr<Sequence>& s, int length, const std::vector<int32_t>& tokens);

  lx::MetisConfig config;
  std::unique_ptr<lx::ModelWeights> weights;               // resident plans
  std::unique_ptr<lx::offload::OffloadedModel> offloaded;  // offloaded plans: the dense weights and the experts
  std::unique_ptr<lx::MetisModel> model;                   // declared after both, so it is destroyed first
  std::unique_ptr<SpecDecoder> spec;                       // verify steps of a lone stream; null when speculation is off
  std::string placement;                                   // where the offloaded experts live, or ""
  std::string speculation_note;                            // why speculation is off, or what it does (speculation_note())
  std::unique_ptr<PrefixCache> cache;  // null when prefix reuse is off or the model lacks slot state
  std::vector<std::string> token_texts;  // every token's text, for the tool grammar's masks
  std::unique_ptr<VocabIndex> vocab_index;  // every token's raw bytes as a trie, for the JSON grammar
  std::mutex json_mu;                       // json_programs: filled by submit on the caller's thread
  std::unordered_map<std::string, std::shared_ptr<JsonProgram>> json_programs;
  std::vector<double> mask_ring = std::vector<double>(4096, 0.0);  // microseconds per timed step (under mu)
  size_t mask_count = 0;
  std::shared_ptr<JsonProgram> json_program(const JsonFormat& format);
  size_t mask_words = 0;                 // 32-bit words per token mask
  DeviceBuffer mask_rows, mask_bits;     // device staging for the masks of one step
  std::unique_ptr<PinnedArray<int32_t>> host_mask_rows;
  std::unique_ptr<PinnedArray<uint32_t>> host_mask_bits;
  std::unique_ptr<lx::text::Tokenizer> tokenizer;
  void* stream = nullptr;
  int V;
  int max_slots, kv_capacity, max_step_tokens, prefill_chunk;
  int grid;  // the prefill grid, or 0
  size_t max_queue;

  DeviceBuffer logits, seen, out_counts, work, params, tokens, token_lp, top_ids, top_lp, prompt_scratch;
  std::unique_ptr<PinnedArray<lx_sampling>> host_params;
  std::unique_ptr<PinnedArray<int32_t>> host_tokens, host_top_ids;
  std::unique_ptr<PinnedArray<float>> host_lp, host_top_lp;

  Clock::time_point epoch;
  std::thread thread;

  // shared with submitters, under mu
  std::mutex mu;
  std::condition_variable cv;
  bool stopping = false;
  bool broken = false;
  std::string broken_reason;
  std::deque<std::shared_ptr<Sequence>> incoming;
  std::set<uint64_t> cancel_ids;
  size_t queued = 0;  // requests submitted and not yet admitted or ended
  EngineStats stats;

  // scheduler thread only
  std::deque<std::shared_ptr<Sequence>> waiting;
  std::vector<std::shared_ptr<Sequence>> running;
  std::vector<int> free_slots;
  unsigned cursor = 0;
};

Engine::Engine(const std::string& model_dir, const Plan& plan, const EngineOptions& options) : plan_(plan) {
  impl_ = std::make_unique<Impl>(model_dir, plan, options);
  impl_->thread = std::thread([this] { impl_->loop(); });
}

Engine::~Engine() {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->stopping = true;
  }
  impl_->cv.notify_all();
  if (impl_->thread.joinable()) impl_->thread.join();
}

const lx::MetisConfig& Engine::config() const { return impl_->config; }

const lx::text::Tokenizer* Engine::tokenizer() const { return impl_->tokenizer.get(); }

double Engine::now() const { return impl_->now(); }

const std::string& Engine::placement() const { return impl_->placement; }
const std::string& Engine::speculation_note() const { return impl_->speculation_note; }

lx::DecodeTraffic Engine::decode_traffic(int context) const { return impl_->model->decode_traffic(context); }
void Engine::set_expert_census(bool on) { impl_->model->set_expert_census(on); }

uint64_t Engine::submit(GenerateRequest request, RequestCallbacks callbacks) {
  Impl& m = *impl_;
  if (request.prompt.empty()) throw std::invalid_argument("the prompt has no tokens");
  if (static_cast<int>(request.prompt.size()) >= m.kv_capacity)
    throw std::invalid_argument("the prompt is " + std::to_string(request.prompt.size()) +
                                " tokens, which leaves no room in the context of " +
                                std::to_string(m.kv_capacity));
  for (int32_t id : request.prompt)
    if (id < 0 || id >= m.V) throw std::invalid_argument("token id " + std::to_string(id) + " is outside the vocabulary");
  if (request.max_tokens < 1) throw std::invalid_argument("max_tokens must be at least 1");
  const SamplingParams& sp = request.sampling;
  if (!(sp.temperature >= 0.0f) || !std::isfinite(sp.temperature))
    throw std::invalid_argument("temperature must be 0 or more");
  if (!(sp.top_p > 0.0f && sp.top_p <= 1.0f)) throw std::invalid_argument("top_p must be above 0 and at most 1");
  if (sp.top_k < 0) throw std::invalid_argument("top_k must be 0 or more");
  if (!(sp.min_p >= 0.0f && sp.min_p <= 1.0f)) throw std::invalid_argument("min_p must be between 0 and 1");
  if (!(sp.repetition_penalty > 0.0f)) throw std::invalid_argument("repetition_penalty must be above 0");
  if (sp.top_logprobs < 0 || sp.top_logprobs > kMaxTopLogprobs)
    throw std::invalid_argument("top_logprobs must be between 0 and " + std::to_string(kMaxTopLogprobs));
  for (const std::string& s : request.stop_strings)
    if (s.empty()) throw std::invalid_argument("a stop string is empty");
  if (!request.stop_strings.empty() && !m.tokenizer)
    throw std::invalid_argument("stop strings need a tokenizer, and this engine has none");
  request.max_tokens = std::min(request.max_tokens, m.kv_capacity - static_cast<int>(request.prompt.size()));

  auto seq = std::make_shared<Sequence>();
  if (request.tool_grammar) {
    if (!m.tokenizer) throw std::invalid_argument("a tool grammar needs a tokenizer, and this engine has none");
    seq->grammar = std::make_unique<ToolGrammar>(*request.tool_grammar, request.thinking);
  }
  seq->stop_ids = request.stop_ids;
  if (!request.ignore_eos && m.tokenizer)
    for (int32_t id : m.tokenizer->stop_ids()) seq->stop_ids.push_back(id);
  if (m.tokenizer) {
    seq->detok = m.tokenizer->stream(false);
    seq->parser = std::make_unique<lx::text::OutputParser>(request.thinking, request.tools_json);
  }
  if (request.json) {
    if (!m.tokenizer) throw std::invalid_argument("a JSON grammar needs a tokenizer, and this engine has none");
    JsonFormat format = *request.json;
    format.thinking = request.thinking;
    seq->json = std::make_unique<JsonGrammar>(m.json_program(format), *m.vocab_index);
  }
  seq->request = std::move(request);
  seq->callbacks = std::move(callbacks);
  seq->submitted_at = m.now();
  seq->id = next_id_.fetch_add(1);
  const uint64_t id = seq->id;
  {
    std::lock_guard<std::mutex> lock(m.mu);
    if (m.broken) throw std::runtime_error("the engine stopped: " + m.broken_reason);
    if (m.stopping) throw std::runtime_error("the engine is shutting down");
    if (m.queued >= m.max_queue) throw std::runtime_error("the request queue is full");
    m.queued += 1;
    m.incoming.push_back(std::move(seq));
  }
  m.cv.notify_all();
  return id;
}

void Engine::cancel(uint64_t id) {
  {
    std::lock_guard<std::mutex> lock(impl_->mu);
    impl_->cancel_ids.insert(id);
  }
  impl_->cv.notify_all();
}

EngineStats Engine::stats() const {
  std::lock_guard<std::mutex> lock(impl_->mu);
  EngineStats out = impl_->stats;
  out.mask_samples = impl_->mask_count;
  const size_t n = std::min(impl_->mask_count, impl_->mask_ring.size());
  if (n > 0) {
    std::vector<double> us(impl_->mask_ring.begin(), impl_->mask_ring.begin() + static_cast<std::ptrdiff_t>(n));
    std::sort(us.begin(), us.end());
    out.mask_us_p50 = us[n / 2];
    out.mask_us_p99 = us[std::min(n - 1, n * 99 / 100)];
  }
  return out;
}

// The compiled schema for a format: compiled once per (thinking, schema) and shared by its requests. Throws what
// compile_json throws for a schema it cannot express. Runs on the submitting thread.
std::shared_ptr<JsonProgram> Engine::Impl::json_program(const JsonFormat& format) {
  const std::string key = std::string(format.thinking ? "T" : "F") + (format.object ? "object" : "schema") + "\n" +
                          format.schema;
  std::lock_guard<std::mutex> lock(json_mu);
  const auto found = json_programs.find(key);
  if (found != json_programs.end()) return found->second;
  const auto t0 = std::chrono::steady_clock::now();
  std::shared_ptr<JsonProgram> program = compile_json(format);
  const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  {
    std::lock_guard<std::mutex> stats_lock(mu);
    stats.grammar_compile_ms = ms;
  }
  if (json_programs.size() >= 32) json_programs.clear();
  json_programs.emplace(key, program);
  return program;
}

// ---- the scheduler thread ------------------------------------------------------------------

void Engine::Impl::loop() {
  while (true) {
    std::deque<std::shared_ptr<Sequence>> arrived;
    std::set<uint64_t> cancels;
    {
      std::unique_lock<std::mutex> lock(mu);
      cv.wait(lock, [this] {
        return stopping || !incoming.empty() || !cancel_ids.empty() || !running.empty() || !waiting.empty();
      });
      if (stopping) break;
      arrived.swap(incoming);
      cancels.swap(cancel_ids);
    }
    try {
      for (auto& s : arrived) waiting.push_back(std::move(s));
      for (uint64_t id : cancels) {
        auto w = std::find_if(waiting.begin(), waiting.end(), [id](const auto& s) { return s->id == id; });
        if (w != waiting.end()) {
          auto s = *w;
          waiting.erase(w);
          {
            std::lock_guard<std::mutex> lock(mu);
            queued -= 1;
          }
          end(s, FinishReason::Cancelled, "", false);
          continue;
        }
        for (auto& r : running)
          if (r->id == id) r->cancelled = true;
      }
      for (auto& r : std::vector<std::shared_ptr<Sequence>>(running))
        if (r->cancelled) {
          end(r, FinishReason::Cancelled, "", false);
          retire(r);
        }
      admit();
      if (!running.empty()) step();
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dromos: the engine stopped: %s\n", e.what());
      fail_all(e.what());
      return;
    }
    publish_stats();
  }
  fail_all("the engine is shutting down");
}

void Engine::Impl::publish_stats() {
  std::lock_guard<std::mutex> lock(mu);
  stats.running = static_cast<int>(running.size());
  stats.waiting = static_cast<int>(waiting.size() + incoming.size());
  if (cache) {
    stats.prefix_device_bytes = cache->device_used();
    stats.prefix_host_bytes = cache->host_used();
  }
}

void Engine::Impl::admit() {
  while (!free_slots.empty() && !waiting.empty()) {
    auto s = waiting.front();
    waiting.pop_front();
    {
      std::lock_guard<std::mutex> lock(mu);
      queued -= 1;
    }
    s->slot = free_slots.back();
    free_slots.pop_back();
    model->reset_slot(s->slot);
    s->history = s->request.prompt;
    restore_prefix(s);
    check(lx_sampler_reset(static_cast<int32_t*>(seen.data()), static_cast<int32_t*>(out_counts.data()), s->slot,
                           V, stream),
          "lx_sampler_reset");
    // the prompt counts toward the repetition penalty from its first token on
    const size_t n = s->request.prompt.size();
    check(lx_rt_copy(prompt_scratch.data(), s->request.prompt.data(), static_cast<int64_t>(sizeof(int32_t) * n),
                     LX_COPY_H2D, stream),
          "copy of the prompt to the device");
    check(lx_sampler_observe(static_cast<int32_t*>(seen.data()), static_cast<int32_t*>(out_counts.data()), s->slot,
                             V, static_cast<const int32_t*>(prompt_scratch.data()), static_cast<int>(n), 0, stream),
          "lx_sampler_observe");
    running.push_back(s);
  }
}

void Engine::Impl::step() {
  // A lone decoding stream verifies the drafts its own text proposes; without one, the plain step below runs.
  if (running.size() == 1 && spec_eligible(running[0]) && spec_step(running[0])) return;
  std::vector<SchedSeq> view;
  view.reserve(running.size());
  for (const auto& s : running)
    view.push_back(SchedSeq{static_cast<int>(s->request.prompt.size()) - s->prompt_fed, s->has_pending,
                            s->prompt_fed});
  const SchedPlan plan = plan_step(view, max_step_tokens, prefill_chunk, cursor, grid);
  if (plan.entries.empty()) return;

  // Entry e is one sequence's tokens; its logits are row e of the model's output.
  lx::StepBatch batch;
  batch.cu_seqlens.push_back(0);
  std::vector<int> rows;
  std::vector<std::shared_ptr<Sequence>> sampled;
  uint64_t prompt_tokens = 0;
  bool all_decode = true;
  std::vector<std::shared_ptr<Sequence>> prompt_moved;  // sequences whose prompt advanced this step
  for (size_t e = 0; e < plan.entries.size(); ++e) {
    const std::shared_ptr<Sequence> s = running[plan.entries[e].seq];
    if (s->has_pending) {
      batch.tokens.push_back(s->pending);
      s->has_pending = false;
      rows.push_back(static_cast<int>(e));
      sampled.push_back(s);
    } else {
      all_decode = false;
      const int n = plan.entries[e].tokens;
      const auto& prompt = s->request.prompt;
      for (int k = 0; k < n; ++k) batch.tokens.push_back(prompt[s->prompt_fed + k]);
      s->prompt_fed += n;
      prompt_tokens += static_cast<uint64_t>(n);
      prompt_moved.push_back(s);
      if (s->prompt_fed == static_cast<int>(prompt.size())) {
        rows.push_back(static_cast<int>(e));
        sampled.push_back(s);
      }
    }
    batch.cu_seqlens.push_back(static_cast<int32_t>(batch.tokens.size()));
    batch.slots.push_back(s->slot);
  }

  model->forward(batch, static_cast<float*>(logits.data()), false, stream);
  if (cache && grid > 0)
    for (const auto& s : prompt_moved)
      if (snapshot_wanted(s->prompt_fed, static_cast<int>(s->request.prompt.size()), grid))
        snapshot(s, s->prompt_fed, s->request.prompt);  // queued before the next step
  {
    std::lock_guard<std::mutex> lock(mu);
    stats.steps += 1;
    if (all_decode) stats.decode_steps += 1;
    stats.prompt_tokens += prompt_tokens;
  }
  if (!sampled.empty()) sample(rows, sampled);
}

void Engine::Impl::sample(const std::vector<int>& rows, const std::vector<std::shared_ptr<Sequence>>& seqs) {
  const int n = static_cast<int>(seqs.size());
  int top_n = 0;
  bool want_lp = false;
  for (int j = 0; j < n; ++j) {
    const Sequence& s = *seqs[j];
    const SamplingParams& sp = s.request.sampling;
    lx_sampling& p = host_params->data()[j];
    p = lx_sampling{};
    p.seed = sp.seed;
    p.position = static_cast<uint64_t>(s.generated);
    p.slot = s.slot;
    p.row = rows[j];
    p.temperature = sp.temperature;
    p.top_p = sp.top_p;
    p.top_k = sp.top_k;
    p.min_p = sp.min_p;
    p.repetition_penalty = sp.repetition_penalty;
    p.presence_penalty = sp.presence_penalty;
    p.frequency_penalty = sp.frequency_penalty;
    if (sp.logprobs) {
      want_lp = true;
      top_n = std::max(top_n, sp.top_logprobs);
    }
  }
  // constrained rows: the logits of the tokens their grammar rules out become -inf before sampling
  int constrained = 0;
  const auto mask_start = std::chrono::steady_clock::now();
  uint64_t json_rows = 0, json_sole = 0;
  for (int j = 0; j < n; ++j) {
    Sequence& s = *seqs[j];
    const bool json = static_cast<bool>(s.json);
    if (!json && !s.grammar) continue;
    const std::vector<uint32_t>& bits = json ? s.json->mask(s.stop_ids) : s.grammar->mask(token_texts, s.stop_ids);
    bool any = false;
    for (uint32_t w : bits) any = any || w != 0;
    if (!any) {
      // A JSON reply with no token left is complete (or dead): the row stays unmasked, and accept() ends the reply.
      if (json) continue;
      throw std::runtime_error("the tool grammar allows no token in a reply");
    }
    if (json) {
      json_rows += 1;
      size_t allowed = 0;
      for (uint32_t w : bits) allowed += static_cast<size_t>(std::popcount(w));
      for (int32_t id : s.stop_ids)
        if (id >= 0 && id < V && (bits[static_cast<size_t>(id) / 32] >> (static_cast<size_t>(id) % 32) & 1u)) allowed -= 1;
      if (allowed == 1) json_sole += 1;
    }
    host_mask_rows->data()[constrained] = rows[j];
    std::memcpy(host_mask_bits->data() + static_cast<size_t>(constrained) * mask_words, bits.data(),
                sizeof(uint32_t) * mask_words);
    constrained += 1;
  }
  if (json_rows > 0) {
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - mask_start).count();
    std::lock_guard<std::mutex> lock(mu);
    mask_ring[mask_count % mask_ring.size()] = us;
    mask_count += 1;
    stats.constrained_steps += 1;
    stats.constrained_rows += json_rows;
    stats.constrained_sole += json_sole;
  }
  if (constrained > 0) {
    check(lx_rt_copy(mask_rows.data(), host_mask_rows->data(), static_cast<int64_t>(sizeof(int32_t) * constrained),
                     LX_COPY_H2D, stream),
          "copy of the mask rows to the device");
    check(lx_rt_copy(mask_bits.data(), host_mask_bits->data(),
                     static_cast<int64_t>(sizeof(uint32_t) * constrained * mask_words), LX_COPY_H2D, stream),
          "copy of the mask bits to the device");
    check(lx_mask_logits(static_cast<float*>(logits.data()), V, static_cast<const int32_t*>(mask_rows.data()),
                         static_cast<const uint32_t*>(mask_bits.data()), constrained, stream),
          "lx_mask_logits");
  }
  check(lx_rt_copy(params.data(), host_params->data(), static_cast<int64_t>(sizeof(lx_sampling) * n),
                   LX_COPY_H2D, stream),
        "copy of the sampling parameters to the device");
  check(lx_sample(static_cast<const float*>(logits.data()), V, static_cast<const lx_sampling*>(params.data()), n,
                  static_cast<int32_t*>(seen.data()), static_cast<int32_t*>(out_counts.data()),
                  static_cast<float*>(work.data()), static_cast<int32_t*>(tokens.data()),
                  want_lp ? static_cast<float*>(token_lp.data()) : nullptr, top_n,
                  static_cast<int32_t*>(top_ids.data()), static_cast<float*>(top_lp.data()), stream),
        "lx_sample");
  check(lx_rt_copy(host_tokens->data(), tokens.data(), static_cast<int64_t>(sizeof(int32_t) * n), LX_COPY_D2H, stream),
        "copy of the sampled tokens to the host");
  if (want_lp) {
    check(lx_rt_copy(host_lp->data(), token_lp.data(), static_cast<int64_t>(sizeof(float) * n), LX_COPY_D2H, stream),
          "copy of the token logprobs to the host");
  }
  if (top_n > 0) {
    check(lx_rt_copy(host_top_ids->data(), top_ids.data(), static_cast<int64_t>(sizeof(int32_t) * n * top_n),
                     LX_COPY_D2H, stream),
          "copy of the top token ids to the host");
    check(lx_rt_copy(host_top_lp->data(), top_lp.data(), static_cast<int64_t>(sizeof(float) * n * top_n),
                     LX_COPY_D2H, stream),
          "copy of the top logprobs to the host");
  }
  check(lx_rt_stream_sync(stream), "waiting for the step's results");

  std::vector<std::shared_ptr<Sequence>> ended;
  for (int j = 0; j < n; ++j) {
    const int top_offset = j * top_n;
    const bool done = accept(seqs[j], host_tokens->data()[j], want_lp ? host_lp->data()[j] : 0.0f,
                             top_n > 0 ? host_top_ids->data() + top_offset : nullptr,
                             top_n > 0 ? host_top_lp->data() + top_offset : nullptr, top_n);
    if (done) ended.push_back(seqs[j]);
  }
  for (const auto& s : ended) retire(s);
}

// One sampled token of a sequence: it is shown (or ends the reply), and the sequence either
// queues the token to feed next step or ends. Returns true when the sequence ended.
bool Engine::Impl::accept(const std::shared_ptr<Sequence>& s, int32_t token, float logprob, const int32_t* top_ids_row,
                          const float* top_lp_row, int top_n, bool feed) {
  Sequence& q = *s;
  if (std::find(q.stop_ids.begin(), q.stop_ids.end(), token) != q.stop_ids.end()) {
    end(s, FinishReason::Stop, "", true);
    return true;
  }
  if (q.grammar && !q.grammar->accept(token_texts[static_cast<size_t>(token)])) {
    end(s, FinishReason::Error, "the tool grammar rejected a sampled token", false);
    return true;
  }
  if (q.json && !q.json->accept(vocab_index->bytes(token))) {
    // Only a complete reply has no token left, so a rejected one ends it: as Stop when the value is complete (an
    // unmasked row, from ignore_eos at the end of a value), as an error otherwise.
    if (q.json->may_end()) end(s, FinishReason::Stop, "", true);
    else end(s, FinishReason::Error, "the JSON grammar rejected a sampled token", false);
    return true;
  }
  q.generated += 1;
  q.history.push_back(token);
  if (q.generated == 1) q.first_token_at = now();
  {
    std::lock_guard<std::mutex> lock(mu);
    stats.generated_tokens += 1;
  }

  TokenDelta delta;
  delta.token = token;
  if (q.request.sampling.logprobs) {
    delta.logprob = logprob;
    const int want = q.request.sampling.top_logprobs;
    for (int k = 0; k < want && k < top_n; ++k) delta.top.emplace_back(top_ids_row[k], top_lp_row[k]);
    if (tokenizer) delta.piece = tokenizer->decode({token}, false);
  }

  if (q.detok) {
    const std::string raw = q.detok->push(token);
    const std::string buf = q.held + raw;
    size_t cut = std::string::npos;
    for (const std::string& stop : q.request.stop_strings) cut = std::min(cut, buf.find(stop));
    std::string shown;
    bool stopped = cut != std::string::npos;
    if (stopped) {
      shown = buf.substr(0, cut);
      q.held.clear();
    } else {
      const size_t hold = held_suffix(buf, q.request.stop_strings);
      shown = buf.substr(0, buf.size() - hold);
      q.held = buf.substr(buf.size() - hold);
    }
    auto parsed = q.parser->push(shown);
    delta.reasoning += parsed.reasoning;
    delta.content += parsed.content;
    delta.tool_calls.insert(delta.tool_calls.end(), parsed.tool_calls.begin(), parsed.tool_calls.end());
    deliver(s, delta);
    if (stopped) {
      end(s, FinishReason::Stop, "", false);
      return true;
    }
  } else {
    deliver(s, delta);
  }

  if (q.generated >= q.request.max_tokens ||
      q.prompt_fed + q.generated >= kv_capacity) {
    end(s, FinishReason::Length, "", true);
    return true;
  }
  if (feed) {
    q.pending = token;
    q.has_pending = true;
  }
  return false;
}

// A decoding sequence may verify drafts: it has its first token and its prompt is in. Constrained sequences draft under
// their grammar's masks (spec_step).
// Its penalties are judged row by row by the verify step, as the plain sampler judges them (SpecDecoder::verify).
// A constrained row never speculates: a verify step has no grammar mask, so a draft the schema forbids would be
// accepted or rejected by the target alone. Constrained rows (q.json for response_format, q.grammar for tool calls)
// always take the plain step, where the mask is applied before every sample.
bool Engine::Impl::spec_eligible(const std::shared_ptr<Sequence>& s) const {
  if (!spec) return false;
  const Sequence& q = *s;
  const SamplingParams& sp = q.request.sampling;
  return q.has_pending && !q.ended && !q.cancelled && q.prompt_fed == static_cast<int>(q.request.prompt.size());
}

// The lone stream's step as a verify step: its pending token and the drafts prompt lookup proposes from its history run
// through the target in one step, and the rows are judged (SpecDecoder::verify). The accepted drafts are shown without
// becoming pending (the step fed them); the token after them is pending, as a plain step's sample is. Returns false,
// having run nothing, when there is no draft: the plain step runs instead.
bool Engine::Impl::spec_step(const std::shared_ptr<Sequence>& s) {
  Sequence& q = *s;
  const SamplingParams& sp = q.request.sampling;
  if (!q.drafter) {
    q.drafter = std::make_unique<NgramDrafter>(4, 1, DraftCost::rtx5070ti_praxis());
    q.drafter->begin(q.request.prompt, stream);
  }
  // Drafts go in the request's remaining tokens and the cache's remaining positions, and at most the verify step's rows.
  const int room = q.request.max_tokens - q.generated;
  const int cache_room = kv_capacity - model->slot_length(q.slot);
  DraftRequest r;
  r.context = &q.history;
  r.max_drafts = std::min({6, model->max_verify_tokens() - 1, room - 1, cache_room - 1});
  r.position = static_cast<uint64_t>(q.generated);
  r.stream = stream;
  Draft draft;
  if (r.max_drafts > 0) draft = q.drafter->propose(r);
  if (static_cast<int>(draft.tokens.size()) > r.max_drafts) draft.tokens.resize(static_cast<size_t>(r.max_drafts));
  if (draft.tokens.empty()) {
    q.drafter->observe(0, 0, stream);  // a step without drafts is a plain one: the adaptive length still learns from it
    return false;
  }

  // A constrained sequence drafts under its grammar. Row j of the step is masked to the tokens the grammar allows after
  // the step's first j tokens, found on copies of the grammar advanced along the drafts. The drafts stop before the first
  // one the grammar rules out, before a stop token (the reply ends there, and the row samples it), and before a state
  // with no token left (the plain step decides that one). Greedy and sampled verify both draw from the masked rows, so
  // an accepted draft is one the plain constrained step could have drawn.
  std::vector<uint32_t> row_masks;  // row j at j * spec_mask_words
  const int spec_mask_words = (V + 31) / 32;
  if (q.grammar || q.json) {
    std::unique_ptr<JsonGrammar> json_copy = q.json ? q.json->clone() : nullptr;
    std::unique_ptr<ToolGrammar> tool_copy = q.grammar ? std::make_unique<ToolGrammar>(*q.grammar) : nullptr;
    auto allowed_now = [&]() -> const std::vector<uint32_t>& {
      return json_copy ? json_copy->mask(q.stop_ids) : tool_copy->mask(token_texts, q.stop_ids);
    };
    auto advance = [&](int32_t t) {
      return json_copy ? json_copy->accept(vocab_index->bytes(t)) : tool_copy->accept(token_texts[static_cast<size_t>(t)]);
    };
    auto bit = [](const uint32_t* bits, int32_t t) { return (bits[t / 32] >> (t % 32)) & 1u; };
    auto any = [](const uint32_t* bits, int words) {
      for (int w = 0; w < words; ++w)
        if (bits[w]) return true;
      return false;
    };
    const std::vector<uint32_t>& first = allowed_now();
    if (!any(first.data(), spec_mask_words)) {
      q.drafter->observe(0, 0, stream);  // no token is allowed: the plain step is the one that ends or refuses the reply
      return false;
    }
    row_masks.assign(first.begin(), first.end());
    size_t kept = 0;
    while (kept < draft.tokens.size()) {
      const int32_t t = draft.tokens[kept];
      const uint32_t* row = row_masks.data() + kept * static_cast<size_t>(spec_mask_words);
      if (t < 0 || t >= V || bit(row, t) == 0) break;
      if (std::find(q.stop_ids.begin(), q.stop_ids.end(), t) != q.stop_ids.end()) break;
      if (!advance(t)) break;
      const std::vector<uint32_t>& next_row = allowed_now();
      if (!any(next_row.data(), spec_mask_words)) break;
      row_masks.insert(row_masks.end(), next_row.begin(), next_row.end());
      kept += 1;
    }
    draft.tokens.resize(kept);
    if (kept == 0) {
      q.drafter->observe(0, 0, stream);
      return false;
    }
  }

  std::vector<int32_t> step{q.pending};
  step.insert(step.end(), draft.tokens.begin(), draft.tokens.end());
  q.has_pending = false;
  SpecConfig config;
  config.row_masks = (q.grammar || q.json) ? row_masks.data() : nullptr;
  config.mask_words = spec_mask_words;
  config.filter = lx_spec_filter{sp.temperature, sp.top_p, sp.top_k, sp.min_p};
  config.seed = sp.seed;
  config.stop_ids = q.stop_ids;
  config.repetition_penalty = sp.repetition_penalty;
  config.presence_penalty = sp.presence_penalty;
  config.frequency_penalty = sp.frequency_penalty;
  // A penalised sequence's counts are its slot's rows of the sampler's tables, as lx_sample keeps them.
  const SpecCounts counts = penalized(config) ? SpecCounts{static_cast<int32_t*>(seen.data()),
                                                           static_cast<int32_t*>(out_counts.data())}
                                              : SpecCounts{};
  int accepted = 0;
  int32_t next = 0;
  // The step's emitted tokens carry the logprobs the plain path would report, when the request asks for them.
  config.logprobs = sp.logprobs;
  config.top_logprobs = sp.top_logprobs;
  SpecLogprobs report;
  spec->verify(q.slot, config, step, draft.q, r.position, counts, accepted, next, stream,
               sp.logprobs ? &report : nullptr);
  q.drafter->observe(accepted, next, stream);
  s->spec_steps += 1;
  s->spec_drafted += static_cast<int>(draft.tokens.size());
  s->spec_accepted += accepted;
  {
    std::lock_guard<std::mutex> lock(mu);
    stats.steps += 1;
    stats.decode_steps += 1;
    stats.spec_steps += 1;
    stats.spec_drafted += draft.tokens.size();
    stats.spec_accepted += static_cast<uint64_t>(accepted);
  }
  // Emitted token k was predicted by row k: the drafts' rows, then the final token's.
  auto row_logprob = [&](size_t k, const int32_t** ids, const float** lp) {
    *ids = nullptr;
    *lp = nullptr;
    if (!sp.logprobs) return 0.0f;
    const size_t top = static_cast<size_t>(report.top_n);
    if (report.top_n > 0) {
      *ids = report.top_ids.data() + k * top;
      *lp = report.top_lp.data() + k * top;
    }
    return report.token[k];
  };
  for (int j = 1; j <= accepted; ++j) {
    const int32_t* ids;
    const float* lp;
    const float value = row_logprob(static_cast<size_t>(j - 1), &ids, &lp);
    if (accept(s, step[static_cast<size_t>(j)], value, ids, lp, report.top_n, /*feed=*/false)) {
      retire(s);
      return true;
    }
  }
  const int32_t* ids;
  const float* lp;
  const float value = row_logprob(static_cast<size_t>(accepted), &ids, &lp);
  if (accept(s, next, value, ids, lp, report.top_n, /*feed=*/true)) retire(s);
  return true;
}

void Engine::Impl::deliver(const std::shared_ptr<Sequence>& s, const TokenDelta& delta) {
  if (!s->callbacks.on_token) return;
  try {
    s->callbacks.on_token(delta);
  } catch (...) {
    // A callback that throws has lost its client; the request ends as cancelled.
    s->cancelled = true;
  }
}

// Finishes a sequence: its held-back text and the parser's tail go out as one last delta (unless
// the reply was cut by a stop string or cancelled), then the result is reported. The caller
// removes it from the batch and frees its slot with retire().
void Engine::Impl::end(const std::shared_ptr<Sequence>& s, FinishReason reason, const std::string& error,
                       bool show_held) {
  if (s->ended) return;
  s->ended = true;
  if (s->parser && reason != FinishReason::Cancelled && reason != FinishReason::Error) {
    std::string tail = show_held ? s->held + s->detok->finish() : std::string();
    s->held.clear();
    TokenDelta d;
    d.token = -1;
    auto pushed = s->parser->push(tail);
    auto fin = s->parser->finish();
    d.reasoning = pushed.reasoning + fin.reasoning;
    d.content = pushed.content + fin.content;
    d.tool_calls = std::move(pushed.tool_calls);
    d.tool_calls.insert(d.tool_calls.end(), fin.tool_calls.begin(), fin.tool_calls.end());
    if (!d.reasoning.empty() || !d.content.empty() || !d.tool_calls.empty()) deliver(s, d);
  }

  RequestResult result;
  result.reason = reason;
  result.error = error;
  result.prompt_tokens = static_cast<int>(s->request.prompt.size());
  result.completion_tokens = s->generated;
  result.first_token_at = s->first_token_at;
  result.finished_at = now();
  result.seconds = result.finished_at - s->submitted_at;
  result.time_to_first_token = s->generated > 0 ? s->first_token_at - s->submitted_at : 0.0;
  result.spec_steps = s->spec_steps;
  result.spec_drafted = s->spec_drafted;
  result.spec_accepted = s->spec_accepted;
  if (s->callbacks.on_finish) {
    try {
      s->callbacks.on_finish(result);
    } catch (...) {
    }
  }
}

// A new sequence starts from the longest cached prefix of its prompt that leaves at least one prompt
// token to feed (that step produces the logits the first token is sampled from).
void Engine::Impl::restore_prefix(const std::shared_ptr<Sequence>& s) {
  s->cached = 0;
  s->prompt_fed = 0;
#ifdef LX_SLOT_STATE
  if (!cache) return;
  PrefixCache::Hit hit;
  const int limit = static_cast<int>(s->request.prompt.size()) - 1;
  if (cache->find(s->request.prompt, limit, grid, hit)) {
    model->restore_slot(s->slot, hit.data, hit.length, stream);
    check(lx_rt_stream_sync(stream), "waiting for the prefix restore");  // the snapshot may be evicted next
    s->cached = hit.length;
    s->prompt_fed = hit.length;
    std::lock_guard<std::mutex> lock(mu);
    stats.prefix_hits += 1;
    stats.prefix_tokens_reused += static_cast<uint64_t>(hit.length);
  }
#endif
}

void Engine::Impl::snapshot(const std::shared_ptr<Sequence>& s, int length, const std::vector<int32_t>& tokens) {
#ifdef LX_SLOT_STATE
  const size_t bytes = model->slot_state_bytes(length);
  const bool kept = cache->store(tokens, length, bytes, [&](void* dst) {
    model->save_slot(s->slot, dst, length, stream);
  });
  if (kept) {
    std::lock_guard<std::mutex> lock(mu);
    stats.prefix_snapshots += 1;
  }
#else
  (void)s;
  (void)length;
  (void)tokens;
#endif
}

void Engine::Impl::retire(const std::shared_ptr<Sequence>& s) {
  auto it = std::find(running.begin(), running.end(), s);
  if (it != running.end()) running.erase(it);
  if (s->slot >= 0) {
    free_slots.push_back(s->slot);
    s->slot = -1;
  }
}

// A device failure or shutdown: the engine cannot continue, so every request ends with an error.
void Engine::Impl::fail_all(const std::string& reason) {
  std::vector<std::shared_ptr<Sequence>> all;
  {
    std::lock_guard<std::mutex> lock(mu);
    broken = true;
    broken_reason = reason;
    while (!incoming.empty()) {
      all.push_back(incoming.front());
      incoming.pop_front();
    }
    queued = 0;
    stats.waiting = 0;
    stats.running = 0;
  }
  for (auto& s : waiting) all.push_back(s);
  for (auto& s : running) all.push_back(s);
  waiting.clear();
  running.clear();
  for (const auto& s : all) {
    if (s->ended) continue;
    s->slot = -1;
    end(s, FinishReason::Error, reason, false);
  }
}

}  // namespace lx::runtime
