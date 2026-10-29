// The engine: owns the model, the tokenizer and the plan, and runs continuous batching on one
// thread. Requests come in from any thread; each step packs the decoding sequences' tokens and
// the next chunks of waiting prompts into one model step, samples the sequences that produced a
// token, and streams each token back through the request's callback. Callbacks run on the engine
// thread, so they must only hand the data off (a queue, a socket buffer), never block.
#pragma once

#include <stdint.h>

#include <atomic>
#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/config.h"
#include "model/traffic.h"
#include "runtime/plan.h"
#include "runtime/json_grammar.h"
#include "runtime/tool_grammar.h"
#include "text/tokenizer.h"

namespace lx::offload {
struct OffloadLoadOptions;
}  // namespace lx::offload

namespace lx::runtime {

constexpr int kMaxTopLogprobs = 20;

struct SamplingParams {
  float temperature = 1.0f;         // 0 is greedy
  float top_p = 1.0f;               // (0, 1]
  int top_k = 0;                    // 0 disables
  float min_p = 0.0f;               // 0 disables
  float repetition_penalty = 1.0f;  // 1 disables
  float presence_penalty = 0.0f;
  float frequency_penalty = 0.0f;
  uint64_t seed = 0;
  bool logprobs = false;            // report each token's log-probability
  int top_logprobs = 0;             // alternatives per token, 0 to 20 (needs logprobs)
};

struct GenerateRequest {
  std::vector<int32_t> prompt;      // the rendered prompt's token ids; at least one
  int max_tokens = 256;             // output tokens; clamped to the room left in the context
  SamplingParams sampling;
  std::vector<int32_t> stop_ids;    // extra ids that end the reply (the tokenizer's end ids always do)
  std::vector<std::string> stop_strings;  // end the reply when one appears; the text before it is kept
  bool thinking = true;             // the reply's reasoning is split off when true, as the template does
  bool ignore_eos = false;          // keep going past end ids (benchmarks)
  // When set, the reply after its reasoning is a tool call the grammar allows (tool_choice required
  // or named). Needs a tokenizer.
  std::optional<ToolGrammarSpec> tool_grammar;
  std::string tools_json;           // the request's tools array (OpenAI shape): types each argument by its schema
  // response_format json_object or json_schema: the reply is a JSON value of that shape (see json_grammar.h).
  // Its thinking flag is the request's. Needs a tokenizer.
  std::optional<JsonFormat> json;
};

// What one generated token brings to the caller. Text is what is now safe to show: the
// tokenizer holds back incomplete UTF-8, and the output parser splits reasoning, content and
// tool calls. A delta is produced for every token, so the token's own fields are always set.
struct TokenDelta {
  std::string reasoning;
  std::string content;
  std::vector<lx::text::ToolCall> tool_calls;  // calls completed by this token
  int32_t token = -1;
  std::string piece;                           // the token's own text, for logprobs
  float logprob = 0.0f;
  std::vector<std::pair<int32_t, float>> top;  // the most likely tokens, when logprobs are wanted
};

enum class FinishReason { Stop, Length, Cancelled, Error };

struct RequestResult {
  FinishReason reason = FinishReason::Stop;
  std::string error;                 // when Error
  int prompt_tokens = 0;
  int completion_tokens = 0;
  double time_to_first_token = 0.0;  // seconds from submit to the first sampled token (0 if none)
  double seconds = 0.0;              // seconds from submit to the finish
  double first_token_at = 0.0;       // engine clock, seconds, for throughput over a set of requests
  double finished_at = 0.0;
  // Prompt-lookup speculation on this request (zero when it never ran a verify step): the verify steps, the draft
  // tokens proposed to them, and the draft tokens the target accepted.
  int spec_steps = 0;
  int spec_drafted = 0;
  int spec_accepted = 0;
};

struct RequestCallbacks {
  std::function<void(const TokenDelta&)> on_token;
  std::function<void(const RequestResult&)> on_finish;  // always called exactly once
};

struct EngineOptions {
  size_t max_queue = 4096;   // waiting requests beyond this are refused
  bool load_tokenizer = true;
  // Reuse the state of prompts already run: a new request starts from the longest cached prefix of
  // its tokens that ends on the prefill grid. Needs the model's slot state (LX_SLOT_STATE).
  bool prefix_cache = true;
  // Prompts are fed in pieces that end on multiples of this (absolute positions), and the state is
  // kept at those points. A request that starts from a snapshot then computes exactly what a cold
  // run computes. 0 feeds prompts in whole chunks, with no snapshots.
  int prefill_grid = 512;
  // Prompt-lookup speculation for a lone decoding stream: its next tokens are drafted from its own text and checked
  // in one verify step (speculative.h, SpecDecoder::verify). Exact: greedy output is the plain output, and sampling
  // keeps the target's distribution. Off for offloaded plans. Several streams decode plainly.
  bool speculate = true;
};

struct EngineStats {
  uint64_t steps = 0;
  uint64_t decode_steps = 0;     // steps where every entry fed one token
  uint64_t prompt_tokens = 0;    // prompt tokens fed
  uint64_t generated_tokens = 0; // tokens sampled
  int running = 0;
  int waiting = 0;
  int slots = 0;
  uint64_t prefix_hits = 0;           // requests that started from a cached prefix
  uint64_t prefix_tokens_reused = 0;  // prompt tokens not fed because a cached prefix covered them
  uint64_t prefix_snapshots = 0;      // state snapshots taken
  uint64_t prefix_device_bytes = 0;   // snapshot bytes held now, in device memory and pinned host memory
  uint64_t prefix_host_bytes = 0;
  uint64_t constrained_steps = 0;     // sampling steps that had a JSON-constrained row
  uint64_t constrained_rows = 0;      // JSON-constrained rows sampled
  uint64_t constrained_sole = 0;      // rows where exactly one non-stop token was allowed (a jump could have fed it)
  uint64_t mask_samples = 0;          // steps timed for the constrained masks (the ring holds the last ones)
  double mask_us_p50 = 0.0;           // CPU time to build a step's constrained masks: median and 99th percentile
  double mask_us_p99 = 0.0;
  double grammar_compile_ms = 0.0;    // the last schema compiled (cache misses only)
  uint64_t spec_steps = 0;            // verify steps of a lone stream (prompt-lookup drafts checked in one step)
  uint64_t spec_drafted = 0;          // draft tokens those verify steps checked
  uint64_t spec_accepted = 0;         // draft tokens the target accepted
};

class Engine {
 public:
  // Loads the checkpoint in model_dir in the plan's format (plan.weights: the file's own, or a conversion the
  // plan allowed), builds the model in the plan's formats and maths, and starts the scheduler. An offloaded plan
  // (plan.offload.enabled) runs the routed experts from the offload runtime: RAM, NVMe and a VRAM cache.
  Engine(const std::string& model_dir, const Plan& plan, const EngineOptions& options = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const lx::MetisConfig& config() const;
  const lx::text::Tokenizer* tokenizer() const;  // null when not loaded
  const Plan& plan() const { return plan_; }
  // Where an offloaded plan's experts live, in lines (empty for a resident plan).
  const std::string& placement() const;
  // One line for serve to print at startup: why speculation is off, or what it does while on. Empty when there is nothing
  // to say.
  const std::string& speculation_note() const;
  // What one decode step moves through memory at `context` cache positions per stream, in this engine's formats
  // (model/traffic.h). The roofline's numerator; an offloaded plan's experts are counted where they are not resident.
  lx::DecodeTraffic decode_traffic(int context) const;
  // The routing census behind the measured experts (metis.h): decode steps count their routed experts while it is on.
  void set_expert_census(bool on);

  // Queues a request. Throws std::invalid_argument for a request the engine cannot run (empty
  // prompt, a prompt longer than the context, out-of-range sampling values) and std::runtime_error
  // when the queue is full. Returns the request id.
  uint64_t submit(GenerateRequest request, RequestCallbacks callbacks);

  // Ends a request as Cancelled at the next step. Unknown or finished ids are ignored.
  void cancel(uint64_t id);

  EngineStats stats() const;

  // Seconds on the engine's clock (the timestamps in RequestResult use it).
  double now() const;

 private:
  struct Impl;

  Plan plan_;
  std::unique_ptr<Impl> impl_;
  std::atomic<uint64_t> next_id_{1};
};

// The offload runtime's options for a plan: the dense weights' target and the experts' format, the VRAM and RAM budgets,
// the CPU lane's share, and the LX_OFFLOAD_* knobs. Every offloaded run loads with them, spec-check included.
lx::offload::OffloadLoadOptions offload_options(const Plan& plan);

}  // namespace lx::runtime
