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
#include <string>
#include <utility>
#include <vector>

#include "core/config.h"
#include "runtime/plan.h"
#include "text/tokenizer.h"

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
};

struct RequestCallbacks {
  std::function<void(const TokenDelta&)> on_token;
  std::function<void(const RequestResult&)> on_finish;  // always called exactly once
};

struct EngineOptions {
  size_t max_queue = 4096;   // waiting requests beyond this are refused
  bool load_tokenizer = true;
};

struct EngineStats {
  uint64_t steps = 0;
  uint64_t decode_steps = 0;     // steps where every entry fed one token
  uint64_t prompt_tokens = 0;    // prompt tokens fed
  uint64_t generated_tokens = 0; // tokens sampled
  int running = 0;
  int waiting = 0;
  int slots = 0;
};

class Engine {
 public:
  // Loads the checkpoint in model_dir (BF16 safetensors) and the tokenizer, and starts the scheduler.
  Engine(const std::string& model_dir, const Plan& plan, const EngineOptions& options = {});
  ~Engine();
  Engine(const Engine&) = delete;
  Engine& operator=(const Engine&) = delete;

  const lx::MetisConfig& config() const;
  const lx::text::Tokenizer* tokenizer() const;  // null when not loaded
  const Plan& plan() const { return plan_; }

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

}  // namespace lx::runtime
