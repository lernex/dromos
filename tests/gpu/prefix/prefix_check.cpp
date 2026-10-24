// The prefix cache, end to end on the model: a request that starts from a cached prefix must produce
// the same greedy tokens as the same request run cold (no cache), token for token.
//
//   lernex_prefix_check MODEL_DIR [N]
//
// Engine A (cache on) runs a prompt, then two requests that extend it: one continues the prompt's own
// reply, one diverges after the prompt. Both start from a snapshot on the prefill grid. Engine B
// (cache off) runs the same two requests cold. Each pair must match exactly.
#include <cstdio>
#include <future>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.h"
#include "runtime/engine.h"
#include "runtime/plan.h"
#include "text/tokenizer.h"

using namespace lx;
using namespace lx::runtime;

namespace {

struct Run {
  std::vector<int32_t> tokens;
  RequestResult result;
};

Run run(Engine& engine, std::vector<int32_t> prompt, int max_tokens) {
  GenerateRequest req;
  req.prompt = std::move(prompt);
  req.max_tokens = max_tokens;
  req.sampling.temperature = 0.0f;
  req.thinking = false;
  req.ignore_eos = true;
  Run out;
  std::promise<RequestResult> done;
  RequestCallbacks cb;
  cb.on_token = [&out](const TokenDelta& d) {
    if (d.token >= 0) out.tokens.push_back(d.token);
  };
  cb.on_finish = [&done](const RequestResult& r) { done.set_value(r); };
  engine.submit(std::move(req), std::move(cb));
  out.result = done.get_future().get();
  return out;
}

int first_difference(const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const size_t n = std::min(a.size(), b.size());
  for (size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return static_cast<int>(i);
  return a.size() == b.size() ? -1 : static_cast<int>(n);
}

void report(const char* label, const std::vector<int32_t>& a, const std::vector<int32_t>& b) {
  const int at = first_difference(a, b);
  std::printf("%-44s %s", label, at < 0 ? "identical" : "DIFFERENT");
  if (at >= 0) std::printf(" (first at token %d)", at);
  std::printf("  [%zu tokens]\n", a.size());
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::fprintf(stderr, "usage: lernex_prefix_check MODEL_DIR [N]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int n = argc > 2 ? std::stoi(argv[2]) : 16;
  const MetisConfig config = load_config(dir);
  const CheckpointInfo checkpoint = inspect_checkpoint(dir);
  PlanRequest request;
  request.sequences = 2;
  request.context = 4096;
  const Plan plan = make_plan(query_device(0), config, checkpoint, request);

  // Prompts built from the tokenizer, so they are real text: a long shared part, then two endings.
  std::string shared;
  for (int i = 0; i < 40; ++i)
    shared += "Photosynthesis turns light, water and carbon dioxide into sugar and oxygen, one step at a time. ";
  std::vector<int32_t> prompt, tail_reuse, tail_other;
  std::vector<int32_t> reuse, other;
  Run first, cached_reuse, cached_other;
  {
    EngineOptions cache_on;
    cache_on.prefix_cache = true;
    Engine a(dir, plan, cache_on);
    const lx::text::Tokenizer* tok = a.tokenizer();
    prompt = tok->encode(shared + "Question one: why do leaves look green?", true);
    tail_reuse = tok->encode(" Question two: and what happens at night?", false);
    tail_other = tok->encode(" A different question entirely about rivers.", false);
    first = run(a, prompt, n);
    reuse = prompt;
    reuse.insert(reuse.end(), first.tokens.begin(), first.tokens.end());  // the reply is in the prompt
    reuse.insert(reuse.end(), tail_reuse.begin(), tail_reuse.end());
    other = prompt;
    other.insert(other.end(), tail_other.begin(), tail_other.end());
    cached_reuse = run(a, reuse, n);
    cached_other = run(a, other, n);
    const EngineStats stats = a.stats();
    std::printf("cache: %llu hits, %llu tokens reused, %llu snapshots\n",
                static_cast<unsigned long long>(stats.prefix_hits),
                static_cast<unsigned long long>(stats.prefix_tokens_reused),
                static_cast<unsigned long long>(stats.prefix_snapshots));
  }

  const int prompt_len = static_cast<int>(prompt.size());
  std::printf("prompt %d tokens; reuse request %zu tokens; other request %zu tokens\n", prompt_len, reuse.size(),
              other.size());

  EngineOptions cache_off;
  cache_off.prefix_cache = false;
  std::vector<int32_t> cold_reuse, cold_other;
  {
    Engine b(dir, plan, cache_off);
    cold_reuse = run(b, reuse, n).tokens;
    cold_other = run(b, other, n).tokens;
  }

  report("reuse (continues the reply): cached vs cold", cached_reuse.tokens, cold_reuse);
  report("other (diverges after the prompt): cached vs cold", cached_other.tokens, cold_other);
  return 0;
}
