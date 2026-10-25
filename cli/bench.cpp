// lernex bench MODEL [--streams 1,8,32,64] [--prompt-tokens N] [--gen-tokens N]
//                    [--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x]
//
// The weights run in the format the file stores unless --weights names another, so a speed is always the speed
// of a named format (the first line of the plan says which).
//
// Runs the engine itself (its scheduler, batching and sampler) on synthetic prompts: every stream
// gets the same number of random prompt tokens and generates the same number of tokens, greedy,
// ignoring end tokens so each run is the same work. Reports, per stream count:
//   decode tok/s    tokens generated while every stream is decoding, over that time
//   per stream      decode tok/s divided by the streams
//   TTFT p50        from submit to the first token, the median across streams
//   prefill tok/s   prompt tokens of every stream, over the time until the last one starts decoding
// --sample draws with the model's own recommended settings (temperature 0.6, top-p 0.95, top-k 20)
// instead of greedy, which is what a chat user gets.
#include <algorithm>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/common.h"
#include "runtime/engine.h"

namespace lx::cli {
namespace {

struct BenchArgs {
  std::string model;
  std::vector<int> streams{1, 8, 32, 64};
  int prompt_tokens = 512;
  int gen_tokens = 256;
  bool sample = false;
  bool prefix_cache = true;  // --no-prefix-cache measures the cold path
  FormatFlags format;
};

std::vector<int> parse_list(const std::string& text) {
  std::vector<int> out;
  size_t pos = 0;
  while (pos <= text.size()) {
    const size_t comma = text.find(',', pos);
    const std::string item = text.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
    if (!item.empty()) out.push_back(std::stoi(item));
    if (comma == std::string::npos) break;
    pos = comma + 1;
  }
  return out;
}

struct StreamTiming {
  std::vector<double> token_times;  // engine clock, one per generated token
  runtime::RequestResult result;
};

// Runs `streams` requests at once and returns each one's timings, when all have finished.
std::vector<StreamTiming> run_batch(runtime::Engine& engine, int streams, int prompt_tokens, int gen_tokens,
                                    int vocab, bool sample) {
  std::vector<StreamTiming> timings(streams);
  std::mutex mu;
  std::condition_variable cv;
  int finished = 0;
  for (int i = 0; i < streams; ++i) {
    runtime::GenerateRequest req;
    req.prompt.resize(prompt_tokens);
    uint64_t x = 0x9E3779B97F4A7C15ull * static_cast<uint64_t>(i + 1);
    for (int& id : req.prompt) {
      x ^= x << 13, x ^= x >> 7, x ^= x << 17;
      id = 1000 + static_cast<int>(x % static_cast<uint64_t>(vocab - 2000));
    }
    req.max_tokens = gen_tokens;
    req.ignore_eos = true;
    req.thinking = false;
    req.sampling.temperature = sample ? 0.6f : 0.0f;
    req.sampling.top_p = sample ? 0.95f : 1.0f;
    req.sampling.top_k = sample ? 20 : 0;
    req.sampling.seed = static_cast<uint64_t>(i) + 1;
    runtime::RequestCallbacks cb;
    StreamTiming* t = &timings[i];
    cb.on_token = [&engine, t](const runtime::TokenDelta& d) {
      if (d.token >= 0) t->token_times.push_back(engine.now());
    };
    cb.on_finish = [&, t](const runtime::RequestResult& r) {
      std::lock_guard<std::mutex> lock(mu);
      t->result = r;
      finished += 1;
      cv.notify_all();
    };
    engine.submit(std::move(req), std::move(cb));
  }
  std::unique_lock<std::mutex> lock(mu);
  cv.wait(lock, [&] { return finished == streams; });
  for (const auto& t : timings)
    if (t.result.reason == runtime::FinishReason::Error) throw std::runtime_error("a bench stream failed: " + t.result.error);
  return timings;
}

struct Row {
  int streams;
  double decode_tps, per_stream_tps, ttft_p50, prefill_tps, ttft_p90;
  int generated_each;
};

Row measure(const std::vector<StreamTiming>& timings, int prompt_tokens, double submitted_at) {
  const int n = static_cast<int>(timings.size());
  double all_decoding = 0.0, end = 0.0;
  std::vector<double> ttft;
  for (const auto& t : timings) {
    all_decoding = std::max(all_decoding, t.token_times.empty() ? 0.0 : t.token_times.front());
    end = std::max(end, t.result.finished_at);
    ttft.push_back(t.result.time_to_first_token);
  }
  uint64_t tokens = 0;
  for (const auto& t : timings)
    for (double at : t.token_times)
      if (at > all_decoding) ++tokens;
  const double decode_window = std::max(end - all_decoding, 1e-9);
  Row row{};
  row.streams = n;
  row.decode_tps = static_cast<double>(tokens) / decode_window;
  row.per_stream_tps = row.decode_tps / n;
  std::sort(ttft.begin(), ttft.end());
  row.ttft_p50 = ttft[ttft.size() / 2];
  row.ttft_p90 = ttft[std::min(ttft.size() - 1, ttft.size() * 9 / 10)];
  const double prefill_window = std::max(all_decoding - submitted_at, 1e-9);
  row.prefill_tps = static_cast<double>(n) * prompt_tokens / prefill_window;
  row.generated_each = static_cast<int>(timings.front().token_times.size());
  return row;
}

}  // namespace

}  // namespace lx::cli

int cmd_bench(int argc, char** argv) {
  using namespace lx::cli;
  using lx::runtime::Engine;
  using lx::runtime::EngineOptions;
  using lx::runtime::Plan;
  BenchArgs args;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
      return argv[++i];
    };
    if (parse_format_flag(argc, argv, i, args.format)) continue;
    if (a == "--streams") args.streams = parse_list(value());
    else if (a == "--prompt-tokens") args.prompt_tokens = std::stoi(value());
    else if (a == "--gen-tokens") args.gen_tokens = std::stoi(value());
    else if (a == "--sample") args.sample = true;
    else if (a == "--no-prefix-cache") args.prefix_cache = false;
    else if (args.model.empty() && a[0] != '-') args.model = a;
    else throw std::runtime_error("unknown option " + a);
  }
  if (args.model.empty()) {
    std::fprintf(stderr, "usage: lernex bench MODEL [--streams 1,8,32,64] [--prompt-tokens N] [--gen-tokens N] %s\n",
                 kFormatUsage);
    return 2;
  }
  if (args.streams.empty() || args.prompt_tokens < 1 || args.gen_tokens < 2)
    throw std::runtime_error("streams, prompt tokens and generated tokens must be positive (at least 2 generated)");
  const int max_streams = *std::max_element(args.streams.begin(), args.streams.end());

  const Model model = find_model(args.model, false);
  EngineOptions options;
  options.load_tokenizer = false;
  options.prefix_cache = args.prefix_cache;
  lx::cli::StartedEngine started = lx::cli::start_engine(model, max_streams, args.prompt_tokens + args.gen_tokens + 16,
                                                         args.format.weights, options, 0.92, args.format.bf16x,
                                                         args.format.fast_math);
  const Plan& plan = started.plan;
  Engine& engine = *started.engine;

  std::printf("%s  (%s)\n", args.model.c_str(), model.dir.c_str());
  for (const std::string& note : plan.notes) std::printf("  %s\n", note.c_str());
  if (!engine.placement().empty()) std::printf("%s", engine.placement().c_str());
  std::printf("  %d slots, %d-token context, %d tokens per step\n", plan.max_slots, plan.kv_capacity,
              plan.max_step_tokens);
  std::printf("  %d prompt tokens, %d generated tokens per stream, %s\n\n", args.prompt_tokens, args.gen_tokens,
              args.sample ? "sampled (temperature 0.6, top-p 0.95, top-k 20)" : "greedy");

  std::printf("%8s %12s %12s %10s %10s %13s\n", "streams", "decode tok/s", "per stream", "TTFT p50", "TTFT p90",
              "prefill tok/s");
  for (int streams : args.streams) {
    run_batch(engine, streams, std::min(args.prompt_tokens, 64), 8, engine.config().vocab_size, args.sample);  // warm up
    const double submitted = engine.now();
    const auto timings =
        run_batch(engine, streams, args.prompt_tokens, args.gen_tokens, engine.config().vocab_size, args.sample);
    const Row row = measure(timings, args.prompt_tokens, submitted);
    std::printf("%8d %12.1f %12.1f %10s %10s %13.0f\n", row.streams, row.decode_tps, row.per_stream_tps,
                human_seconds(row.ttft_p50).c_str(), human_seconds(row.ttft_p90).c_str(), row.prefill_tps);
    std::fflush(stdout);
  }
  return 0;
}
