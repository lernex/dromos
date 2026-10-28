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
//
// The roofline columns (cli/roofline.h): the tokens per second the card's memory bandwidth allows for the streams at the
// run's mean context (prompt plus half the generation), at the spec and at the measured read rate, and the run's share
// of each. The bandwidth is probed before the engine takes its memory, and the traffic is the engine's own
// (Engine::decode_traffic), checked against the config's count for this format.
#include <algorithm>
#include <condition_variable>
#include <cmath>
#include <cstdio>
#include <mutex>
#include <stdexcept>
#include <string>
#include <sys/wait.h>
#include <vector>

#include "cli/common.h"
#include "cli/roofline.h"
#include "core/install.h"
#include "model/traffic.h"
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
  bool one_count = false;    // a child run: one stream count, the plan and its row (no header); see cmd_bench
  FormatFlags format;
};

// POSIX shell quoting of one argument.
std::string shell_quote(const std::string& text) {
  std::string out = "'";
  for (const char c : text) {
    if (c == '\'') out += "'\\''";
    else out += c;
  }
  return out + "'";
}

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
    else if (a == "--one-count") args.one_count = true;
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

  const Model model = find_model(args.model, false);
  EngineOptions options;
  options.load_tokenizer = false;
  options.prefix_cache = args.prefix_cache;
  // A child run that refuses its count exits with this status: the plan holds fewer sequences than the count.
  constexpr int kRefusedExit = 3;

  if (!args.one_count) {
    // Each stream count is a run of its own process, so its plan is the plan a standalone run at that count gets: the
    // engines an earlier count loaded leave device memory in use, which the planner would count as taken. The child
    // prints its plan, its roofline and its row; this process prints the header once and then each child's output.
    std::printf("%s  (%s)\n", args.model.c_str(), model.dir.c_str());
    std::printf("  %d prompt tokens, %d generated tokens per stream, %s\n", args.prompt_tokens, args.gen_tokens,
                args.sample ? "sampled (temperature 0.6, top-p 0.95, top-k 20)" : "greedy");
    std::printf("  each stream count loads its own engine, so it runs on the plan a standalone run at that count gets\n");
    std::printf("%8s %12s %12s %10s %10s %13s %15s %9s %9s\n", "streams", "decode tok/s", "per stream", "TTFT p50",
                "TTFT p90", "prefill tok/s", "roofline tok/s", "% of spec", "% of peak");
    std::fflush(stdout);
    bool refused = false;
    for (int streams : args.streams) {
      std::string command = shell_quote(lx::install::executable_path());
      for (int i = 0; i < argc; ++i) {  // argv[0] is the subcommand, "bench": the child is run as this one
        const std::string a = argv[i];
        if (i > 0 && a == "--streams") {
          ++i;  // the list is replaced by this count
          continue;
        }
        command += " " + shell_quote(a);
      }
      command += " --streams " + std::to_string(streams) + " --one-count 2>&1";
      FILE* child = popen(command.c_str(), "r");
      if (!child) throw std::runtime_error("bench: cannot start a run at " + std::to_string(streams) + " streams");
      char line[4096];
      while (std::fgets(line, sizeof line, child)) std::fputs(line, stdout);
      const int status = pclose(child);
      if (WIFEXITED(status) && WEXITSTATUS(status) == kRefusedExit) refused = true;
      else if (status != 0) throw std::runtime_error("bench: the run at " + std::to_string(streams) + " streams failed");
      std::fflush(stdout);
    }
    return refused ? 1 : 0;
  }

  // One stream count in this process: the probe first (its 1 GiB buffer is returned before the engine plans its memory).
  const int streams = args.streams.front();
  const int plan_context = args.prompt_tokens + args.gen_tokens + 16;
  const lx::runtime::DeviceInfo device = lx::runtime::query_device(0);
  const lx::cli::Bandwidth bw = lx::cli::measure_bandwidth(device.bandwidth_gbs);
  const lx::cli::ComputePeak peak = device.cpu ? lx::cli::ComputePeak{} : lx::cli::compute_peak(device.cc());
  const double tflops = peak.known ? peak.bf16_tflops : 0.0;
  lx::cli::StartedEngine started = lx::cli::start_engine(model, streams, plan_context, args.format.weights, options, 0.92,
                                                         args.format.bf16x, args.format.fast_math);
  const Plan& plan = started.plan;
  Engine& engine = *started.engine;
  if (streams > plan.max_slots) {
    // The queue would take the rest, and decode and prefill would then time the queue: refuse rather than measure it.
    std::printf("%8d  refused: this plan holds %d sequences at this context, and the rest would wait in the queue\n",
                streams, plan.max_slots);
    std::fflush(stdout);
    return kRefusedExit;
  }

  const int context = args.prompt_tokens + args.gen_tokens / 2;  // the mean context a stream reads over the run
  const lx::DecodeTraffic traffic = engine.decode_traffic(context);
  // The roofline needs the whole model on the device: an offloaded plan's experts come over PCIe, which is another bound.
  const bool roofline_ok = !plan.offload.enabled && bw.measured_gbs > 0;

  std::printf("\n%d streams: %d slots, %d-token context, %d tokens per step; %s\n", streams, plan.max_slots, plan.kv_capacity,
              plan.max_step_tokens, plan.bf16x_reason.c_str());
  for (const std::string& note : plan.notes) std::printf("  %s\n", note.c_str());
  if (!engine.placement().empty()) std::printf("%s", engine.placement().c_str());

  // The traffic the engine counts, against the config's count for the same format: they must agree (traffic.h).
  const lx::DecodeTraffic from_config = lx::config_decode_traffic(engine.config(), plan.weights, plan.bf16x_decode, context, &plan.bf16x_dropped);
  bool agree = true;
  for (int n : {1, 8, 32}) {
    const double a = traffic.step_bytes(n), b = from_config.step_bytes(n);
    agree = agree && std::fabs(a - b) <= 1e-9 * std::max(a, b);
  }
  agree = agree && std::fabs(traffic.macs_per_token() - from_config.macs_per_token()) <= 1e-9 * traffic.macs_per_token();
  if (roofline_ok) {
    std::printf("  roofline  %.0f GB/s spec, %.0f GB/s measured%s, %s, context %d (mean over the run)\n", bw.spec_gbs,
                bw.measured_gbs, peak.known ? "" : ", no tensor peak for this device",
                peak.known ? (std::to_string(static_cast<int>(tflops)) + " TFLOPS BF16 tensor").c_str() : "no compute bound",
                context);
    std::printf("  traffic   %.1f MB per token at one stream: %s\n", traffic.bytes_per_token() / 1e6,
                agree ? "the model's count and the config's agree" : "MISMATCH: the model and the config count differently");
  } else {
    std::printf("  roofline  not shown: %s\n", plan.offload.enabled ? "the routed experts are offloaded"
                                                                     : "no bandwidth measured");
  }

  run_batch(engine, streams, std::min(args.prompt_tokens, 64), 8, engine.config().vocab_size, args.sample);  // warm up
  const double submitted = engine.now();
  const auto timings = run_batch(engine, streams, args.prompt_tokens, args.gen_tokens, engine.config().vocab_size, args.sample);
  const Row row = measure(timings, args.prompt_tokens, submitted);
  // The ceilings for this many streams: at the spec, and at the measured read rate (the tighter one in practice).
  const double spec_ceiling = roofline_ok && bw.spec_gbs > 0 ? lx::cli::ceiling_tps(traffic, streams, bw.spec_gbs, tflops) : 0;
  const double peak_ceiling = roofline_ok ? lx::cli::ceiling_tps(traffic, streams, bw.measured_gbs, tflops) : 0;
  std::string roof = "-", of_spec = "-", of_peak = "-";
  if (peak_ceiling > 0) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.0f", peak_ceiling);
    roof = buf;
    std::snprintf(buf, sizeof buf, "%.1f%%", 100.0 * row.decode_tps / peak_ceiling);
    of_peak = buf;
  }
  if (spec_ceiling > 0) {
    char buf[32];
    std::snprintf(buf, sizeof buf, "%.1f%%", 100.0 * row.decode_tps / spec_ceiling);
    of_spec = buf;
  }
  std::printf("%8d %12.1f %12.1f %10s %10s %13.0f %15s %9s %9s\n", row.streams, row.decode_tps, row.per_stream_tps,
              human_seconds(row.ttft_p50).c_str(), human_seconds(row.ttft_p90).c_str(), row.prefill_tps, roof.c_str(),
              of_spec.c_str(), of_peak.c_str());
  std::fflush(stdout);
  return 0;
}
