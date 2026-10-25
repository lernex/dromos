// lernex_gpu_check MODEL_DIR IDS N: runs the engine (scheduler, batching, sampler) on one prompt
// given as comma-separated token ids, greedy, and prints the N tokens it generates, comma-separated.
// tests/gpu/check_greedy.py compares them with the argmax of the reference path (lernex eval).
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/engine.h"
#include "runtime/plan.h"

int main(int argc, char** argv) {
  try {
    if (argc != 4) {
      std::fprintf(stderr, "usage: lernex_gpu_check MODEL_DIR IDS N\n");
      return 2;
    }
    std::vector<int32_t> prompt;
    std::stringstream ids(argv[2]);
    for (std::string item; std::getline(ids, item, ',');) prompt.push_back(std::stoi(item));
    const int n = std::stoi(argv[3]);

    lx::runtime::PlanRequest request;
    request.sequences = 1;
    request.context = static_cast<int>(prompt.size()) + n + 8;
    lx::MetisConfig config = lx::load_config(argv[1]);
    lx::runtime::Plan plan =
        lx::runtime::make_plan(lx::runtime::query_device(0), config, lx::runtime::inspect_checkpoint(argv[1]), request);
    lx::runtime::EngineOptions options;
    options.load_tokenizer = false;
    lx::runtime::Engine engine(argv[1], plan, options);

    std::mutex mu;
    std::condition_variable cv;
    bool done = false;
    std::vector<int32_t> generated;
    lx::runtime::GenerateRequest req;
    req.prompt = prompt;
    req.max_tokens = n;
    req.ignore_eos = true;
    req.thinking = false;
    req.sampling.temperature = 0.0f;
    lx::runtime::RequestCallbacks cb;
    cb.on_token = [&](const lx::runtime::TokenDelta& d) {
      if (d.token >= 0) generated.push_back(d.token);
    };
    cb.on_finish = [&](const lx::runtime::RequestResult& r) {
      std::lock_guard<std::mutex> lock(mu);
      if (r.reason == lx::runtime::FinishReason::Error) std::fprintf(stderr, "error: %s\n", r.error.c_str());
      done = true;
      cv.notify_all();
    };
    engine.submit(req, cb);
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&] { return done; });
    for (size_t i = 0; i < generated.size(); ++i) std::printf(i ? ",%d" : "%d", generated[i]);
    std::printf("\n");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
