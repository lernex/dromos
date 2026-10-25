// lernex prefill: time one sequence's prefill of T tokens (only the last row's logits, as serving asks for
// them), for measuring prefill throughput. Tokens are taken from PROMPTS_JSON, concatenated and repeated
// to T, so the experts see text-like routing. The first run warms up; the median of the rest is reported.
// The weights load in the folder's format or the one --weights names (cli/common.h, load_weights).
#include <algorithm>
#include <cstdio>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/common.h"
#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"

int cmd_prefill(int argc, char** argv) {
  try {
    lx::cli::FormatFlags format;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
      if (!lx::cli::parse_format_flag(argc, argv, i, format)) args.push_back(argv[i]);
    if (args.size() != 3 && args.size() != 4) {
      std::fprintf(stderr, "usage: lernex prefill MODEL_DIR PROMPTS_JSON T [RUNS] %s\n", lx::cli::kFormatUsage);
      return 2;
    }
    const std::string dir = args[0];
    const int T = std::stoi(args[2]);
    const int runs = args.size() == 4 ? std::stoi(args[3]) : 5;
    nlohmann::json prompts;
    std::ifstream(args[1]) >> prompts;
    std::vector<int32_t> text;
    for (auto& [name, ids] : prompts.items())
      for (int32_t id : ids.get<std::vector<int32_t>>()) text.push_back(id);
    if (text.empty()) throw std::runtime_error("no tokens in the prompts");
    std::vector<int32_t> tokens(T);
    for (int i = 0; i < T; ++i) tokens[i] = text[i % text.size()];

    const lx::MetisConfig config = lx::load_config(dir);
    const lx::cli::LoadedWeights loaded = lx::cli::load_weights(dir, config, format);
    std::fprintf(stderr, "%s\n", loaded.description.c_str());
    lx::MetisModel model(config, *loaded.weights, loaded.formats, 1, std::max(4096, T), T);
    float* d_logits = nullptr;
    const int64_t logit_bytes = static_cast<int64_t>(sizeof(float) * config.vocab_size);
    if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
      throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
    void* stream = nullptr;
    if (lx_rt_stream_create(&stream) != 0) throw std::runtime_error("creating a device stream");
    void* e0 = nullptr;
    void* e1 = nullptr;
    if (lx_rt_event_create_timed(&e0) != 0) throw std::runtime_error("creating a timing event");
    if (lx_rt_event_create_timed(&e1) != 0) throw std::runtime_error("creating a timing event");
    std::vector<float> ms;
    for (int r = 0; r <= runs; ++r) {
      model.reset_slot(0);
      lx::StepBatch step{tokens, {0, T}, {0}};
      lx_rt_event_record(e0, stream);
      model.forward(step, d_logits, false, stream);
      lx_rt_event_record(e1, stream);
      lx_rt_event_sync(e1);
      float t = 0;
      lx_rt_event_elapsed(e0, e1, &t);
      if (r > 0) ms.push_back(t);
    }
    if (const int err = lx_rt_last_error())
      throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the run");
    std::sort(ms.begin(), ms.end());
    const float med = ms[ms.size() / 2];
    std::printf("prefill %d tokens: %.1f ms median of %d (min %.1f), %.0f tok/s\n", T, med, runs, ms.front(), T / (med / 1e3));
    lx_rt_free(d_logits);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
