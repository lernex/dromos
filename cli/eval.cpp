// lernex eval: run reference prompts through the model and write every position's logits, for
// tools/compare.py. With a split point N the first N tokens go in as one prefill and the rest as
// single-token decode steps, which checks that the caches carry a sequence exactly.
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace {

using Clock = std::chrono::steady_clock;

}  // namespace

int cmd_eval(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: lernex eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]\n");
    return 2;
  }
  try {
    const std::string dir = argv[1];
    const int split = argc == 5 ? std::stoi(argv[4]) : 0;
    const std::filesystem::path out_dir = argv[3];
    std::filesystem::create_directories(out_dir);
    nlohmann::json prompts;
    std::ifstream(argv[2]) >> prompts;

    const lx::MetisConfig config = lx::load_config(dir);
    const lx::ModelWeights weights = lx::ModelWeights::load(dir, config, lx::LoadOptions{LX_W_BF16});
    size_t longest = 1;
    for (auto& [name, ids] : prompts.items()) longest = std::max(longest, ids.size());
    lx::MetisModel model(config, weights, lx::ModelFormats{}, 1, 4096, static_cast<int>(longest));

    const int V = config.vocab_size;
    float* d_logits = nullptr;
    const auto logit_bytes = static_cast<int64_t>(sizeof(float) * longest * V);
    if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
      throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
    void* stream = nullptr;
    if (lx_rt_stream_create(&stream) != 0) throw std::runtime_error("creating a device stream");

    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
      const int T = static_cast<int>(ids.size());
      const int first = split > 0 && split < T ? split : T;
      std::vector<float> logits(static_cast<size_t>(T) * V);
      model.reset_slot(0);
      const auto t0 = Clock::now();
      // the prefill
      lx::StepBatch step{{ids.begin(), ids.begin() + first}, {0, first}, {0}};
      model.forward(step, d_logits, true, stream);
      lx_rt_copy(logits.data(), d_logits, static_cast<int64_t>(sizeof(float) * first * V), LX_COPY_D2H, stream);
      lx_rt_stream_sync(stream);
      const double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
      // then one token at a time
      const auto t1 = Clock::now();
      for (int t = first; t < T; ++t) {
        lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
        model.forward(one, d_logits, true, stream);
        lx_rt_copy(logits.data() + static_cast<size_t>(t) * V, d_logits, static_cast<int64_t>(sizeof(float) * V),
                   LX_COPY_D2H, stream);
      }
      lx_rt_stream_sync(stream);
      const double decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
      if (const int err = lx_rt_last_error())
        throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the run");
      std::ofstream(out_dir / (name + ".f32"), std::ios::binary)
          .write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
      std::printf("%-20s %4d tokens: prefill %d in %.1f ms", name.c_str(), T, first, prefill_s * 1e3);
      if (first < T) std::printf(", decode %d in %.1f ms (%.1f tok/s)", T - first, decode_s * 1e3, (T - first) / decode_s);
      std::printf("\n");
    }
    lx_rt_free(d_logits);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
