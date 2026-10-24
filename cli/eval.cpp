// lernex eval: run reference prompts through the model and write every position's logits, for
// tools/compare.py. With a split point N the first N tokens go in as one prefill and the rest as
// single-token decode steps, which checks that the caches carry a sequence exactly.
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace {

using Clock = std::chrono::steady_clock;

lx::WeightLookup lookup_for(const lx::Weights& weights) {
  return [&weights](const std::string& name) {
    lx::WeightView view;
    const lx::Tensor* t = weights.find(name);
    if (!t) return view;
    view.data = t->data;
    view.dtype = t->dtype == lx::DType::F32 ? LX_F32 : t->dtype == lx::DType::BF16 ? LX_BF16 : -1;
    view.shape = t->shape;
    return view;
  };
}

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
    const lx::SafetensorsSet set = lx::SafetensorsSet::open(dir);
    const lx::Weights weights = lx::Weights::upload(set);
    size_t longest = 1;
    for (auto& [name, ids] : prompts.items()) longest = std::max(longest, ids.size());
    lx::MetisModel model(config, lookup_for(weights), 1, 4096, static_cast<int>(longest));

    const int V = config.vocab_size;
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, sizeof(float) * longest * V) != cudaSuccess) throw std::runtime_error("cudaMalloc");
    cudaStream_t stream;
    cudaStreamCreate(&stream);

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
      cudaMemcpyAsync(logits.data(), d_logits, sizeof(float) * first * V, cudaMemcpyDeviceToHost, stream);
      cudaStreamSynchronize(stream);
      const double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
      // then one token at a time
      const auto t1 = Clock::now();
      for (int t = first; t < T; ++t) {
        lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
        model.forward(one, d_logits, true, stream);
        cudaMemcpyAsync(logits.data() + static_cast<size_t>(t) * V, d_logits, sizeof(float) * V,
                        cudaMemcpyDeviceToHost, stream);
      }
      cudaStreamSynchronize(stream);
      const double decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
      if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error during the run");
      std::ofstream(out_dir / (name + ".f32"), std::ios::binary)
          .write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
      std::printf("%-20s %4d tokens: prefill %d in %.1f ms", name.c_str(), T, first, prefill_s * 1e3);
      if (first < T) std::printf(", decode %d in %.1f ms (%.1f tok/s)", T - first, decode_s * 1e3, (T - first) / decode_s);
      std::printf("\n");
    }
    cudaFree(d_logits);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
