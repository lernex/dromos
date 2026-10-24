// lernex prefill: time one sequence's prefill of T tokens (only the last row's logits, as serving asks for
// them), for measuring prefill throughput. Tokens are taken from PROMPTS_JSON, concatenated and repeated
// to T, so the experts see text-like routing. The first run warms up; the median of the rest is reported.
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
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

int cmd_prefill(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: lernex prefill MODEL_DIR PROMPTS_JSON T [RUNS]\n");
    return 2;
  }
  try {
    const std::string dir = argv[1];
    const int T = std::stoi(argv[3]);
    const int runs = argc == 5 ? std::stoi(argv[4]) : 5;
    nlohmann::json prompts;
    std::ifstream(argv[2]) >> prompts;
    std::vector<int32_t> text;
    for (auto& [name, ids] : prompts.items())
      for (int32_t id : ids.get<std::vector<int32_t>>()) text.push_back(id);
    if (text.empty()) throw std::runtime_error("no tokens in the prompts");
    std::vector<int32_t> tokens(T);
    for (int i = 0; i < T; ++i) tokens[i] = text[i % text.size()];

    const lx::MetisConfig config = lx::load_config(dir);
    const lx::SafetensorsSet set = lx::SafetensorsSet::open(dir);
    const lx::Weights weights = lx::Weights::upload(set);
    auto lookup = [&weights](const std::string& name) {
      lx::WeightView view;
      const lx::Tensor* t = weights.find(name);
      if (!t) return view;
      view.data = t->data;
      view.dtype = t->dtype == lx::DType::F32 ? LX_F32 : t->dtype == lx::DType::BF16 ? LX_BF16 : -1;
      view.shape = t->shape;
      return view;
    };
    lx::MetisModel model(config, lookup, 1, std::max(4096, T), T);
    float* d_logits = nullptr;
    if (cudaMalloc(&d_logits, sizeof(float) * config.vocab_size) != cudaSuccess) throw std::runtime_error("cudaMalloc");
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    std::vector<float> ms;
    for (int r = 0; r <= runs; ++r) {
      model.reset_slot(0);
      lx::StepBatch step{tokens, {0, T}, {0}};
      cudaEventRecord(e0, stream);
      model.forward(step, d_logits, false, stream);
      cudaEventRecord(e1, stream);
      cudaEventSynchronize(e1);
      float t = 0;
      cudaEventElapsedTime(&t, e0, e1);
      if (r > 0) ms.push_back(t);
    }
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error during the run");
    std::sort(ms.begin(), ms.end());
    const float med = ms[ms.size() / 2];
    std::printf("prefill %d tokens: %.1f ms median of %d (min %.1f), %.0f tok/s\n", T, med, runs, ms.front(), T / (med / 1e3));
    cudaFree(d_logits);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
