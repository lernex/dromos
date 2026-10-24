// lernex offload: run a model larger than the GPU, and measure what offloading has to work with.
//
//   lernex offload probe [FILE]                       the machine's bandwidths (PCIe, RAM, NVMe, CPU experts)
//   lernex offload eval MODEL PROMPTS OUT [N]         like `lernex eval`, with the experts offloaded
//   lernex offload bench MODEL PROMPTS [N] [REPEAT]   single-stream decode speed: prefill N tokens of each
//                                                     prompt, then decode the rest one token at a time
//
// The plan comes from the environment:
//   LX_VRAM_BUDGET_GB   everything this process may put on the GPU (default: the card)
//   LX_RAM_BUDGET_GB    pinned RAM for experts (default: all of them; the rest stay on NVMe)
//   LX_EXPERTS          bf16 | int8 | fp8 (default: the checkpoint's format; int8 converts from BF16)
//   LX_ROUTING_STATS    per-layer expert frequencies (tools/routing_stats.py); placement uses them
//   LX_OFFLOAD_DENSE_HOST=1   bf16 projections decode reads as int8 stay in RAM (read by prefill)
//   LX_OFFLOAD_DYNAMIC  share of the VRAM slots under the replacement policy (default 0.25)
//   LX_OFFLOAD_CPU_FRACTION, LX_OFFLOAD_CPU_PAIRS   the CPU lane's share of misses, pairs per step
//   LX_OFFLOAD_FILL=0   GPU misses are not written into VRAM
#include <cuda_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.h"
#include "model/metis.h"
#include "runtime/offload.h"
#include "runtime/offload_model.h"

namespace {

using Clock = std::chrono::steady_clock;
namespace off = lx::offload;

double seconds_since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atof(v) : fallback;
}

off::OffloadLoadOptions options_from_env() {
  off::OffloadLoadOptions o;
  if (const char* e = std::getenv("LX_EXPERTS")) {
    const std::string s = e;
    o.experts = s == "bf16" ? LX_W_BF16 : s == "int8" ? LX_W_INT8 : s == "fp8" ? LX_W_FP8 : -1;
    if (o.experts < 0) throw std::runtime_error("LX_EXPERTS must be bf16, int8 or fp8");
  }
  o.vram_budget = off::vram_budget_bytes();
  o.ram_budget = off::ram_budget_bytes();
  if (const char* p = std::getenv("LX_ROUTING_STATS")) o.routing_stats = p;
  o.dense_prefill_on_host = env_double("LX_OFFLOAD_DENSE_HOST", 0) != 0;
  o.w8_decode = std::getenv("LX_NO_W8") == nullptr;
  o.policy.dynamic_fraction = env_double("LX_OFFLOAD_DYNAMIC", 0.25);
  o.policy.cpu_fraction = env_double("LX_OFFLOAD_CPU_FRACTION", 0);
  o.policy.max_cpu_pairs = static_cast<int>(env_double("LX_OFFLOAD_CPU_PAIRS", 0));
  o.policy.fill = env_double("LX_OFFLOAD_FILL", 1) != 0;
  return o;
}

struct Loaded {
  std::unique_ptr<off::OffloadedModel> weights;
  std::unique_ptr<lx::MetisModel> model;
};

Loaded load(const std::string& dir, int max_tokens) {
  const auto t0 = Clock::now();
  const lx::MetisConfig config = lx::load_config(dir);
  Loaded l;
  l.weights = off::OffloadedModel::load(dir, config, options_from_env());
  l.model = std::make_unique<lx::MetisModel>(config, l.weights->lookup(), 1, 4096, max_tokens, &l.weights->experts());
  l.weights->finalize();
  std::printf("loaded in %.1f s\n%s", seconds_since(t0), l.weights->describe().c_str());
  return l;
}

void print_counters(off::ExpertOffload& x, double tokens) {
  const off::OffloadCounters c = x.counters();
  const double pairs = static_cast<double>(c.hits + c.gpu_misses + c.cpu_misses);
  if (pairs > 0)
    std::printf("  decode pairs: %.1f%% VRAM hits, %.1f%% read across PCIe, %.1f%% on the CPU; per token %.1f hits, "
                "%.2f PCIe, %.2f CPU, %.2f fills, %.2f prefetched (%.2f used)\n",
                100 * c.hits / pairs, 100 * c.gpu_misses / pairs, 100 * c.cpu_misses / pairs, c.hits / tokens,
                c.gpu_misses / tokens, c.cpu_misses / tokens, c.fills / tokens, c.prefetches / tokens,
                c.prefetch_hits / tokens);
  if (c.prefill_pairs) std::printf("  prefill: %llu pairs, %llu experts staged\n", (unsigned long long)c.prefill_pairs,
                                   (unsigned long long)c.prefill_staged);
  if (c.cpu_jobs)
    std::printf("  CPU lane: %llu jobs, %.1f us each%s\n", (unsigned long long)c.cpu_jobs, c.cpu_busy_us / c.cpu_jobs,
                c.disk_reads ? (", " + std::to_string(c.disk_reads) + " experts read from NVMe").c_str() : "");
}

int cmd_probe(int argc, char** argv) {
  off::ProbeOptions o;
  if (argc >= 2) o.nvme_file = argv[1];
  if (const char* e = std::getenv("LX_EXPERTS")) {
    const std::string s = e;
    o.expert_type = s == "int8" ? LX_W_INT8 : s == "fp8" ? LX_W_FP8 : LX_W_BF16;
  }
  const off::MachineProfile p = off::probe_machine(o);
  for (const auto& line : p.lines) std::printf("%s\n", line.c_str());
  return 0;
}

int cmd_eval(int argc, char** argv) {
  if (argc != 4 && argc != 5) {
    std::fprintf(stderr, "usage: lernex offload eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int split = argc == 5 ? std::stoi(argv[4]) : 0;
  const std::filesystem::path out_dir = argv[3];
  std::filesystem::create_directories(out_dir);
  nlohmann::json prompts;
  std::ifstream(argv[2]) >> prompts;
  size_t longest = 1;
  for (auto& [name, ids] : prompts.items()) longest = std::max(longest, ids.size());
  Loaded l = load(dir, static_cast<int>(longest));
  const int V = l.model->vocab_size();
  float* d_logits = nullptr;
  if (cudaMalloc(&d_logits, sizeof(float) * longest * V) != cudaSuccess) throw std::runtime_error("cudaMalloc");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  for (auto& [name, ids_json] : prompts.items()) {
    const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
    const int T = static_cast<int>(ids.size());
    const int first = split > 0 && split < T ? split : T;
    std::vector<float> logits(static_cast<size_t>(T) * V);
    l.model->reset_slot(0);
    l.weights->experts().reset_counters();
    const auto t0 = Clock::now();
    lx::StepBatch step{{ids.begin(), ids.begin() + first}, {0, first}, {0}};
    l.model->forward(step, d_logits, true, stream);
    cudaMemcpyAsync(logits.data(), d_logits, sizeof(float) * first * V, cudaMemcpyDeviceToHost, stream);
    cudaStreamSynchronize(stream);
    const double prefill_s = seconds_since(t0);
    const auto t1 = Clock::now();
    for (int t = first; t < T; ++t) {
      lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
      l.model->forward(one, d_logits, true, stream);
      cudaMemcpyAsync(logits.data() + static_cast<size_t>(t) * V, d_logits, sizeof(float) * V, cudaMemcpyDeviceToHost,
                      stream);
    }
    cudaStreamSynchronize(stream);
    const double decode_s = seconds_since(t1);
    if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error during the run");
    if (l.weights->experts().error()) throw std::runtime_error("the CPU lane stopped answering");
    std::ofstream(out_dir / (name + ".f32"), std::ios::binary)
        .write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
    std::printf("%-20s %4d tokens: prefill %d in %.1f ms", name.c_str(), T, first, prefill_s * 1e3);
    if (first < T) std::printf(", decode %d in %.1f ms (%.1f tok/s)", T - first, decode_s * 1e3, (T - first) / decode_s);
    std::printf("\n");
    print_counters(l.weights->experts(), std::max(1, T - first));
  }
  cudaFree(d_logits);
  return 0;
}

int cmd_bench(int argc, char** argv) {
  if (argc < 3 || argc > 5) {
    std::fprintf(stderr, "usage: lernex offload bench MODEL_DIR PROMPTS_JSON [N] [REPEAT]\n");
    return 2;
  }
  const std::string dir = argv[1];
  const int split = argc >= 4 ? std::stoi(argv[3]) : 16;
  const int repeat = argc >= 5 ? std::stoi(argv[4]) : 1;
  nlohmann::json prompts;
  std::ifstream(argv[2]) >> prompts;
  size_t longest = 1;
  for (auto& [name, ids] : prompts.items()) longest = std::max(longest, ids.size());
  Loaded l = load(dir, static_cast<int>(longest));
  float* d_logits = nullptr;
  if (cudaMalloc(&d_logits, sizeof(float) * l.model->vocab_size()) != cudaSuccess) throw std::runtime_error("cudaMalloc");
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  double total_s = 0;
  int total_tokens = 0;
  for (int r = 0; r < repeat; ++r) {
    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
      const int T = static_cast<int>(ids.size());
      if (T <= split + 4) continue;
      l.model->reset_slot(0);
      lx::StepBatch step{{ids.begin(), ids.begin() + split}, {0, split}, {0}};
      l.model->forward(step, d_logits, false, stream);
      // two decode steps warm up (the first runs eagerly, the second captures the graph)
      for (int t = split; t < split + 2; ++t) {
        lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
        l.model->forward(one, d_logits, false, stream);
      }
      cudaStreamSynchronize(stream);
      l.weights->experts().reset_counters();
      const auto t0 = Clock::now();
      for (int t = split + 2; t < T; ++t) {
        lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
        l.model->forward(one, d_logits, false, stream);
      }
      cudaStreamSynchronize(stream);
      const double s = seconds_since(t0);
      const int n = T - split - 2;
      if (cudaGetLastError() != cudaSuccess) throw std::runtime_error("CUDA error during the run");
      std::printf("%-20s decode %d tokens in %.1f ms: %.1f tok/s (%.3f ms/token)\n", name.c_str(), n, s * 1e3, n / s,
                  s * 1e3 / n);
      print_counters(l.weights->experts(), n);
      total_s += s;
      total_tokens += n;
    }
  }
  if (l.weights->experts().error()) throw std::runtime_error("the CPU lane stopped answering");
  std::printf("all: %d tokens, %.1f tok/s\n", total_tokens, total_tokens / total_s);
  size_t free_b = 0, total_b = 0;
  cudaMemGetInfo(&free_b, &total_b);
  std::printf("device memory in use at the end: %.2f GiB of %.2f\n", (total_b - free_b) / 1073741824.0,
              total_b / 1073741824.0);
  cudaFree(d_logits);
  return 0;
}

}  // namespace

int cmd_offload(int argc, char** argv) {
  try {
    if (argc >= 2 && !std::strcmp(argv[1], "probe")) return cmd_probe(argc - 1, argv + 1);
    if (argc >= 2 && !std::strcmp(argv[1], "eval")) return cmd_eval(argc - 1, argv + 1);
    if (argc >= 2 && !std::strcmp(argv[1], "bench")) return cmd_bench(argc - 1, argv + 1);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  std::fprintf(stderr,
               "usage: lernex offload probe [FILE]\n"
               "       lernex offload eval MODEL_DIR PROMPTS_JSON OUT_DIR [N]\n"
               "       lernex offload bench MODEL_DIR PROMPTS_JSON [N] [REPEAT]\n");
  return 2;
}
