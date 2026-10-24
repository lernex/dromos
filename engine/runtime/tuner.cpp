#include "runtime/tuner.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

#ifndef LX_TUNING_DIR
#define LX_TUNING_DIR "tuning"
#endif

namespace lx::runtime {
namespace {

using Clock = std::chrono::steady_clock;

// Relative to the largest reference value: two correct variants differ by accumulation order (and bf16
// rounding of a value near a rounding boundary), never by this much.
constexpr double kAgreement = 2e-2;

void check(cudaError_t err, const char* what) {
  if (err != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
}

// Buckets as lx_tune_get sees them: the smallest power of two that holds M.
std::vector<int> buckets() {
  std::vector<int> out;
  for (int m = 1; m <= 4096; m *= 2) out.push_back(m);
  return out;
}

struct Buffer {
  float* data = nullptr;
  size_t capacity = 0;
  void ensure(size_t n) {
    if (n <= capacity) return;
    if (data) cudaFree(data);
    check(cudaMalloc(reinterpret_cast<void**>(&data), n * sizeof(float)), "cudaMalloc");
    capacity = n;
  }
  ~Buffer() {
    if (data) cudaFree(data);
  }
};

// Runs one variant `count` times on the stream; returns cudaSuccess or the first error.
cudaError_t run_many(TuneRunner runner, int variant, const TuneProblem& p, float* out, cudaStream_t stream, int count) {
  for (int i = 0; i < count; ++i) {
    const cudaError_t err = static_cast<cudaError_t>(runner(variant, p, out, stream));
    if (err != cudaSuccess) return err;
  }
  return cudaSuccess;
}

// Median time per launch, in microseconds, or a negative value when a launch fails.
double time_variant(TuneRunner runner, int variant, const TuneProblem& p, float* out, cudaStream_t stream,
                    const TuneOptions& o) {
  if (run_many(runner, variant, p, out, stream, o.warmup) != cudaSuccess) return -1;
  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");
  double result = -1;
  do {
    // one launch to size the samples, so each lasts about sample_ms
    cudaEventRecord(start, stream);
    if (run_many(runner, variant, p, out, stream, 1) != cudaSuccess) break;
    cudaEventRecord(stop, stream);
    if (cudaEventSynchronize(stop) != cudaSuccess) break;
    float first_ms = 0;
    cudaEventElapsedTime(&first_ms, start, stop);
    const int count = std::clamp(static_cast<int>(std::ceil(o.sample_ms / std::max(first_ms, 1e-3f))), 1, 100000);
    std::vector<double> samples;
    bool ok = true;
    for (int r = 0; r < o.repeats && ok; ++r) {
      cudaEventRecord(start, stream);
      ok = run_many(runner, variant, p, out, stream, count) == cudaSuccess;
      cudaEventRecord(stop, stream);
      ok = ok && cudaEventSynchronize(stop) == cudaSuccess;
      float ms = 0;
      cudaEventElapsedTime(&ms, start, stop);
      samples.push_back(static_cast<double>(ms) * 1e3 / count);
    }
    if (!ok || samples.empty()) break;
    std::sort(samples.begin(), samples.end());
    result = samples[samples.size() / 2];
  } while (false);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return result;
}

// Whether `got` agrees with `want`: finite everywhere, and within kAgreement of the largest value.
bool agrees(const std::vector<float>& got, const std::vector<float>& want, double& worst) {
  double scale = 0, err = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    if (!std::isfinite(got[i])) return false;
    scale = std::max(scale, static_cast<double>(std::fabs(want[i])));
    err = std::max(err, static_cast<double>(std::fabs(got[i] - want[i])));
  }
  worst = err;
  return err <= kAgreement * scale + 1e-30;
}

std::string gpu_dir() {
  const char* home = std::getenv("HOME");
  return std::string(home ? home : ".") + "/.cache/lernex/tune";
}

}  // namespace

std::vector<TuneProblem> model_problems(const MetisConfig& c, const Plan& plan) {
  const int V = c.vocab_size, D = c.hidden_size, S = c.n_streams, Mm = c.memory_dim, R = c.route_feature_dim;
  const int Lat = c.latent_dim, Iexp = c.expert_intermediate_size, Ish = c.shared_expert_intermediate_size;
  const int I = c.mamba_intermediate(), H = c.mamba_heads();
  const int G = c.mamba_n_groups, N = c.mamba_d_state;
  const int proj = 2 * I + 2 * G * N + H;
  const int qkv = (c.num_attention_heads + 2 * c.num_key_value_heads) * c.head_dim;
  const int att = c.num_attention_heads * c.head_dim;

  // N x K of each dense product, in the dense format of the plan
  const std::pair<int, int> dense[] = {{V, D},       {Mm, D},   {D, Mm},  {R, 3 * D + R}, {qkv, D}, {D, att},
                                       {proj, D},    {D, I},    {Lat, D}, {D, Lat},       {2 * Ish, Lat}, {Lat, Ish}};
  std::vector<TuneProblem> out;
  for (const auto& [n, k] : dense) {
    TuneProblem p;
    p.N = n;
    p.K = k;
    p.weight = plan.recipe.dense;
    p.math = plan.prefill_math;
    out.push_back(p);
  }
  // expert stacks: gate_up is [2 * Iexp, Lat], down is [Lat, Iexp], each per expert
  const std::pair<int, int> experts[] = {{2 * Iexp, Lat}, {Lat, Iexp}};
  for (const auto& [n, k] : experts) {
    TuneProblem p;
    p.N = n;
    p.K = k;
    p.experts = c.n_routed_experts;
    p.weight = plan.recipe.experts;
    p.math = plan.expert_math;
    out.push_back(p);
  }
  // the hyper-connection and depth-memory rows: N = D, K = S, no format choice
  TuneProblem row;
  row.N = D;
  row.K = S;
  out.push_back(row);

  std::vector<TuneProblem> all;
  for (const int m : buckets()) {
    for (TuneProblem p : out) {
      p.M = m;
      all.push_back(p);
    }
  }
  return all;
}

TuneReport tune(const std::vector<TuneProblem>& problems, const TuneOptions& options) {
  const auto t0 = Clock::now();
  TuneReport report;
  report.gpu = query_device(0).slug();
  cudaStream_t stream = nullptr;
  check(cudaStreamCreate(&stream), "cudaStreamCreate");
  Buffer out, reference;

  const char* names[256];
  const int total = lx_tune_kernels(names, 256);
  for (int k = 0; k < total && k < 256; ++k) {
    const char* kernel = names[k];
    const TuneRunner runner = tunable_runner(kernel);
    if (!runner) continue;
    ++report.kernels;
    const int variants = lx_tune_variants(kernel);
    for (const TuneProblem& p : problems) {
      const size_t elems = static_cast<size_t>(p.M) * p.N;
      out.ensure(elems);
      reference.ensure(elems);
      // variant 0 is the reference every other variant must agree with
      cudaError_t ref_status = static_cast<cudaError_t>(runner(0, p, reference.data, stream));
      if (ref_status == cudaErrorNotSupported) continue;  // this kernel does not run this problem
      ++report.problems;
      if (ref_status != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
        ++report.rejected;
        continue;
      }
      std::vector<float> want(elems);
      check(cudaMemcpy(want.data(), reference.data, elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
      const double ref_us = time_variant(runner, 0, p, reference.data, stream, options);
      if (ref_us < 0) {
        ++report.rejected;
        continue;
      }
      int best = 0;
      double best_us = ref_us;
      for (int v = 1; v < variants; ++v) {
        const cudaError_t status = static_cast<cudaError_t>(runner(v, p, out.data, stream));
        if (status == cudaErrorNotSupported) {
          ++report.skipped;
          continue;
        }
        if (status != cudaSuccess || cudaStreamSynchronize(stream) != cudaSuccess) {
          ++report.rejected;
          continue;
        }
        std::vector<float> got(elems);
        check(cudaMemcpy(got.data(), out.data, elems * sizeof(float), cudaMemcpyDeviceToHost), "cudaMemcpy");
        double worst = 0;
        if (!agrees(got, want, worst)) {
          ++report.rejected;
          if (options.log)
            std::printf("  %-28s M=%-5d variant %d rejected: worst error %.3e\n", tune_key(kernel, p).c_str(), p.M,
                        v, worst);
          continue;
        }
        const double us = time_variant(runner, v, p, out.data, stream, options);
        if (us < 0) {
          ++report.rejected;
          continue;
        }
        ++report.timed;
        if (us < best_us) {
          best = v;
          best_us = us;
        }
        if (options.log)
          std::printf("  %-28s M=%-5d %-14s %9.2f us  (reference %9.2f us)\n", tune_key(kernel, p).c_str(), p.M,
                      lx_tune_variant_name(kernel, v), us, ref_us);
      }
      TuneEntry e;
      e.key = tune_key(kernel, p);
      e.M = p.M;
      e.N = p.N;
      e.K = p.K;
      e.variant = best;
      const char* vname = lx_tune_variant_name(kernel, best);
      e.variant_name = vname ? vname : "";
      e.us = best_us;
      e.us_reference = ref_us;
      report.entries.push_back(e);
    }
  }
  cudaStreamDestroy(stream);
  report.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return report;
}

std::string tune_cache_path(const DeviceInfo& device) {
  return gpu_dir() + "/" + device.slug() + "_" + device.driver_text + "_" + kEngineVersion + ".json";
}

std::string shipped_table_path(const DeviceInfo& device) {
  return std::string(LX_TUNING_DIR) + "/" + device.slug() + "_" + kEngineVersion + ".json";
}

void save_table(const TuneReport& report, const DeviceInfo& device, const std::string& path) {
  nlohmann::json doc;
  doc["engine"] = kEngineVersion;
  doc["gpu"] = device.slug();
  doc["name"] = device.name;
  doc["cc"] = device.cc();
  doc["driver"] = device.driver_text;
  doc["entries"] = nlohmann::json::array();
  for (const TuneEntry& e : report.entries) {
    nlohmann::json j;
    j["key"] = e.key;
    j["M"] = e.M;
    j["N"] = e.N;
    j["K"] = e.K;
    j["variant"] = e.variant;
    j["name"] = e.variant_name;
    j["us"] = e.us;
    j["us_reference"] = e.us_reference;
    doc["entries"].push_back(j);
  }
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path);
  if (!out) throw std::runtime_error("cannot write " + path);
  out << doc.dump(1) << "\n";
}

int load_table(const std::string& path, const DeviceInfo& device, std::string* why) {
  std::ifstream in(path);
  if (!in) {
    if (why) *why = "no table at " + path;
    return -1;
  }
  nlohmann::json doc;
  try {
    in >> doc;
  } catch (const std::exception& e) {
    if (why) *why = std::string("unreadable table: ") + e.what();
    return -1;
  }
  if (doc.value("engine", std::string()) != kEngineVersion) {
    if (why) *why = "table is for engine " + doc.value("engine", std::string()) + ", not " + kEngineVersion;
    return -1;
  }
  if (doc.value("gpu", std::string()) != device.slug()) {
    if (why) *why = "table is for " + doc.value("gpu", std::string()) + ", not " + device.slug();
    return -1;
  }
  int applied = 0;
  for (const auto& j : doc["entries"]) {
    const std::string key = j["key"].get<std::string>();
    const int M = j["M"].get<int>(), N = j["N"].get<int>(), K = j["K"].get<int>(), variant = j["variant"].get<int>();
    if (lx_tune_set(key.c_str(), M, N, K, variant) != 0) {
      // A table is all or nothing. An entry its kernel refuses was tuned under other rules (an order class
      // another batch size would not share), so the whole table is dropped rather than part of it applied.
      lx_tune_clear();
      if (why)
        *why = "entry " + key + " M=" + std::to_string(M) + " N=" + std::to_string(N) + " K=" + std::to_string(K) +
               " variant " + std::to_string(variant) + " is refused by its kernel";
      return -1;
    }
    ++applied;
  }
  return applied;
}

std::string load_tuning(const DeviceInfo& device) {
  for (const std::string& path : {tune_cache_path(device), shipped_table_path(device)}) {
    if (std::filesystem::exists(path) && load_table(path, device) >= 0) return path;
  }
  return "";
}

}  // namespace lx::runtime
