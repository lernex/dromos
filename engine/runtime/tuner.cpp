#include "runtime/tuner.h"

#include "cuda/kernels.h"

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

#include "core/install.h"
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

// Throws when a backend call failed, naming what was being done.
void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + ": " + lx_rt_error_string(status));
}

// Copies `count` floats from device memory to the host, then waits for the copy to finish.
void download(float* host, const float* device, size_t count) {
  const std::string what = "copy of " + std::to_string(count * sizeof(float)) + " bytes to the host";
  check(lx_rt_copy(host, device, static_cast<int64_t>(count * sizeof(float)), LX_COPY_D2H, nullptr), what);
  check(lx_rt_stream_sync(nullptr), what);
}

// The backend's form of a problem: the same fields, read by the backend's tuner.
lx_tune_problem to_abi(const TuneProblem& p) {
  lx_tune_problem cp;
  cp.M = p.M;
  cp.N = p.N;
  cp.K = p.K;
  cp.experts = p.experts;
  cp.weight = p.weight;
  cp.math = p.math;
  return cp;
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
    if (data) lx_rt_free(data);
    check(lx_rt_alloc(reinterpret_cast<void**>(&data), static_cast<int64_t>(n * sizeof(float))),
          "device allocation of " + std::to_string(n * sizeof(float)) + " bytes");
    capacity = n;
  }
  ~Buffer() {
    if (data) lx_rt_free(data);
  }
};

// Runs one variant `count` times on the stream; returns 0 or the first error.
int run_many(const char* kernel, int variant, const TuneProblem& p, float* out, void* stream, int count) {
  const lx_tune_problem cp = to_abi(p);
  for (int i = 0; i < count; ++i) {
    const int status = lx_tune_run(kernel, variant, &cp, out, stream);
    if (status != 0) return status;
  }
  return 0;
}

// Median time per launch, in microseconds, or a negative value when a launch fails.
double time_variant(const char* kernel, int variant, const TuneProblem& p, float* out, void* stream,
                    const TuneOptions& o) {
  if (run_many(kernel, variant, p, out, stream, o.warmup) != 0) return -1;
  void* start = nullptr;
  void* stop = nullptr;
  check(lx_rt_event_create_timed(&start), "timing event creation");
  check(lx_rt_event_create_timed(&stop), "timing event creation");
  double result = -1;
  do {
    // one launch to size the samples, so each lasts about sample_ms
    lx_rt_event_record(start, stream);
    if (run_many(kernel, variant, p, out, stream, 1) != 0) break;
    lx_rt_event_record(stop, stream);
    if (lx_rt_event_sync(stop) != 0) break;
    float first_ms = 0;
    lx_rt_event_elapsed(start, stop, &first_ms);
    const int count = std::clamp(static_cast<int>(std::ceil(o.sample_ms / std::max(first_ms, 1e-3f))), 1, 100000);
    std::vector<double> samples;
    bool ok = true;
    for (int r = 0; r < o.repeats && ok; ++r) {
      lx_rt_event_record(start, stream);
      ok = run_many(kernel, variant, p, out, stream, count) == 0;
      lx_rt_event_record(stop, stream);
      ok = ok && lx_rt_event_sync(stop) == 0;
      float ms = 0;
      lx_rt_event_elapsed(start, stop, &ms);
      samples.push_back(static_cast<double>(ms) * 1e3 / count);
    }
    if (!ok || samples.empty()) break;
    std::sort(samples.begin(), samples.end());
    result = samples[samples.size() / 2];
  } while (false);
  lx_rt_event_destroy(start);
  lx_rt_event_destroy(stop);
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
  return std::string(home ? home : ".") + "/.cache/dromos/tune";
}

// The table key of a kernel's problem, as the backend writes it (lx_tune_key), so host and backend agree.
std::string tune_key(const char* kernel, const TuneProblem& p) {
  const lx_tune_problem cp = to_abi(p);
  char key[256];
  lx_tune_key(kernel, &cp, key, static_cast<int>(sizeof(key)));
  return key;
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
  void* stream = nullptr;
  check(lx_rt_stream_create(&stream), "stream creation");
  Buffer out, reference;

  const char* names[256];
  const int total = lx_tune_kernels(names, 256);
  for (int k = 0; k < total && k < 256; ++k) {
    const char* kernel = names[k];
    if (!lx_tune_runnable(kernel)) continue;
    ++report.kernels;
    const int variants = lx_tune_variants(kernel);
    for (const TuneProblem& p : problems) {
      const lx_tune_problem cp = to_abi(p);
      const size_t elems = static_cast<size_t>(p.M) * p.N;
      out.ensure(elems);
      reference.ensure(elems);
      // variant 0 is the reference every other variant must agree with
      const int ref_status = lx_tune_run(kernel, 0, &cp, reference.data, stream);
      if (ref_status == LX_ERR_NOT_SUPPORTED) continue;  // this kernel does not run this problem
      ++report.problems;
      if (ref_status != 0 || lx_rt_stream_sync(stream) != 0) {
        ++report.rejected;
        continue;
      }
      std::vector<float> want(elems);
      download(want.data(), reference.data, elems);
      const double ref_us = time_variant(kernel, 0, p, reference.data, stream, options);
      if (ref_us < 0) {
        ++report.rejected;
        continue;
      }
      int best = 0;
      double best_us = ref_us;
      for (int v = 1; v < variants; ++v) {
        const int status = lx_tune_run(kernel, v, &cp, out.data, stream);
        if (status == LX_ERR_NOT_SUPPORTED) {
          ++report.skipped;
          continue;
        }
        if (status != 0 || lx_rt_stream_sync(stream) != 0) {
          ++report.rejected;
          continue;
        }
        std::vector<float> got(elems);
        download(got.data(), out.data, elems);
        double worst = 0;
        if (!agrees(got, want, worst)) {
          ++report.rejected;
          if (options.log)
            std::printf("  %-28s M=%-5d variant %d rejected: worst error %.3e\n",
                        lx::runtime::tune_key(kernel, p).c_str(), p.M, v, worst);
          continue;
        }
        const double us = time_variant(kernel, v, p, out.data, stream, options);
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
          std::printf("  %-28s M=%-5d %-14s %9.2f us  (reference %9.2f us)\n",
                      lx::runtime::tune_key(kernel, p).c_str(), p.M, lx_tune_variant_name(kernel, v), us, ref_us);
      }
      TuneEntry e;
      e.key = lx::runtime::tune_key(kernel, p);
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
  lx_rt_stream_destroy(stream);
  report.seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  return report;
}

std::string tune_cache_path(const DeviceInfo& device) {
  return gpu_dir() + "/" + device.slug() + "_" + device.driver_text + "_" + kEngineVersion + ".json";
}

std::string shipped_table_path(const DeviceInfo& device) {
  const std::string name = device.slug() + "_" + kEngineVersion + ".json";
  // The install's own tables (<executable>/../share/dromos/tuning) first, then the source tree the build came from.
  const std::string share = lx::install::share_dir();
  if (!share.empty() && std::filesystem::exists(share + "/tuning/" + name)) return share + "/tuning/" + name;
  return std::string(LX_TUNING_DIR) + "/" + name;
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
