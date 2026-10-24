#include "cuda/tunable.h"

#include <cuda_runtime.h>

#include "cuda/kernels.h"

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

lx::TuneProblem to_problem(const lx_tune_problem& p) {
  lx::TuneProblem t;
  t.M = p.M;
  t.N = p.N;
  t.K = p.K;
  t.experts = p.experts;
  t.weight = p.weight;
  t.math = p.math;
  return t;
}

std::mutex& lock() {
  static std::mutex m;
  return m;
}

int g_override = -1;  // -1: not read from the environment yet

int device_arch() {
  int dev = 0, major = 0, minor = 0;
  cudaGetDevice(&dev);
  cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
  cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor, dev);
  return major * 10 + minor;
}

std::map<std::string, std::vector<std::string>>& registry() {
  static std::map<std::string, std::vector<std::string>> r;
  return r;
}

std::map<std::string, lx::TuneRunner>& runners() {
  static std::map<std::string, lx::TuneRunner> r;
  return r;
}

using Key = std::tuple<std::string, int, int, int>;
std::map<Key, int>& table() {
  static std::map<Key, int> t;
  return t;
}

int bucket(int m) {
  int b = 1;
  while (b < m) b <<= 1;
  return b;
}

}  // namespace

extern "C" {

int lx_arch(void) {
  static thread_local int cached_dev = -1, cached_real = 0;
  int dev = 0;
  cudaGetDevice(&dev);
  if (dev != cached_dev) {
    cached_dev = dev;
    cached_real = device_arch();
  }
  int over;
  {
    std::lock_guard<std::mutex> g(lock());
    if (g_override < 0) {
      const char* env = std::getenv("LX_ARCH_OVERRIDE");
      g_override = env ? std::atoi(env) : 0;
    }
    over = g_override;
  }
  return over > 0 && over < cached_real ? over : cached_real;
}

int lx_set_arch_override(int arch) {
  {
    std::lock_guard<std::mutex> g(lock());
    g_override = arch > 0 ? arch : 0;
  }
  return lx_arch();
}

int lx_tune_get(const char* kernel, int M, int N, int K) {
  std::lock_guard<std::mutex> g(lock());
  auto it = table().find(Key{kernel, bucket(M), N, K});
  return it == table().end() ? -1 : it->second;
}

void lx_tune_set(const char* kernel, int M, int N, int K, int variant) {
  std::lock_guard<std::mutex> g(lock());
  table()[Key{kernel, bucket(M), N, K}] = variant;
}

void lx_tune_clear(void) {
  std::lock_guard<std::mutex> g(lock());
  table().clear();
}

int lx_tune_kernels(const char** names, int capacity) {
  std::lock_guard<std::mutex> g(lock());
  int i = 0;
  for (auto& [name, variants] : registry()) {
    if (i < capacity) names[i] = name.c_str();
    ++i;
  }
  return i;
}

int lx_tune_variants(const char* kernel) {
  std::lock_guard<std::mutex> g(lock());
  auto it = registry().find(kernel);
  return it == registry().end() ? 0 : static_cast<int>(it->second.size());
}

const char* lx_tune_variant_name(const char* kernel, int variant) {
  std::lock_guard<std::mutex> g(lock());
  auto it = registry().find(kernel);
  if (it == registry().end() || variant < 0 || variant >= static_cast<int>(it->second.size())) return nullptr;
  return it->second[variant].c_str();
}

int lx_tune_run(const char* kernel, int variant, const lx_tune_problem* problem, float* out, void* stream) {
  const lx::TuneRunner runner = lx::tunable_runner(kernel);
  if (!runner) return LX_ERR_NOT_SUPPORTED;
  return runner(variant, to_problem(*problem), out, stream);
}

int lx_tune_runnable(const char* kernel) { return lx::tunable_runner(kernel) ? 1 : 0; }

int lx_tune_key(const char* kernel, const lx_tune_problem* problem, char* out, int capacity) {
  const std::string key = lx::tune_key(kernel, to_problem(*problem));
  if (capacity > 0) {
    const size_t n = std::min(key.size(), static_cast<size_t>(capacity - 1));
    std::memcpy(out, key.data(), n);
    out[n] = '\0';
  }
  return static_cast<int>(key.size());
}

}  // extern "C"

namespace lx {

std::string tune_key(const char* kernel, const TuneProblem& p) {
  std::string key = kernel;
  if (p.weight != 0 || p.math != 0) key += "@w" + std::to_string(p.weight) + "m" + std::to_string(p.math);
  if (p.experts != 1) key += "@e" + std::to_string(p.experts);
  return key;
}

TuneRunner tunable_runner(const char* kernel) {
  std::lock_guard<std::mutex> g(lock());
  auto it = runners().find(kernel);
  return it == runners().end() ? nullptr : it->second;
}

TunableRegistration::TunableRegistration(const char* kernel, std::initializer_list<const char*> variants) {
  std::lock_guard<std::mutex> g(lock());
  auto& v = registry()[kernel];
  v.clear();
  for (const char* name : variants) v.emplace_back(name);
}

TunableRegistration::TunableRegistration(const char* kernel, std::initializer_list<const char*> variants,
                                         TuneRunner runner)
    : TunableRegistration(kernel, variants) {
  std::lock_guard<std::mutex> g(lock());
  runners()[kernel] = runner;
}

}  // namespace lx
