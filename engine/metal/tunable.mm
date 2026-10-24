// tunable.h on Metal. The architecture is the Apple GPU family, as 1000 + N for MTLGPUFamilyAppleN
// (1010 for an M5), lowered by LX_ARCH_OVERRIDE or lx_set_arch_override so the paths of older GPUs
// run and are tested on a newer one. Below 1000 every kernel takes its plainest path (the analogue of
// CUDA's pre-cluster paths, so `--lx-arch=80` means the same thing on every backend). The variant
// table is the same as CUDA's.
#include "cuda/tunable.h"

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "metal/backend.h"

namespace {

std::mutex& lock() {
  static std::mutex m;
  return m;
}

int g_override = -1;  // -1: not read from the environment yet

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
  static const int real = 1000 + lx::metal::Device::get().family();
  int over;
  {
    std::lock_guard<std::mutex> g(lock());
    if (g_override < 0) {
      const char* env = std::getenv("LX_ARCH_OVERRIDE");
      g_override = env ? std::atoi(env) : 0;
    }
    over = g_override;
  }
  return over > 0 && over < real ? over : real;
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
