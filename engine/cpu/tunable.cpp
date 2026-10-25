// The kernel variant table (tunable.h) on the CPU. lx_arch and its override live in isa.cpp.
//
// The CPU's kernels have no variants: each product has one order (dot form, kern/isa.cpp), so nothing
// registers here today and the tuner finds nothing to time. The table, the admission rule and the runners
// work as tunable.cu has them, so a kernel that grows alternatives registers them the same way.
#include <algorithm>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "cpu/cpu.h"

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

std::map<std::string, std::vector<std::string>>& registry() {
  static auto* r = new std::map<std::string, std::vector<std::string>>;
  return *r;
}

std::map<std::string, lx::TuneRunner>& runners() {
  static auto* r = new std::map<std::string, lx::TuneRunner>;
  return *r;
}

std::map<std::string, lx::TuneAdmit>& admitters() {
  static auto* a = new std::map<std::string, lx::TuneAdmit>;
  return *a;
}

using Key = std::tuple<std::string, int, int, int>;
std::map<Key, int>& table() {
  static auto* t = new std::map<Key, int>;
  return *t;
}

int bucket(int m) {
  int b = 1;
  while (b < m) b <<= 1;
  return b;
}

}  // namespace

extern "C" {

int lx_tune_get(const char* kernel, int M, int N, int K) {
  std::lock_guard<std::mutex> g(lock());
  auto it = table().find(Key{kernel, bucket(M), N, K});
  return it == table().end() ? -1 : it->second;
}

int lx_tune_set(const char* kernel, int M, int N, int K, int variant) {
  std::lock_guard<std::mutex> g(lock());
  auto reg = registry().find(kernel);
  if (reg != registry().end()) {
    if (variant < 0 || variant >= static_cast<int>(reg->second.size())) return LX_ERR_TUNE_VARIANT;
    auto adm = admitters().find(kernel);
    if (adm != admitters().end() && !adm->second(variant, M, N, K)) return LX_ERR_TUNE_REJECTED;
  }
  table()[Key{kernel, bucket(M), N, K}] = variant;
  return 0;
}

int lx_tune_force(const char* kernel, int M, int N, int K, int variant) {
  std::lock_guard<std::mutex> g(lock());
  auto reg = registry().find(kernel);
  if (reg != registry().end() && (variant < 0 || variant >= static_cast<int>(reg->second.size())))
    return LX_ERR_TUNE_VARIANT;
  table()[Key{kernel, bucket(M), N, K}] = variant;
  return 0;
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
  if (!kernel || !problem || !out) return lx::cpu::fail(lx::cpu::kInvalid);
  const lx::TuneRunner runner = lx::tunable_runner(kernel);
  if (!runner) return LX_ERR_NOT_SUPPORTED;
  return runner(variant, to_problem(*problem), out, stream);
}

int lx_tune_runnable(const char* kernel) { return kernel && lx::tunable_runner(kernel) ? 1 : 0; }

int lx_tune_key(const char* kernel, const lx_tune_problem* problem, char* out, int capacity) {
  if (!kernel || !problem) return 0;
  const std::string key = lx::tune_key(kernel, to_problem(*problem));
  if (out && capacity > 0) {
    const size_t n = std::min(key.size(), static_cast<size_t>(capacity - 1));
    memcpy(out, key.data(), n);
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

TunableRegistration::TunableRegistration(const char* kernel, std::initializer_list<const char*> variants,
                                         TuneAdmit admit)
    : TunableRegistration(kernel, variants) {
  std::lock_guard<std::mutex> g(lock());
  admitters()[kernel] = admit;
}

}  // namespace lx
