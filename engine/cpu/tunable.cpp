// The kernel variant table (tunable.h) on the CPU. lx_arch and its override live in isa.cpp.
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

#include "cpu/cpu.h"

namespace {

std::mutex& lock() {
  static std::mutex m;
  return m;
}

std::map<std::string, std::vector<std::string>>& registry() {
  static auto* r = new std::map<std::string, std::vector<std::string>>;
  return *r;
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

TunableRegistration::TunableRegistration(const char* kernel, std::initializer_list<const char*> variants) {
  std::lock_guard<std::mutex> g(lock());
  auto& v = registry()[kernel];
  v.clear();
  for (const char* name : variants) v.emplace_back(name);
}

}  // namespace lx
