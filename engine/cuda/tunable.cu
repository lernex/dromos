#include "cuda/tunable.h"

#include <cuda_runtime.h>

#include <cstdlib>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>

namespace {

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

}  // extern "C"

namespace lx {

TunableRegistration::TunableRegistration(const char* kernel, std::initializer_list<const char*> variants) {
  std::lock_guard<std::mutex> g(lock());
  auto& v = registry()[kernel];
  v.clear();
  for (const char* name : variants) v.emplace_back(name);
}

}  // namespace lx
