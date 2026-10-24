#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cctype>
#include <span>
#include <stdexcept>
#include <utility>

#include "cuda/tunable.h"
#include "runtime/plan.h"

namespace lx {
namespace {

// Tensors start on this boundary so every one can feed a vectorized kernel directly.
constexpr size_t kAlignment = 256;

void check(cudaError_t err, const std::string& what) {
  if (err != cudaSuccess) throw std::runtime_error(what + ": " + cudaGetErrorString(err));
}

size_t align_up(size_t n) { return (n + kAlignment - 1) / kAlignment * kAlignment; }

std::string format_shape(std::span<const int64_t> shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

}  // namespace

DeviceBuffer::DeviceBuffer(size_t bytes) {
  if (bytes == 0) return;
  void* p = nullptr;
  check(cudaMalloc(&p, bytes), "cudaMalloc of " + std::to_string(bytes) + " bytes");
  ptr_ = p;
  bytes_ = bytes;
}

DeviceBuffer::DeviceBuffer(DeviceBuffer&& other) noexcept
    : ptr_(std::exchange(other.ptr_, nullptr)), bytes_(std::exchange(other.bytes_, 0)) {}

DeviceBuffer& DeviceBuffer::operator=(DeviceBuffer&& other) noexcept {
  if (this != &other) {
    if (ptr_) cudaFree(ptr_);
    ptr_ = std::exchange(other.ptr_, nullptr);
    bytes_ = std::exchange(other.bytes_, 0);
  }
  return *this;
}

DeviceBuffer::~DeviceBuffer() {
  if (ptr_) cudaFree(ptr_);
}

int64_t Tensor::numel() const noexcept {
  int64_t n = 1;
  for (const int64_t dim : shape) n *= dim;
  return n;
}

Weights Weights::upload(const SafetensorsSet& set) {
  // File order keeps the reads from the disk sequential.
  std::vector<const TensorInfo*> order;
  order.reserve(set.tensors().size());
  for (const auto& entry : set.tensors()) order.push_back(&entry.second);
  std::sort(order.begin(), order.end(),
            [](const TensorInfo* a, const TensorInfo* b) { return a->data < b->data; });

  std::vector<size_t> offsets;
  offsets.reserve(order.size());
  size_t total = 0;
  for (const TensorInfo* t : order) {
    offsets.push_back(total);
    total += align_up(t->bytes);
  }

  Weights w;
  w.buffer_ = DeviceBuffer(total);
  auto* base = static_cast<std::byte*>(w.buffer_.data());
  for (size_t i = 0; i < order.size(); ++i) {
    const TensorInfo& t = *order[i];
    void* dst = base + offsets[i];
    if (t.bytes) check(cudaMemcpy(dst, t.data, t.bytes, cudaMemcpyHostToDevice), "uploading \"" + t.name + "\"");
    w.tensors_.emplace(t.name, Tensor{t.name, t.dtype, t.shape, t.bytes, dst});
  }
  return w;
}

const Tensor* Weights::find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

const Tensor& Weights::get(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const {
  const Tensor* t = find(name);
  if (!t) throw std::runtime_error("weights: no tensor named \"" + name + "\"");
  if (t->dtype != dtype)
    throw std::runtime_error("weights: \"" + name + "\" is " + std::string(dtype_name(t->dtype)) +
                             ", expected " + std::string(dtype_name(dtype)));
  const std::span<const int64_t> want(shape.begin(), shape.size());
  if (!std::ranges::equal(t->shape, want))
    throw std::runtime_error("weights: \"" + name + "\" has shape " + format_shape(t->shape) + ", expected " +
                             format_shape(want));
  return *t;
}

}  // namespace lx

namespace lx::runtime {
namespace {

int attribute(cudaDeviceAttr attr, int ordinal) {
  int value = 0;
  if (cudaDeviceGetAttribute(&value, attr, ordinal) != cudaSuccess) throw std::runtime_error("cudaDeviceGetAttribute");
  return value;
}

}  // namespace

std::string DeviceInfo::slug() const {
  std::string out;
  for (char ch : name) {
    if (std::isalnum(static_cast<unsigned char>(ch))) {
      out += ch;
    } else if (!out.empty() && out.back() != '_') {
      out += '_';
    }
  }
  while (!out.empty() && out.back() == '_') out.pop_back();
  out += "_sm" + std::to_string(real_cc());
  if (cc() != real_cc()) out += "_as" + std::to_string(cc());  // an arch override is part of the identity
  return out;
}

DeviceInfo query_device(int ordinal) {
  if (cudaSetDevice(ordinal) != cudaSuccess) throw std::runtime_error("no CUDA device " + std::to_string(ordinal));
  DeviceInfo d;
  d.ordinal = ordinal;
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, ordinal) != cudaSuccess) throw std::runtime_error("cudaGetDeviceProperties");
  d.name = prop.name;

  d.real_cc_major = attribute(cudaDevAttrComputeCapabilityMajor, ordinal);
  d.real_cc_minor = attribute(cudaDevAttrComputeCapabilityMinor, ordinal);
  const int effective = lx_arch();  // the real capability, lowered by any override
  d.cc_major = effective / 10;
  d.cc_minor = effective % 10;

  d.sms = attribute(cudaDevAttrMultiProcessorCount, ordinal);
  d.l2_bytes = static_cast<size_t>(attribute(cudaDevAttrL2CacheSize, ordinal));
  // Peak bandwidth: the memory clock (kHz) with two transfers per clock, times the bus width in bytes.
  const double clock_khz = attribute(cudaDevAttrMemoryClockRate, ordinal);
  const double bus_bits = attribute(cudaDevAttrGlobalMemoryBusWidth, ordinal);
  d.bandwidth_gbs = clock_khz * 1e3 * 2.0 * (bus_bits / 8.0) / 1e9;

  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) throw std::runtime_error("cudaMemGetInfo");
  d.free_bytes = free_bytes;
  d.total_bytes = total_bytes;

  // The tensor-core instructions the engine's kernel variants use, by capability. FP4 block-scaled
  // multiplies are the arch-specific 100a / 120a instructions.
  const int cc = d.cc();
  d.bf16_mma = cc >= 80;
  d.int8_mma = cc >= 80;
  d.fp8_mma = cc >= 89;
  d.fp4_mma = cc == 100 || cc == 120;
  d.clusters = cc >= kClusterArch;
  d.built = kernels_built_for(d.real_cc());

  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  d.driver = driver;
  d.runtime = runtime;
  d.driver_text = std::to_string(driver / 1000) + "." + std::to_string((driver % 1000) / 10);
  d.runtime_text = std::to_string(runtime / 1000) + "." + std::to_string((runtime % 1000) / 10);
  return d;
}

}  // namespace lx::runtime
