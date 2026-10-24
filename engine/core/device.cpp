#include "core/device.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <utility>

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
