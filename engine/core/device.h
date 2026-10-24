// Device memory for the weights. This header is free of CUDA headers; device.cpp does the work.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

#include "core/safetensors.h"

namespace lx {

// Move-only owner of one device block. A zero-byte buffer owns nothing.
class DeviceBuffer {
 public:
  DeviceBuffer() noexcept = default;
  explicit DeviceBuffer(size_t bytes);
  DeviceBuffer(DeviceBuffer&& other) noexcept;
  DeviceBuffer& operator=(DeviceBuffer&& other) noexcept;
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
  ~DeviceBuffer();

  void* data() const noexcept { return ptr_; }
  size_t size() const noexcept { return bytes_; }

 private:
  void* ptr_ = nullptr;
  size_t bytes_ = 0;
};

// A tensor on the device: a view into the Weights' buffer.
struct Tensor {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;
  size_t bytes = 0;
  void* data = nullptr;  // device pointer, 256-byte aligned

  int64_t numel() const noexcept;
};

// Every tensor of a checkpoint in one device allocation, looked up by name.
class Weights {
 public:
  // Uploads all tensors of the set, in file order, each starting on a 256-byte boundary.
  static Weights upload(const SafetensorsSet& set);

  Weights() = default;
  Weights(Weights&&) = default;
  Weights& operator=(Weights&&) = default;
  Weights(const Weights&) = delete;
  Weights& operator=(const Weights&) = delete;
  ~Weights() = default;

  const Tensor* find(const std::string& name) const;
  // Throws unless the tensor exists with exactly this dtype and shape. The message names it.
  const Tensor& get(const std::string& name, DType dtype, std::initializer_list<int64_t> shape) const;

  const std::map<std::string, Tensor>& tensors() const noexcept { return tensors_; }
  size_t device_bytes() const noexcept { return buffer_.size(); }

 private:
  DeviceBuffer buffer_;
  std::map<std::string, Tensor> tensors_;
};

}  // namespace lx
