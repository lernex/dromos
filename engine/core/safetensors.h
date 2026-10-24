// safetensors checkpoints, memory-mapped. Host side only: the device upload lives in device.h.
#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <map>
#include <string>
#include <string_view>
#include <vector>

namespace lx {

// The dtypes the safetensors format names. The engine computes in F32 and BF16; the rest are
// accepted so that every checkpoint reads.
enum class DType : uint8_t { F64, F32, F16, BF16, F8_E4M3, F8_E5M2, I64, I32, I16, I8, U8, BOOL };

size_t dtype_bytes(DType dtype) noexcept;
std::string_view dtype_name(DType dtype) noexcept;
// Throws std::runtime_error for a name the format does not define.
DType parse_dtype(std::string_view name);

// One tensor of a checkpoint: a view of the mapped file, valid while the SafetensorsSet lives.
struct TensorInfo {
  std::string name;
  DType dtype = DType::F32;
  std::vector<int64_t> shape;  // rank 0 (a scalar) is an empty shape with one element
  const std::byte* data = nullptr;
  size_t bytes = 0;

  int64_t numel() const noexcept;
};

// A read-only mapping of one file.
class MappedFile {
 public:
  explicit MappedFile(const std::filesystem::path& path);
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;
  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  ~MappedFile();

  const std::byte* data() const noexcept { return data_; }
  size_t size() const noexcept { return size_; }

 private:
  const std::byte* data_ = nullptr;
  size_t size_ = 0;
};

// All tensors of a checkpoint, from model.safetensors or from the shards that
// model.safetensors.index.json names. Names are unique across shards.
class SafetensorsSet {
 public:
  static SafetensorsSet open(const std::filesystem::path& dir);

  SafetensorsSet(SafetensorsSet&&) = default;
  SafetensorsSet& operator=(SafetensorsSet&&) = default;
  SafetensorsSet(const SafetensorsSet&) = delete;
  SafetensorsSet& operator=(const SafetensorsSet&) = delete;
  ~SafetensorsSet() = default;

  const std::map<std::string, TensorInfo>& tensors() const noexcept { return tensors_; }
  const TensorInfo* find(const std::string& name) const;
  size_t total_bytes() const noexcept;
  size_t file_count() const noexcept { return files_.size(); }

 private:
  SafetensorsSet() = default;

  std::vector<MappedFile> files_;  // the TensorInfo data pointers point into these
  std::map<std::string, TensorInfo> tensors_;
};

}  // namespace lx
