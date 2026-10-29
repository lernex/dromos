#include "core/safetensors.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

#include "platform/posix.h"

namespace lx {
namespace {

using json = nlohmann::json;

constexpr size_t kLengthPrefix = 8;  // the header length: a little-endian u64

std::string read_text(const std::filesystem::path& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return text.str();
}

// Adds every tensor of one shard. Each entry is checked against the file bounds and its shape, so
// a truncated download fails here and not as garbage weights later.
void read_shard(const MappedFile& file, const std::string& where, std::map<std::string, TensorInfo>& out) {
  if (file.size() < kLengthPrefix) throw std::runtime_error(where + ": shorter than its header length");
  uint64_t header_len = 0;
  for (size_t i = 0; i < kLengthPrefix; ++i)
    header_len |= uint64_t{std::to_integer<uint8_t>(file.data()[i])} << (8 * i);
  if (header_len > file.size() - kLengthPrefix)
    throw std::runtime_error(where + ": header length runs past the end of the file");

  const auto* header_begin = reinterpret_cast<const char*>(file.data() + kLengthPrefix);
  json header;
  try {
    header = json::parse(header_begin, header_begin + header_len);
  } catch (const json::exception& e) {
    throw std::runtime_error(where + ": header is not valid JSON (" + e.what() + ")");
  }
  if (!header.is_object()) throw std::runtime_error(where + ": header must be a JSON object");

  const std::byte* body = file.data() + kLengthPrefix + header_len;
  const uint64_t body_len = file.size() - kLengthPrefix - header_len;

  for (const auto& [name, entry] : header.items()) {
    if (name == "__metadata__") continue;
    TensorInfo t;
    t.name = name;
    std::vector<uint64_t> offsets;
    try {
      t.dtype = parse_dtype(entry.at("dtype").get<std::string>());
      t.shape = entry.at("shape").get<std::vector<int64_t>>();
      offsets = entry.at("data_offsets").get<std::vector<uint64_t>>();
    } catch (const json::exception&) {
      throw std::runtime_error(where + ": tensor \"" + name + "\" has a malformed header entry");
    }
    for (const int64_t dim : t.shape)
      if (dim < 0) throw std::runtime_error(where + ": tensor \"" + name + "\" has a negative dimension");
    if (offsets.size() != 2 || offsets[0] > offsets[1] || offsets[1] > body_len)
      throw std::runtime_error(where + ": tensor \"" + name + "\" points outside the file");

    t.bytes = static_cast<size_t>(offsets[1] - offsets[0]);
    t.data = body + offsets[0];
    if (static_cast<size_t>(t.numel()) * dtype_bytes(t.dtype) != t.bytes)
      throw std::runtime_error(where + ": tensor \"" + name + "\" has " + std::to_string(t.bytes) +
                               " bytes, which does not match its dtype and shape");
    if (!out.emplace(name, std::move(t)).second)
      throw std::runtime_error(where + ": tensor \"" + name + "\" appears in more than one shard");
  }
}

}  // namespace

size_t dtype_bytes(DType dtype) noexcept {
  switch (dtype) {
    case DType::F64:
    case DType::I64: return 8;
    case DType::F32:
    case DType::I32: return 4;
    case DType::F16:
    case DType::BF16:
    case DType::I16: return 2;
    case DType::F8_E4M3:
    case DType::F8_E5M2:
    case DType::I8:
    case DType::U8:
    case DType::BOOL: return 1;
  }
  return 0;
}

std::string_view dtype_name(DType dtype) noexcept {
  switch (dtype) {
    case DType::F64: return "F64";
    case DType::F32: return "F32";
    case DType::F16: return "F16";
    case DType::BF16: return "BF16";
    case DType::F8_E4M3: return "F8_E4M3";
    case DType::F8_E5M2: return "F8_E5M2";
    case DType::I64: return "I64";
    case DType::I32: return "I32";
    case DType::I16: return "I16";
    case DType::I8: return "I8";
    case DType::U8: return "U8";
    case DType::BOOL: return "BOOL";
  }
  return "?";
}

DType parse_dtype(std::string_view name) {
  for (const DType d : {DType::F64, DType::F32, DType::F16, DType::BF16, DType::F8_E4M3, DType::F8_E5M2,
                        DType::I64, DType::I32, DType::I16, DType::I8, DType::U8, DType::BOOL})
    if (dtype_name(d) == name) return d;
  throw std::runtime_error("unknown safetensors dtype \"" + std::string(name) + "\"");
}

int64_t TensorInfo::numel() const noexcept {
  int64_t n = 1;
  for (const int64_t dim : shape) n *= dim;
  return n;
}

MappedFile::MappedFile(const std::filesystem::path& path) {
#if defined(_WIN32)
  // The UTF-16 path, and the 64-bit size: a checkpoint is larger than the 2 GB plain fstat reports on Windows.
  const int fd = ::_wopen(path.c_str(), _O_RDONLY | _O_BINARY);
  if (fd < 0) throw std::runtime_error("cannot open " + path.string() + ": " + std::strerror(errno));
  struct _stat64 st {};
  if (::_fstat64(fd, &st) != 0) {
#else
  const int fd = ::open(path.c_str(), O_RDONLY);
  if (fd < 0) throw std::runtime_error("cannot open " + path.string() + ": " + std::strerror(errno));
  struct stat st {};
  if (::fstat(fd, &st) != 0) {
#endif
    const int err = errno;
    ::close(fd);
    throw std::runtime_error("cannot stat " + path.string() + ": " + std::strerror(err));
  }
  if (st.st_size <= 0) {
    ::close(fd);
    throw std::runtime_error(path.string() + " is empty");
  }
  void* p = ::mmap(nullptr, static_cast<size_t>(st.st_size), PROT_READ, MAP_PRIVATE, fd, 0);
  const int err = errno;
  ::close(fd);  // the mapping outlives the descriptor
  if (p == MAP_FAILED) throw std::runtime_error("cannot map " + path.string() + ": " + std::strerror(err));
  data_ = static_cast<const std::byte*>(p);
  size_ = static_cast<size_t>(st.st_size);
}

MappedFile::MappedFile(MappedFile&& other) noexcept : data_(other.data_), size_(other.size_) {
  other.data_ = nullptr;
  other.size_ = 0;
}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
    data_ = other.data_;
    size_ = other.size_;
    other.data_ = nullptr;
    other.size_ = 0;
  }
  return *this;
}

MappedFile::~MappedFile() {
  if (data_) ::munmap(const_cast<std::byte*>(data_), size_);
}

SafetensorsSet SafetensorsSet::open(const std::filesystem::path& dir) {
  SafetensorsSet set;
  const std::filesystem::path index_path = dir / "model.safetensors.index.json";
  if (!std::filesystem::exists(index_path)) {
    set.files_.emplace_back(dir / "model.safetensors");
    read_shard(set.files_.back(), (dir / "model.safetensors").string(), set.tensors_);
    return set;
  }

  // A sharded checkpoint: the index maps every tensor name to the shard that holds it.
  json index;
  try {
    index = json::parse(read_text(index_path));
  } catch (const json::exception& e) {
    throw std::runtime_error(index_path.string() + ": not valid JSON (" + e.what() + ")");
  }
  const auto weight_map = index.find("weight_map");
  if (weight_map == index.end() || !weight_map->is_object())
    throw std::runtime_error(index_path.string() + ": missing \"weight_map\" object");

  std::set<std::string> shard_names;
  for (const auto& [tensor, shard] : weight_map->items()) {
    if (!shard.is_string()) throw std::runtime_error(index_path.string() + ": shard for \"" + tensor + "\" is not a file name");
    shard_names.insert(shard.get<std::string>());
  }
  for (const std::string& shard : shard_names) {
    set.files_.emplace_back(dir / shard);
    read_shard(set.files_.back(), (dir / shard).string(), set.tensors_);
  }
  for (const auto& [tensor, shard] : weight_map->items())
    if (!set.tensors_.contains(tensor))
      throw std::runtime_error("index lists \"" + tensor + "\" in " + shard.get<std::string>() +
                               ", which does not contain it");
  return set;
}

const TensorInfo* SafetensorsSet::find(const std::string& name) const {
  const auto it = tensors_.find(name);
  return it == tensors_.end() ? nullptr : &it->second;
}

size_t SafetensorsSet::total_bytes() const noexcept {
  size_t total = 0;
  for (const auto& entry : tensors_) total += entry.second.bytes;
  return total;
}

}  // namespace lx
