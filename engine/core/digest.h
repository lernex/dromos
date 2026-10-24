// SHA-256 and SHA-1 (git's blob ids), streaming. Used to verify downloads.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <array>
#include <string>
#include <string_view>

namespace lx {

class Sha256 {
 public:
  Sha256();
  void update(const void* data, size_t size);
  std::array<uint8_t, 32> finish();
  // Lower-case hex of the digest of everything fed so far.
  std::string hex_digest();

 private:
  void block(const uint8_t* p);
  uint32_t h_[8];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t total_ = 0;
};

class Sha1 {
 public:
  Sha1();
  void update(const void* data, size_t size);
  std::array<uint8_t, 20> finish();
  std::string hex_digest();

 private:
  void block(const uint8_t* p);
  uint32_t h_[5];
  uint8_t buf_[64];
  size_t used_ = 0;
  uint64_t total_ = 0;
};

// The SHA-256 of a whole string, lower-case hex.
std::string sha256_hex(std::string_view data);

// A git blob id: SHA-1 of "blob <size>\0" followed by the content. This is what the hub reports as
// the ETag of a file that is not stored with Git LFS.
class GitBlobId {
 public:
  explicit GitBlobId(uint64_t size);
  void update(const void* data, size_t size) { sha_.update(data, size); }
  std::string hex_digest() { return sha_.hex_digest(); }

 private:
  Sha1 sha_;
};

}  // namespace lx
