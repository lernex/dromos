// Byte-pair merges over byte-level symbols, as tokenizers' BPE model applies them: the lowest-rank
// pair merges first, ties go to the leftmost position, and a merge is valid only if the pair is
// still adjacent with the same tokens.
#pragma once

#include <array>
#include <cstdint>
#include <mutex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace lx::text {

class BytePair {
 public:
  void set_byte_token(uint8_t byte, int32_t id) { byte_ids_[byte] = id; }
  int32_t byte_token(uint8_t byte) const { return byte_ids_[byte]; }

  // Merge `left right` into `merged`; rank is the merge's position in the merge list.
  void add_merge(int32_t left, int32_t right, uint32_t rank, int32_t merged);

  // Appends the token ids of one pre-token. Its bytes are the byte-level symbols.
  void encode(std::string_view piece, std::vector<int32_t>& out) const;

 private:
  struct Merge {
    uint32_t rank;
    int32_t merged;
  };

  static uint64_t key(int32_t left, int32_t right) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(left)) << 32) | static_cast<uint32_t>(right);
  }
  void merge_word(std::string_view word, std::vector<int32_t>& out) const;

  std::array<int32_t, 256> byte_ids_{};
  std::unordered_map<uint64_t, Merge> merges_;

  // Words that were merged before (the tokenizer cache). Guarded: encode may run on several threads.
  static constexpr size_t kMaxCachedWords = 1 << 16;
  mutable std::mutex cache_mutex_;
  mutable std::unordered_map<std::string, std::vector<int32_t>> cache_;
};

}  // namespace lx::text
