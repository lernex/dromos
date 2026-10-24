#include "text/bpe.h"

#include <functional>
#include <queue>
#include <utility>

namespace lx::text {

void BytePair::add_merge(int32_t left, int32_t right, uint32_t rank, int32_t merged) {
  merges_.emplace(key(left, right), Merge{rank, merged});
}

void BytePair::encode(std::string_view piece, std::vector<int32_t>& out) const {
  if (piece.size() == 1) {
    out.push_back(byte_ids_[static_cast<uint8_t>(piece[0])]);
    return;
  }
  {
    std::lock_guard<std::mutex> lock(cache_mutex_);
    const auto hit = cache_.find(std::string(piece));
    if (hit != cache_.end()) {
      out.insert(out.end(), hit->second.begin(), hit->second.end());
      return;
    }
  }
  std::vector<int32_t> ids;
  merge_word(piece, ids);
  out.insert(out.end(), ids.begin(), ids.end());
  std::lock_guard<std::mutex> lock(cache_mutex_);
  if (cache_.size() >= kMaxCachedWords) cache_.clear();
  cache_.emplace(std::string(piece), std::move(ids));
}

void BytePair::merge_word(std::string_view word, std::vector<int32_t>& out) const {
  const auto n = static_cast<int32_t>(word.size());
  std::vector<int32_t> ids(n);
  std::vector<int32_t> prev(n), next(n);
  std::vector<bool> alive(n, true);
  for (int32_t i = 0; i < n; ++i) {
    ids[i] = byte_ids_[static_cast<uint8_t>(word[i])];
    prev[i] = i - 1;
    next[i] = i + 1 < n ? i + 1 : -1;
  }

  // (rank, position): the lowest rank first, and the leftmost position among equal ranks.
  using Item = std::pair<uint32_t, int32_t>;
  std::priority_queue<Item, std::vector<Item>, std::greater<Item>> heap;
  auto push_pair = [&](int32_t pos) {
    const int32_t right = next[pos];
    if (right < 0) return;
    const auto it = merges_.find(key(ids[pos], ids[right]));
    if (it != merges_.end()) heap.emplace(it->second.rank, pos);
  };
  for (int32_t i = 0; i + 1 < n; ++i) push_pair(i);

  while (!heap.empty()) {
    const auto [rank, pos] = heap.top();
    heap.pop();
    if (!alive[pos]) continue;
    const int32_t right = next[pos];
    if (right < 0) continue;
    const auto it = merges_.find(key(ids[pos], ids[right]));
    if (it == merges_.end() || it->second.rank != rank) continue;  // stale entry
    ids[pos] = it->second.merged;
    alive[right] = false;
    next[pos] = next[right];
    if (next[right] >= 0) prev[next[right]] = pos;
    if (prev[pos] >= 0) push_pair(prev[pos]);
    push_pair(pos);
  }

  // Position 0 is never merged into a left neighbour, so the list starts there.
  for (int32_t pos = 0; pos >= 0; pos = next[pos]) out.push_back(ids[pos]);
}

}  // namespace lx::text
