#include "runtime/prefix_cache.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "cuda/kernels.h"

namespace lx::runtime {

namespace {

constexpr uint64_t kFnvOffset = 1469598103934665603ull;
constexpr uint64_t kFnvPrime = 1099511628211ull;

}  // namespace

std::vector<uint64_t> prefix_hashes(const std::vector<int32_t>& tokens, int limit) {
  const int n = std::min(limit, static_cast<int>(tokens.size()));
  std::vector<uint64_t> out(static_cast<size_t>(n) + 1);
  uint64_t h = kFnvOffset;
  out[0] = h;
  for (int i = 0; i < n; ++i) {
    const uint32_t t = static_cast<uint32_t>(tokens[static_cast<size_t>(i)]);
    for (int b = 0; b < 4; ++b) {
      h ^= static_cast<uint64_t>((t >> (8 * b)) & 0xFFu);
      h *= kFnvPrime;
    }
    out[static_cast<size_t>(i) + 1] = h;
  }
  return out;
}

PrefixCache::PrefixCache(size_t device_bytes, size_t host_bytes, void* stream)
    : device_budget_(device_bytes), host_budget_(host_bytes), stream_(stream) {}

PrefixCache::~PrefixCache() {
  for (Entry& e : entries_)
    if (e.host) lx_rt_host_free(e.host);
}

bool PrefixCache::find(const std::vector<int32_t>& tokens, int limit, int multiple, Hit& hit) {
  if (entries_.empty() || tokens.empty()) return false;
  const std::vector<uint64_t> hashes = prefix_hashes(tokens, limit);
  auto best = entries_.end();
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    const size_t len = it->tokens.size();
    if (len == 0 || len >= hashes.size()) continue;  // must fit under the limit and the query
    if (multiple > 1 && len % static_cast<size_t>(multiple) != 0) continue;
    if (best != entries_.end() && len <= best->tokens.size()) continue;
    if (it->hash != hashes[len]) continue;
    if (!std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin())) continue;  // exact check
    best = it;
  }
  if (best == entries_.end()) return false;
  hit.length = static_cast<int>(best->tokens.size());
  hit.on_host = best->on_host();
  hit.data = best->on_host() ? best->host : best->device.data();
  entries_.splice(entries_.begin(), entries_, best);  // most recently used
  return true;
}

bool PrefixCache::store(const std::vector<int32_t>& tokens, int length, size_t bytes,
                        const std::function<void(void* dst)>& fill) {
  if (length <= 0 || static_cast<size_t>(length) > tokens.size()) return false;
  const std::vector<uint64_t> hashes = prefix_hashes(tokens, length);
  const uint64_t h = hashes[static_cast<size_t>(length)];
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    if (it->tokens.size() == static_cast<size_t>(length) && it->hash == h &&
        std::equal(it->tokens.begin(), it->tokens.end(), tokens.begin())) {
      entries_.splice(entries_.begin(), entries_, it);
      return true;
    }
  }

  Entry e;
  e.tokens.assign(tokens.begin(), tokens.begin() + length);
  e.hash = h;
  e.bytes = bytes;
  if (bytes <= device_budget_) {
    make_room(bytes, true);
    e.device = DeviceBuffer(bytes);
    fill(e.device.data());
  } else if (bytes <= host_budget_) {
    make_room(bytes, false);
    if (lx_rt_host_alloc(&e.host, static_cast<int64_t>(bytes)) != 0) return false;
    fill(e.host);
  } else {
    return false;
  }
  // the copies are queued on the stream: wait for them before the entry is ever read or evicted
  if (lx_rt_stream_sync(stream_) != 0) throw std::runtime_error("the prefix snapshot copy failed");
  if (e.host) host_used_ += bytes;
  else device_used_ += bytes;
  entries_.push_front(std::move(e));
  return true;
}

// Frees room for `bytes` in the device budget (or the host one) by spilling or dropping the least
// recently used entries that live there.
void PrefixCache::make_room(size_t bytes, bool for_device) {
  while (true) {
    const size_t used = for_device ? device_used_ : host_used_;
    const size_t budget = for_device ? device_budget_ : host_budget_;
    if (used + bytes <= budget) return;
    auto victim = entries_.end();
    for (auto it = entries_.end(); it != entries_.begin();) {
      --it;
      if (it->on_host() == !for_device) {  // device entries when freeing the device budget
        victim = it;
        break;
      }
    }
    if (victim == entries_.end()) return;  // nothing left to give up; the caller checks the budget
    if (for_device) spill_or_drop(victim);
    else drop(victim);
  }
}

void PrefixCache::spill_or_drop(std::list<Entry>::iterator it) {
  if (host_used_ + it->bytes > host_budget_) {
    drop(it);
    return;
  }
  void* host = nullptr;
  if (lx_rt_host_alloc(&host, static_cast<int64_t>(it->bytes)) != 0) {
    drop(it);
    return;
  }
  if (lx_rt_copy(host, it->device.data(), static_cast<int64_t>(it->bytes), LX_COPY_D2H, stream_) != 0 ||
      lx_rt_stream_sync(stream_) != 0) {
    lx_rt_host_free(host);
    drop(it);
    return;
  }
  device_used_ -= it->bytes;
  host_used_ += it->bytes;
  it->device = DeviceBuffer();
  it->host = host;
}

void PrefixCache::drop(std::list<Entry>::iterator it) {
  if (it->on_host()) host_used_ -= it->bytes;
  else device_used_ -= it->bytes;
  if (it->host) lx_rt_host_free(it->host);
  entries_.erase(it);
}

}  // namespace lx::runtime
