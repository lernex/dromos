// Snapshots of a sequence's model state, keyed by the token prefix each one was taken after.
// Metis keeps a recurrent Mamba state, which cannot be cut back to an earlier length, so a prefix
// is reusable only as a whole snapshot taken at exactly that length. The cache keeps snapshots in
// device memory up to a byte budget, spilling the least recently used to pinned host memory up to
// a second budget, and dropping them past that. Used on the engine thread only.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <cstddef>
#include <functional>
#include <list>
#include <vector>

#include "core/device.h"

namespace lx::runtime {

class PrefixCache {
 public:
  PrefixCache(size_t device_bytes, size_t host_bytes, void* stream);
  ~PrefixCache();
  PrefixCache(const PrefixCache&) = delete;
  PrefixCache& operator=(const PrefixCache&) = delete;

  struct Hit {
    int length = 0;             // tokens of the prefix
    const void* data = nullptr; // the snapshot: device memory, or pinned host memory when on_host
    bool on_host = false;
  };

  // The longest cached prefix of `tokens` that is at most `limit` tokens long and a multiple of
  // `multiple` (the prefill grid). Marks it recently used.
  bool find(const std::vector<int32_t>& tokens, int limit, int multiple, Hit& hit);

  // Keeps a snapshot of tokens[0..length), `bytes` long: `fill` writes it into the memory it is
  // given (device or pinned host) and the call returns once the copies are done. A prefix already
  // cached is only marked recently used. Returns false when the snapshot cannot be kept at all, the
  // memory for it included.
  bool store(const std::vector<int32_t>& tokens, int length, size_t bytes,
             const std::function<void(void* dst)>& fill);

  size_t device_used() const { return device_used_; }
  size_t host_used() const { return host_used_; }
  size_t entries() const { return entries_.size(); }

 private:
  struct Entry {
    std::vector<int32_t> tokens;
    uint64_t hash = 0;  // of tokens, FNV-1a over the ids
    size_t bytes = 0;
    DeviceBuffer device;  // when on the device
    void* host = nullptr; // pinned, when spilled
    bool on_host() const { return host != nullptr; }
  };

  void make_room(size_t bytes, bool for_device);
  void spill_or_drop(std::list<Entry>::iterator it);
  void drop(std::list<Entry>::iterator it);
  void free_host(void* p);

  size_t device_budget_, host_budget_;
  size_t device_used_ = 0, host_used_ = 0;
  void* stream_;
  std::list<Entry> entries_;  // most recently used first
};

// The FNV-1a hash of a token prefix, one value per length: out[k] hashes tokens[0..k).
std::vector<uint64_t> prefix_hashes(const std::vector<int32_t>& tokens, int limit);

// Whether a prompt of `prompt` tokens is snapshotted at grid point `length`. Each snapshot holds the whole prefix, so
// one at every grid point would grow with the square of the prompt's length: a 20K-token prompt made 40 of them. A
// prompt keeps its last grid point, where a follow-up turn starts, and the grid points at a power of two times the
// grid (512, 1024, 2048, ...), where another prompt that shares only its start picks up at most twice as early as it
// could have.
inline bool snapshot_wanted(int length, int prompt, int grid) {
  if (grid <= 0 || length <= 0 || length % grid != 0) return false;
  if (length + grid > prompt) return true;  // the prompt's last grid point
  const int k = length / grid;
  return (k & (k - 1)) == 0;
}

}  // namespace lx::runtime
