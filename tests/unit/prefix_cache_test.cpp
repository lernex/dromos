// The prefix cache: which grid points a prompt is snapshotted at, and store and find on the backend's memory.
#include <stdexcept>
#include <vector>

#include "core/device.h"
#include "doctest.h"
#include "runtime/prefix_cache.h"

using lx::runtime::PrefixCache;
using lx::runtime::snapshot_wanted;

namespace {

std::vector<int32_t> prompt(int n) {
  std::vector<int32_t> t(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) t[static_cast<size_t>(i)] = (i * 7919) % 151643;
  return t;
}

// Whether this test binary has a backend to allocate from (the unit tests also run where none is loaded).
bool have_backend() {
  try {
    lx::DeviceBuffer probe(64);
    return probe.data() != nullptr;
  } catch (const std::exception&) {
    return false;
  }
}

}  // namespace

TEST_CASE("a prompt is snapshotted at its last grid point and at the grid times a power of two") {
  // a 20K-token prompt: 512, 1024, 2048, 4096, 8192, 16384 and its last grid point, 20480 (40 grid points in all)
  std::vector<int> taken;
  for (int length = 512; length <= 20500; length += 512)
    if (snapshot_wanted(length, 20500, 512)) taken.push_back(length);
  CHECK(taken == std::vector<int>{512, 1024, 2048, 4096, 8192, 16384, 20480});
  // a prompt that ends on the grid is snapshotted at its end
  CHECK(snapshot_wanted(3072, 3072, 512));
  CHECK_FALSE(snapshot_wanted(2560, 3072, 512));
  // only grid points, and none without a grid
  CHECK_FALSE(snapshot_wanted(513, 20500, 512));
  CHECK_FALSE(snapshot_wanted(0, 20500, 512));
  CHECK_FALSE(snapshot_wanted(1024, 20500, 0));
}

TEST_CASE("store keeps a snapshot once, find takes the longest, and one too big for every budget is refused") {
  if (!have_backend()) {
    MESSAGE("no backend to allocate from: skipped");
    return;
  }
  PrefixCache cache(1 << 20, 0, nullptr);
  const std::vector<int32_t> t = prompt(2048);
  const auto fill = [](void*) {};
  CHECK(cache.store(t, 512, 4096, fill));
  CHECK(cache.store(t, 512, 4096, fill));  // already there: only marked recently used
  CHECK(cache.entries() == 1);
  CHECK(cache.store(t, 1536, 4096, fill));
  CHECK(cache.entries() == 2);
  CHECK(cache.device_used() == 8192);
  PrefixCache::Hit hit;
  CHECK(cache.find(t, 2047, 512, hit));
  CHECK(hit.length == 1536);
  CHECK(cache.find(t, 1535, 512, hit));  // under the limit, the shorter one
  CHECK(hit.length == 512);
  CHECK_FALSE(cache.store(t, 1024, 2u << 20, fill));
  CHECK(cache.entries() == 2);
}
