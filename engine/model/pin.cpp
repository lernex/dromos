#include "model/pin.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>

#include "cuda/kernels.h"

namespace lx::pin {
namespace {

constexpr int64_t kAlign = 256;

int64_t aligned(int64_t n) { return (n + kAlign - 1) / kAlign * kAlign; }

bool logging() { return std::getenv("LX_L2_PIN_LOG") != nullptr; }

// LX_L2_PIN_MIN_READS=n: only regions a token reads at least n times are held (the depth memory's projections read
// 6 to 26 times a token; the per-pass projections 2). Experiments use it to pin the memory alone.
double min_reads_from_env() {
  const char* v = std::getenv("LX_L2_PIN_MIN_READS");
  return v && *v ? std::atof(v) : 0.0;
}

// LX_L2_PIN_MAX_TOKENS=n: steps of at most n tokens run with the window. The default is 1, the one-stream decode: on
// the measured formats it is faster by 1.4 to 2.6 percent there, while at 8 streams NVFP4 loses 0.2 to 1.6 percent
// (BF16 gains 1.1) and at 32 the window is a wash or a cost.
int max_tokens_from_env() {
  static const int n = [] {
    const char* v = std::getenv("LX_L2_PIN_MAX_TOKENS");
    return v && *v ? std::atoi(v) : 1;
  }();
  return n;
}

}  // namespace

Hold hold_from_env() {
  if (const char* off = std::getenv("LX_NO_L2_PIN"); off && *off && std::strcmp(off, "0") != 0) return Hold::Off;
  const char* v = std::getenv("LX_L2_PIN");
  if (!v || !*v) return Hold::Off;  // opt-in: the rebased measurements found no reliable gain (see pin.h)
  const std::string s = v;
  if (s == "off" || s == "0") return Hold::Off;
  if (s == "weights") return Hold::Weights;
  if (s == "states") return Hold::States;
  if (s == "both") return Hold::Both;
  throw std::runtime_error("LX_L2_PIN: unknown hold " + s + " (weights, states, both or off)");
}

int64_t budget_bytes() {
  if (const char* mb = std::getenv("LX_L2_PIN_MB"); mb && *mb)
    return static_cast<int64_t>(std::atof(mb) * 1048576.0);
  lx_device_props props{};
  if (lx_rt_device_props(0, &props) != 0 || props.l2_bytes <= 0) return 0;
  return props.l2_bytes * 3 / 4;
}

Arena::Arena() : hold_(hold_from_env()) {}

Arena::~Arena() {
  if (window_ > 0) lx_rt_l2_persist(nullptr, 0, 0, nullptr);
  if (base_) lx_rt_free(base_);
}

bool step_pinned(int tokens) { return tokens > 0 && tokens <= max_tokens_from_env(); }

void Arena::set_on(bool on) {
  if (window_ <= 0 || on == on_) return;
  if (on) {
    int64_t applied = 0;
    if (lx_rt_l2_persist(base_, window_, 1.0f, &applied) != 0) return;  // the window stays off
    on_ = true;
  } else {
    lx_rt_l2_persist(nullptr, 0, 0, nullptr);
    on_ = false;
  }
  if (logging()) std::fprintf(stderr, "L2 carve-out %s\n", on_ ? "on" : "off");
}

void Arena::weight(std::vector<Part> parts, double reads) {
  if (!holds_weights()) return;
  Item it;
  it.parts = std::move(parts);
  it.reads = reads;
  for (const Part& p : it.parts) {
    if (!p.src || p.bytes <= 0 || !p.point) return;  // a region the model cannot make read the copy of: left alone
    it.bytes += aligned(p.bytes);
  }
  if (it.parts.empty()) return;
  items_.push_back(std::move(it));
}

int Arena::state(int64_t bytes, double reads) {
  if (!holds_states() || bytes <= 0) return -1;
  Item it;
  it.state = true;
  it.bytes = aligned(bytes);
  it.reads = reads;
  items_.push_back(std::move(it));
  return static_cast<int>(items_.size()) - 1;
}

void* Arena::state_buffer(int index) const {
  if (index < 0 || index >= static_cast<int>(items_.size())) return nullptr;
  const Item& it = items_[static_cast<size_t>(index)];
  return it.chosen ? it.buffer : nullptr;
}

bool Arena::open(int64_t budget) {
  if (hold_ == Hold::Off || budget <= 0 || items_.empty()) return false;
  // What the device grants for the budget, clamped to its persisting carve-out and window limit. The arena is packed
  // to that, so it holds no bytes the window cannot keep. Asking allocates nothing and changes no state.
  int64_t granted = 0;
  if (lx_rt_l2_persist(nullptr, budget, 1.0f, &granted) != 0 || granted <= 0) return false;
  // Highest reads per token first; ties keep their registration order (weights before states).
  std::vector<size_t> order(items_.size());
  std::iota(order.begin(), order.end(), size_t{0});
  std::stable_sort(order.begin(), order.end(),
                   [this](size_t a, size_t b) { return items_[a].reads > items_[b].reads; });
  std::vector<size_t> chosen;
  int64_t room = granted;
  const double min_reads = min_reads_from_env();
  for (size_t i : order) {
    if (room <= 0) break;
    const Item& it = items_[i];
    if (it.reads < min_reads) continue;
    if (it.state && it.bytes > room) continue;  // a state is never split: it is whole or in DRAM
    chosen.push_back(i);
    room -= it.bytes;  // a weight may straddle the budget: its head is then the window's, the rest streams
  }
  if (chosen.empty()) return false;

  // The arena holds the chosen regions in priority order; the window is its first `budget` bytes.
  int64_t offset = 0;
  for (size_t i : chosen) {
    items_[i].offset = offset;
    offset += items_[i].bytes;
  }
  const int64_t bytes = offset;
  const int64_t window = std::min(granted, bytes);
  void* base = nullptr;
  if (lx_rt_alloc(&base, bytes) != 0 || !base) throw std::runtime_error("L2 arena: device allocation failed");
  int64_t applied = 0;
  const int status = lx_rt_l2_persist(base, window, 1.0f, &applied);
  if (status != 0) {
    lx_rt_free(base);
    if (status != LX_ERR_NOT_SUPPORTED)
      std::fprintf(stderr, "dromos: persisting L2 window refused (%s); running unpinned\n",
                   lx_rt_error_string(status));
    return false;
  }
  base_ = base;
  arena_bytes_ = bytes;
  window_ = applied;
  on_ = true;
  auto* at = static_cast<char*>(base);
  for (size_t i : chosen) {
    Item& it = items_[i];
    it.chosen = true;
    if (it.state) {
      it.buffer = at + it.offset;  // zeroed by lx_rt_alloc, as the model's own allocations are
      continue;
    }
    int64_t where = it.offset;
    for (Part& p : it.parts) {
      void* dst = at + where;
      if (lx_rt_copy(dst, p.src, p.bytes, LX_COPY_D2D, nullptr) != 0 || lx_rt_stream_sync(nullptr) != 0)
        throw std::runtime_error("L2 arena: copying a weight failed");
      p.point(dst);
      where += aligned(p.bytes);
    }
  }
  if (logging()) {
    size_t weights = 0, states = 0;
    for (size_t i : chosen) (items_[i].state ? states : weights) += 1;
    std::fprintf(stderr,
                 "L2 window: %lld bytes granted for %lld asked, arena %lld bytes, %zu weight regions, %zu states\n",
                 static_cast<long long>(applied), static_cast<long long>(budget), static_cast<long long>(bytes),
                 weights, states);
    for (size_t i : chosen)
      std::fprintf(stderr, "  %s %10lld bytes at %10lld, %g reads a token\n", items_[i].state ? "state " : "weight",
                   static_cast<long long>(items_[i].bytes), static_cast<long long>(items_[i].offset), items_[i].reads);
  }
  return true;
}

}  // namespace lx::pin
