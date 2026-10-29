#include "runtime/offload.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#if defined(__x86_64__)
#include <immintrin.h>
#endif
#include <time.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace lx::offload {
namespace {

using Clock = std::chrono::steady_clock;

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed with error " + std::to_string(status));
}

double seconds_since(Clock::time_point start) { return std::chrono::duration<double>(Clock::now() - start).count(); }

std::string format(const char* f, ...) {
  char buf[512];
  va_list args;
  va_start(args, f);
  std::vsnprintf(buf, sizeof(buf), f, args);
  va_end(args);
  return buf;
}

double env_double(const char* name, double fallback) {
  const char* v = std::getenv(name);
  return v && *v ? std::atof(v) : fallback;
}

constexpr int64_t kGiB = 1ll << 30;
constexpr size_t kHuge = 2u << 20;

size_t round_up(size_t n, size_t to) { return (n + to - 1) / to * to; }

// A read-only descriptor that bypasses the page cache: O_DIRECT on Linux, F_NOCACHE where there is no
// O_DIRECT (macOS).
int open_direct(const char* path) {
#if defined(O_DIRECT)
  return open(path, O_RDONLY | O_DIRECT);
#else
  const int fd = open(path, O_RDONLY);
  if (fd >= 0) fcntl(fd, F_NOCACHE, 1);
  return fd;
#endif
}

// Anonymous memory, backed by huge pages where the kernel allows (the CPU lane streams whole experts,
// so fewer TLB misses matter), then pinned and mapped for the GPU.
void* pin_region(size_t bytes, void** device) {
  void* p = mmap(nullptr, bytes, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) throw std::runtime_error("mmap of " + std::to_string(bytes) + " bytes failed");
#if defined(MADV_HUGEPAGE)
  madvise(p, bytes, MADV_HUGEPAGE);
#endif
  if (const int e = lx_rt_host_register(p, static_cast<int64_t>(bytes), device); e != 0) {
    munmap(p, bytes);
    throw std::runtime_error("pinning " + std::to_string(bytes >> 20) + " MiB of host memory failed (error " +
                             std::to_string(e) + ")");
  }
  return p;
}

void unpin_region(void* p, size_t bytes) {
  if (!p) return;
  lx_rt_host_unregister(p);
  munmap(p, bytes);
}

// Reads [offset, offset + bytes) of a file with O_DIRECT into dst (any alignment): the aligned
// superset goes through `bounce` (aligned to 4096, at least bytes + 8192).
void read_direct(int fd, int64_t offset, int64_t bytes, void* dst, char* bounce) {
  const int64_t start = offset / 4096 * 4096;
  const int64_t end = round_up(static_cast<size_t>(offset + bytes), 4096);
  int64_t done = 0;
  while (done < end - start) {
    const ssize_t r = pread(fd, bounce + done, static_cast<size_t>(end - start - done), start + done);
    if (r <= 0) {
      if (r == 0) break;  // the file ends inside the last aligned block
      throw std::runtime_error("pread failed");
    }
    done += r;
  }
  if (done < offset - start + bytes) throw std::runtime_error("short read from the checkpoint");
  std::memcpy(dst, bounce + (offset - start), static_cast<size_t>(bytes));
}

// The CPU backend library beside the GPU kernel library (it exports the same lx_* names, so it is
// loaded by path, never linked).
void* open_cpu_backend() {
  std::string path;
  if (const char* env = std::getenv("LX_CPU_KERNELS")) path = env;
  if (path.empty()) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&lx_offload_plan), &info) && info.dli_fname) {
      path = info.dli_fname;
      const size_t slash = path.rfind('/');
      path = (slash == std::string::npos ? std::string(".") : path.substr(0, slash)) + "/libdromos_cpu.so";
    }
  }
  void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
  if (!lib) throw std::runtime_error("cannot load the CPU backend " + path + ": " + dlerror());
  return lib;
}

}  // namespace

// ---- the device side ------------------------------------------------------------------------------

struct ExpertOffload::Device {
  std::vector<void*> allocations;
  template <typename T>
  T* alloc(size_t count) {
    void* p = nullptr;
    check(lx_rt_alloc(&p, static_cast<int64_t>(std::max<size_t>(count, 1) * sizeof(T))), "device allocation");
    allocations.push_back(p);
    return static_cast<T*>(p);
  }
  ~Device() {
    for (void* p : allocations) lx_rt_free(p);
    if (pinned) lx_rt_host_free(pinned);
  }

  lx_offload_state host_state{};
  lx_offload_state* state = nullptr;
  char *gu = nullptr, *gus = nullptr, *dn = nullptr, *dns = nullptr;
  int32_t *vram_slot, *host_slot, *layer_first, *layer_pinned, *slot_expert, *job_pending, *error;
  uint8_t *cpu_first, *slot_prefetched;
  uint32_t *slot_stamp, *clock, *job_seq;
  unsigned long long* counters;
  lx_expert_ref *src, *fill, *psrc, *pdst;
  int32_t* pcount;
  void* workspace = nullptr;

  // host mirrors of the tables, as last uploaded or downloaded
  std::vector<int32_t> h_layer_first, h_layer_pinned;

  // prefill: a growing workspace, staging slots, pinned host scratch
  void* pf_workspace = nullptr;
  int64_t pf_workspace_bytes = 0;
  int pf_stage_slots = 0;
  char *st_gu = nullptr, *st_gus = nullptr, *st_dn = nullptr, *st_dns = nullptr;
  void* pinned = nullptr;  // prefill host scratch (ids, tables, group arrays, a disk bounce buffer)
  int64_t pinned_bytes = 0;
  lx_expert_ref* pf_groups = nullptr;
  int32_t* pf_ints = nullptr;  // offsets then pairs
  int64_t pf_capacity = 0;     // pairs
};

// ---- construction and load ------------------------------------------------------------------------

ExpertOffload::ExpertOffload(const ExpertShape& shape, const OffloadOptions& options)
    : shape_(shape), options_(options) {
  if (shape.n_layers <= 0 || shape.n_experts <= 0 || shape.top_k <= 0 || shape.top_k > 8)
    throw std::runtime_error("offload: bad expert shape");
  if (shape.type != LX_W_BF16 && shape.type != LX_W_INT8 && shape.type != LX_W_FP8)
    throw std::runtime_error("offload: experts must be BF16, INT8 or FP8");
  const int LE = shape.n_layers * shape.n_experts;
  host_slot_.assign(LE, -1);
  disk_.resize(LE);
  loaded_.assign(LE, 0);
  dev_ = std::make_unique<Device>();
  cpu_lib_ = open_cpu_backend();
  cpu_experts_ = reinterpret_cast<ExpertsFn>(dlsym(cpu_lib_, "lx_moe_experts_decode"));
  if (!cpu_experts_) throw std::runtime_error("the CPU backend has no lx_moe_experts_decode");
  auto ws_bytes = reinterpret_cast<int64_t (*)(int, int, int)>(dlsym(cpu_lib_, "lx_moe_decode_workspace_bytes"));
  cpu_workspace_.resize(static_cast<size_t>(ws_bytes ? ws_bytes(8, shape.I, 8) : 8ll * 8 * shape.I * 4));
}

ExpertOffload::~ExpertOffload() {
  stop_.store(true);
  if (worker_.joinable()) worker_.join();
  dev_.reset();
  if (mapped_block_) lx_rt_host_free(mapped_block_);
  unpin_region(host_region_, host_region_bytes_);
  std::free(d_gu_);
  if (cpu_lib_) dlclose(cpu_lib_);
}

RoutingStats RoutingStats::load(const std::filesystem::path& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open routing stats " + path.string());
  RoutingStats s;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream row(line);
    std::vector<double> counts;
    double v;
    while (row >> v) counts.push_back(v);
    if (counts.empty()) continue;
    if (s.n_experts == 0) s.n_experts = static_cast<int>(counts.size());
    if (static_cast<int>(counts.size()) != s.n_experts) throw std::runtime_error("routing stats: ragged rows");
    s.frequency.insert(s.frequency.end(), counts.begin(), counts.end());
    ++s.n_layers;
  }
  return s;
}

void ExpertOffload::plan_host(const RoutingStats& stats) {
  const int LE = shape_.n_layers * shape_.n_experts;
  const int64_t eb = shape_.expert_bytes();
  int slots = LE;
  if (options_.ram_bytes >= 0) slots = static_cast<int>(std::min<int64_t>(LE, options_.ram_bytes / eb));
  host_slots_ = slots;
  // The hottest experts get RAM slots; with no stats, the first ones. Slot order follows (layer,
  // expert) order among the chosen, so a full host tier is one stack per layer.
  std::vector<int> order(LE);
  std::iota(order.begin(), order.end(), 0);
  if (slots < LE && static_cast<int>(stats.frequency.size()) == LE)
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return stats.frequency[a] > stats.frequency[b]; });
  std::vector<int> chosen(order.begin(), order.begin() + slots);
  std::sort(chosen.begin(), chosen.end());
  std::fill(host_slot_.begin(), host_slot_.end(), -1);
  for (int s = 0; s < slots; ++s) host_slot_[chosen[s]] = s;

  const size_t gu = round_up(static_cast<size_t>(slots) * shape_.gate_up_bytes(), kHuge);
  const size_t gus = round_up(static_cast<size_t>(slots) * shape_.gate_up_scale_bytes(), kHuge);
  const size_t dn = round_up(static_cast<size_t>(slots) * shape_.down_bytes(), kHuge);
  const size_t dns = round_up(static_cast<size_t>(slots) * shape_.down_scale_bytes(), kHuge);
  host_region_bytes_ = std::max<size_t>(gu + gus + dn + dns, kHuge);
  host_region_ = pin_region(host_region_bytes_, &host_region_dev_);
  h_gu_ = static_cast<char*>(host_region_);
  h_gus_ = h_gu_ + gu;
  h_dn_ = h_gus_ + gus;
  h_dns_ = h_dn_ + dn;
  host_bytes_ = static_cast<int64_t>(slots) * eb;

  // The CPU lane's RAM cache for experts that live only on disk.
  if (slots < LE) {
    disk_cache_slots_ = static_cast<int>(env_double("LX_DISK_CACHE_SLOTS", 64));
    const size_t total = static_cast<size_t>(disk_cache_slots_) * eb + 4 * 4096;
    d_gu_ = static_cast<char*>(std::aligned_alloc(4096, round_up(total, 4096)));
    d_gus_ = d_gu_ + static_cast<size_t>(disk_cache_slots_) * shape_.gate_up_bytes();
    d_dn_ = d_gus_ + static_cast<size_t>(disk_cache_slots_) * shape_.gate_up_scale_bytes();
    d_dns_ = d_dn_ + static_cast<size_t>(disk_cache_slots_) * shape_.down_bytes();
    disk_cache_owner_.assign(disk_cache_slots_, -1);
    disk_cache_stamp_.assign(disk_cache_slots_, 0);
  }
}

void ExpertOffload::add_expert(int layer, int expert, const ExpertSource& src, const DiskLocation& disk) {
  if (!host_region_) throw std::runtime_error("offload: plan_host before add_expert");
  const int id = layer * shape_.n_experts + expert;
  disk_[id] = disk;
  const int s = host_slot_[id];
  if (s >= 0) {
    std::memcpy(h_gu_ + s * shape_.gate_up_bytes(), src.gate_up, shape_.gate_up_bytes());
    std::memcpy(h_dn_ + s * shape_.down_bytes(), src.down, shape_.down_bytes());
    if (shape_.scaled()) {
      std::memcpy(h_gus_ + s * shape_.gate_up_scale_bytes(), src.gate_up_scale, shape_.gate_up_scale_bytes());
      std::memcpy(h_dns_ + s * shape_.down_scale_bytes(), src.down_scale, shape_.down_scale_bytes());
    }
  } else if (disk.file.empty() || disk.gate_up < 0) {
    throw std::runtime_error("offload: an expert without a RAM slot needs its disk location");
  }
  loaded_[id] = 1;
}

SlotLayout layout_slots(int Lr, int E, int C, const RoutingStats& stats, double dynamic_fraction) {
  const int LE = Lr * E;
  C = std::clamp(C, 0, LE);
  const bool have_stats = static_cast<int>(stats.frequency.size()) == LE;
  SlotLayout out;
  out.slots.assign(Lr, 0);
  out.pinned.assign(Lr, 0);
  out.contents.assign(Lr, {});
  out.cpu_first.assign(LE, 0);
  std::vector<int>& n_slots = out.slots;
  std::vector<int>& n_pinned = out.pinned;
  std::vector<std::vector<int>>& contents = out.contents;
  std::vector<uint8_t>& cpu_first = out.cpu_first;
  if (C == LE) {
    for (int l = 0; l < Lr; ++l) {
      n_slots[l] = n_pinned[l] = E;
      for (int e = 0; e < E; ++e) contents[l].push_back(e);
    }
  } else if (have_stats) {
    const int n_static = static_cast<int>(C * (1.0 - dynamic_fraction));
    std::vector<int> order(LE);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return stats.frequency[a] > stats.frequency[b]; });
    std::vector<uint8_t> placed(LE, 0);
    for (int j = 0; j < n_static; ++j) {
      placed[order[j]] = 1;
      contents[order[j] / E].push_back(order[j] % E);
    }
    for (int l = 0; l < Lr; ++l) {
      std::sort(contents[l].begin(), contents[l].end(),
                [&](int a, int b) { return stats.frequency[l * E + a] > stats.frequency[l * E + b]; });
      n_pinned[l] = static_cast<int>(contents[l].size());
    }
    // dynamic slots in proportion to the routing mass the pinned sets leave uncovered
    std::vector<double> residual(Lr, 1e-9);
    double total = 0;
    for (int l = 0; l < Lr; ++l) {
      for (int e = 0; e < E; ++e)
        if (!placed[l * E + e]) residual[l] += stats.frequency[l * E + e];
      total += residual[l];
    }
    int left = C - n_static;
    std::vector<int> extra(Lr, 0);
    for (int l = 0; l < Lr; ++l) {
      extra[l] = std::min(E - n_pinned[l], static_cast<int>(std::floor((C - n_static) * residual[l] / total)));
      left -= extra[l];
    }
    for (int round = 0; left > 0 && round < E; ++round)
      for (int l = 0; l < Lr && left > 0; ++l)
        if (n_pinned[l] + extra[l] < E) {
          ++extra[l];
          --left;
        }
    for (int l = 0; l < Lr; ++l) {
      std::vector<int> rest;
      for (int e = 0; e < E; ++e)
        if (!placed[l * E + e]) rest.push_back(e);
      std::stable_sort(rest.begin(), rest.end(),
                       [&](int a, int b) { return stats.frequency[l * E + a] > stats.frequency[l * E + b]; });
      for (int j = 0; j < extra[l]; ++j) contents[l].push_back(rest[j]);
      n_slots[l] = n_pinned[l] + extra[l];
      // Cold experts (outside the initial set): the CPU computes their misses first.
      for (size_t j = extra[l]; j < rest.size(); ++j) cpu_first[l * E + rest[j]] = 1;
    }
  } else {
    for (int l = 0; l < Lr; ++l) {
      n_slots[l] = C / Lr + (l < C % Lr ? 1 : 0);
      for (int e = 0; e < n_slots[l]; ++e) contents[l].push_back(e);
    }
  }
  return out;
}

void ExpertOffload::finalize(const RoutingStats& stats, int64_t vram_cache_bytes) {
  options_.vram_cache_bytes = vram_cache_bytes;
  Device& d = *dev_;
  const int Lr = shape_.n_layers, E = shape_.n_experts, LE = Lr * E;
  for (int i = 0; i < LE; ++i)
    if (!loaded_[i]) throw std::runtime_error("offload: expert " + std::to_string(i % E) + " of layer " +
                                              std::to_string(i / E) + " was never added");
  const int64_t eb = shape_.expert_bytes();
  const bool have_stats = static_cast<int>(stats.frequency.size()) == LE;
  int C = static_cast<int>(std::min<int64_t>(LE, std::max<int64_t>(0, options_.vram_cache_bytes / eb)));

  const SlotLayout layout = layout_slots(Lr, E, C, have_stats ? stats : RoutingStats{}, options_.dynamic_fraction);
  const std::vector<int>& n_slots = layout.slots;
  const std::vector<int>& n_pinned = layout.pinned;
  const std::vector<std::vector<int>>& contents = layout.contents;
  const std::vector<uint8_t>& cpu_first = layout.cpu_first;
  C = std::accumulate(n_slots.begin(), n_slots.end(), 0);
  n_slots_ = C;

  // VRAM pool
  const int alloc_slots = std::max(C, 1);
  d.gu = d.alloc<char>(static_cast<size_t>(alloc_slots) * shape_.gate_up_bytes());
  d.dn = d.alloc<char>(static_cast<size_t>(alloc_slots) * shape_.down_bytes());
  if (shape_.scaled()) {
    d.gus = d.alloc<char>(static_cast<size_t>(alloc_slots) * shape_.gate_up_scale_bytes());
    d.dns = d.alloc<char>(static_cast<size_t>(alloc_slots) * shape_.down_scale_bytes());
  }
  vram_bytes_ = static_cast<int64_t>(alloc_slots) * eb;

  // tables
  std::vector<int32_t> vram_slot(LE, -1), slot_expert(alloc_slots, -1), layer_first(Lr + 1, 0);
  for (int l = 0; l < Lr; ++l) layer_first[l + 1] = layer_first[l] + n_slots[l];
  std::vector<char> bounce;
  int fd = -1;
  std::string fd_path;
  for (int l = 0; l < Lr; ++l) {
    for (int j = 0; j < n_slots[l]; ++j) {
      const int e = contents[l][j], s = layer_first[l] + j, id = l * E + e;
      vram_slot[id] = s;
      slot_expert[s] = e;
      const int h = host_slot_[id];
      const void *gu, *gus = nullptr, *dn, *dns = nullptr;
      if (h >= 0) {
        gu = h_gu_ + h * shape_.gate_up_bytes();
        dn = h_dn_ + h * shape_.down_bytes();
        if (shape_.scaled()) {
          gus = h_gus_ + h * shape_.gate_up_scale_bytes();
          dns = h_dns_ + h * shape_.down_scale_bytes();
        }
      } else {  // only on disk: read it once for its VRAM slot
        const DiskLocation& loc = disk_[id];
        if (fd < 0 || fd_path != loc.file.string()) {
          if (fd >= 0) close(fd);
          fd = open_direct(loc.file.c_str());
          fd_path = loc.file.string();
          if (fd < 0) throw std::runtime_error("cannot open " + fd_path);
        }
        bounce.resize(static_cast<size_t>(eb) * 2 + 4 * 8192);
        char* b = reinterpret_cast<char*>(round_up(reinterpret_cast<size_t>(bounce.data()), 4096));
        static std::vector<char> tmp;
        tmp.resize(static_cast<size_t>(eb));
        char* t = tmp.data();
        read_direct(fd, loc.gate_up, shape_.gate_up_bytes(), t, b);
        read_direct(fd, loc.down, shape_.down_bytes(), t + shape_.gate_up_bytes(), b);
        gu = t;
        dn = t + shape_.gate_up_bytes();
        if (shape_.scaled()) {
          read_direct(fd, loc.gate_up_scale, shape_.gate_up_scale_bytes(), t + shape_.gate_up_bytes() + shape_.down_bytes(), b);
          read_direct(fd, loc.down_scale, shape_.down_scale_bytes(),
                      t + shape_.gate_up_bytes() + shape_.down_bytes() + shape_.gate_up_scale_bytes(), b);
          gus = t + shape_.gate_up_bytes() + shape_.down_bytes();
          dns = static_cast<const char*>(gus) + shape_.gate_up_scale_bytes();
        }
        check(lx_rt_copy(d.gu + s * shape_.gate_up_bytes(), gu, shape_.gate_up_bytes(), LX_COPY_H2D, nullptr), "copy");
        check(lx_rt_copy(d.dn + s * shape_.down_bytes(), dn, shape_.down_bytes(), LX_COPY_H2D, nullptr), "copy");
        if (shape_.scaled()) {
          check(lx_rt_copy(d.gus + s * shape_.gate_up_scale_bytes(), gus, shape_.gate_up_scale_bytes(), LX_COPY_H2D, nullptr), "copy");
          check(lx_rt_copy(d.dns + s * shape_.down_scale_bytes(), dns, shape_.down_scale_bytes(), LX_COPY_H2D, nullptr), "copy");
        }
        check(lx_rt_stream_sync(nullptr), "sync");
        continue;
      }
      check(lx_rt_copy(d.gu + s * shape_.gate_up_bytes(), gu, shape_.gate_up_bytes(), LX_COPY_H2D, nullptr), "copy");
      check(lx_rt_copy(d.dn + s * shape_.down_bytes(), dn, shape_.down_bytes(), LX_COPY_H2D, nullptr), "copy");
      if (shape_.scaled()) {
        check(lx_rt_copy(d.gus + s * shape_.gate_up_scale_bytes(), gus, shape_.gate_up_scale_bytes(), LX_COPY_H2D, nullptr), "copy");
        check(lx_rt_copy(d.dns + s * shape_.down_scale_bytes(), dns, shape_.down_scale_bytes(), LX_COPY_H2D, nullptr), "copy");
      }
    }
  }
  if (fd >= 0) close(fd);
  check(lx_rt_stream_sync(nullptr), "sync");

  d.vram_slot = d.alloc<int32_t>(LE);
  d.host_slot = d.alloc<int32_t>(LE);
  d.cpu_first = d.alloc<uint8_t>(LE);
  d.layer_first = d.alloc<int32_t>(Lr + 1);
  d.layer_pinned = d.alloc<int32_t>(Lr);
  d.slot_expert = d.alloc<int32_t>(alloc_slots);
  d.slot_stamp = d.alloc<uint32_t>(alloc_slots);
  d.slot_prefetched = d.alloc<uint8_t>(alloc_slots);
  d.clock = d.alloc<uint32_t>(1);
  d.job_seq = d.alloc<uint32_t>(1);
  d.job_pending = d.alloc<int32_t>(1);
  d.error = d.alloc<int32_t>(1);
  d.counters = d.alloc<unsigned long long>(static_cast<size_t>(Lr) * LX_OFFLOAD_COUNTERS);
  d.src = d.alloc<lx_expert_ref>(LX_OFFLOAD_MAX_PAIRS);
  d.fill = d.alloc<lx_expert_ref>(LX_OFFLOAD_MAX_PAIRS);
  d.psrc = d.alloc<lx_expert_ref>(16);
  d.pdst = d.alloc<lx_expert_ref>(16);
  d.pcount = d.alloc<int32_t>(1);
  d.workspace = d.alloc<char>(static_cast<size_t>(lx_offload_workspace_bytes(8, shape_.L, shape_.I, shape_.top_k)));
  check(lx_rt_copy(d.vram_slot, vram_slot.data(), LE * 4ll, LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_copy(d.host_slot, host_slot_.data(), LE * 4ll, LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_copy(d.cpu_first, cpu_first.data(), LE, LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_copy(d.layer_first, layer_first.data(), (Lr + 1) * 4ll, LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_copy(d.layer_pinned, n_pinned.data(), Lr * 4ll, LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_copy(d.slot_expert, slot_expert.data(), alloc_slots * 4ll, LX_COPY_H2D, nullptr), "copy");
  d.h_layer_first = layer_first;
  d.h_layer_pinned = n_pinned;

  // the CPU lane's mapped block: mailbox, latent rows, result rows
  const size_t mb = round_up(sizeof(lx_offload_mailbox), 4096);
  const size_t lat = round_up(8ull * shape_.L * 2, 4096), res = round_up(8ull * shape_.L * 4, 4096);
  void* dev_ptr = nullptr;
  check(lx_rt_host_alloc_mapped(&mapped_block_, &dev_ptr, static_cast<int64_t>(mb + lat + res)), "mapped allocation");
  std::memset(mapped_block_, 0, mb + lat + res);
  mailbox_ = static_cast<lx_offload_mailbox*>(mapped_block_);
  cpu_latent_ = reinterpret_cast<uint16_t*>(static_cast<char*>(mapped_block_) + mb);
  cpu_result_ = reinterpret_cast<float*>(static_cast<char*>(mapped_block_) + mb + lat);
  char* dev_base = static_cast<char*>(dev_ptr);

  lx_offload_state& st = d.host_state;
  st.n_layers = Lr;
  st.n_experts = E;
  st.top_k = shape_.top_k;
  st.L = shape_.L;
  st.I = shape_.I;
  st.type = shape_.type;
  st.vram_slot = d.vram_slot;
  st.host_slot = d.host_slot;
  st.cpu_first = d.cpu_first;
  st.layer_first = d.layer_first;
  st.layer_pinned = d.layer_pinned;
  st.slot_expert = d.slot_expert;
  st.slot_stamp = d.slot_stamp;
  st.slot_prefetched = d.slot_prefetched;
  st.clock = d.clock;
  st.vram_pool = lx_expert_ref{d.gu, d.gus, d.dn, d.dns};
  char* hdev = static_cast<char*>(host_region_dev_);
  const int64_t off_gus = h_gus_ - h_gu_, off_dn = h_dn_ - h_gu_, off_dns = h_dns_ - h_gu_;
  st.host_pool = lx_expert_ref{hdev, shape_.scaled() ? hdev + off_gus : nullptr, hdev + off_dn,
                               shape_.scaled() ? hdev + off_dns : nullptr};
  st.gate_up_stride = shape_.gate_up_bytes();
  st.gate_up_scale_stride = shape_.gate_up_scale_bytes();
  st.down_stride = shape_.down_bytes();
  st.down_scale_stride = shape_.down_scale_bytes();
  st.max_cpu_pairs = options_.max_cpu_pairs;
  st.cpu_fraction = static_cast<float>(options_.cpu_fraction);
  st.fill = options_.fill ? 1 : 0;
  st.mailbox = reinterpret_cast<lx_offload_mailbox*>(dev_base);
  st.cpu_latent = dev_base + mb;
  st.cpu_result = reinterpret_cast<const float*>(dev_base + mb + lat);
  st.job_seq = d.job_seq;
  st.job_pending = d.job_pending;
  st.error = d.error;
  st.counters = d.counters;
  d.state = d.alloc<lx_offload_state>(1);
  check(lx_rt_copy(d.state, &st, sizeof(st), LX_COPY_H2D, nullptr), "copy");
  check(lx_rt_stream_sync(nullptr), "sync");

  const bool any_disk = std::any_of(host_slot_.begin(), host_slot_.end(), [](int32_t s) { return s < 0; });
  if (options_.max_cpu_pairs > 0 || any_disk) worker_ = std::thread([this] { cpu_worker(); });
}

// ---- steps ----------------------------------------------------------------------------------------

void ExpertOffload::decode(int layer, const int32_t* ids, const float* weights, const void* latent, float* out, int T,
                           void* stream) {
  Device& d = *dev_;
  check(lx_offload_plan(ids, weights, latent, layer, d.state, d.src, d.fill, T, stream), "lx_offload_plan");
  check(lx_offload_experts(latent, weights, d.src, d.fill, shape_.type, out, d.workspace, T, shape_.L, shape_.I,
                           shape_.top_k, stream),
        "lx_offload_experts");
  check(lx_offload_join(d.state, out, T, shape_.L, stream), "lx_offload_join");
}

void ExpertOffload::prefetch(int layer, const int32_t* predicted, int n, void* stream) {
  Device& d = *dev_;
  n = std::min(n, 16);
  check(lx_offload_prefetch(predicted, n, layer, d.state, d.psrc, d.pdst, d.pcount, stream), "lx_offload_prefetch");
  check(lx_offload_copy(d.psrc, d.pdst, d.pcount, n, shape_.type, shape_.L, shape_.I, stream), "lx_offload_copy");
}

void ExpertOffload::prefill(int layer, const int32_t* ids, const float* weights, const void* latent, float* out, int T,
                            void* stream) {
  Device& d = *dev_;
  const int K = shape_.top_k, P = T * K, E = shape_.n_experts;
  const int first = d.h_layer_first[layer], last = d.h_layer_first[layer + 1], pinned = d.h_layer_pinned[layer];
  const int nl = last - first;

  // pinned scratch: ids [P] | vram row [E] | slot_expert [nl] | slot_stamp [nl] | clock | tables out | groups
  const int64_t need = 4ll * P + 4ll * E + 8ll * nl + 64 + 4ll * E + 8ll * nl + 64 +
                       static_cast<int64_t>(sizeof(lx_expert_ref)) * (E + 1) + 4ll * (E + 2 + P) + 4096;
  if (need > d.pinned_bytes) {
    if (d.pinned) {
      check(lx_rt_stream_sync(stream), "sync");
      lx_rt_host_free(d.pinned);
    }
    d.pinned_bytes = need * 2;
    check(lx_rt_host_alloc(&d.pinned, d.pinned_bytes), "pinned allocation");
  }
  const int64_t ws_need = lx_offload_workspace_bytes(T, shape_.L, shape_.I, K);
  if (ws_need > d.pf_workspace_bytes) {
    check(lx_rt_stream_sync(stream), "sync");
    if (d.pf_workspace) lx_rt_free(d.pf_workspace);
    check(lx_rt_alloc(&d.pf_workspace, ws_need), "device allocation");
    d.pf_workspace_bytes = ws_need;
  }
  if (P + E + 2 > d.pf_capacity) {
    check(lx_rt_stream_sync(stream), "sync");
    const int64_t cap = std::max<int64_t>(P + E + 2, 4096);
    d.pf_groups = d.alloc<lx_expert_ref>(static_cast<size_t>(E + 1));
    d.pf_ints = d.alloc<int32_t>(static_cast<size_t>(cap + E + 2));
    d.pf_capacity = cap;
  }
  char* hp = static_cast<char*>(d.pinned);
  int32_t* h_ids = reinterpret_cast<int32_t*>(hp);
  int32_t* h_row = h_ids + P;
  int32_t* h_owner = h_row + E;
  uint32_t* h_stamp = reinterpret_cast<uint32_t*>(h_owner + nl);
  uint32_t* h_clock = h_stamp + nl;
  int32_t* o_row = reinterpret_cast<int32_t*>(h_clock + 16);  // tables to upload
  int32_t* o_owner = o_row + E;
  uint32_t* o_stamp = reinterpret_cast<uint32_t*>(o_owner + nl);
  lx_expert_ref* h_groups =
      reinterpret_cast<lx_expert_ref*>(round_up(reinterpret_cast<size_t>(o_stamp + nl + 16), 64));
  int32_t* h_ints = reinterpret_cast<int32_t*>(h_groups + E + 1);  // offsets [G + 1] then pairs [P]

  check(lx_rt_copy(h_ids, ids, 4ll * P, LX_COPY_D2H, stream), "copy");
  check(lx_rt_copy(h_row, d.vram_slot + static_cast<int64_t>(layer) * E, 4ll * E, LX_COPY_D2H, stream), "copy");
  if (nl > 0) {
    check(lx_rt_copy(h_owner, d.slot_expert + first, 4ll * nl, LX_COPY_D2H, stream), "copy");
    check(lx_rt_copy(h_stamp, d.slot_stamp + first, 4ll * nl, LX_COPY_D2H, stream), "copy");
  }
  check(lx_rt_copy(h_clock, d.clock, 4, LX_COPY_D2H, stream), "copy");
  check(lx_rt_stream_sync(stream), "sync");

  std::vector<int> count(E, 0);
  for (int p = 0; p < P; ++p) ++count[h_ids[p]];
  std::vector<int> resident, missing;
  std::vector<uint8_t> used_slot(nl, 0);
  for (int e = 0; e < E; ++e) {
    if (!count[e]) continue;
    if (h_row[e] >= 0) {
      resident.push_back(e);
      used_slot[h_row[e] - first] = 1;
    } else {
      missing.push_back(e);
    }
  }
  // victims: the layer's dynamic slots not used by this step, least recently used first
  std::vector<int> victims;
  for (int j = pinned; j < nl; ++j)
    if (!used_slot[j]) victims.push_back(first + j);
  std::stable_sort(victims.begin(), victims.end(), [&](int a, int b) { return h_stamp[a - first] < h_stamp[b - first]; });
  bool staging = false;
  if (victims.empty() && !missing.empty()) {
    // a layer with no free dynamic slot stages through slots of its own, outside the cache
    if (!d.pf_stage_slots) {
      d.pf_stage_slots = 8;
      d.st_gu = d.alloc<char>(8ull * shape_.gate_up_bytes());
      d.st_dn = d.alloc<char>(8ull * shape_.down_bytes());
      if (shape_.scaled()) {
        d.st_gus = d.alloc<char>(8ull * shape_.gate_up_scale_bytes());
        d.st_dns = d.alloc<char>(8ull * shape_.down_scale_bytes());
      }
    }
    staging = true;
    for (int j = 0; j < d.pf_stage_slots; ++j) victims.push_back(j);
  }
  int batch = static_cast<int>(victims.size());
  if (options_.prefill_batch > 0) batch = std::min(batch, options_.prefill_batch);
  if (batch <= 0) batch = 1;

  auto vram_ref = [&](int s) {
    return lx_expert_ref{d.gu + s * shape_.gate_up_bytes(), d.gus ? d.gus + s * shape_.gate_up_scale_bytes() : nullptr,
                         d.dn + s * shape_.down_bytes(), d.dns ? d.dns + s * shape_.down_scale_bytes() : nullptr};
  };
  auto stage_ref = [&](int s) {
    return lx_expert_ref{d.st_gu + s * shape_.gate_up_bytes(),
                         d.st_gus ? d.st_gus + s * shape_.gate_up_scale_bytes() : nullptr,
                         d.st_dn + s * shape_.down_bytes(), d.st_dns ? d.st_dns + s * shape_.down_scale_bytes() : nullptr};
  };

  // Group arrays for every batch, in one upload: batch 0 is the resident experts plus the first
  // `batch` missing ones; later batches reuse the same victims.
  std::vector<int> group_expert, group_slot;  // per group: expert, VRAM slot (or staging slot)
  std::vector<int> batch_first{0};            // group index where each batch starts
  for (int e : resident) {
    group_expert.push_back(e);
    group_slot.push_back(h_row[e]);
  }
  for (size_t m = 0; m < missing.size(); ++m) {
    if (m > 0 && m % batch == 0) batch_first.push_back(static_cast<int>(group_expert.size()));
    group_expert.push_back(missing[m]);
    group_slot.push_back(victims[m % batch]);
  }
  batch_first.push_back(static_cast<int>(group_expert.size()));
  const int G = static_cast<int>(group_expert.size());
  int* offsets = h_ints;
  int* pairs = h_ints + G + 1;
  {
    std::vector<int> start(E, -1);
    int pos = 0;
    for (int g = 0; g < G; ++g) {
      start[group_expert[g]] = pos;
      offsets[g] = pos;
      pos += count[group_expert[g]];
    }
    offsets[G] = pos;
    std::vector<int> fillpos(start);
    for (int p = 0; p < P; ++p) pairs[fillpos[h_ids[p]]++] = p;
  }
  const int resident_groups = static_cast<int>(resident.size());
  for (int g = 0; g < G; ++g)
    h_groups[g] = (g >= resident_groups && staging) ? stage_ref(group_slot[g]) : vram_ref(group_slot[g]);
  check(lx_rt_copy(d.pf_groups, h_groups, static_cast<int64_t>(sizeof(lx_expert_ref)) * std::max(G, 1), LX_COPY_H2D,
                   stream),
        "copy");
  check(lx_rt_copy(d.pf_ints, h_ints, 4ll * (G + 1 + P), LX_COPY_H2D, stream), "copy");

  // tables after the step: start from what the device holds
  std::memcpy(o_row, h_row, 4ll * E);
  std::memcpy(o_owner, h_owner, 4ll * nl);
  std::memcpy(o_stamp, h_stamp, 4ll * nl);
  const uint32_t now = *h_clock;
  for (int e : resident) o_stamp[h_row[e] - first] = now;

  std::vector<char> disk_tmp;
  for (size_t b = 0; b + 1 < batch_first.size(); ++b) {
    const int g0 = batch_first[b], g1 = batch_first[b + 1];
    for (int g = std::max(g0, resident_groups); g < g1; ++g) {
      const int e = group_expert[g], s = group_slot[g], id = layer * E + e;
      const lx_expert_ref dst = staging ? stage_ref(s) : vram_ref(s);
      const int h = host_slot_[id];
      if (h >= 0) {
        check(lx_rt_copy(const_cast<void*>(dst.gate_up), h_gu_ + h * shape_.gate_up_bytes(), shape_.gate_up_bytes(),
                         LX_COPY_H2D, stream),
              "copy");
        check(lx_rt_copy(const_cast<void*>(dst.down), h_dn_ + h * shape_.down_bytes(), shape_.down_bytes(), LX_COPY_H2D,
                         stream),
              "copy");
        if (shape_.scaled()) {
          check(lx_rt_copy(const_cast<void*>(dst.gate_up_scale), h_gus_ + h * shape_.gate_up_scale_bytes(),
                           shape_.gate_up_scale_bytes(), LX_COPY_H2D, stream),
                "copy");
          check(lx_rt_copy(const_cast<void*>(dst.down_scale), h_dns_ + h * shape_.down_scale_bytes(),
                           shape_.down_scale_bytes(), LX_COPY_H2D, stream),
                "copy");
        }
      } else {
        // only on disk: through the CPU lane's RAM cache, synchronously (the slow tier)
        check(lx_rt_stream_sync(stream), "sync");
        int cs;
        auto it = disk_cache_index_.find(id);
        if (it != disk_cache_index_.end()) {
          cs = it->second;
        } else {
          cs = static_cast<int>(std::min_element(disk_cache_stamp_.begin(), disk_cache_stamp_.end()) -
                                disk_cache_stamp_.begin());
          load_disk_expert(layer, e, cs);
        }
        disk_cache_stamp_[cs] = ++disk_clock_;
        check(lx_rt_copy(const_cast<void*>(dst.gate_up), d_gu_ + cs * shape_.gate_up_bytes(), shape_.gate_up_bytes(),
                         LX_COPY_H2D, stream),
              "copy");
        check(lx_rt_copy(const_cast<void*>(dst.down), d_dn_ + cs * shape_.down_bytes(), shape_.down_bytes(),
                         LX_COPY_H2D, stream),
              "copy");
        if (shape_.scaled()) {
          check(lx_rt_copy(const_cast<void*>(dst.gate_up_scale), d_gus_ + cs * shape_.gate_up_scale_bytes(),
                           shape_.gate_up_scale_bytes(), LX_COPY_H2D, stream),
                "copy");
          check(lx_rt_copy(const_cast<void*>(dst.down_scale), d_dns_ + cs * shape_.down_scale_bytes(),
                           shape_.down_scale_bytes(), LX_COPY_H2D, stream),
                "copy");
        }
        check(lx_rt_stream_sync(stream), "sync");
      }
      if (!staging) {
        const int j = s - first;
        if (o_owner[j] >= 0) o_row[o_owner[j]] = -1;
        o_owner[j] = e;
        o_row[e] = s;
        o_stamp[j] = now;
      }
    }
    const int flags = (b == 0 ? LX_OG_BEGIN : 0) | (b + 2 == batch_first.size() ? LX_OG_END : 0);
    check(lx_offload_experts_grouped(latent, weights, d.pf_groups + g0, d.pf_ints + g0, d.pf_ints + G + 1, g1 - g0,
                                     shape_.type, out, d.pf_workspace, T, shape_.L, shape_.I, K, flags, stream),
          "lx_offload_experts_grouped");
  }
  check(lx_rt_copy(d.vram_slot + static_cast<int64_t>(layer) * E, o_row, 4ll * E, LX_COPY_H2D, stream), "copy");
  if (nl > 0) {
    check(lx_rt_copy(d.slot_expert + first, o_owner, 4ll * nl, LX_COPY_H2D, stream), "copy");
    check(lx_rt_copy(d.slot_stamp + first, o_stamp, 4ll * nl, LX_COPY_H2D, stream), "copy");
    check(lx_rt_memset(d.slot_prefetched + first, 0, nl, stream), "memset");
  }
  prefill_pairs_ += P;
  prefill_staged_ += missing.size();
}

// ---- the CPU lane -----------------------------------------------------------------------------

void ExpertOffload::load_disk_expert(int layer, int expert, int cs) {
  const int id = layer * shape_.n_experts + expert;
  const DiskLocation& loc = disk_[id];
  const int fd = open_direct(loc.file.c_str());
  if (fd < 0) throw std::runtime_error("cannot open " + loc.file.string());
  static thread_local std::vector<char> bounce;
  bounce.resize(static_cast<size_t>(shape_.gate_up_bytes()) + 3 * 8192);
  char* b = reinterpret_cast<char*>(round_up(reinterpret_cast<size_t>(bounce.data()), 4096));
  read_direct(fd, loc.gate_up, shape_.gate_up_bytes(), d_gu_ + cs * shape_.gate_up_bytes(), b);
  read_direct(fd, loc.down, shape_.down_bytes(), d_dn_ + cs * shape_.down_bytes(), b);
  if (shape_.scaled()) {
    read_direct(fd, loc.gate_up_scale, shape_.gate_up_scale_bytes(), d_gus_ + cs * shape_.gate_up_scale_bytes(), b);
    read_direct(fd, loc.down_scale, shape_.down_scale_bytes(), d_dns_ + cs * shape_.down_scale_bytes(), b);
  }
  close(fd);
  if (disk_cache_owner_[cs] >= 0) disk_cache_index_.erase(disk_cache_owner_[cs]);
  disk_cache_owner_[cs] = id;
  disk_cache_index_[id] = cs;
  disk_reads_.fetch_add(1);
}

void ExpertOffload::serve_job() {
  const auto start = Clock::now();
  const int layer = mailbox_->layer, T = mailbox_->T, n = mailbox_->n, L = shape_.L, E = shape_.n_experts;
  std::memset(cpu_result_, 0, sizeof(float) * T * L);
  const lx_weight host_gu{shape_.type, h_gu_, shape_.scaled() ? h_gus_ : nullptr, 1.0f};
  const lx_weight host_dn{shape_.type, h_dn_, shape_.scaled() ? h_dns_ : nullptr, 1.0f};
  const lx_weight disk_gu{shape_.type, d_gu_, shape_.scaled() ? d_gus_ : nullptr, 1.0f};
  const lx_weight disk_dn{shape_.type, d_dn_, shape_.scaled() ? d_dns_ : nullptr, 1.0f};
  static thread_local std::vector<float> tmp;
  tmp.resize(L);
  for (int t = 0; t < T; ++t) {
    int32_t ids[2][LX_OFFLOAD_MAX_PAIRS];
    float w[2][LX_OFFLOAD_MAX_PAIRS];
    int cnt[2] = {0, 0};
    for (int j = 0; j < n; ++j) {
      if (mailbox_->token[j] != t) continue;
      const int id = layer * E + mailbox_->expert[j];
      const int h = host_slot_[id];
      if (h >= 0) {
        ids[0][cnt[0]] = h;
        w[0][cnt[0]++] = mailbox_->weight[j];
      } else {
        int cs;
        auto it = disk_cache_index_.find(id);
        if (it != disk_cache_index_.end()) {
          cs = it->second;
        } else {
          // never evict a slot this token already uses
          cs = -1;
          uint64_t best = UINT64_MAX;
          for (int s = 0; s < disk_cache_slots_; ++s) {
            bool taken = false;
            for (int q = 0; q < cnt[1]; ++q) taken |= ids[1][q] == s;
            if (!taken && disk_cache_stamp_[s] < best) {
              best = disk_cache_stamp_[s];
              cs = s;
            }
          }
          load_disk_expert(layer, mailbox_->expert[j], cs);
        }
        disk_cache_stamp_[cs] = ++disk_clock_;
        ids[1][cnt[1]] = cs;
        w[1][cnt[1]++] = mailbox_->weight[j];
      }
    }
    for (int tier = 0; tier < 2; ++tier) {
      if (!cnt[tier]) continue;
      // the contract allows top_k up to the pairs of one token; chunk at 8 like the GPU kernels
      for (int c0 = 0; c0 < cnt[tier]; c0 += 8) {
        const int k = std::min(8, cnt[tier] - c0);
        const int status = cpu_experts_(cpu_latent_ + static_cast<int64_t>(t) * L, ids[tier] + c0, w[tier] + c0,
                                        tier == 0 ? &host_gu : &disk_gu, tier == 0 ? &host_dn : &disk_dn, tmp.data(),
                                        cpu_workspace_.data(), 1, L, shape_.I, k, nullptr);
        if (status != 0) throw std::runtime_error("CPU experts failed");
        float* r = cpu_result_ + static_cast<int64_t>(t) * L;
        for (int i = 0; i < L; ++i) r[i] += tmp[i];
      }
    }
  }
  cpu_busy_us_.store(cpu_busy_us_.load() + seconds_since(start) * 1e6);
  cpu_jobs_.fetch_add(1);
}

void ExpertOffload::cpu_worker() {
  uint32_t seen = __atomic_load_n(&mailbox_->doorbell, __ATOMIC_ACQUIRE);
  auto last = Clock::now();
  while (!stop_.load(std::memory_order_relaxed)) {
    const uint32_t bell = __atomic_load_n(&mailbox_->doorbell, __ATOMIC_ACQUIRE);
    if (bell != seen) {
      seen = bell;
      try {
        serve_job();
      } catch (const std::exception& e) {
        std::fprintf(stderr, "offload: CPU lane failed: %s\n", e.what());
        return;  // the GPU's join times out and flags the error
      }
      __atomic_store_n(&mailbox_->done, bell, __ATOMIC_RELEASE);
      last = Clock::now();
      continue;
    }
    // Spin while the model is generating (a job every ~100 us); after 50 ms idle, nap 20 us at a time.
    if (seconds_since(last) < 0.05) {
#if defined(__x86_64__)
      _mm_pause();
#elif defined(__aarch64__)
      __asm__ __volatile__("yield");
#endif
    } else {
      timespec ts{0, 20000};
      nanosleep(&ts, nullptr);
    }
  }
}

// ---- reporting ------------------------------------------------------------------------------------

OffloadCounters ExpertOffload::counters() {
  OffloadCounters c;
  const int Lr = shape_.n_layers;
  std::vector<unsigned long long> raw(static_cast<size_t>(Lr) * LX_OFFLOAD_COUNTERS);
  check(lx_rt_stream_sync(nullptr), "sync");
  check(lx_rt_copy(raw.data(), dev_->counters, static_cast<int64_t>(raw.size() * 8), LX_COPY_D2H, nullptr), "copy");
  check(lx_rt_stream_sync(nullptr), "sync");
  for (int l = 0; l < Lr; ++l) {
    const unsigned long long* r = raw.data() + l * LX_OFFLOAD_COUNTERS;
    c.hits += r[LX_OC_HITS];
    c.gpu_misses += r[LX_OC_GPU_MISSES];
    c.cpu_misses += r[LX_OC_CPU_MISSES];
    c.fills += r[LX_OC_FILLS];
    c.prefetches += r[LX_OC_PREFETCHES];
    c.prefetch_hits += r[LX_OC_PREFETCH_HITS];
    c.calls += r[LX_OC_CALLS];
  }
  c.prefill_pairs = prefill_pairs_;
  c.prefill_staged = prefill_staged_;
  c.disk_reads = disk_reads_.load();
  c.cpu_busy_us = cpu_busy_us_.load();
  c.cpu_jobs = cpu_jobs_.load();
  return c;
}

void ExpertOffload::reset_counters() {
  check(lx_rt_stream_sync(nullptr), "sync");
  check(lx_rt_memset(dev_->counters, 0, static_cast<int64_t>(shape_.n_layers) * LX_OFFLOAD_COUNTERS * 8, nullptr),
        "memset");
  check(lx_rt_stream_sync(nullptr), "sync");
  prefill_pairs_ = prefill_staged_ = 0;
  disk_reads_.store(0);
  cpu_busy_us_.store(0);
  cpu_jobs_.store(0);
}

bool ExpertOffload::error() {
  int32_t e = 0;
  check(lx_rt_copy(&e, dev_->error, 4, LX_COPY_D2H, nullptr), "copy");
  check(lx_rt_stream_sync(nullptr), "sync");
  return e != 0;
}

std::string ExpertOffload::describe() const {
  const int LE = shape_.n_layers * shape_.n_experts;
  const int on_disk = static_cast<int>(std::count(host_slot_.begin(), host_slot_.end(), -1));
  const char* names[] = {"BF16", "INT8", "FP8", "NVFP4"};
  std::string s = format("experts: %d layers x %d, %s, %.2f MB each, %.2f GB in all\n", shape_.n_layers,
                         shape_.n_experts, names[shape_.type], shape_.expert_bytes() / 1e6,
                         static_cast<double>(LE) * shape_.expert_bytes() / 1e9);
  s += format("  VRAM: %d slots (%.1f%% of the experts), %.2f GB\n", n_slots_, 100.0 * n_slots_ / LE, vram_bytes_ / 1e9);
  s += format("  RAM:  %d experts, %.2f GB pinned%s\n", LE - on_disk, host_bytes_ / 1e9,
              on_disk ? format(", %d on NVMe (CPU lane cache %d)", on_disk, disk_cache_slots_).c_str() : "");
  s += format("  CPU lane: up to %d pairs per step, %.0f%% of misses; fill %s\n", options_.max_cpu_pairs,
              100 * options_.cpu_fraction, options_.fill ? "on" : "off");
  return s;
}

// ---- budgets and placement --------------------------------------------------------------------------

int64_t vram_budget_bytes() {
  int64_t free_b = 0, total_b = 0;
  check(lx_rt_mem_info(&free_b, &total_b), "lx_rt_mem_info");
  const double gb = env_double("LX_VRAM_BUDGET_GB", 0);
  return gb > 0 ? static_cast<int64_t>(gb * kGiB) : total_b;
}

// The GPU memory the driver charges this process (its CUDA context included), or -1 where the backend
// cannot tell (lx_rt_process_memory).
int64_t process_vram_bytes() {
  int64_t bytes = 0;
  return lx_rt_process_memory(&bytes) == 0 ? bytes : -1;
}

int64_t ram_budget_bytes() {
  const double gb = env_double("LX_RAM_BUDGET_GB", 0);
  return gb > 0 ? static_cast<int64_t>(gb * kGiB) : -1;
}

Placement plan_placement(const ExpertShape& shape, int64_t vram_free, int64_t ram_available,
                         const MachineProfile* machine) {
  Placement p;
  const int64_t all = static_cast<int64_t>(shape.n_layers) * shape.n_experts * shape.expert_bytes();
  p.vram_cache_bytes = std::max<int64_t>(0, std::min(all, vram_free));
  p.ram_bytes = ram_available < 0 || ram_available >= all ? -1 : ram_available;
  if (p.vram_cache_bytes < all) {
    // The CPU lane computes a miss from RAM without bringing the expert into VRAM, so the expert stays
    // cold and misses again: with most experts cached, read-through fills win. With fewer cached, the
    // GPU's misses crowd PCIe and the CPU's RAM bandwidth adds to it. Logos decode on the RTX 5070 Ti
    // with the lane (half the misses, 2 pairs a step) against without: 73% cached BF16 2% slower, 57%
    // FP8 1% slower; 43% FP8 2% faster, 31% 3% faster, 19% 5% faster. It takes a share of the misses
    // matching the two rates, cold experts first.
    const double resident = static_cast<double>(p.vram_cache_bytes) / static_cast<double>(all);
    if (machine && machine->cpu_expert_gbs > 0 && machine->zero_copy_gbs > 0 && resident < kCpuLaneBelow) {
      const double share = machine->cpu_expert_gbs / (machine->cpu_expert_gbs + machine->zero_copy_gbs);
      p.cpu_fraction = std::min(0.5, share);
      p.max_cpu_pairs = p.cpu_fraction > 0.05 ? 2 : 0;
    }
    p.notes.push_back(format("experts: %.0f%% in VRAM (%.2f of %.2f GB), the rest from %s", 100.0 * p.vram_cache_bytes / all,
                             p.vram_cache_bytes / 1e9, all / 1e9, p.ram_bytes < 0 ? "pinned RAM" : "pinned RAM and NVMe"));
  }
  if (const char* v = std::getenv("LX_OFFLOAD_CPU_FRACTION")) p.cpu_fraction = std::atof(v);
  if (const char* v = std::getenv("LX_OFFLOAD_CPU_PAIRS")) p.max_cpu_pairs = std::atoi(v);
  return p;
}

}  // namespace lx::offload
