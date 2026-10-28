// The runtime half of the backend ABI (kernels.h, "Runtime") on the CPU.
//
// "Device memory" is host memory. Large allocations are anonymous mappings (zeroed by the kernel, and on
// Linux backed by transparent huge pages, which keeps the TLB out of the way of a weight stream); small
// ones come from the heap. Every op runs to completion on the calling thread (with the pool), so a
// stream is an in-order queue whose work is always already done: syncs and events have nothing to wait
// for, and the order of calls is the order of execution. Graph capture has no meaning here and returns
// LX_ERR_NOT_SUPPORTED, so the model runs eagerly. A timed event records the clock when it is recorded,
// which is when the work issued before it finished.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <mutex>
#include <unordered_map>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif
#if defined(__x86_64__) || defined(_M_X64)
#include <cpuid.h>
#endif

#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

constexpr size_t kMapThreshold = size_t{1} << 20;  // allocations from 1 MB up are mapped
constexpr size_t kHuge = size_t{2} << 20;

std::mutex g_alloc_mu;
std::unordered_map<void*, size_t>& mapped() {
  static auto* m = new std::unordered_map<void*, size_t>;
  return *m;
}

thread_local int t_last_error = 0;

int alloc(void** ptr, int64_t bytes) {
  if (!ptr) return fail(kInvalid);
  *ptr = nullptr;
  if (bytes <= 0) return kOk;
  const size_t n = static_cast<size_t>(bytes);
  if (n < kMapThreshold) {
    void* p = nullptr;
    if (posix_memalign(&p, 64, n) != 0 || !p) return fail(kNoMemory);
    memset(p, 0, n);
    *ptr = p;
    return kOk;
  }
  // Rounded up to whole huge pages, so the kernel can back the range with them.
  const size_t len = (n + kHuge - 1) / kHuge * kHuge;
  void* p = mmap(nullptr, len, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (p == MAP_FAILED) return fail(kNoMemory);
#if defined(MADV_HUGEPAGE)
  madvise(p, len, MADV_HUGEPAGE);
#endif
  {
    std::lock_guard<std::mutex> g(g_alloc_mu);
    mapped()[p] = len;
  }
  *ptr = p;
  return kOk;
}

int release(void* ptr) {
  if (!ptr) return kOk;
  size_t len = 0;
  {
    std::lock_guard<std::mutex> g(g_alloc_mu);
    auto it = mapped().find(ptr);
    if (it != mapped().end()) {
      len = it->second;
      mapped().erase(it);
    }
  }
  if (len) munmap(ptr, len);
  else free(ptr);
  return kOk;
}

// Copies and fills of a few MB and up are split over the pool: one core cannot saturate the memory bus.
constexpr int64_t kParallelBytes = int64_t{4} << 20;
constexpr int64_t kCopyGrain = int64_t{1} << 20;

struct Stream {
  int id;
};
struct Event {
  bool timed = false;     // made by lx_rt_event_create_timed
  bool recorded = false;  // recorded at least once
  std::chrono::steady_clock::time_point at;
};

// The processor's name: the brand string on x86 and macOS, the first "model name" (or "Hardware") line of
// /proc/cpuinfo elsewhere on Linux, else "CPU".
void cpu_name(char* out, size_t capacity) {
  snprintf(out, capacity, "CPU");
#if defined(__x86_64__) || defined(_M_X64)
  unsigned regs[12] = {};
  if (__get_cpuid_max(0x80000000u, nullptr) >= 0x80000004u) {
    for (unsigned i = 0; i < 3; ++i)
      __get_cpuid(0x80000002u + i, &regs[4 * i], &regs[4 * i + 1], &regs[4 * i + 2], &regs[4 * i + 3]);
    char brand[49] = {};
    memcpy(brand, regs, 48);
    const char* b = brand;
    while (*b == ' ') ++b;
    if (*b) snprintf(out, capacity, "%s", b);
  }
#elif defined(__APPLE__)
  char brand[256] = {};
  size_t len = sizeof brand;
  if (sysctlbyname("machdep.cpu.brand_string", brand, &len, nullptr, 0) == 0 && brand[0])
    snprintf(out, capacity, "%s", brand);
#elif defined(__linux__)
  FILE* f = fopen("/proc/cpuinfo", "r");
  if (!f) return;
  char line[512];
  while (fgets(line, sizeof line, f)) {
    if (strncmp(line, "model name", 10) != 0 && strncmp(line, "Hardware", 8) != 0) continue;
    const char* v = strchr(line, ':');
    if (!v) continue;
    ++v;
    while (*v == ' ' || *v == '\t') ++v;
    snprintf(out, capacity, "%s", v);
    out[strcspn(out, "\n")] = 0;
    break;
  }
  fclose(f);
#endif
}

// The data (or unified) cache of one core at `level` (1, 2 or 3), in bytes, or 0 where the system does not
// say. On a hybrid chip it is the first core's (a performance core); an L3 is shared by every core.
int64_t cache_bytes(int level) {
#if defined(__APPLE__)
  // The performance cores' caches (an L2 is shared by a cluster). Apple lists no L3.
  const char* names[2][2] = {{"hw.perflevel0.l1dcachesize", "hw.l1dcachesize"},
                             {"hw.perflevel0.l2cachesize", "hw.l2cachesize"}};
  if (level < 1 || level > 2) return 0;
  for (const char* name : names[level - 1]) {
    int64_t v = 0;
    size_t len = sizeof v;
    if (sysctlbyname(name, &v, &len, nullptr, 0) == 0 && v > 0) return v;
  }
  return 0;
#elif defined(__linux__)
  for (int i = 0; i < 10; ++i) {
    char path[96], type[32] = {}, size[32] = {};
    int lv = 0;
    auto read = [&](const char* leaf, const char* fmt, void* out) {
      snprintf(path, sizeof path, "/sys/devices/system/cpu/cpu0/cache/index%d/%s", i, leaf);
      FILE* f = fopen(path, "r");
      if (!f) return false;
      const bool ok = fscanf(f, fmt, out) == 1;
      fclose(f);
      return ok;
    };
    if (!read("level", "%d", &lv)) break;
    if (lv != level || !read("type", "%31s", type) || !strcmp(type, "Instruction") || !read("size", "%31s", size))
      continue;
    char* unit = nullptr;
    const long v = strtol(size, &unit, 10);
    return static_cast<int64_t>(v) * (unit && *unit == 'M' ? 1 << 20 : unit && *unit == 'K' ? 1 << 10 : 1);
  }
  return 0;
#else
  (void)level;
  return 0;
#endif
}

// The read bandwidth the kernels see: every thread of the pool summing a buffer several times the last-level
// cache, the best of a few passes. Measured once per process, in a few tens of milliseconds.
int64_t read_bandwidth() {
  static const int64_t measured = [] {
    const int64_t bytes = std::min<int64_t>(int64_t{512} << 20, std::max<int64_t>(int64_t{128} << 20, 8 * cache_bytes(3)));
    void* buf = nullptr;
    if (alloc(&buf, bytes) != kOk) return int64_t{0};
    // Written first: untouched pages would all read the one zero page.
    uint64_t* w = static_cast<uint64_t*>(buf);
    const int64_t words = bytes / 8;
    parallel_for(words, grain_for(words, 1 << 16), [&](int64_t b, int64_t e) {
      for (int64_t i = b; i < e; ++i) w[i] = static_cast<uint64_t>(i);
    });
    std::atomic<uint64_t> sink{0};
    double best = 0;
    for (int pass = 0; pass < 5; ++pass) {
      const auto t0 = std::chrono::steady_clock::now();
      parallel_for(words, grain_for(words, 1 << 16, 1), [&](int64_t b, int64_t e) {
        uint64_t a[8] = {};
        int64_t i = b;
        for (; i + 8 <= e; i += 8)
          for (int j = 0; j < 8; ++j) a[j] += w[i + j];
        for (; i < e; ++i) a[0] += w[i];
        sink.fetch_add(a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7], std::memory_order_relaxed);
      });
      const double t = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (pass > 0 && (best == 0 || t < best)) best = t;  // the first pass also faults the pages in
    }
    release(buf);
    return best > 0 && sink.load() != 1 ? static_cast<int64_t>(static_cast<double>(bytes) / best) : int64_t{0};
  }();
  return measured;
}

}  // namespace

int fail(int status) {
  if (status != kOk) t_last_error = status;
  return status;
}

}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_rt_backend(void) { return LX_BACKEND_CPU; }
int lx_rt_device_count(void) { return 1; }
int lx_rt_set_device(int ordinal) { return ordinal == 0 ? kOk : fail(kBadDevice); }

int lx_rt_alloc(void** ptr, int64_t bytes) { return alloc(ptr, bytes); }
int lx_rt_free(void* ptr) { return release(ptr); }
int lx_rt_host_alloc(void** ptr, int64_t bytes) { return alloc(ptr, bytes); }
int lx_rt_host_free(void* ptr) { return release(ptr); }

// Host memory is device memory here, so a mapped block is one ordinary allocation whose address serves
// as both the host and the device pointer.
int lx_rt_host_alloc_mapped(void** host, void** device, int64_t bytes) {
  if (!device) return fail(kInvalid);
  *device = nullptr;
  if (const int status = alloc(host, bytes); status != kOk) return status;
  *device = *host;
  return kOk;
}

// Nothing to pin: the range is already where the kernels read it, so registering cannot fail.
int lx_rt_host_register(void* host, int64_t bytes, void** device) {
  if (!device || (!host && bytes > 0)) return fail(kInvalid);
  *device = host;
  return kOk;
}
int lx_rt_host_unregister(void* host) {
  (void)host;
  return kOk;
}

int lx_rt_memset(void* ptr, int value, int64_t bytes, void* stream) {
  (void)stream;
  if (bytes <= 0) return kOk;
  if (!ptr) return fail(kInvalid);
  char* p = static_cast<char*>(ptr);
  if (bytes < kParallelBytes) {
    memset(p, value, static_cast<size_t>(bytes));
    return kOk;
  }
  parallel_for(bytes, kCopyGrain, [&](int64_t b, int64_t e) { memset(p + b, value, static_cast<size_t>(e - b)); });
  return kOk;
}

// Every direction is the same copy here: host and device memory are one memory.
int lx_rt_copy(void* dst, const void* src, int64_t bytes, lx_copy_kind kind, void* stream) {
  (void)stream;
  if (bytes <= 0) return kOk;
  if (!dst || !src || kind < LX_COPY_H2D || kind > LX_COPY_DEFAULT) return fail(kInvalid);
  char* d = static_cast<char*>(dst);
  const char* s = static_cast<const char*>(src);
  if (bytes < kParallelBytes) {
    memmove(d, s, static_cast<size_t>(bytes));
    return kOk;
  }
  // Overlapping ranges keep memmove's meaning only when copied by one thread.
  if ((d < s + bytes) && (s < d + bytes)) {
    memmove(d, s, static_cast<size_t>(bytes));
    return kOk;
  }
  parallel_for(bytes, kCopyGrain, [&](int64_t b, int64_t e) { memcpy(d + b, s + b, static_cast<size_t>(e - b)); });
  return kOk;
}

int lx_rt_stream_create(void** stream) {
  if (!stream) return fail(kInvalid);
  *stream = new Stream{0};
  return kOk;
}
int lx_rt_stream_destroy(void* stream) {
  delete static_cast<Stream*>(stream);
  return kOk;
}
int lx_rt_stream_sync(void* stream) {
  (void)stream;
  return kOk;
}

int lx_rt_event_create(void** event) {
  if (!event) return fail(kInvalid);
  *event = new Event;
  return kOk;
}
int lx_rt_event_create_timed(void** event) {
  if (!event) return fail(kInvalid);
  Event* e = new Event;
  e->timed = true;
  *event = e;
  return kOk;
}
int lx_rt_event_destroy(void* event) {
  delete static_cast<Event*>(event);
  return kOk;
}
int lx_rt_event_record(void* event, void* stream) {
  (void)stream;
  if (!event) return fail(kBadHandle);
  Event* e = static_cast<Event*>(event);
  e->recorded = true;
  if (e->timed) e->at = std::chrono::steady_clock::now();
  return kOk;
}
// As on CUDA: both events must be timed and recorded.
int lx_rt_event_elapsed(void* start, void* stop, float* ms) {
  if (!ms) return fail(kInvalid);
  const Event* a = static_cast<const Event*>(start);
  const Event* b = static_cast<const Event*>(stop);
  if (!a || !b || !a->timed || !b->timed || !a->recorded || !b->recorded) return fail(kBadHandle);
  *ms = std::chrono::duration<float, std::milli>(b->at - a->at).count();
  return kOk;
}
int lx_rt_event_sync(void* event) {
  (void)event;
  return kOk;
}
// Streams are synchronous, so the event has already happened by the time anything waits on it.
int lx_rt_stream_wait(void* stream, void* event) {
  (void)stream;
  (void)event;
  return kOk;
}

// Every op has finished by the time it returns, on every stream.
int lx_rt_device_sync(void) { return kOk; }

// Declining a capture is the answer to a question the model asks, not a failure, so it does not set
// lx_rt_last_error (the model checks that after a run).
int lx_rt_capture_begin(void* stream) {
  (void)stream;
  return kNotSupported;
}
int lx_rt_capture_end(void* stream, void** graph) {
  (void)stream;
  if (graph) *graph = nullptr;
  return fail(kNotSupported);
}
int lx_rt_graph_launch(void* graph, void* stream) {
  (void)graph;
  (void)stream;
  return fail(kNotSupported);
}
int lx_rt_graph_destroy(void* graph) {
  (void)graph;
  return kOk;
}

// No persisting L2 on the CPU (kernels.h, lx_rt_l2_persist): the model runs unpinned. Declining is an answer the
// model asks for, not a failure, so it does not set lx_rt_last_error.
int lx_rt_l2_persist(const void* base, int64_t bytes, float hit_ratio, int64_t* granted_bytes) {
  (void)base;
  (void)bytes;
  (void)hit_ratio;
  if (granted_bytes) *granted_bytes = 0;
  return kNotSupported;
}

// Free and total memory of the machine, in bytes: what the offload planner sizes RAM tiers from. Free is
// the memory the kernel can hand out without reclaiming (MemAvailable on Linux, free pages on macOS).
int lx_rt_mem_info(int64_t* free_bytes, int64_t* total_bytes) {
  if (!free_bytes || !total_bytes) return fail(kInvalid);
#if defined(__APPLE__)
  uint64_t total = 0;
  size_t len = sizeof total;
  if (sysctlbyname("hw.memsize", &total, &len, nullptr, 0) != 0) return fail(kNotSupported);
  vm_statistics64_data_t vm{};
  mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
  if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vm), &count) !=
      KERN_SUCCESS)
    return fail(kNotSupported);
  *free_bytes = static_cast<int64_t>(vm.free_count) * static_cast<int64_t>(sysconf(_SC_PAGESIZE));
  *total_bytes = static_cast<int64_t>(total);
  return kOk;
#elif defined(__linux__)
  FILE* f = fopen("/proc/meminfo", "r");
  if (!f) return fail(kNotSupported);
  unsigned long long total_kb = 0, avail_kb = 0;
  int found = 0;
  char line[256];
  while (found < 2 && fgets(line, sizeof line, f)) {
    if (sscanf(line, "MemTotal: %llu kB", &total_kb) == 1) ++found;
    else if (sscanf(line, "MemAvailable: %llu kB", &avail_kb) == 1) ++found;
  }
  fclose(f);
  if (found < 2) return fail(kNotSupported);
  *free_bytes = static_cast<int64_t>(avail_kb) * 1024;
  *total_bytes = static_cast<int64_t>(total_kb) * 1024;
  return kOk;
#else
  return fail(kNotSupported);
#endif
}

// One device: this machine. Its compute capability is the ISA level (lx_arch's code before any override,
// major * 10 + minor), its multiprocessors the threads the kernels run on, and its bandwidth what they
// read at, measured. The memory clock and bus width are not something a process can read, so they are 0,
// as are the driver and runtime versions, which do not exist here.
int lx_rt_device_props(int ordinal, lx_device_props* props) {
  if (!props) return fail(kInvalid);
  if (ordinal != 0) return fail(kBadDevice);
  memset(props, 0, sizeof *props);
  cpu_name(props->name, sizeof props->name);
  const int level = real_level();
  props->cc_major = level / 10;
  props->cc_minor = level % 10;
  props->sms = threads();
  props->l1_bytes = cache_bytes(1);
  props->l2_bytes = cache_bytes(2);
  props->l3_bytes = cache_bytes(3);
  props->read_bandwidth = read_bandwidth();
  snprintf(props->isa, sizeof props->isa, "%s%s", level_isa(kern().code),
           amx_present() ? "; AMX tiles on the chip, which these kernels do not use" : "");
  props->native_math = native_math();
  return kOk;
}

// Device memory is host memory, so what this process holds is its resident memory: the physical footprint
// on macOS (what Activity Monitor shows), the resident set on Linux.
int lx_rt_process_memory(int64_t* bytes) {
  if (!bytes) return fail(kInvalid);
#if defined(__APPLE__)
  task_vm_info_data_t info{};
  mach_msg_type_number_t count = TASK_VM_INFO_COUNT;
  if (task_info(mach_task_self(), TASK_VM_INFO, reinterpret_cast<task_info_t>(&info), &count) != KERN_SUCCESS)
    return fail(kNotSupported);
  *bytes = static_cast<int64_t>(info.phys_footprint);
  return kOk;
#elif defined(__linux__)
  FILE* f = fopen("/proc/self/statm", "r");
  if (!f) return fail(kNotSupported);
  unsigned long long size = 0, resident = 0;
  const int n = fscanf(f, "%llu %llu", &size, &resident);
  fclose(f);
  if (n != 2) return fail(kNotSupported);
  *bytes = static_cast<int64_t>(resident) * static_cast<int64_t>(sysconf(_SC_PAGESIZE));
  return kOk;
#else
  return fail(kNotSupported);
#endif
}

// Nothing here is memory the device reads across a bus: the kernels read host memory directly.
int lx_rt_is_host_memory(const void* ptr) {
  (void)ptr;
  return 0;
}

// The CUDA runtime's text for the codes this library returns, and the tuner's two.
const char* lx_rt_error_string(int status) {
  switch (status) {
    case kOk: return "no error";
    case kInvalid: return "invalid argument";
    case kNoMemory: return "out of memory";
    case kBadDevice: return "invalid device ordinal";
    case kBadHandle: return "invalid resource handle";
    case kNotSupported: return "operation not supported";
    case LX_ERR_TUNE_REJECTED: return "the variant is outside the shape's order class";
    case LX_ERR_TUNE_VARIANT: return "the kernel has no such variant";
  }
  return "unrecognized error code";
}

int lx_rt_last_error(void) {
  const int e = t_last_error;
  t_last_error = 0;
  return e;
}

}  // extern "C"
