// The runtime half of the backend ABI (kernels.h, "Runtime") on the CPU.
//
// "Device memory" is host memory. Large allocations are anonymous mappings (zeroed by the kernel, and on
// Linux backed by transparent huge pages, which keeps the TLB out of the way of a weight stream); small
// ones come from the heap. Every op runs to completion on the calling thread (with the pool), so a
// stream is an in-order queue whose work is always already done: syncs and events have nothing to wait
// for, and the order of calls is the order of execution. Graph capture has no meaning here and returns
// LX_ERR_NOT_SUPPORTED, so the model runs eagerly.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

#include <mutex>
#include <unordered_map>

#if defined(__APPLE__)
#include <mach/mach.h>
#include <sys/sysctl.h>
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
  int id;
};

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

int lx_rt_copy(void* dst, const void* src, int64_t bytes, lx_copy_kind kind, void* stream) {
  (void)kind;
  (void)stream;
  if (bytes <= 0) return kOk;
  if (!dst || !src) return fail(kInvalid);
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
  *event = new Event{0};
  return kOk;
}
int lx_rt_event_destroy(void* event) {
  delete static_cast<Event*>(event);
  return kOk;
}
int lx_rt_event_record(void* event, void* stream) {
  (void)event;
  (void)stream;
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

int lx_rt_last_error(void) {
  const int e = t_last_error;
  t_last_error = 0;
  return e;
}

}  // extern "C"
