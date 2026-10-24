// The runtime half of the backend ABI (kernels.h, "Runtime") on Metal, and the device, registry and
// streams every op encodes through (backend.h has the design).
#include "cuda/tunable.h"
#include "metal/backend.h"

#include <IOKit/IOKitLib.h>
#include <mach/mach.h>
#include <mach/mach_vm.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <memory>

#include "metal/kernels/args.h"

extern const char* lx_metal_source();      // the kernel library, embedded at build time
extern const char* lx_metal_nax_source();  // the neural accelerator kernels (kernels/nax)

namespace lx::metal {
namespace {

// Dispatches per command buffer before it is committed, so the GPU starts while the host encodes on.
int flush_interval() {
  static const int n = [] {
    const char* env = std::getenv("LX_METAL_FLUSH");
    const int v = env ? std::atoi(env) : 0;
    return v > 0 ? v : 96;
  }();
  return n;
}

// LX_METAL_SERIAL=1: a serial encoder, every dispatch ordered after the one before (no range tracking).
bool serial() {
  static const bool v = [] {
    const char* env = std::getenv("LX_METAL_SERIAL");
    return env && env[0] && env[0] != '0';
  }();
  return v;
}

constexpr size_t kRingBytes = size_t{32} << 20;  // upload staging per stream
constexpr MTLResourceOptions kShared = MTLResourceStorageModeShared | MTLResourceHazardTrackingModeUntracked;

size_t page_size() {
  static const size_t p = static_cast<size_t>(getpagesize());
  return p;
}

// The end of the virtual memory region holding address a (0 if it is not mapped).
uintptr_t region_end(uintptr_t a, size_t limit) {
  // The kernel keeps a large anonymous allocation as several map entries (of 128 MB on macOS), so the
  // entries that follow without a gap and stay readable belong to the same range.
  uintptr_t end = 0;
  for (mach_vm_address_t at = a;;) {
    mach_vm_address_t addr = at;
    mach_vm_size_t size = 0;
    vm_region_basic_info_data_64_t info;
    mach_msg_type_number_t count = VM_REGION_BASIC_INFO_COUNT_64;
    mach_port_t object = MACH_PORT_NULL;
    if (mach_vm_region(mach_task_self(), &addr, &size, VM_REGION_BASIC_INFO_64,
                       reinterpret_cast<vm_region_info_t>(&info), &count, &object) != KERN_SUCCESS)
      break;
    if (addr > at || !(info.protection & VM_PROT_READ)) break;  // a hole, or memory the GPU cannot read
    end = static_cast<uintptr_t>(addr + size);
    if (end - a >= limit) return a + limit;
    at = end;
  }
  return end;
}

// Errors from command buffers that failed on the GPU, reported by the next sync and lx_rt_last_error.
std::atomic<int> g_async_error{0};

// LX_METAL_PROFILE=1: command buffers, dispatches and GPU busy time, printed at exit. =2 also commits
// every dispatch on its own and prints the GPU time per kernel (slower: for attribution, not timing).
// =3 keeps the command buffers but gives every dispatch its own encoder, ordered after the one before,
// and prints the GPU time per kernel from timestamps at the encoders' ends: in place, as the model runs
// (serialized, so independent kernels no longer overlap).
struct Profile {
  int level = 0;
  std::atomic<uint64_t> buffers{0}, dispatches{0}, gpu_ns{0}, barriers{0};
  std::mutex mu;
  std::unordered_map<std::string, std::pair<uint64_t, uint64_t>> kernels;  // name: (count, ns)
  double ns_per_tick = 1.0;  // GPU timestamp ticks (level 3), calibrated before printing
  std::function<double()> calibrate;

  static Profile* get() {
    static Profile* p = []() -> Profile* {
      const char* env = std::getenv("LX_METAL_PROFILE");
      if (!env || !*env || *env == '0') return nullptr;
      auto* q = new Profile();
      q->level = std::atoi(env);
      std::atexit([] { get()->print(); });
      return q;
    }();
    return p;
  }
  void add_kernel(const std::string& name, uint64_t ns) {
    std::lock_guard<std::mutex> g(mu);
    auto& k = kernels[name];
    k.first += 1;
    k.second += ns;
  }
  void print() {
    if (calibrate) ns_per_tick = calibrate();
    std::fprintf(stderr, "lernex metal: %llu command buffers, %llu dispatches, %llu barriers, GPU busy %.1f ms\n",
                 (unsigned long long)buffers.load(), (unsigned long long)dispatches.load(),
                 (unsigned long long)barriers.load(), gpu_ns.load() / 1e6);
    std::vector<std::pair<std::string, std::pair<uint64_t, uint64_t>>> rows(kernels.begin(), kernels.end());
    std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.second.second > b.second.second; });
    const double scale = level >= 3 ? ns_per_tick : 1.0;
    double total = 0;
    for (const auto& [name, k] : rows) {
      total += k.second * scale;
      std::fprintf(stderr, "  %-40s %8llu x %8.2f us = %9.2f ms\n", name.c_str(), (unsigned long long)k.first,
                   k.second * scale / 1e3 / double(k.first), k.second * scale / 1e6);
    }
    if (level >= 3) std::fprintf(stderr, "  kernels in all %.1f ms\n", total / 1e6);
  }
};
thread_local int t_last_error = 0;

int note(int status) {
  if (status != kOk) t_last_error = status;
  return status;
}

// Whether every dispatch is committed on its own (LX_METAL_PROFILE=2).
bool profile_each() {
  static const bool each = Profile::get() && Profile::get()->level == 2;
  return each;
}

// Whether every dispatch is timed in its own encoder (LX_METAL_PROFILE=3), where the GPU can.
constexpr size_t kTimedMax = 128;  // timed encoders per command buffer
bool profile_timed() {
  static const bool timed = [] {
    Profile* p = Profile::get();
    if (!p || p->level < 3) return false;
    id<MTLDevice> d = Device::get().mtl();
    if (![d supportsCounterSampling:MTLCounterSamplingPointAtStageBoundary]) {
      std::fprintf(stderr, "lernex metal: this GPU cannot time encoders; LX_METAL_PROFILE=3 counts only\n");
      return false;
    }
    MTLTimestamp cpu0 = 0, gpu0 = 0;
    [d sampleTimestamps:&cpu0 gpuTimestamp:&gpu0];
    p->calibrate = [d, cpu0, gpu0] {  // the CPU side of the pair counts nanoseconds
      MTLTimestamp cpu1 = 0, gpu1 = 0;
      [d sampleTimestamps:&cpu1 gpuTimestamp:&gpu1];
      return gpu1 > gpu0 ? double(cpu1 - cpu0) / double(gpu1 - gpu0) : 1.0;
    };
    return true;
  }();
  return timed;
}

// Sample buffers for the timed encoders, reused once their command buffer finished.
std::mutex g_samples_mu;
std::vector<id<MTLCounterSampleBuffer>> g_samples;

id<MTLCounterSampleBuffer> take_samples() {
  {
    std::lock_guard<std::mutex> g(g_samples_mu);
    if (!g_samples.empty()) {
      id<MTLCounterSampleBuffer> b = g_samples.back();
      g_samples.pop_back();
      return b;
    }
  }
  id<MTLDevice> d = Device::get().mtl();
  MTLCounterSampleBufferDescriptor* desc = [MTLCounterSampleBufferDescriptor new];
  for (id<MTLCounterSet> set in d.counterSets)
    if ([set.name isEqualToString:MTLCommonCounterSetTimestamp]) desc.counterSet = set;
  desc.storageMode = MTLStorageModeShared;
  desc.sampleCount = 2 * kTimedMax;
  NSError* err = nil;
  id<MTLCounterSampleBuffer> b = [d newCounterSampleBufferWithDescriptor:desc error:&err];
  if (!b) {
    std::fprintf(stderr, "lernex metal: no counter sample buffer: %s\n", err.localizedDescription.UTF8String);
    std::abort();
  }
  return b;
}

}  // namespace

// ---- Device ---------------------------------------------------------------------------------

Device& Device::get() {
  static Device* d = new Device();  // never destroyed: teardown order at exit is not ours to rely on
  return *d;
}

Device::Device() {
  device_ = MTLCreateSystemDefaultDevice();
  if (!device_) {
    std::fprintf(stderr, "lernex metal: no Metal device\n");
    return;
  }
  for (int n = 12; n >= 1; --n) {
    if ([device_ supportsFamily:static_cast<MTLGPUFamily>(MTLGPUFamilyApple1 + n - 1)]) {
      family_ = n;
      break;
    }
  }
  MTLResidencySetDescriptor* desc = [MTLResidencySetDescriptor new];
  desc.label = @"lernex";
  desc.initialCapacity = 512;
  NSError* err = nil;
  residency_ = [device_ newResidencySetWithDescriptor:desc error:&err];
  if (!residency_) std::fprintf(stderr, "lernex metal: no residency set (%s)\n", err.localizedDescription.UTF8String);
}

id<MTLLibrary> Device::build(const char* source, MTLLanguageVersion version, const char* what) {
  @autoreleasepool {
    MTLCompileOptions* opt = [MTLCompileOptions new];
    // IEEE semantics and precise transcendentals: the tests hold f32 kernels to 1e-5. Kernels opt into
    // the fast functions (fast::exp and friends) where the CUDA kernels use the intrinsics.
    opt.mathMode = MTLMathModeSafe;
    opt.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
    opt.languageVersion = version;
    NSError* err = nil;
    id<MTLLibrary> lib = [device_ newLibraryWithSource:@(source) options:opt error:&err];
    if (!lib) std::fprintf(stderr, "lernex metal: %s failed to compile:\n%s\n", what, err.localizedDescription.UTF8String);
    return lib;
  }
}

void Device::compile() {
  library_ = build(lx_metal_source(), MTLLanguageVersion3_2, "the kernel library");
  if (!library_) std::abort();
}

bool Device::nax() {
  static const bool wanted = [] {
    const char* env = std::getenv("LX_METAL_NAX");
    return !(env && env[0] == '0');
  }();
  if (!wanted || family_ < 10) return false;
  std::call_once(nax_compiled_, [this] {
    nax_library_ = build(lx_metal_nax_source(), MTLLanguageVersion4_0, "the neural accelerator library");
    if (!nax_library_) std::fprintf(stderr, "lernex metal: running the GEMMs on simdgroup matrices instead\n");
  });
  return nax_library_ != nil;
}

id<MTLComputePipelineState> Device::pipeline(const std::string& name, bool nax) {
  std::call_once(compiled_, [this] { compile(); });
  if (nax && !this->nax()) {
    std::fprintf(stderr, "lernex metal: %s needs the neural accelerator library\n", name.c_str());
    std::abort();
  }
  std::lock_guard<std::mutex> g(mu_);
  auto it = pipelines_.find(name);
  if (it != pipelines_.end()) return it->second;
  @autoreleasepool {
    id<MTLFunction> fn = [nax ? nax_library_ : library_ newFunctionWithName:@(name.c_str())];
    if (!fn) {
      std::fprintf(stderr, "lernex metal: no kernel named %s\n", name.c_str());
      std::abort();
    }
    MTLComputePipelineDescriptor* desc = [MTLComputePipelineDescriptor new];
    desc.computeFunction = fn;
    desc.label = @(name.c_str());
    NSError* err = nil;
    id<MTLComputePipelineState> ps = [device_ newComputePipelineStateWithDescriptor:desc
                                                                            options:MTLPipelineOptionNone
                                                                         reflection:nil
                                                                              error:&err];
    if (!ps) {
      std::fprintf(stderr, "lernex metal: pipeline %s: %s\n", name.c_str(), err.localizedDescription.UTF8String);
      std::abort();
    }
    pipelines_.emplace(name, ps);
    return ps;
  }
}

id<MTLCommandQueue> Device::new_queue() {
  id<MTLCommandQueue> q = [device_ newCommandQueueWithMaxCommandBufferCount:64];
  if (residency_) [q addResidencySet:residency_];
  std::lock_guard<std::mutex> g(mu_);
  queues_.push_back(q);
  return q;
}

Stream* Device::default_stream() {
  static std::once_flag once;
  std::call_once(once, [this] { default_stream_ = new Stream(new_queue()); });
  return default_stream_;
}

int Device::alloc(void** ptr, size_t bytes) {
  *ptr = nullptr;
  if (bytes == 0) return kOk;
  if (!device_) return kNotSupported;
  // Whole pages, so a buffer never shares a page with another allocation.
  const size_t rounded = (bytes + page_size() - 1) / page_size() * page_size();
  if (rounded > device_.maxBufferLength) return kOutOfMemory;
  id<MTLBuffer> b = [device_ newBufferWithLength:rounded options:kShared];
  if (!b) return kOutOfMemory;
  std::memset(b.contents, 0, rounded);
  const auto base = reinterpret_cast<uintptr_t>(b.contents);
  {
    std::lock_guard<std::mutex> g(mu_);
    const auto at = std::lower_bound(entries_.begin(), entries_.end(), base,
                                     [](const auto& e, uintptr_t v) { return e.first < v; });
    entries_.insert(at, {base, Entry{b, rounded, true}});
    [residency_ addAllocation:b];
    [residency_ commit];
  }
  *ptr = b.contents;
  return kOk;
}

int Device::map_host(void* ptr, size_t bytes) {
  if (!ptr || bytes == 0) return kInvalidValue;
  if (!device_) return kNotSupported;
  const uintptr_t lo = reinterpret_cast<uintptr_t>(ptr) & ~(page_size() - 1);
  const uintptr_t hi = (reinterpret_cast<uintptr_t>(ptr) + bytes + page_size() - 1) & ~(page_size() - 1);
  if (hi - lo > device_.maxBufferLength) return kOutOfMemory;
  id<MTLBuffer> b = [device_ newBufferWithBytesNoCopy:reinterpret_cast<void*>(lo) length:hi - lo options:kShared
                                          deallocator:nil];
  if (!b) return kInvalidValue;
  std::lock_guard<std::mutex> g(mu_);
  const auto at = std::lower_bound(entries_.begin(), entries_.end(), lo,
                                   [](const auto& e, uintptr_t v) { return e.first < v; });
  entries_.insert(at, {lo, Entry{b, hi - lo, false}});
  [residency_ addAllocation:b];
  [residency_ commit];
  return kOk;
}

// Drops the registration holding `key`. Like cudaFree this waits for every stream first, so a
// buffer still in flight stays valid until the GPU is done with it.
int Device::release(uintptr_t key, bool owned) {
  std::vector<Stream*> streams;
  {
    std::lock_guard<std::mutex> g(mu_);
    streams = streams_;
  }
  for (Stream* s : streams) s->wait();
  std::lock_guard<std::mutex> g(mu_);
  for (auto it = entries_.begin(); it != entries_.end(); ++it) {
    const uintptr_t base = it->first;
    if (key >= base && key < base + it->second.bytes && it->second.owned == owned) {
      [residency_ removeAllocation:it->second.buffer];
      [residency_ commit];
      entries_.erase(it);
      return kOk;
    }
  }
  return kInvalidValue;
}

int Device::free(void* ptr) { return ptr ? release(reinterpret_cast<uintptr_t>(ptr), true) : kOk; }

int Device::unmap_host(void* ptr) {
  if (!ptr) return kOk;
  return release(reinterpret_cast<uintptr_t>(ptr) & ~(page_size() - 1), false);
}

int Device::sync_all() {
  std::vector<Stream*> streams;
  {
    std::lock_guard<std::mutex> g(mu_);
    streams = streams_;
  }
  int first = kOk;
  for (Stream* s : streams) {
    const int e = s->capturing() ? kOk : s->wait();
    if (first == kOk) first = e;
  }
  return first;
}

void Device::add_stream(Stream* s) {
  std::lock_guard<std::mutex> g(mu_);
  streams_.push_back(s);
}

void Device::remove_stream(Stream* s) {
  std::lock_guard<std::mutex> g(mu_);
  streams_.erase(std::remove(streams_.begin(), streams_.end(), s), streams_.end());
}

id<MTLBuffer> Device::lookup(const void* p, size_t bytes, uintptr_t* base) {
  const auto a = reinterpret_cast<uintptr_t>(p);
  std::lock_guard<std::mutex> g(mu_);
  // The last entries starting at or below p; registrations may overlap (two tensors on one page),
  // so walk back until one holds the whole range.
  auto it = std::upper_bound(entries_.begin(), entries_.end(), a,
                             [](uintptr_t v, const auto& e) { return v < e.first; });
  for (int steps = 0; it != entries_.begin() && steps < 8; ++steps) {
    --it;
    if (a + bytes <= it->first + it->second.bytes) {
      *base = it->first;
      return it->second.buffer;
    }
  }
  return nil;
}

size_t weight_data_bytes(lx_wtype type, int64_t N, int64_t K) {
  switch (type) {
    case LX_W_BF16: return static_cast<size_t>(N * K * 2);
    case LX_W_INT8:
    case LX_W_FP8: return static_cast<size_t>(N * K);
    case LX_W_NVFP4: return static_cast<size_t>(N * (K / 2));
    case LX_W_BF16X: break;  // not run on Metal
  }
  return 0;
}

size_t weight_scale_bytes(lx_wtype type, int64_t N, int64_t K) {
  switch (type) {
    case LX_W_BF16: return 0;
    case LX_W_INT8:
    case LX_W_FP8: return static_cast<size_t>(N * 4);
    case LX_W_NVFP4: return static_cast<size_t>(N * (K / 16));
    case LX_W_BF16X: break;
  }
  return 0;
}

// ---- Stream ---------------------------------------------------------------------------------

Stream::Stream(id<MTLCommandQueue> queue) : queue_(queue) {
  timeline_ = [Device::get().mtl() newSharedEvent];
  Device::get().add_stream(this);
}

Stream::~Stream() {
  wait();
  Device::get().remove_stream(this);
  if (ring_) Device::get().free(ring_.contents);
}

void Stream::open() {
  if (cb_) return;
  cb_ = [queue_ commandBufferWithUnretainedReferences];
  // Untracked buffers: nothing orders this command buffer after the previous one but this wait.
  if (committed_ > 0) [cb_ encodeWaitForEvent:timeline_ value:committed_];
  for (const auto& [event, value] : waits_) [cb_ encodeWaitForEvent:event value:value];
  waits_.clear();
  enc_ = [cb_ computeCommandEncoderWithDispatchType:serial() ? MTLDispatchTypeSerial : MTLDispatchTypeConcurrent];
  bound_ = nil;
  // A new command buffer starts after the last one finished, so nothing before it is a hazard.
  reads_.clear();
  writes_.clear();
  barrier_next_ = false;
}

bool Stream::hazard() {
  bool need = false;
  for (const Access& t : taking_) {
    for (const Access& w : writes_) need = need || (t.lo < w.hi && w.lo < t.hi);
    if (t.write)
      for (const Access& r : reads_) need = need || (t.lo < r.hi && r.lo < t.hi);
    if (need) break;
  }
  if (need) {
    reads_.clear();
    writes_.clear();
  }
  for (const Access& t : taking_) (t.write ? writes_ : reads_).push_back(t);
  last_ = std::move(taking_);
  taking_.clear();
  return need;
}

uint64_t Stream::address(const void* p, size_t bytes, bool write) {
  if (!p) return 0;
  const auto a = reinterpret_cast<uintptr_t>(p);
  const bool track = !serial();
  uintptr_t base = 0;
  if (id<MTLBuffer> b = Device::get().lookup(p, bytes == kUnknownExtent || bytes == 0 ? 1 : bytes, &base)) {
    if (track) taking_.push_back(Access{a, bytes == kUnknownExtent ? base + b.length : a + std::max<size_t>(bytes, 1), write});
    return b.gpuAddress + (a - base);
  }
  // Unregistered host memory: wrap its pages for this command buffer, reusing a wrap that covers it.
  const uintptr_t lo = a & ~(page_size() - 1);
  uintptr_t hi = 0;
  if (bytes == kUnknownExtent) {
    hi = region_end(a, static_cast<size_t>([Device::get().mtl() maxBufferLength]) - (a - lo));
    hi = (hi + page_size() - 1) & ~(page_size() - 1);
    if (!hi) {
      error_ = note(kInvalidValue);
      return 0;
    }
  } else {
    hi = (a + std::max<size_t>(bytes, 1) + page_size() - 1) & ~(page_size() - 1);
  }
  if (track) taking_.push_back(Access{a, bytes == kUnknownExtent ? hi : a + std::max<size_t>(bytes, 1), write});
  for (const Wrap& w : wraps_)
    if (lo >= w.lo && hi <= w.hi) return w.buffer.gpuAddress + (a - w.lo);
  id<MTLBuffer> b = [Device::get().mtl() newBufferWithBytesNoCopy:reinterpret_cast<void*>(lo) length:hi - lo
                                                          options:kShared deallocator:nil];
  if (!b) {
    error_ = note(kInvalidValue);
    return 0;
  }
  Device::get().count_transient();
  if (graph_) {
    graph_->keep.push_back(b);
  } else {
    open();
    [enc_ useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  wraps_.push_back(Wrap{lo, hi, b});
  return b.gpuAddress + (a - lo);
}

static bool skipped(id<MTLComputePipelineState> pipeline) {
  static const char* env = std::getenv("LX_METAL_SKIP");
  if (!env || !*env || !pipeline.label) return false;
  const bool invert = env[0] == '!';
  const std::string name = pipeline.label.UTF8String, list = env + (invert ? 1 : 0);
  size_t at = 0;
  while (at <= list.size()) {
    size_t end = list.find(',', at);
    if (end == std::string::npos) end = list.size();
    const std::string want = list.substr(at, end - at);
    if (!want.empty() && name.rfind(want, 0) == 0) return !invert;
    at = end + 1;
  }
  return invert;
}

void Stream::dispatch(id<MTLComputePipelineState> pipeline, const void* args, size_t args_bytes, MTLSize groups,
                      MTLSize threads, NSUInteger threadgroup_bytes) {
  if (skipped(pipeline)) {
    if (!serial()) hazard();
    return;
  }
  dispatch_one(pipeline, args, args_bytes, groups, threads, threadgroup_bytes);
}

void Stream::timed_encoder(id<MTLComputePipelineState> pipeline, const std::vector<id<MTLBuffer>>* keep) {
  if (timed_.size() >= kTimedMax) {
    flush();
    open();
    if (keep)
      for (id<MTLBuffer> b : *keep) [enc_ useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  }
  if (!samples_) samples_ = take_samples();
  if (!fence_) fence_ = [Device::get().mtl() newFence];
  [enc_ updateFence:fence_];
  [enc_ endEncoding];
  MTLComputePassDescriptor* d = [MTLComputePassDescriptor computePassDescriptor];
  d.dispatchType = MTLDispatchTypeSerial;
  d.sampleBufferAttachments[0].sampleBuffer = samples_;
  d.sampleBufferAttachments[0].startOfEncoderSampleIndex = 2 * timed_.size();
  d.sampleBufferAttachments[0].endOfEncoderSampleIndex = 2 * timed_.size() + 1;
  enc_ = [cb_ computeCommandEncoderWithDescriptor:d];
  [enc_ waitForFence:fence_];
  for (const Wrap& w : wraps_) [enc_ useResource:w.buffer usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  if (keep)
    for (id<MTLBuffer> b : *keep) [enc_ useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  bound_ = nil;
  timed_.push_back(pipeline.label ? pipeline.label.UTF8String : "?");
}

void Stream::dispatch_one(id<MTLComputePipelineState> pipeline, const void* args, size_t args_bytes, MTLSize groups,
                          MTLSize threads, NSUInteger threadgroup_bytes) {
  if (graph_) {
    Command c;
    c.pipeline = pipeline;
    c.args.assign(static_cast<const uint8_t*>(args), static_cast<const uint8_t*>(args) + args_bytes);
    c.groups = groups;
    c.threads = threads;
    c.threadgroup_bytes = threadgroup_bytes;
    c.barrier = !serial() && hazard();
    graph_->commands.push_back(std::move(c));
    return;
  }
  open();
  const bool barrier = !serial() && (hazard() || barrier_next_);
  if (profile_timed()) {
    timed_encoder(pipeline, nullptr);
  } else if (barrier) {
    [enc_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (Profile* p = Profile::get()) p->barriers += 1;
  }
  barrier_next_ = false;
  if (pipeline != bound_) {
    [enc_ setComputePipelineState:pipeline];
    bound_ = pipeline;
  }
  [enc_ setBytes:args length:args_bytes atIndex:0];
  if (threadgroup_bytes) [enc_ setThreadgroupMemoryLength:(threadgroup_bytes + 15) / 16 * 16 atIndex:0];
  [enc_ dispatchThreadgroups:groups threadsPerThreadgroup:threads];
  if (++pending_ >= flush_interval() || profile_each()) flush();
}

uint64_t Stream::flush() {
  if (!cb_) return committed_;
  [enc_ endEncoding];
  const uint64_t value = ++committed_;
  [cb_ encodeSignalEvent:timeline_ value:value];
  Profile* profile = Profile::get();
  const int dispatches = pending_;
  NSString* kernel = profile && profile->level == 2 && bound_ ? bound_.label : nil;
  id<MTLCounterSampleBuffer> samples = samples_;
  auto timed = std::make_shared<std::vector<std::string>>(std::move(timed_));
  samples_ = nil;
  timed_.clear();
  [cb_ addCompletedHandler:^(id<MTLCommandBuffer> b) {
    if (profile) {
      const auto ns = b.GPUStartTime > 0 && b.GPUEndTime > b.GPUStartTime
                          ? static_cast<uint64_t>((b.GPUEndTime - b.GPUStartTime) * 1e9)
                          : 0;
      profile->buffers += 1;
      profile->dispatches += static_cast<uint64_t>(dispatches);
      profile->gpu_ns += ns;
      if (kernel && dispatches == 1) profile->add_kernel(kernel.UTF8String, ns);
    }
    if (samples) {
      NSData* data = [samples resolveCounterRange:NSMakeRange(0, 2 * timed->size())];
      const auto* ts = static_cast<const MTLCounterResultTimestamp*>(data.bytes);
      for (size_t i = 0; ts && i < timed->size(); ++i)
        if (ts[2 * i + 1].timestamp > ts[2 * i].timestamp && ts[2 * i].timestamp != MTLCounterErrorValue)
          profile->add_kernel((*timed)[i], ts[2 * i + 1].timestamp - ts[2 * i].timestamp);
      std::lock_guard<std::mutex> g(g_samples_mu);
      g_samples.push_back(samples);
    }
    if (b.status == MTLCommandBufferStatusError) {
      std::fprintf(stderr, "lernex metal: command buffer failed: %s\n", b.error.localizedDescription.UTF8String);
      g_async_error.store(kLaunchFailure);
    }
  }];
  [cb_ commit];
  if (!wraps_.empty()) retiring_.emplace_back(value, std::move(wraps_));
  wraps_.clear();
  if (ring_) ring_marks_.emplace_back(value, ring_head_);
  last_cb_ = cb_;
  cb_ = nil;
  enc_ = nil;
  bound_ = nil;
  pending_ = 0;
  return value;
}

bool Stream::done(uint64_t value) const { return timeline_.signaledValue >= value; }

int Stream::wait_for(id<MTLSharedEvent> event, uint64_t value) {
  if (event == timeline_ || value == 0 || event.signaledValue >= value) return kOk;  // already ordered, or done
  if (graph_) return note(kNotSupported);  // a recording is one stream's work
  // The work so far goes now; the next command buffer, which holds whatever comes after, waits.
  flush();
  waits_.emplace_back(event, value);
  return kOk;
}

int Stream::wait(uint64_t value) {
  if (value == 0) value = flush();
  if (value > 0 && timeline_.signaledValue < value) {
    if (![timeline_ waitUntilSignaledValue:value timeoutMS:600000]) return note(kLaunchFailure);
  }
  retire();
  const int async = g_async_error.exchange(0);
  return note(async);
}

void Stream::retire() {
  const uint64_t reached = timeline_.signaledValue;
  retiring_.erase(std::remove_if(retiring_.begin(), retiring_.end(),
                                 [reached](const auto& r) { return r.first <= reached; }),
                  retiring_.end());
}

int Stream::upload(void* dst, const void* src, size_t bytes) {
  static Kernel copy_kernel("copy_bytes");
  if (graph_) return note(kNotSupported);  // a host source cannot be captured (as in CUDA)
  if (!ring_) {
    void* p = nullptr;
    if (int e = Device::get().alloc(&p, kRingBytes)) return note(e);
    uintptr_t base = 0;
    ring_ = Device::get().lookup(p, kRingBytes, &base);
  }
  auto* d = static_cast<uint8_t*>(dst);
  const auto* s = static_cast<const uint8_t*>(src);
  while (bytes > 0) {
    const size_t chunk = std::min(bytes, kRingBytes / 2);
    size_t at = (ring_head_ + 15) & ~size_t{15};
    if (at + chunk > kRingBytes) {
      // The ring is full: everything that read it must finish before it is reused from the start.
      wait();
      ring_marks_.clear();
      at = 0;
    }
    std::memcpy(static_cast<uint8_t*>(ring_.contents) + at, s, chunk);
    ring_head_ = at + chunk;
    CopyArgs a{ring_.gpuAddress + at, address(d, chunk, true), static_cast<int64_t>(chunk)};
    const int64_t units = static_cast<int64_t>((chunk + 15) / 16);
    copy_kernel.launch(this, a, size1(ceil_div(units, 256)), size1(256));
    d += chunk;
    s += chunk;
    bytes -= chunk;
  }
  return error_;
}

int Stream::begin_capture() {
  if (graph_) return note(kInvalidValue);
  graph_ = new Graph();
  // The graph's barriers are among its own commands; the eager ranges wait until capture ends.
  saved_reads_ = std::move(reads_);
  saved_writes_ = std::move(writes_);
  reads_.clear();
  writes_.clear();
  taking_.clear();
  return kOk;
}

Graph* Stream::end_capture() {
  Graph* g = graph_;
  graph_ = nullptr;
  reads_ = std::move(saved_reads_);
  writes_ = std::move(saved_writes_);
  return g;
}

void Stream::replay(const Graph& graph) {
  // The replay is ordered after everything encoded before it, and whatever follows after all of it.
  const bool after = pending_ > 0;
  open();
  for (id<MTLBuffer> b : graph.keep) [enc_ useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
  bool first = true;
  // A step usually starts on an idle GPU: the first few commands go at once, the rest in the usual chunks.
  int limit = std::min(flush_interval(), 16);
  for (const Command& c : graph.commands) {
    if (profile_timed()) {
      timed_encoder(c.pipeline, &graph.keep);
    } else if (!serial() && (c.barrier || (first && after))) {
      [enc_ memoryBarrierWithScope:MTLBarrierScopeBuffers];
      if (Profile* p = Profile::get()) p->barriers += 1;
    }
    first = false;
    if (c.pipeline != bound_) {
      [enc_ setComputePipelineState:c.pipeline];
      bound_ = c.pipeline;
    }
    [enc_ setBytes:c.args.data() length:c.args.size() atIndex:0];
    if (c.threadgroup_bytes) [enc_ setThreadgroupMemoryLength:(c.threadgroup_bytes + 15) / 16 * 16 atIndex:0];
    [enc_ dispatchThreadgroups:c.groups threadsPerThreadgroup:c.threads];
    if (++pending_ >= limit || profile_each()) {
      flush();
      open();
      for (id<MTLBuffer> b : graph.keep) [enc_ useResource:b usage:MTLResourceUsageRead | MTLResourceUsageWrite];
      limit = flush_interval();
    }
  }
  reads_.clear();
  writes_.clear();
  taking_.clear();
  barrier_next_ = true;
}

}  // namespace lx::metal

// ---- The ABI --------------------------------------------------------------------------------

using namespace lx::metal;

namespace {

struct Event {
  Stream* stream = nullptr;
  uint64_t value = 0;
  // Timed events (lx_rt_event_create_timed): the command buffer committed with the event, whose GPU end
  // time is when the stream reached it.
  bool timed = false;
  id<MTLCommandBuffer> buffer = nil;
};

// The number of GPU cores, from the GPU's IORegistry entry (0 when it does not say).
int gpu_cores() {
  int cores = 0;
  io_service_t gpu = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
  if (!gpu) return 0;
  if (CFTypeRef v = IORegistryEntryCreateCFProperty(gpu, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0)) {
    if (CFGetTypeID(v) == CFNumberGetTypeID()) CFNumberGetValue(static_cast<CFNumberRef>(v), kCFNumberIntType, &cores);
    CFRelease(v);
  }
  IOObjectRelease(gpu);
  return cores;
}

// The memory of Apple's chips, which no API reports: transfers per second (MT/s) and bus width, by chip
// name, and for the Max chips whose bus width follows their GPU's size, by core count. Unknown chips get
// zeros (the bandwidth is then reported as unknown).
void chip_memory(const std::string& name, int cores, int64_t* mts, int* bus_bits) {
  struct Chip {
    const char* name;
    int min_cores;  // the row applies from this many GPU cores up
    int64_t mts;
    int bus;
  };
  static const Chip chips[] = {
      {"Apple M1", 0, 4266, 128},          {"Apple M1 Pro", 0, 6400, 256},     {"Apple M1 Max", 0, 6400, 512},
      {"Apple M1 Ultra", 0, 6400, 1024},   {"Apple M2", 0, 6400, 128},         {"Apple M2 Pro", 0, 6400, 256},
      {"Apple M2 Max", 0, 6400, 512},      {"Apple M2 Ultra", 0, 6400, 1024},  {"Apple M3", 0, 6400, 128},
      {"Apple M3 Pro", 0, 6400, 192},      {"Apple M3 Max", 0, 6400, 384},     {"Apple M3 Max", 40, 6400, 512},
      {"Apple M3 Ultra", 0, 6400, 1024},   {"Apple M4", 0, 7500, 128},         {"Apple M4 Pro", 0, 8533, 256},
      {"Apple M4 Max", 0, 8533, 384},      {"Apple M4 Max", 40, 8533, 512},    {"Apple M5", 0, 9600, 128},
  };
  *mts = 0;
  *bus_bits = 0;
  for (const Chip& c : chips) {
    if (name == c.name && cores >= c.min_cores) {
      *mts = c.mts;
      *bus_bits = c.bus;
    }
  }
}

}  // namespace

extern "C" {

int lx_rt_backend(void) { return LX_BACKEND_METAL; }

int lx_rt_device_count(void) { return Device::get().mtl() ? 1 : 0; }

int lx_rt_set_device(int ordinal) { return note(ordinal == 0 && Device::get().mtl() ? kOk : kInvalidValue); }

int lx_rt_alloc(void** ptr, int64_t bytes) {
  if (!ptr || bytes < 0) return note(kInvalidValue);
  return note(Device::get().alloc(ptr, static_cast<size_t>(bytes)));
}

int lx_rt_free(void* ptr) { return note(Device::get().free(ptr)); }

// Unified memory: "pinned host memory" is a registered shared buffer, which kernels may also read.
int lx_rt_host_alloc(void** ptr, int64_t bytes) { return lx_rt_alloc(ptr, bytes); }
int lx_rt_host_free(void* ptr) { return lx_rt_free(ptr); }

int lx_rt_memset(void* ptr, int value, int64_t bytes, void* stream) {
  static Kernel fill("fill_bytes");
  if (bytes <= 0) return kOk;
  Stream* s = stream_of(stream);
  FillArgs a{s->address(ptr, static_cast<size_t>(bytes), true), bytes, static_cast<uint32_t>(value & 0xff)};
  return note(fill.launch(s, a, size1(ceil_div((bytes + 15) / 16, 256)), size1(256)));
}

int lx_rt_copy(void* dst, const void* src, int64_t bytes, lx_copy_kind kind, void* stream) {
  static Kernel copy_kernel("copy_bytes");
  if (bytes <= 0) return kOk;
  if (!dst || !src) return note(kInvalidValue);
  Stream* s = stream_of(stream);
  switch (kind) {
    case LX_COPY_H2D: return note(s->upload(dst, src, static_cast<size_t>(bytes)));
    case LX_COPY_D2H: {
      // The GPU's writes are visible to the host once the stream has drained up to here.
      if (s->capturing()) return note(kNotSupported);
      if (int e = s->wait()) return e;
      std::memcpy(dst, src, static_cast<size_t>(bytes));
      return kOk;
    }
    case LX_COPY_D2D: {
      CopyArgs a{s->address(src, static_cast<size_t>(bytes)), s->address(dst, static_cast<size_t>(bytes), true), bytes};
      return note(copy_kernel.launch(s, a, size1(ceil_div((bytes + 15) / 16, 256)), size1(256)));
    }
    case LX_COPY_DEFAULT: {
      // Device memory is what the registry holds; anything else is host memory.
      uintptr_t base = 0;
      const bool dst_device = Device::get().lookup(dst, static_cast<size_t>(bytes), &base) != nil;
      const bool src_device = Device::get().lookup(src, static_cast<size_t>(bytes), &base) != nil;
      const lx_copy_kind k = !dst_device ? LX_COPY_D2H : src_device ? LX_COPY_D2D : LX_COPY_H2D;
      return lx_rt_copy(dst, src, bytes, k, stream);
    }
  }
  return note(kInvalidValue);
}

int lx_rt_stream_create(void** stream) {
  if (!stream) return note(kInvalidValue);
  *stream = new Stream(Device::get().new_queue());
  return kOk;
}

int lx_rt_stream_destroy(void* stream) {
  if (!stream) return kOk;
  delete static_cast<Stream*>(stream);
  return kOk;
}

int lx_rt_stream_sync(void* stream) {
  Stream* s = stream_of(stream);
  if (s->capturing()) return note(kNotSupported);
  if (int e = s->take_error()) return note(e);
  return s->wait();
}

int lx_rt_event_create(void** event) {
  if (!event) return note(kInvalidValue);
  *event = new Event();
  return kOk;
}

int lx_rt_event_destroy(void* event) {
  delete static_cast<Event*>(event);
  return kOk;
}

int lx_rt_event_record(void* event, void* stream) {
  static Kernel mark("fill_bytes");
  auto* e = static_cast<Event*>(event);
  if (!e) return note(kInvalidValue);
  e->stream = stream_of(stream);
  e->buffer = nil;
  if (e->timed && !e->stream->capturing()) {
    // An empty fill goes in the command buffer the event closes, so there is one even on an idle stream:
    // its end on the GPU's clock is the event's time.
    FillArgs a{0, 0, 0};
    if (int err = mark.launch(e->stream, a, size1(1), size1(1))) return note(err);
    e->value = e->stream->flush();
    e->buffer = e->stream->last_buffer();
    return kOk;
  }
  e->value = e->stream->flush();
  return kOk;
}

int lx_rt_event_sync(void* event) {
  auto* e = static_cast<Event*>(event);
  if (!e) return note(kInvalidValue);
  if (!e->stream || e->value == 0) return kOk;
  return e->stream->wait(e->value);
}

// Capture records the stream's dispatches (argument structs included) instead of running them; a
// launch re-encodes the recording, skipping every op's host-side logic.
int lx_rt_capture_begin(void* stream) { return note(stream_of(stream)->begin_capture()); }

int lx_rt_capture_end(void* stream, void** graph) {
  if (!graph) return note(kInvalidValue);
  *graph = stream_of(stream)->end_capture();
  return *graph ? kOk : note(kInvalidValue);
}

int lx_rt_graph_launch(void* graph, void* stream) {
  if (!graph) return note(kInvalidValue);
  Stream* s = stream_of(stream);
  s->replay(*static_cast<Graph*>(graph));
  return note(s->error());
}

int lx_rt_graph_destroy(void* graph) {
  delete static_cast<Graph*>(graph);
  return kOk;
}

int lx_rt_last_error(void) {
  int e = t_last_error;
  t_last_error = 0;
  if (e == 0) e = g_async_error.exchange(0);
  return e;
}

int lx_rt_event_create_timed(void** event) {
  if (!event) return note(kInvalidValue);
  auto* e = new Event();
  e->timed = true;
  *event = e;
  return kOk;
}

int lx_rt_event_elapsed(void* start, void* stop, float* ms) {
  auto* a = static_cast<Event*>(start);
  auto* b = static_cast<Event*>(stop);
  if (!a || !b || !ms || !a->buffer || !b->buffer) return note(kInvalidValue);
  if (int e = lx_rt_event_sync(a)) return e;
  if (int e = lx_rt_event_sync(b)) return e;
  // The timeline is signalled from inside the buffer, a little before the buffer has its times.
  [a->buffer waitUntilCompleted];
  [b->buffer waitUntilCompleted];
  *ms = static_cast<float>((b->buffer.GPUEndTime - a->buffer.GPUEndTime) * 1e3);
  return kOk;
}

int lx_rt_device_sync(void) { return note(Device::get().sync_all()); }

const char* lx_rt_error_string(int status) {
  switch (status) {
    case kOk: return "no error";
    case kInvalidValue: return "invalid argument";
    case kOutOfMemory: return "out of memory";
    case kNotSupported: return "operation not supported on Metal";
    case kLaunchFailure: return "a command buffer failed on the GPU";
  }
  thread_local char text[48];
  std::snprintf(text, sizeof text, "error %d", status);
  return text;
}

// Unified memory: nothing the GPU reads is across a bus.
int lx_rt_is_host_memory(const void* ptr) { return 0; }

int lx_rt_device_props(int ordinal, lx_device_props* props) {
  id<MTLDevice> d = Device::get().mtl();
  if (!props || ordinal != 0 || !d) return note(kInvalidValue);
  *props = lx_device_props{};
  std::snprintf(props->name, sizeof props->name, "%s", d.name.UTF8String);
  const int arch = 1000 + Device::get().family();  // lx_arch without an override
  props->cc_major = arch / 10;
  props->cc_minor = arch % 10;
  props->sms = gpu_cores();
  props->l2_bytes = 0;  // not published for Apple GPUs
  int64_t mts = 0;
  chip_memory(d.name.UTF8String, props->sms, &mts, &props->bus_bits);
  props->mem_clock_khz = mts * 1000 / 2;  // two transfers per clock
  // The driver ships with the OS: its version. The runtime: the Metal version the GPU supports.
  const NSOperatingSystemVersion os = NSProcessInfo.processInfo.operatingSystemVersion;
  props->driver_version = static_cast<int>(1000 * os.majorVersion + 10 * os.minorVersion);
  props->runtime_version = [d supportsFamily:MTLGPUFamilyMetal4] ? 4000 : 3020;
  return kOk;
}

// The GPU's working set: what Metal recommends this process keep resident, less what it holds.
int lx_rt_mem_info(int64_t* free_bytes, int64_t* total_bytes) {
  id<MTLDevice> d = Device::get().mtl();
  if (!free_bytes || !total_bytes || !d) return note(kInvalidValue);
  const auto total = static_cast<int64_t>(d.recommendedMaxWorkingSetSize);
  const auto held = static_cast<int64_t>(d.currentAllocatedSize);
  *total_bytes = total;
  *free_bytes = std::max<int64_t>(0, total - held);
  return kOk;
}

int lx_rt_process_memory(int64_t* bytes) {
  id<MTLDevice> d = Device::get().mtl();
  if (!bytes || !d) return note(kInvalidValue);
  *bytes = static_cast<int64_t>(d.currentAllocatedSize);
  return kOk;
}

int lx_rt_host_alloc_mapped(void** host, void** device, int64_t bytes) {
  if (!host || !device) return note(kInvalidValue);
  *device = nullptr;
  if (int e = lx_rt_alloc(host, bytes)) return e;
  *device = *host;  // unified memory: kernels read it where it is
  return kOk;
}

int lx_rt_host_register(void* host, int64_t bytes, void** device) {
  if (!device) return note(kInvalidValue);
  *device = nullptr;
  if (bytes <= 0) return note(kInvalidValue);
  if (int e = Device::get().map_host(host, static_cast<size_t>(bytes))) return note(e);
  *device = host;
  return kOk;
}

int lx_rt_host_unregister(void* host) { return note(Device::get().unmap_host(host)); }

int lx_rt_stream_wait(void* stream, void* event) {
  auto* e = static_cast<Event*>(event);
  if (!e) return note(kInvalidValue);
  if (!e->stream) return kOk;  // never recorded: nothing to wait for
  return stream_of(stream)->wait_for(e->stream->timeline(), e->value);
}

}  // extern "C"

namespace lx::metal {

bool math_supported(lx_wtype w, lx_math math) {
  const int arch = lx_arch();
  const int family = arch >= 1000 ? 1000 : arch;
  switch (math) {
    case LX_MATH_BF16: return true;
    case LX_MATH_INT8: return w == LX_W_INT8 && family >= 80;
    case LX_MATH_FP8: return (w == LX_W_FP8 && family >= 89) || (w == LX_W_NVFP4 && family >= 120);
    case LX_MATH_FP4: return w == LX_W_NVFP4 && family >= 120;
  }
  return false;
}

}  // namespace lx::metal
