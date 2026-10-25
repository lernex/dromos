// The Metal backend's host side, shared by every op file. Objective-C++ only (engine/metal/*.mm).
//
// Memory: every pointer the ABI passes is the CPU address of shared storage on unified memory. The
// device keeps a registry from address ranges to their MTLBuffer, so a kernel argument becomes a GPU
// address (buffer.gpuAddress + offset). Kernels are bindless: each takes one constant struct of GPU
// addresses and scalars (kernels/args.h), set with setBytes, so a dispatch is three encoder calls.
// Registered buffers are resident through one residency set attached to every queue. A pointer that
// is not registered (host memory from malloc, as the kernel tests pass) is wrapped without a copy
// for the command buffer that uses it: page-aligned newBufferWithBytesNoCopy over its range.
//
// Streams: a stream is a command queue with one open command buffer and one concurrent compute
// encoder. Ops append dispatches; the buffer is committed at sync points and every kLx flush interval,
// so the GPU starts while the host is still encoding. Untracked buffers give no ordering between
// command buffers of a queue, so each buffer signals a timeline value and the next one waits for it.
// The values are signaled twice: on a GPU-only event, which the stream's next buffer and other streams
// wait on (a wait on a shared event costs a quarter of a millisecond or more when the GPU is not busy),
// and on a shared event, which the host waits on (stream syncs, lx_rt_event_*).
//
// Within an encoder, dispatches run concurrently unless one depends on another: every address an op
// takes names the range a kernel may touch and whether it writes there (addr / addr_w), and a dispatch
// that reads what an earlier one since the last barrier wrote, or writes what it touched, gets a memory
// barrier first. A stream's ops still behave as if they ran in order, but independent neighbours (the
// four copies of a concat, a shared expert beside the routed ones) overlap, which matters most where
// many small kernels each wait on memory latency. LX_METAL_SERIAL=1 orders every dispatch instead.
#pragma once

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "cuda/kernels.h"

namespace lx::metal {

// Status codes (kernels.h uses the CUDA values).
constexpr int kOk = 0;
constexpr int kInvalidValue = 1;
constexpr int kOutOfMemory = 2;
constexpr int kNotSupported = LX_ERR_NOT_SUPPORTED;
constexpr int kLaunchFailure = 719;
constexpr size_t kUnknownExtent = ~size_t{0};

// One recorded dispatch, for capture and replay; or a point where the command buffer ends (kCut), or
// where it ends and the next waits for the CPU lane of an offload state (kMailbox, ops_offload.mm).
struct Command {
  enum Kind { kDispatch, kCut, kMailbox };
  Kind kind = kDispatch;
  id<MTLComputePipelineState> pipeline = nil;
  std::vector<uint8_t> args;
  MTLSize groups{}, threads{};
  NSUInteger threadgroup_bytes = 0;
  bool barrier = false;  // depends on an earlier command of the graph
  const void* state = nullptr;  // kMailbox: the lx_offload_state
};

struct Graph {
  std::vector<Command> commands;
  std::vector<id<MTLBuffer>> keep;  // transient wraps the recorded commands point into
};

class Stream {
 public:
  explicit Stream(id<MTLCommandQueue> queue);
  ~Stream();

  // The GPU address of p, which the next dispatch may read (and with `write`, write) over
  // [p, p + bytes). Null maps to 0. kUnknownExtent: the caller cannot tell (rows picked by ids from a
  // table of unknown height); the range then runs to the end of p's buffer, and an unregistered pointer
  // is wrapped to the end of its virtual memory region.
  uint64_t address(const void* p, size_t bytes, bool write = false);
  // Forgets the ranges taken for a dispatch that is not launched after all.
  void drop_pending() { taking_.clear(); }
  // Takes the last dispatch's ranges again, for a next dispatch with the same arguments (the kernels of
  // one op sharing an argument struct), so the two are ordered and the second is tracked in full.
  void again() { taking_ = last_; }
  // The next dispatch may touch any memory (pointers it reads out of memory): it is ordered after
  // everything before it, and everything after it after it.
  void touch_all();
  // Keeps `buffer` resident and alive for the next dispatch (and its graph, while capturing); returns
  // its GPU address.
  uint64_t retain(id<MTLBuffer> buffer);
  // Ends the command buffer here, so the GPU's writes so far reach the CPU as soon as it gets here.
  void cut();
  // Has the work after this point wait until the CPU lane of offload state `state` answered the job
  // posted by the plan before the last cut (or there was none, or it timed out).
  void mailbox_wait(const void* state);

  // Appends one dispatch of `pipeline` with its argument struct.
  void dispatch(id<MTLComputePipelineState> pipeline, const void* args, size_t args_bytes, MTLSize groups,
                MTLSize threads, NSUInteger threadgroup_bytes = 0);
  void dispatch_one(id<MTLComputePipelineState> pipeline, const void* args, size_t args_bytes, MTLSize groups,
                    MTLSize threads, NSUInteger threadgroup_bytes);

  // Commits the open command buffer (if it has work) and returns the timeline value that marks it.
  uint64_t flush();
  // Waits until the timeline reaches `value` (0: everything committed so far).
  int wait(uint64_t value = 0);
  uint64_t committed() const { return committed_; }
  bool done(uint64_t value) const;
  // The command buffer committed last (for timing).
  id<MTLCommandBuffer> last_buffer() const { return last_cb_; }
  // Work issued after this call waits until stream `other` reaches `value` (lx_rt_stream_wait).
  int wait_for(Stream* other, uint64_t value);

  // Host to device copy in stream order: the source is staged now (so the caller may reuse it), and a
  // kernel moves it into place when the stream gets there.
  int upload(void* dst, const void* src, size_t bytes);

  bool capturing() const { return graph_ != nullptr; }
  int begin_capture();
  Graph* end_capture();
  void replay(const Graph& graph);

  int error() const { return error_; }
  int take_error() {
    const int e = error_;
    error_ = kOk;
    return e;
  }

 private:
  void open();
  void retire();
  // Queues the block that lets ticket number `ticket` of the mailbox event go; returns the ticket.
  uint64_t schedule_mailbox(uint64_t after, const void* state);
  // Work encoded after this call waits until `event` reaches `value`, within the open command buffer.
  void wait_here(id<MTLEvent> event, uint64_t value, const std::vector<id<MTLBuffer>>* keep);
  bool hazard();  // whether the pending ranges need a barrier; then tracks them
  // LX_METAL_PROFILE=3: ends the encoder and opens one for the next dispatch alone, timed at its ends.
  void timed_encoder(id<MTLComputePipelineState> pipeline, const std::vector<id<MTLBuffer>>* keep);

  // Ranges the dispatch being built touches, and those touched since the last barrier.
  struct Access {
    uintptr_t lo, hi;
    bool write;
  };
  std::vector<Access> taking_, last_, reads_, writes_, saved_reads_, saved_writes_;
  bool barrier_next_ = false;  // after a graph replay: order the next dispatch after it

  id<MTLCommandQueue> queue_;
  id<MTLSharedEvent> timeline_;
  id<MTLEvent> chain_;  // the same values, signaled for the stream's next command buffer to wait on
  id<MTLCommandBuffer> cb_ = nil;
  id<MTLCommandBuffer> last_cb_ = nil;
  std::vector<std::pair<id<MTLEvent>, uint64_t>> waits_;  // for the next command buffer to wait on
  id<MTLComputeCommandEncoder> enc_ = nil;
  id<MTLComputePipelineState> bound_ = nil;
  uint64_t committed_ = 0;  // timeline value of the last committed command buffer
  int pending_ = 0;         // dispatches in the open command buffer
  int error_ = kOk;
  // Unregistered host ranges wrapped for the open command buffer, and those waiting for theirs to finish.
  struct Wrap {
    uintptr_t lo, hi;
    id<MTLBuffer> buffer;
  };
  std::vector<Wrap> wraps_;
  std::vector<std::pair<uint64_t, std::vector<Wrap>>> retiring_;
  // Upload staging: a ring of shared memory, reused once the command buffers reading it finished.
  id<MTLBuffer> ring_ = nil;
  size_t ring_head_ = 0;
  std::vector<std::pair<uint64_t, size_t>> ring_marks_;  // (timeline value, ring head after it)
  Graph* graph_ = nullptr;
  // Offload joins: the event the next command buffer waits on, its last value, and the queue whose
  // block watches the mailbox and signals it.
  id<MTLSharedEvent> mailbox_ = nil;
  uint64_t mailbox_ticket_ = 0;
  uint64_t cut_ = 0;  // the timeline value of the last cut (the plan's command buffer)
  dispatch_queue_t poller_ = nil;
  // LX_METAL_PROFILE=3: the open command buffer's samples, its timed kernels, and the fence ordering them.
  id<MTLCounterSampleBuffer> samples_ = nil;
  std::vector<std::string> timed_;
  id<MTLFence> fence_ = nil;
};

// The process-wide device: buffers, the registry, the residency set, the library and pipelines.
class Device {
 public:
  static Device& get();

  id<MTLDevice> mtl() const { return device_; }
  int family() const { return family_; }  // highest MTLGPUFamilyApple N supported, as N

  // Allocations and registrations. `owned` buffers were made by alloc and are released by free.
  int alloc(void** ptr, size_t bytes);
  int free(void* ptr);
  int map_host(void* ptr, size_t bytes);  // registers caller-owned host memory without a copy
  int unmap_host(void* ptr);

  // The registered buffer holding [p, p + bytes), or nil. Also returns its base address.
  id<MTLBuffer> lookup(const void* p, size_t bytes, uintptr_t* base);
  // The registry as a table kernels translate CPU addresses with (args.h OxRange, sorted), and its
  // length. Rebuilt when the registry changed; a buffer handed out is never written again.
  id<MTLBuffer> table(int* n);

  // A compiled pipeline for a kernel of the library, by function name. Fatal on a missing kernel.
  // nax: from the neural accelerator library (kernels/nax), which only nax() GPUs build.
  id<MTLComputePipelineState> pipeline(const std::string& name, bool nax = false);
  // Whether the GPU has neural accelerators the tensor-op kernels run on (Apple GPU family 10, the M5,
  // and later; LX_METAL_NAX=0 turns them off) and their library compiled.
  bool nax();

  Stream* default_stream();
  id<MTLCommandQueue> new_queue();
  // Waits for every stream's work (lx_rt_device_sync); the first error any of them reports.
  int sync_all();
  void add_stream(Stream* s);
  void remove_stream(Stream* s);

  // Unregistered pointers that were wrapped on the fly; the model is expected to keep this at zero.
  uint64_t transient_wraps() const { return transient_wraps_; }
  void count_transient() { ++transient_wraps_; }

 private:
  Device();
  void compile();
  id<MTLLibrary> build(const char* source, MTLLanguageVersion version, const char* what);
  int release(uintptr_t key, bool owned);

  struct Entry {
    id<MTLBuffer> buffer;
    size_t bytes;
    bool owned;
  };
  id<MTLDevice> device_;
  id<MTLResidencySet> residency_;
  id<MTLLibrary> library_ = nil, nax_library_ = nil;
  std::once_flag compiled_, nax_compiled_;
  int family_ = 0;
  std::mutex mu_;
  std::vector<std::pair<uintptr_t, Entry>> entries_;  // sorted by base address
  uint64_t generation_ = 0, table_generation_ = ~uint64_t{0};  // registry changes, and the table's
  id<MTLBuffer> table_ = nil;
  int table_n_ = 0;
  std::unordered_map<std::string, id<MTLComputePipelineState>> pipelines_;
  std::vector<id<MTLCommandQueue>> queues_;
  std::vector<Stream*> streams_;
  Stream* default_stream_ = nullptr;
  uint64_t transient_wraps_ = 0;
};

inline Stream* stream_of(void* s) { return s ? static_cast<Stream*>(s) : Device::get().default_stream(); }

// A kernel of the library, resolved once: `static Kernel k("rmsnorm"); k.launch(...)`.
class Kernel {
 public:
  explicit Kernel(const char* name) : name_(name) {}
  id<MTLComputePipelineState> pipeline() {
    std::call_once(once_, [this] { pipeline_ = Device::get().pipeline(name_); });
    return pipeline_;
  }
  template <typename Args>
  int launch(void* stream, const Args& args, MTLSize groups, MTLSize threads, NSUInteger threadgroup_bytes = 0) {
    Stream* s = stream_of(stream);
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) {
      s->drop_pending();
      return kOk;
    }
    s->dispatch(pipeline(), &args, sizeof(Args), groups, threads, threadgroup_bytes);
    return s->error();
  }
  int max_threads() { return static_cast<int>(pipeline().maxTotalThreadsPerThreadgroup); }

 private:
  const char* name_;
  std::once_flag once_;
  id<MTLComputePipelineState> pipeline_ = nil;
};

// A family of kernel variants, numbered 0 .. N - 1 by the caller and named by `namer` (the kernel
// template's host names); each pipeline is built on its first launch.
template <int N>
class KernelSet {
 public:
  using Namer = std::string (*)(int);
  explicit KernelSet(Namer namer, bool nax = false) : namer_(namer), nax_(nax) {}
  id<MTLComputePipelineState> pipeline(int i) {
    std::call_once(once_[i], [this, i] { pipes_[i] = Device::get().pipeline(namer_(i), nax_); });
    return pipes_[i];
  }
  template <typename Args>
  int launch(int i, void* stream, const Args& args, MTLSize groups, MTLSize threads, NSUInteger threadgroup_bytes = 0) {
    Stream* s = stream_of(stream);
    if (groups.width == 0 || groups.height == 0 || groups.depth == 0) {
      s->drop_pending();
      return kOk;
    }
    s->dispatch(pipeline(i), &args, sizeof(Args), groups, threads, threadgroup_bytes);
    return s->error();
  }

 private:
  Namer namer_;
  bool nax_;
  std::once_flag once_[N];
  id<MTLComputePipelineState> pipes_[N];
};

inline MTLSize size1(NSUInteger x) { return MTLSizeMake(x, 1, 1); }
inline MTLSize size2(NSUInteger x, NSUInteger y) { return MTLSizeMake(x, y, 1); }
inline MTLSize size3(NSUInteger x, NSUInteger y, NSUInteger z) { return MTLSizeMake(x, y, z); }
inline NSUInteger ceil_div(int64_t a, int64_t b) { return static_cast<NSUInteger>((a + b - 1) / b); }

// GPU address of p over `bytes` on stream `s` (the ABI's void* stream), for the next dispatch to read
// (addr) or to write, and maybe read too (addr_w).
inline uint64_t addr(void* stream, const void* p, size_t bytes) { return stream_of(stream)->address(p, bytes); }
inline uint64_t addr_w(void* stream, const void* p, size_t bytes) {
  return stream_of(stream)->address(p, bytes, true);
}

// The end of the virtual memory region holding address a, at most a + limit (0 if a is not mapped).
uintptr_t region_end(uintptr_t a, size_t limit);

inline size_t dtype_size(lx_dtype d) { return d == LX_BF16 ? 2 : 4; }
inline bool valid_dtype(lx_dtype d) { return d == LX_F32 || d == LX_BF16; }

// Bytes of a [rows, cols] matrix with row stride ld (elements of `elem` bytes): what a kernel may touch.
inline size_t span(int64_t rows, int64_t cols, int64_t ld, size_t elem) {
  if (rows <= 0 || cols <= 0) return 0;
  return static_cast<size_t>(((rows - 1) * ld + cols) * static_cast<int64_t>(elem));
}

// Whether a tensor-core math can run on a weight format here. Every pair runs on an Apple GPU (the
// quantized maths are emulated exactly in f32); an architecture override below 1000 names a CUDA family
// and gets that family's answer, so the tests' NotSupported cases mean the same on every backend.
bool math_supported(lx_wtype w, lx_math math);

// Bytes of an [N, K] weight in its format: (data, scale).
size_t weight_data_bytes(lx_wtype type, int64_t N, int64_t K);
size_t weight_scale_bytes(lx_wtype type, int64_t N, int64_t K);

}  // namespace lx::metal
