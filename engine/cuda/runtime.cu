// The runtime half of the backend ABI (kernels.h, "Runtime"), on the CUDA runtime.
#include <cuda_runtime.h>

#include <dlfcn.h>
#include <unistd.h>

#include <cstdio>

#include "cuda/kernels.h"

namespace {

int status(cudaError_t e) { return static_cast<int>(e); }
cudaStream_t as_stream(void* s) { return static_cast<cudaStream_t>(s); }

}  // namespace

extern "C" {

// The persisting L2 window's graph hook (l2.cu).
void lx_l2_apply(void* graph);

int lx_rt_backend(void) { return LX_BACKEND_CUDA; }

int lx_rt_device_count(void) {
  int n = 0;
  return cudaGetDeviceCount(&n) == cudaSuccess ? n : 0;
}

int lx_rt_set_device(int ordinal) { return status(cudaSetDevice(ordinal)); }

int lx_rt_alloc(void** ptr, int64_t bytes) {
  *ptr = nullptr;
  if (bytes <= 0) return 0;
  if (cudaError_t e = cudaMalloc(ptr, static_cast<size_t>(bytes)); e != cudaSuccess) return status(e);
  return status(cudaMemset(*ptr, 0, static_cast<size_t>(bytes)));
}

int lx_rt_free(void* ptr) { return status(cudaFree(ptr)); }

int lx_rt_host_alloc(void** ptr, int64_t bytes) {
  *ptr = nullptr;
  return bytes <= 0 ? 0 : status(cudaMallocHost(ptr, static_cast<size_t>(bytes)));
}

int lx_rt_host_free(void* ptr) { return status(cudaFreeHost(ptr)); }

int lx_rt_memset(void* ptr, int value, int64_t bytes, void* stream) {
  return bytes <= 0 ? 0 : status(cudaMemsetAsync(ptr, value, static_cast<size_t>(bytes), as_stream(stream)));
}

int lx_rt_copy(void* dst, const void* src, int64_t bytes, lx_copy_kind kind, void* stream) {
  if (bytes <= 0) return 0;
  const cudaMemcpyKind k = kind == LX_COPY_H2D      ? cudaMemcpyHostToDevice
                           : kind == LX_COPY_D2H    ? cudaMemcpyDeviceToHost
                           : kind == LX_COPY_DEFAULT ? cudaMemcpyDefault
                                                    : cudaMemcpyDeviceToDevice;
  return status(cudaMemcpyAsync(dst, src, static_cast<size_t>(bytes), k, as_stream(stream)));
}

int lx_rt_stream_create(void** stream) {
  cudaStream_t s = nullptr;
  const int e = status(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  *stream = s;
  return e;
}

int lx_rt_stream_destroy(void* stream) { return status(cudaStreamDestroy(as_stream(stream))); }
int lx_rt_stream_sync(void* stream) { return status(cudaStreamSynchronize(as_stream(stream))); }

int lx_rt_event_create(void** event) {
  cudaEvent_t e = nullptr;
  const int r = status(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
  *event = e;
  return r;
}

int lx_rt_event_destroy(void* event) { return status(cudaEventDestroy(static_cast<cudaEvent_t>(event))); }
int lx_rt_event_record(void* event, void* stream) {
  return status(cudaEventRecord(static_cast<cudaEvent_t>(event), as_stream(stream)));
}
int lx_rt_event_sync(void* event) { return status(cudaEventSynchronize(static_cast<cudaEvent_t>(event))); }

int lx_rt_capture_begin(void* stream) {
  return status(cudaStreamBeginCapture(as_stream(stream), cudaStreamCaptureModeThreadLocal));
}

int lx_rt_capture_end(void* stream, void** graph) {
  *graph = nullptr;
  cudaGraph_t g = nullptr;
  if (cudaError_t e = cudaStreamEndCapture(as_stream(stream), &g); e != cudaSuccess) return status(e);
  lx_l2_apply(g);  // the persisting L2 window, if one is set (l2.cu)
  cudaGraphExec_t exec = nullptr;
  const cudaError_t e = cudaGraphInstantiate(&exec, g, 0);
  cudaGraphDestroy(g);
  *graph = exec;
  return status(e);
}

int lx_rt_graph_launch(void* graph, void* stream) {
  return status(cudaGraphLaunch(static_cast<cudaGraphExec_t>(graph), as_stream(stream)));
}
int lx_rt_graph_destroy(void* graph) { return status(cudaGraphExecDestroy(static_cast<cudaGraphExec_t>(graph))); }
int lx_rt_last_error(void) { return status(cudaGetLastError()); }

int lx_rt_event_create_timed(void** event) {
  cudaEvent_t e = nullptr;
  const int r = status(cudaEventCreate(&e));
  *event = e;
  return r;
}

int lx_rt_event_elapsed(void* start, void* stop, float* ms) {
  return status(cudaEventElapsedTime(ms, static_cast<cudaEvent_t>(start), static_cast<cudaEvent_t>(stop)));
}

int lx_rt_device_sync(void) { return status(cudaDeviceSynchronize()); }

const char* lx_rt_error_string(int code) { return cudaGetErrorString(static_cast<cudaError_t>(code)); }

int lx_rt_is_host_memory(const void* ptr) {
  cudaPointerAttributes attr{};
  return cudaPointerGetAttributes(&attr, ptr) == cudaSuccess && attr.type == cudaMemoryTypeHost ? 1 : 0;
}

int lx_rt_device_props(int ordinal, lx_device_props* props) {
  cudaDeviceProp prop{};
  if (cudaError_t e = cudaGetDeviceProperties(&prop, ordinal); e != cudaSuccess) return status(e);
  std::snprintf(props->name, sizeof(props->name), "%s", prop.name);
  // The attributes in lx_device_props order: capability major and minor, SMs, L2, memory clock, bus width.
  const cudaDeviceAttr attrs[6] = {cudaDevAttrComputeCapabilityMajor, cudaDevAttrComputeCapabilityMinor,
                                   cudaDevAttrMultiProcessorCount,    cudaDevAttrL2CacheSize,
                                   cudaDevAttrMemoryClockRate,        cudaDevAttrGlobalMemoryBusWidth};
  int v[6] = {};
  for (int i = 0; i < 6; ++i)
    if (cudaError_t e = cudaDeviceGetAttribute(&v[i], attrs[i], ordinal); e != cudaSuccess) return status(e);
  props->cc_major = v[0];
  props->cc_minor = v[1];
  props->sms = v[2];
  props->l2_bytes = v[3];
  props->mem_clock_khz = v[4];
  props->bus_bits = v[5];
  int driver = 0, runtime = 0;
  cudaDriverGetVersion(&driver);
  cudaRuntimeGetVersion(&runtime);
  props->driver_version = driver;
  props->runtime_version = runtime;
  return 0;
}

int lx_rt_process_memory(int64_t* bytes) {
  // NVML, loaded by path so nothing links against it: the GPU memory the driver charges this process,
  // summed over the devices it has a context on.
  struct ProcessInfo {  // nvmlProcessInfo_v2_st
    unsigned int pid;
    unsigned long long used;
    unsigned int gpu_instance, compute_instance;
  };
  using InitFn = int (*)();
  using CountFn = int (*)(unsigned int*);
  using HandleFn = int (*)(unsigned int, void**);
  using ProcsFn = int (*)(void*, unsigned int*, ProcessInfo*);
  static void* lib = dlopen("libnvidia-ml.so.1", RTLD_NOW | RTLD_LOCAL);
  if (!lib) return LX_ERR_NOT_SUPPORTED;
  static const bool ready = [] {
    auto init = reinterpret_cast<InitFn>(dlsym(lib, "nvmlInit_v2"));
    return init && init() == 0;
  }();
  auto count_fn = reinterpret_cast<CountFn>(dlsym(lib, "nvmlDeviceGetCount_v2"));
  auto handle_fn = reinterpret_cast<HandleFn>(dlsym(lib, "nvmlDeviceGetHandleByIndex_v2"));
  auto procs_fn = reinterpret_cast<ProcsFn>(dlsym(lib, "nvmlDeviceGetComputeRunningProcesses_v3"));
  if (!ready || !count_fn || !handle_fn || !procs_fn) return LX_ERR_NOT_SUPPORTED;
  unsigned int devices = 0;
  if (count_fn(&devices) != 0) return LX_ERR_NOT_SUPPORTED;
  const unsigned int pid = static_cast<unsigned int>(getpid());
  int64_t total = 0;
  bool found = false;
  for (unsigned int d = 0; d < devices; ++d) {
    void* h = nullptr;
    if (handle_fn(d, &h) != 0) continue;
    ProcessInfo info[64];
    unsigned int n = 64;
    if (procs_fn(h, &n, info) != 0) continue;
    for (unsigned int i = 0; i < n; ++i)
      if (info[i].pid == pid && info[i].used != ~0ull) {  // ~0: NVML_VALUE_NOT_AVAILABLE
        total += static_cast<int64_t>(info[i].used);
        found = true;
      }
  }
  if (!found) return LX_ERR_NOT_SUPPORTED;
  *bytes = total;
  return 0;
}

}  // extern "C"
