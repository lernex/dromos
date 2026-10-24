// The runtime half of the backend ABI (kernels.h, "Runtime"), on the CUDA runtime.
#include <cuda_runtime.h>

#include "cuda/kernels.h"

namespace {

int status(cudaError_t e) { return static_cast<int>(e); }
cudaStream_t as_stream(void* s) { return static_cast<cudaStream_t>(s); }

}  // namespace

extern "C" {

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
  const cudaMemcpyKind k = kind == LX_COPY_H2D   ? cudaMemcpyHostToDevice
                           : kind == LX_COPY_D2H ? cudaMemcpyDeviceToHost
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

// Device memory is the GPU's own: a mapped checkpoint is copied instead (kernels.h).
int lx_rt_map_host(void* ptr, int64_t bytes, void** device_ptr) {
  (void)ptr;
  (void)bytes;
  if (device_ptr) *device_ptr = nullptr;
  return LX_ERR_NOT_SUPPORTED;
}

int lx_rt_unmap_host(void* ptr) {
  (void)ptr;
  return LX_ERR_NOT_SUPPORTED;
}

}  // extern "C"
