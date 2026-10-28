// The persisting L2 window (kernels.h, "Runtime"): a carve-out of the L2 that the decode's accesses to one region
// may use, set on every kernel node of a graph as it is captured. The graphs are the decode steps (runtime.cu's
// capture end calls lx_l2_apply), so a window applies to what the model replays, not to its eager launches.
#include <cuda_runtime.h>

#include <algorithm>
#include <vector>

#include "cuda/kernels.h"

namespace {

struct Window {
  const void* base = nullptr;
  int64_t bytes = 0;
  float hit = 1.0f;
};

Window g_window;  // one window per process: the model that set it owns the region until it clears it

}  // namespace

extern "C" {

int lx_rt_l2_persist(const void* base, int64_t bytes, float hit_ratio, int64_t* granted_bytes) {
  if (granted_bytes) *granted_bytes = 0;
  if (bytes <= 0) {
    g_window = Window{};
    if (cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, 0); e != cudaSuccess)
      return static_cast<int>(e);
    return static_cast<int>(cudaCtxResetPersistingL2Cache());
  }
  int device = 0, persist = 0, window = 0;
  if (cudaGetDevice(&device) != cudaSuccess) return LX_ERR_NOT_SUPPORTED;
  if (cudaDeviceGetAttribute(&persist, cudaDevAttrMaxPersistingL2CacheSize, device) != cudaSuccess) persist = 0;
  if (cudaDeviceGetAttribute(&window, cudaDevAttrMaxAccessPolicyWindowSize, device) != cudaSuccess) window = 0;
  if (persist <= 0 || window <= 0) return LX_ERR_NOT_SUPPORTED;
  // The carve-out and the window hold the same bytes: a window larger than the carve-out would evict its own lines.
  const int64_t size = std::min<int64_t>(bytes, std::min<int64_t>(persist, window));
  if (granted_bytes) *granted_bytes = size;
  if (!base) return 0;  // only asked
  if (cudaError_t e = cudaDeviceSetLimit(cudaLimitPersistingL2CacheSize, static_cast<size_t>(size)); e != cudaSuccess)
    return static_cast<int>(e);
  g_window.base = base;
  g_window.bytes = size;
  g_window.hit = std::min(1.0f, std::max(0.0f, hit_ratio));
  return 0;
}

// Called by lx_rt_capture_end with the captured template graph, before it is instantiated: every kernel node gets the
// window, whichever stream it was captured from (the model's decode forks work onto side streams).
void lx_l2_apply(void* graph) {
  if (g_window.bytes <= 0 || !graph) return;
  const cudaGraph_t g = static_cast<cudaGraph_t>(graph);
  size_t count = 0;
  if (cudaGraphGetNodes(g, nullptr, &count) != cudaSuccess || count == 0) return;
  std::vector<cudaGraphNode_t> nodes(count);
  if (cudaGraphGetNodes(g, nodes.data(), &count) != cudaSuccess) return;
  cudaKernelNodeAttrValue value{};
  value.accessPolicyWindow.base_ptr = const_cast<void*>(g_window.base);
  value.accessPolicyWindow.num_bytes = static_cast<size_t>(g_window.bytes);
  value.accessPolicyWindow.hitRatio = g_window.hit;
  value.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
  value.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
  for (size_t i = 0; i < count; ++i) {
    cudaGraphNodeType type{};
    if (cudaGraphNodeGetType(nodes[i], &type) != cudaSuccess || type != cudaGraphNodeTypeKernel) continue;
    cudaGraphKernelNodeSetAttribute(nodes[i], cudaKernelNodeAttributeAccessPolicyWindow, &value);
  }
}

}  // extern "C"
