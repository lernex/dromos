// The HIP backend's persisting L2 window (kernels.h, lx_rt_l2_persist). HIP has no access policy window on kernel
// nodes that the decode could use, so the window declines and graphs are captured with none: the model runs unpinned.
// Replaces engine/cuda/l2.cu in src/ (translate.py HANDWRITTEN).
#include "kernels.h"

extern "C" {

int lx_rt_l2_persist(const void* base, int64_t bytes, float hit_ratio) {
  (void)base;
  (void)bytes;
  (void)hit_ratio;
  return LX_ERR_NOT_SUPPORTED;
}

void lx_l2_apply(void* graph) { (void)graph; }

}  // extern "C"
