// The slice of CUB the CUDA sources use, for the HIP port. BlockScan<T, BLOCK> is a block-wide
// exclusive prefix sum over one value per thread, with CUB's TempStorage and ExclusiveSum interface:
// a Hillis-Steele scan through shared memory. The block must be exactly BLOCK threads.
#pragma once

#include "lx_hip.h"

namespace cub {

template <typename T, int BLOCK>
class BlockScan {
 public:
  struct TempStorage {
    T buf[BLOCK];
  };

  __device__ explicit BlockScan(TempStorage& storage) : s_(storage) {}

  // out = sum of `in` over the threads before this one; total = sum over the whole block.
  __device__ void ExclusiveSum(T in, T& out, T& total) {
    const int t = threadIdx.x;
    s_.buf[t] = in;
    __syncthreads();
    for (int off = 1; off < BLOCK; off <<= 1) {
      const T add = t >= off ? s_.buf[t - off] : T(0);
      __syncthreads();
      s_.buf[t] += add;
      __syncthreads();
    }
    out = s_.buf[t] - in;
    total = s_.buf[BLOCK - 1];
    __syncthreads();  // the storage is free for the caller's next use
  }

 private:
  TempStorage& s_;
};

}  // namespace cub
