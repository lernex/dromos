// Round-to-nearest quantization of a bf16 weight into the formats of kernels.h, "Weight formats"
// (lx_quantize_weight). Used at load, when the engine is asked for a format the file is not stored in.
//
// INT8 is lx_quantize_w8 (gemv.cu). FP8 takes a per-row absmax scale (/448) and rounds each value
// to e4m3, saturating. NVFP4 takes a global per matrix (max|w| / (448 * 6), read back to the host
// because lx_weight carries it as a host value) and a per-block e4m3 scale (block absmax / (6 *
// global)); each value is rounded to the nearest e2m1 of w / (block scale * global), ties to the even
// code, as the checkpoints' own quantizer does. Divisions are IEEE, so the bytes match a reference
// that divides the same way.

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>

#include "cuda/common.cuh"
#include "cuda/kernels.h"
#include "cuda/wformat.cuh"

namespace {

using bf16 = __nv_bfloat16;
constexpr int kThreads = 256;
constexpr int kWarps = kThreads / 32;

// The nearest e2m1 code of x: 0, .5, 1, 1.5, 2, 3, 4, 6 with a sign bit (bit 3). The code is the number
// of midpoints below |x|; a value exactly on a midpoint goes to the even code. Saturates at 6. Zero
// (or a value that rounds to it) is code 0 with no sign, as the checkpoints store it.
__device__ __forceinline__ uint32_t e2m1_encode(float x) {
  const float mag = fminf(fabsf(x), 6.0f);
  constexpr float mids[7] = {0.25f, 0.75f, 1.25f, 1.75f, 2.5f, 3.5f, 5.0f};
  uint32_t idx = 0;
#pragma unroll
  for (int i = 0; i < 7; ++i) idx += mag > mids[i];
  uint32_t tie = 0;  // mag sits on midpoint idx between codes idx and idx + 1: go to the even one
#pragma unroll
  for (int i = 0; i < 7; ++i) tie |= (static_cast<uint32_t>(i) == idx && mag == mids[i] && (i & 1)) ? 1u : 0u;
  idx += tie;
  const uint32_t sign = (x < 0.0f && idx > 0) ? 8u : 0u;
  return idx | sign;
}

// One thread per row: absmax of a row, then the row's e4m3 value, codes rounded per element.
// FP8 scale = absmax / 448 (1 for an all-zero row), q = e4m3(w / scale), saturating.
__global__ void quantize_fp8_kernel(const bf16* __restrict__ w, uint8_t* __restrict__ q, float* __restrict__ scale,
                                    int64_t rows, int cols) {
  const int64_t row = static_cast<int64_t>(blockIdx.x) * kWarps + (threadIdx.x >> 5);
  if (row >= rows) return;
  const int lane = threadIdx.x & 31;
  const bf16* src = w + row * cols;
  uint8_t* dst = q + row * cols;

  float amax = 0.0f;
  for (int c = lane; c < cols; c += 32) amax = fmaxf(amax, fabsf(lx::to_f32(src[c])));
  amax = lx::warp_max(amax);
  const float s = amax > 0.0f ? __fdiv_rn(amax, 448.0f) : 1.0f;
  if (lane == 0) scale[row] = s;
  for (int c = lane; c < cols; c += 32) {
    const float v = fminf(fmaxf(__fdiv_rn(lx::to_f32(src[c]), s), -448.0f), 448.0f);
    dst[c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
  }
}

// Absolute maximum of the whole matrix. |w| is non-negative, so its bit pattern orders like the value.
__global__ void absmax_kernel(const bf16* __restrict__ w, int64_t n, unsigned int* __restrict__ out) {
  unsigned int local = 0;
  for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += static_cast<int64_t>(gridDim.x) * blockDim.x) {
    local = max(local, __float_as_uint(fabsf(lx::to_f32(w[i]))));
  }
  // Reduce the block through shared memory, then one atomic per block.
  __shared__ unsigned int partial[kThreads];
  partial[threadIdx.x] = local;
  __syncthreads();
  for (int s = kThreads / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) partial[threadIdx.x] = max(partial[threadIdx.x], partial[threadIdx.x + s]);
    __syncthreads();
  }
  if (threadIdx.x == 0) atomicMax(out, partial[0]);
}

// One thread per (row, block of 16 values): the block's e4m3 scale, then its 16 codes packed into 8 bytes.
// The effective scale is the e4m3 block scale times the global, the same product the kernel uses.
__global__ void quantize_nvfp4_kernel(const bf16* __restrict__ w, uint8_t* __restrict__ q, uint8_t* __restrict__ scale,
                                      float global, int64_t rows, int cols) {
  const int64_t blocks_per_row = cols / 16;
  const int64_t t = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (t >= rows * blocks_per_row) return;
  const int64_t row = t / blocks_per_row, b = t % blocks_per_row;
  const bf16* src = w + row * cols + 16 * b;

  float v[16];
  float amax = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; ++j) {
    v[j] = lx::to_f32(src[j]);
    amax = fmaxf(amax, fabsf(v[j]));
  }
  // The block scale: absmax / (6 * global), rounded to e4m3. An all-zero block (or a zero global) has
  // scale 0, and its codes are 0.
  const uint8_t sb = __nv_cvt_float_to_fp8(__fdiv_rn(amax, 6.0f * global), __NV_SATFINITE, __NV_E4M3);
  scale[row * blocks_per_row + b] = sb;
  const float eff = lx::e4m3_to_f32(sb) * global;

  uint8_t* dst = q + row * (cols / 2) + 8 * b;
#pragma unroll
  for (int p = 0; p < 8; ++p) {
    uint32_t lo = 0, hi = 0;
    if (eff > 0.0f) {
      lo = e2m1_encode(__fdiv_rn(v[2 * p], eff));
      hi = e2m1_encode(__fdiv_rn(v[2 * p + 1], eff));
    }
    dst[p] = static_cast<uint8_t>(lo | (hi << 4));  // the low nibble is the even k
  }
}

// Element (n, k) of any format as bf16, with its row factor: the per-row scale, or the global. One
// thread per element.
template <int F>
__global__ void dequantize_kernel(const lx_weight w, bf16* __restrict__ out, int64_t rows, int cols) {
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= rows * cols) return;
  const int64_t n = i / cols, k = i % cols;
  const float v = lx::element<F>(w, n, k, cols) * lx::WordFormat<F>::row_factor(w, n);
  out[i] = lx::from_f32<bf16>(v);
}

}  // namespace

extern "C" int lx_quantize_weight(const void* w, lx_weight* out, int64_t rows, int cols, void* stream) {
  if (!out || !w) return static_cast<int>(cudaErrorInvalidValue);
  if (rows <= 0) return 0;
  if (cols <= 0) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  switch (out->type) {
    case LX_W_BF16:
      return static_cast<int>(cudaErrorInvalidValue);  // nothing to quantize: the bf16 weight is the result
    case LX_W_BF16X: {
      // Lossless: the encoding of the bf16 matrix (sized by lx_bf16x_bytes, which synchronizes the stream).
      if (!out->data) return static_cast<int>(cudaErrorInvalidValue);
      const int64_t bytes = lx_bf16x_bytes(w, rows, cols, stream);
      if (bytes < 0) return static_cast<int>(-bytes);
      return lx_bf16x_encode(w, rows, cols, const_cast<void*>(out->data), bytes, stream);
    }
    case LX_W_INT8:
      if (!out->data || !out->scale) return static_cast<int>(cudaErrorInvalidValue);
      return lx_quantize_w8(w, static_cast<int8_t*>(const_cast<void*>(out->data)), static_cast<float*>(const_cast<void*>(out->scale)), rows, cols, stream);
    case LX_W_FP8: {
      if (!out->data || !out->scale) return static_cast<int>(cudaErrorInvalidValue);
      const int64_t blocks = (rows + kWarps - 1) / kWarps;
      quantize_fp8_kernel<<<static_cast<unsigned int>(blocks), kThreads, 0, s>>>(
          static_cast<const bf16*>(w), static_cast<uint8_t*>(const_cast<void*>(out->data)),
          static_cast<float*>(const_cast<void*>(out->scale)), rows, cols);
      return lx::launch_status();
    }
    case LX_W_NVFP4: {
      if (!out->data || !out->scale || cols % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
      const int64_t n = rows * cols;
      // The global: max|w| / (448 * 6), read back as a host value (this call is a load-time step).
      unsigned int* amax_dev = nullptr;
      if (cudaMalloc(&amax_dev, sizeof(unsigned int)) != cudaSuccess) return static_cast<int>(cudaErrorMemoryAllocation);
      cudaMemsetAsync(amax_dev, 0, sizeof(unsigned int), s);
      const unsigned int grid = static_cast<unsigned int>(std::min<int64_t>((n + kThreads - 1) / kThreads, 1024));
      absmax_kernel<<<grid, kThreads, 0, s>>>(static_cast<const bf16*>(w), n, amax_dev);
      unsigned int amax_bits = 0;
      cudaMemcpyAsync(&amax_bits, amax_dev, sizeof(amax_bits), cudaMemcpyDeviceToHost, s);
      const cudaError_t sync = cudaStreamSynchronize(s);
      cudaFree(amax_dev);
      if (sync != cudaSuccess) return static_cast<int>(sync);
      float amax = 0.0f;
      std::memcpy(&amax, &amax_bits, sizeof(amax));
      const float global = amax / 2688.0f;  // 448 * 6, as float32
      out->global = global;
      const int64_t t = rows * (cols / 16);
      const unsigned int blocks = static_cast<unsigned int>((t + kThreads - 1) / kThreads);
      quantize_nvfp4_kernel<<<blocks, kThreads, 0, s>>>(static_cast<const bf16*>(w),
                                                         static_cast<uint8_t*>(const_cast<void*>(out->data)),
                                                         static_cast<uint8_t*>(const_cast<void*>(out->scale)), global,
                                                         rows, cols);
      return lx::launch_status();
    }
    default:
      return static_cast<int>(cudaErrorInvalidValue);
  }
}

extern "C" int lx_dequantize_weight(const lx_weight* w, void* out, int64_t rows, int cols, void* stream) {
  if (!w || !w->data || !out) return static_cast<int>(cudaErrorInvalidValue);
  if (rows <= 0) return 0;
  if (cols <= 0) return static_cast<int>(cudaErrorInvalidValue);
  if (w->type != LX_W_BF16 && w->type != LX_W_BF16X && w->type != LX_W_INT8 && w->type != LX_W_FP8 &&
      w->type != LX_W_NVFP4)
    return static_cast<int>(cudaErrorInvalidValue);
  if (w->type != LX_W_BF16 && w->type != LX_W_BF16X && !w->scale) return static_cast<int>(cudaErrorInvalidValue);
  if (w->type == LX_W_NVFP4 && cols % 16 != 0) return static_cast<int>(cudaErrorInvalidValue);
  if (w->type == LX_W_BF16X) return lx_bf16x_decode(w->data, rows, cols, out, stream);
  if (w->type == LX_W_BF16) {
    // Nothing to decode: a plain copy of the rows.
    const cudaError_t e = cudaMemcpyAsync(out, w->data, sizeof(bf16) * rows * cols, cudaMemcpyDeviceToDevice,
                                          static_cast<cudaStream_t>(stream));
    return static_cast<int>(e);
  }
  const int64_t n = rows * cols;
  const unsigned int blocks = static_cast<unsigned int>((n + kThreads - 1) / kThreads);
  auto s = static_cast<cudaStream_t>(stream);
  auto* o = static_cast<bf16*>(out);
  switch (w->type) {
    case LX_W_INT8: dequantize_kernel<LX_W_INT8><<<blocks, kThreads, 0, s>>>(*w, o, rows, cols); break;
    case LX_W_FP8: dequantize_kernel<LX_W_FP8><<<blocks, kThreads, 0, s>>>(*w, o, rows, cols); break;
    default: dequantize_kernel<LX_W_NVFP4><<<blocks, kThreads, 0, s>>>(*w, o, rows, cols); break;
  }
  return lx::launch_status();
}
