// Peak tensor-core throughput per mma.sync form on this GPU: the ceilings the GEMM kernels are
// measured against. Every warp runs independent chains of one instruction on register operands
// (no memory traffic), enough of them to cover the pipeline latency.
//
//   nvcc -O3 -gencode arch=compute_120a,code=sm_120a bench/mma_peak.cu -o build/mma_peak && build/mma_peak
#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>

constexpr int kChains = 8;
constexpr int kIters = 4096;

#define MMA_BODY(INSTR, NA, NB, ...)                                                         \
  float d[kChains][4] = {};                                                                  \
  uint32_t a[4] = {seed, seed ^ 1u, seed ^ 2u, seed ^ 3u}, b[2] = {seed * 3u, seed * 5u};   \
  for (int it = 0; it < kIters; ++it) {                                                      \
    _Pragma("unroll") for (int c = 0; c < kChains; ++c) { INSTR; }                          \
  }                                                                                          \
  float s = 0;                                                                               \
  for (int c = 0; c < kChains; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];             \
  if (s == 1234.5f) out[threadIdx.x] = s;

#define D4(c) "+f"(d[c][0]), "+f"(d[c][1]), "+f"(d[c][2]), "+f"(d[c][3])

__global__ void bf16_k16(float* out, uint32_t seed) {
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])), 4, 2)
}
__global__ void f16_k16_f16acc(float* out, uint32_t seed) {
  uint32_t h[kChains][2] = {};
  uint32_t a[4] = {seed, seed ^ 1u, seed ^ 2u, seed ^ 3u}, b[2] = {seed * 3u, seed * 5u};
  for (int it = 0; it < kIters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c)
      asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};"
                   : "+r"(h[c][0]), "+r"(h[c][1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  uint32_t s = 0;
  for (int c = 0; c < kChains; ++c) s ^= h[c][0] ^ h[c][1];
  if (s == 12345u) out[threadIdx.x] = s;
}
__global__ void s8_k32(float* out, uint32_t seed) {
  int d[kChains][4] = {};
  uint32_t a[4] = {seed, seed ^ 1u, seed ^ 2u, seed ^ 3u}, b[2] = {seed * 3u, seed * 5u};
  for (int it = 0; it < kIters; ++it) {
#pragma unroll
    for (int c = 0; c < kChains; ++c)
      asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                   : "+r"(d[c][0]), "+r"(d[c][1]), "+r"(d[c][2]), "+r"(d[c][3])
                   : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
  }
  int s = 0;
  for (int c = 0; c < kChains; ++c) s += d[c][0] + d[c][1] + d[c][2] + d[c][3];
  if (s == 12345) out[threadIdx.x] = s;
}
__global__ void e4m3_k32(float* out, uint32_t seed) {
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])), 4, 2)
}
__global__ void e4m3_k16(float* out, uint32_t seed) {
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(b[0])), 2, 1)
}
__global__ void f8f6f4_e4m3_e2m1(float* out, uint32_t seed) {
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::f8f6f4.f32.e4m3.e2m1.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1])), 4, 2)
}
__global__ void nvf4_k64(float* out, uint32_t seed) {
  const uint32_t sa = 0x38383838u, sb = 0x38383838u;
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12, %12}, {%11}, {%12, %12};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
                          "h"((uint16_t)0)), 4, 2)
}
__global__ void mxf8_e4m3_k32(float* out, uint32_t seed) {
  const uint32_t sa = 0x7f7f7f7fu, sb = 0x7f7f7f7fu;
  MMA_BODY(asm volatile("mma.sync.aligned.m16n8k32.row.col.kind::mxf8f6f4.block_scale.scale_vec::1X.f32.e4m3.e4m3.f32.ue8m0 "
                        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%12, %12}, {%11}, {%12, %12};"
                        : D4(c) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]), "r"(sa), "r"(sb),
                          "h"((uint16_t)0)), 4, 2)
}

template <typename K>
void run(const char* name, K kernel, double macs_per_mma) {
  int sms = 0;
  cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
  float* out;
  cudaMalloc(&out, 4096);
  const int blocks = sms * 4, threads = 128;  // 16 warps per SM
  kernel<<<blocks, threads>>>(out, 1u);
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  cudaEventRecord(e0);
  for (int r = 0; r < 5; ++r) kernel<<<blocks, threads>>>(out, 1u + r);
  cudaEventRecord(e1);
  cudaEventSynchronize(e1);
  float ms = 0;
  cudaEventElapsedTime(&ms, e0, e1);
  const double mmas = 5.0 * blocks * (threads / 32) * kIters * kChains;
  std::printf("%-28s %8.1f TFLOPS (%s)\n", name, 2.0 * mmas * macs_per_mma / (ms * 1e-3) / 1e12,
              cudaGetErrorString(cudaGetLastError()));
  cudaFree(out);
}

int main() {
  run("bf16 m16n8k16 f32acc", bf16_k16, 16 * 8 * 16);
  run("f16 m16n8k16 f16acc", f16_k16_f16acc, 16 * 8 * 16);
  run("s8 m16n8k32 s32acc", s8_k32, 16 * 8 * 32);
  run("e4m3 m16n8k32 f32acc", e4m3_k32, 16 * 8 * 32);
  run("e4m3 m16n8k16 f32acc", e4m3_k16, 16 * 8 * 16);
  run("f8f6f4 e4m3 x e2m1 k32", f8f6f4_e4m3_e2m1, 16 * 8 * 32);
  run("mxf8f6f4 e4m3 k32 (ue8m0)", mxf8_e4m3_k32, 16 * 8 * 32);
  run("mxf4nvf4 e2m1 m16n8k64", nvf4_k64, 16 * 8 * 64);
  return 0;
}
