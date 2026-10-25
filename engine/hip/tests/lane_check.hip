// Lane-by-lane check of lx_matrix_native.h against a scalar reference. Build and run on an AMD card:
//   hipcc -O2 -std=c++20 -I../include --offload-arch=gfx942 lane_check.hip -o lane_check && ./lane_check
// (use gfx1100 or gfx1201 for the WMMA families). It runs the family the device supports, prints the
// mismatches with their lane and element, and exits nonzero on any. Compile-only on this Mac and the Linux
// server: every kernel builds for all three targets, but none has run on hardware yet.
#include "lx_matrix_native.h"
#include "lx_mma.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

uint16_t f2bf(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even
  return static_cast<uint16_t>(u >> 16);
}

float bf2f(uint16_t b) {
  const uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// e4m3: bias 7, no infinities. Only the finite codes are generated below, so no NaN case is needed.
float e4m3_to_f(uint8_t v) {
  const int s = v >> 7, e = (v >> 3) & 0xf, m = v & 0x7;
  const float r = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
  return s ? -r : r;
}

// Each lane loads its fragment with the layout in lx_matrix_native.h, runs the instruction, and writes
// its accumulator to out[lane * elements + element]. The bodies exist only for their own target.

__global__ void k_mfma_bf16(const uint16_t* A, const uint16_t* B, const float* C, float* out) {
#if defined(LX_NATIVE_MFMA)
  const int l = threadIdx.x, r = l % 16, g = l / 16;
  lx::native::bf16x4_t a, b;
  lx::native::f32x4_t c;
  for (int j = 0; j < 4; ++j) {
    a[j] = static_cast<short>(A[r * 16 + 4 * g + j]);
    b[j] = static_cast<short>(B[(4 * g + j) * 16 + r]);
  }
  for (int i = 0; i < 4; ++i) c[i] = C[(4 * g + i) * 16 + r];
  const lx::native::f32x4_t d = lx::native::mfma_16x16x16_bf16(a, b, c);
  for (int i = 0; i < 4; ++i) out[l * 4 + i] = d[i];
#endif
}

// A is 16x32 and B is 32x16 of e4m3 bytes, row-major.
__global__ void k_mfma_fp8(const uint8_t* A, const uint8_t* B, const float* C, float* out) {
#if defined(LX_NATIVE_MFMA)
  const int l = threadIdx.x, r = l % 16, g = l / 16;
  uint64_t a = 0, b = 0;
  for (int j = 0; j < 8; ++j) {
    a |= static_cast<uint64_t>(A[r * 32 + 8 * g + j]) << (8 * j);
    b |= static_cast<uint64_t>(B[(8 * g + j) * 16 + r]) << (8 * j);
  }
  lx::native::f32x4_t c;
  for (int i = 0; i < 4; ++i) c[i] = C[(4 * g + i) * 16 + r];
  const lx::native::f32x4_t d = lx::native::mfma_16x16x32_fp8(a, b, c);
  for (int i = 0; i < 4; ++i) out[l * 4 + i] = d[i];
#endif
}

__global__ void k_wmma_rdna3(const uint16_t* A, const uint16_t* B, const float* C, float* out) {
#if defined(LX_NATIVE_WMMA_GFX11)
  const int l = threadIdx.x, r = l % 16, g = l / 16;
  uint16_t ra[16], rb[16];
  for (int j = 0; j < 16; ++j) {
    ra[j] = A[r * 16 + j];
    rb[j] = B[j * 16 + r];
  }
  lx::native::bf16x16_t a, b;
  __builtin_memcpy(&a, ra, sizeof a);
  __builtin_memcpy(&b, rb, sizeof b);
  lx::native::f32x8_t c;
  for (int i = 0; i < 8; ++i) c[i] = C[(2 * i + g) * 16 + r];
  const lx::native::f32x8_t d = lx::native::wmma_16x16x16_bf16(a, b, c);
  for (int i = 0; i < 8; ++i) out[l * 8 + i] = d[i];
#endif
}

__global__ void k_wmma_rdna4(const uint16_t* A, const uint16_t* B, const float* C, float* out) {
#if defined(LX_NATIVE_WMMA_GFX12)
  const int l = threadIdx.x, r = l % 16, g = l / 16;
  lx::native::bf16x8_t a, b;
  for (int j = 0; j < 8; ++j) {
    a[j] = static_cast<short>(A[r * 16 + 8 * g + j]);
    b[j] = static_cast<short>(B[(8 * g + j) * 16 + r]);
  }
  lx::native::f32x8_t c;
  for (int i = 0; i < 8; ++i) c[i] = C[(2 * i + g) * 16 + r];
  const lx::native::f32x8_t d = lx::native::wmma_16x16x16_bf16(a, b, c);
  for (int i = 0; i < 8; ++i) out[l * 8 + i] = d[i];
#endif
}

// The portable emulation of the same products (lx_mma.cuh), fed the PTX fragment layouts: each lane
// group of 32 computes an m16n8 tile per half of the 16 columns. Element (row, col) lands at
// emu[row * 16 + col]. On a wave64 both lane groups compute the same tile, and only the first writes.
__device__ __forceinline__ uint32_t pack2(uint16_t lo, uint16_t hi) {
  return lo | (static_cast<uint32_t>(hi) << 16);
}
__device__ __forceinline__ uint32_t pack4(uint8_t b0, uint8_t b1, uint8_t b2, uint8_t b3) {
  return static_cast<uint32_t>(b0) | (static_cast<uint32_t>(b1) << 8) | (static_cast<uint32_t>(b2) << 16) |
         (static_cast<uint32_t>(b3) << 24);
}

__global__ void k_emul_bf16(const uint16_t* A, const uint16_t* B, const float* C, float* emu) {
  const int l = threadIdx.x & 31, g = l >> 2, t = l & 3;
  for (int h = 0; h < 2; ++h) {
    const int col = 8 * h + g, c0 = 8 * h + 2 * t;
    uint32_t a[4];
    a[0] = pack2(A[g * 16 + 2 * t], A[g * 16 + 2 * t + 1]);
    a[1] = pack2(A[(g + 8) * 16 + 2 * t], A[(g + 8) * 16 + 2 * t + 1]);
    a[2] = pack2(A[g * 16 + 2 * t + 8], A[g * 16 + 2 * t + 9]);
    a[3] = pack2(A[(g + 8) * 16 + 2 * t + 8], A[(g + 8) * 16 + 2 * t + 9]);
    const uint32_t b0 = pack2(B[(2 * t) * 16 + col], B[(2 * t + 1) * 16 + col]);
    const uint32_t b1 = pack2(B[(2 * t + 8) * 16 + col], B[(2 * t + 9) * 16 + col]);
    float c[4] = {C[g * 16 + c0], C[g * 16 + c0 + 1], C[(g + 8) * 16 + c0], C[(g + 8) * 16 + c0 + 1]};
    lx::mma::mma_bf16(c, a, b0, b1);
    if (threadIdx.x < 32) {
      emu[g * 16 + c0] = c[0];
      emu[g * 16 + c0 + 1] = c[1];
      emu[(g + 8) * 16 + c0] = c[2];
      emu[(g + 8) * 16 + c0 + 1] = c[3];
    }
  }
}

// A is 16x32 and B is 32x16 of e4m3 bytes, row-major.
__global__ void k_emul_fp8(const uint8_t* A, const uint8_t* B, const float* C, float* emu) {
  const int l = threadIdx.x & 31, g = l >> 2, t = l & 3;
  for (int h = 0; h < 2; ++h) {
    const int col = 8 * h + g, c0 = 8 * h + 2 * t;
    uint32_t a[4];
    a[0] = pack4(A[g * 32 + 4 * t], A[g * 32 + 4 * t + 1], A[g * 32 + 4 * t + 2], A[g * 32 + 4 * t + 3]);
    a[1] = pack4(A[(g + 8) * 32 + 4 * t], A[(g + 8) * 32 + 4 * t + 1], A[(g + 8) * 32 + 4 * t + 2],
                 A[(g + 8) * 32 + 4 * t + 3]);
    a[2] = pack4(A[g * 32 + 4 * t + 16], A[g * 32 + 4 * t + 17], A[g * 32 + 4 * t + 18], A[g * 32 + 4 * t + 19]);
    a[3] = pack4(A[(g + 8) * 32 + 4 * t + 16], A[(g + 8) * 32 + 4 * t + 17], A[(g + 8) * 32 + 4 * t + 18],
                 A[(g + 8) * 32 + 4 * t + 19]);
    const uint32_t b0 = pack4(B[(4 * t) * 16 + col], B[(4 * t + 1) * 16 + col], B[(4 * t + 2) * 16 + col],
                              B[(4 * t + 3) * 16 + col]);
    const uint32_t b1 = pack4(B[(4 * t + 16) * 16 + col], B[(4 * t + 17) * 16 + col], B[(4 * t + 18) * 16 + col],
                              B[(4 * t + 19) * 16 + col]);
    float c[4] = {C[g * 16 + c0], C[g * 16 + c0 + 1], C[(g + 8) * 16 + c0], C[(g + 8) * 16 + c0 + 1]};
    lx::mma::mma_e4m3(c, a, b0, b1);
    if (threadIdx.x < 32) {
      emu[g * 16 + c0] = c[0];
      emu[g * 16 + c0 + 1] = c[1];
      emu[(g + 8) * 16 + c0] = c[2];
      emu[(g + 8) * 16 + c0 + 1] = c[3];
    }
  }
}

// out[l * elems + i] against ref[row(l, i) * 16 + col(l, i)]. Returns the mismatch count.
template <typename Row, typename Col>
int compare(const char* name, const std::vector<float>& out, const std::vector<double>& ref, int lanes,
            int elems, Row row, Col col) {
  int bad = 0;
  for (int l = 0; l < lanes; ++l)
    for (int i = 0; i < elems; ++i) {
      const double want = ref[row(l, i) * 16 + col(l, i)];
      const double got = out[l * elems + i];
      if (std::fabs(got - want) > 1e-4 * (1.0 + std::fabs(want))) {
        if (bad < 8)
          std::printf("%s: lane %d element %d (row %d col %d) got %.6f want %.6f\n", name, l, i, row(l, i),
                      col(l, i), got, want);
        ++bad;
      }
    }
  std::printf("%s: %d lanes x %d elements, %d mismatches\n", name, lanes, elems, bad);
  return bad;
}

template <typename T>
bool run(void (*kernel)(const T*, const T*, const float*, float*), const std::vector<T>& a,
         const std::vector<T>& b, const std::vector<float>& c, std::vector<float>& out, int threads) {
  T *da = nullptr, *db = nullptr;
  float *dc = nullptr, *dout = nullptr;
  const size_t outs = static_cast<size_t>(threads) * (out.size() / threads);
  if (hipMalloc(&da, a.size() * sizeof(T)) || hipMalloc(&db, b.size() * sizeof(T)) ||
      hipMalloc(&dc, c.size() * sizeof(float)) || hipMalloc(&dout, outs * sizeof(float)))
    return false;
  hipMemcpy(da, a.data(), a.size() * sizeof(T), hipMemcpyHostToDevice);
  hipMemcpy(db, b.data(), b.size() * sizeof(T), hipMemcpyHostToDevice);
  hipMemcpy(dc, c.data(), c.size() * sizeof(float), hipMemcpyHostToDevice);
  kernel<<<1, threads>>>(da, db, dc, dout);
  const bool ok = hipDeviceSynchronize() == hipSuccess;
  hipMemcpy(out.data(), dout, outs * sizeof(float), hipMemcpyDeviceToHost);
  hipFree(da);
  hipFree(db);
  hipFree(dc);
  hipFree(dout);
  return ok;
}

}  // namespace

int main() {
  hipDeviceProp_t prop;
  if (hipGetDeviceProperties(&prop, 0) != hipSuccess) {
    std::fprintf(stderr, "no device\n");
    return 3;
  }
  const auto fam = lx::native::family_of(prop);
  if (fam == lx::native::Family::none) {
    std::printf("%s has no native matrix family here; nothing to check\n", prop.gcnArchName);
    return 0;
  }

  std::mt19937 rng(7);
  std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
  int bad = 0;

  // bf16 tile: A and B are rounded to bf16, and the reference uses the rounded values.
  std::vector<uint16_t> A16(256), B16(256);
  std::vector<float> C(256);
  std::vector<double> ref(256, 0.0);
  std::vector<float> Af(256), Bf(256);
  for (int i = 0; i < 256; ++i) {
    A16[i] = f2bf(uni(rng));
    B16[i] = f2bf(uni(rng));
    C[i] = uni(rng);
    Af[i] = bf2f(A16[i]);
    Bf[i] = bf2f(B16[i]);
  }
  for (int r = 0; r < 16; ++r)
    for (int c = 0; c < 16; ++c) {
      double s = C[r * 16 + c];
      for (int k = 0; k < 16; ++k) s += static_cast<double>(Af[r * 16 + k]) * Bf[k * 16 + c];
      ref[r * 16 + c] = s;
    }

  if (fam == lx::native::Family::mfma_cdna3) {
    std::vector<float> out(64 * 4);
    if (!run(k_mfma_bf16, A16, B16, C, out, 64)) return 3;
    bad += compare("mfma_16x16x16_bf16", out, ref, 64, 4,
                   [](int l, int i) { return 4 * (l / 16) + i; }, [](int l, int) { return l % 16; });
    std::vector<float> emu(256);
    if (!run(k_emul_bf16, A16, B16, C, emu, 64)) return 3;
    bad += compare("mfma_16x16x16_bf16 vs emulation", out, std::vector<double>(emu.begin(), emu.end()), 64, 4,
                   [](int l, int i) { return 4 * (l / 16) + i; }, [](int l, int) { return l % 16; });

    // fp8 tile: A is 16x32 and B is 32x16 of e4m3 bytes, sampled away from the NaN codes.
    std::vector<uint8_t> A8(512), B8(512);
    std::vector<double> ref8(256, 0.0);
    std::vector<float> C8(256);
    for (auto& v : A8) v = static_cast<uint8_t>(rng() & 0xff), v = (v & 0x7f) == 0x7f ? v & 0xfe : v;
    for (auto& v : B8) v = static_cast<uint8_t>(rng() & 0xff), v = (v & 0x7f) == 0x7f ? v & 0xfe : v;
    for (auto& v : C8) v = uni(rng);
    for (int r = 0; r < 16; ++r)
      for (int c = 0; c < 16; ++c) {
        double s = C8[r * 16 + c];
        for (int k = 0; k < 32; ++k) s += static_cast<double>(e4m3_to_f(A8[r * 32 + k])) * e4m3_to_f(B8[k * 16 + c]);
        ref8[r * 16 + c] = s;
      }
    std::vector<float> out8(64 * 4);
    if (!run(k_mfma_fp8, A8, B8, C8, out8, 64)) return 3;
    bad += compare("mfma_16x16x32_fp8", out8, ref8, 64, 4,
                   [](int l, int i) { return 4 * (l / 16) + i; }, [](int l, int) { return l % 16; });
    std::vector<float> emu8(256);
    if (!run(k_emul_fp8, A8, B8, C8, emu8, 64)) return 3;
    bad += compare("mfma_16x16x32_fp8 vs emulation", out8, std::vector<double>(emu8.begin(), emu8.end()), 64, 4,
                   [](int l, int i) { return 4 * (l / 16) + i; }, [](int l, int) { return l % 16; });
  } else {
    // WMMA tiles are 32 lanes with 8 accumulator elements each: row 2*i + lane/16, column lane%16.
    std::vector<float> out(32 * 8);
    const bool rdna4 = fam == lx::native::Family::wmma_rdna4;
    const bool ok = rdna4 ? run(k_wmma_rdna4, A16, B16, C, out, 32) : run(k_wmma_rdna3, A16, B16, C, out, 32);
    if (!ok) return 3;
    bad += compare(rdna4 ? "wmma_16x16x16_bf16 (gfx12)" : "wmma_16x16x16_bf16 (gfx11)", out, ref, 32, 8,
                   [](int l, int i) { return 2 * i + l / 16; }, [](int l, int) { return l % 16; });
    std::vector<float> emu(256);
    if (!run(k_emul_bf16, A16, B16, C, emu, 32)) return 3;
    bad += compare("wmma_16x16x16_bf16 vs emulation", out, std::vector<double>(emu.begin(), emu.end()), 32, 8,
                   [](int l, int i) { return 2 * i + l / 16; }, [](int l, int) { return l % 16; });
  }
  return bad ? 1 : 0;
}
