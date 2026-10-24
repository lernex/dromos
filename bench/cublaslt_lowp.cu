// cuBLASLt's low-precision GEMMs at Praxis's prefill shapes, to compare with lx_gemm_w (bench_gemm_w.py):
//   fp8_vec    e4m3 x e4m3, per-row weight scale and per-token activation scale (OUTER_VEC_32F): the FP8
//              format and math of kernels.h exactly
//   mxfp8      e4m3 x e4m3 with ue8m0 scales per 32 values (VEC32_UE8M0), the block-scaled kind
//   nvfp4      e2m1 x e2m1 with ue4m3 scales per 16 values (VEC16_UE4M3) and one scalar alpha, the
//              standard NVFP4 (a per-tensor global; kernels.h's activations carry a per-token one)
// TFLOPS of the GEMM alone (operands already in format, f32 out), best of the heuristic's top 8 algorithms.
// Data is random: only speed is measured. A mode the library cannot run here prints "-".
//
// Build and run on the GPU server:
//   nvcc -O2 -std=c++17 -gencode arch=compute_120a,code=sm_120a bench/cublaslt_lowp.cu -lcublasLt -o /tmp/lowp
//   /tmp/lowp
#include <cublasLt.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace {

#define CK(x)                                                              \
  do {                                                                     \
    auto e_ = (x);                                                         \
    if (static_cast<int>(e_) != 0) {                                       \
      std::fprintf(stderr, "%s:%d %s -> %d\n", __FILE__, __LINE__, #x, (int)e_); \
      return -1.0;                                                         \
    }                                                                      \
  } while (0)

enum Mode { kFp8Vec, kMxFp8, kNvFp4 };

double run(cublasLtHandle_t lt, Mode mode, int M, int N, int K, void* ws, size_t ws_bytes) {
  const bool fp4 = mode == kNvFp4;
  const size_t a_bytes = static_cast<size_t>(N) * K / (fp4 ? 2 : 1), b_bytes = static_cast<size_t>(M) * K / (fp4 ? 2 : 1);
  void *A, *B, *D, *sa, *sb;
  CK(cudaMalloc(&A, a_bytes));
  CK(cudaMalloc(&B, b_bytes));
  CK(cudaMalloc(&D, static_cast<size_t>(M) * N * 4));
  const size_t pad_rows_a = (N + 127) / 128 * 128, pad_rows_b = (M + 127) / 128 * 128;
  const int vec = mode == kMxFp8 ? 32 : 16;
  const size_t cols = ((K / vec) + 3) / 4 * 4;
  const size_t sa_bytes = mode == kFp8Vec ? N * 4 : pad_rows_a * cols, sb_bytes = mode == kFp8Vec ? M * 4 : pad_rows_b * cols;
  CK(cudaMalloc(&sa, sa_bytes));
  CK(cudaMalloc(&sb, sb_bytes));
  cudaMemset(A, 0x22, a_bytes);
  cudaMemset(B, 0x23, b_bytes);
  if (mode == kFp8Vec) {
    std::vector<float> one(std::max(M, N), 1.0f);
    cudaMemcpy(sa, one.data(), N * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(sb, one.data(), M * 4, cudaMemcpyHostToDevice);
  } else {
    cudaMemset(sa, mode == kMxFp8 ? 0x7f : 0x38, sa_bytes);
    cudaMemset(sb, mode == kMxFp8 ? 0x7f : 0x38, sb_bytes);
  }

  cublasLtMatmulDesc_t op;
  CK(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
  const cublasOperation_t tA = CUBLAS_OP_T, tB = CUBLAS_OP_N;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &tA, sizeof(tA)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tB, sizeof(tB)));
  const cublasLtMatmulMatrixScale_t smode = mode == kFp8Vec ? CUBLASLT_MATMUL_MATRIX_SCALE_OUTER_VEC_32F
                                            : mode == kMxFp8 ? CUBLASLT_MATMUL_MATRIX_SCALE_VEC32_UE8M0
                                                             : CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3;
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_MODE, &smode, sizeof(smode)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_MODE, &smode, sizeof(smode)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_A_SCALE_POINTER, &sa, sizeof(sa)));
  CK(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_B_SCALE_POINTER, &sb, sizeof(sb)));
  const cudaDataType_t t = fp4 ? CUDA_R_4F_E2M1 : CUDA_R_8F_E4M3;
  cublasLtMatrixLayout_t la, lb, ld;
  CK(cublasLtMatrixLayoutCreate(&la, t, K, N, K));
  CK(cublasLtMatrixLayoutCreate(&lb, t, K, M, K));
  CK(cublasLtMatrixLayoutCreate(&ld, CUDA_R_32F, N, M, N));
  cublasLtMatmulPreference_t pref;
  CK(cublasLtMatmulPreferenceCreate(&pref));
  CK(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws_bytes, sizeof(ws_bytes)));
  cublasLtMatmulHeuristicResult_t res[8];
  int n = 0;
  if (cublasLtMatmulAlgoGetHeuristic(lt, op, la, lb, ld, ld, pref, 8, res, &n) != CUBLAS_STATUS_SUCCESS || n == 0) {
    cudaFree(A), cudaFree(B), cudaFree(D), cudaFree(sa), cudaFree(sb);
    return 0.0;
  }
  const float alpha = 1.0f, beta = 0.0f;
  double best = 1e30;
  cudaEvent_t e0, e1;
  cudaEventCreate(&e0);
  cudaEventCreate(&e1);
  const double flops = 2.0 * M * N * K;
  const int iters = static_cast<int>(std::max(3.0, std::min(1000.0, 0.02 / (flops / 300e12))));
  for (int i = 0; i < n; ++i) {
    if (cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &res[i].algo, ws, ws_bytes, 0) !=
        CUBLAS_STATUS_SUCCESS)
      continue;
    cudaDeviceSynchronize();
    for (int r = 0; r < 3; ++r) {
      cudaEventRecord(e0);
      for (int it = 0; it < iters; ++it)
        cublasLtMatmul(lt, op, &alpha, A, la, B, lb, &beta, D, ld, D, ld, &res[i].algo, ws, ws_bytes, 0);
      cudaEventRecord(e1);
      cudaEventSynchronize(e1);
      float ms = 0;
      cudaEventElapsedTime(&ms, e0, e1);
      best = std::min(best, ms / 1e3 / iters);
    }
  }
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lb), cublasLtMatrixLayoutDestroy(ld);
  cublasLtMatmulDescDestroy(op);
  cudaFree(A), cudaFree(B), cudaFree(D), cudaFree(sa), cudaFree(sb);
  return best < 1e29 ? flops / best / 1e12 : 0.0;
}

}  // namespace

int main() {
  cublasLtHandle_t lt;
  cublasLtCreate(&lt);
  const size_t ws_bytes = 64 << 20;
  void* ws;
  cudaMalloc(&ws, ws_bytes);
  struct Shape {
    const char* name;
    int N, K;
  } shapes[] = {{"in_proj", 10304, 2048}, {"out_proj", 2048, 4096}, {"qkv", 3072, 2048}, {"latent_down", 1024, 2048},
                {"latent_up", 2048, 1024}};
  const int ms[] = {64, 256, 1024, 2048, 4096, 8192};
  std::printf("%-12s %5s %9s %9s %9s   (TFLOPS, f32 out)\n", "shape", "M", "fp8_vec", "mxfp8", "nvfp4");
  for (const Shape& s : shapes)
    for (int M : ms) {
      double r[3];
      for (int m = 0; m < 3; ++m) r[m] = run(lt, static_cast<Mode>(m), M, s.N, s.K, ws, ws_bytes);
      std::printf("%-12s %5d", s.name, M);
      for (double v : r) v > 0 ? std::printf(" %9.1f", v) : std::printf(" %9s", "-");
      std::printf("\n");
    }
  return 0;
}
