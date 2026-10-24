// Dense GEMM on cuBLASLt: y[M, N] = alpha * (x[M, K] @ w[N, K]^T + bias[N]) + beta * y.
//
// cuBLASLt is column-major, and a row-major y[M, N] is the column-major y^T[N, M] with leading
// dimension ld_y. So the product is computed as y^T = w x^T: w is the left operand, transposed
// (its row-major [N, K] memory is a column-major [K, N] matrix), and x the right operand as stored.
// M = 1 (decode) is the n = 1 case of the same call; the heuristic picks a GEMV-shaped kernel.
//
// The bias goes through cuBLASLt's bias epilogue, which adds after alpha and beta, so the bias
// has to be alpha * bias. cuBLASLt also wants the bias in the output's dtype. Those two rules mean
// the checkpoint's bias goes straight into the epilogue only when alpha == 1 and the dtypes agree
// (the bf16 GEMM with bf16 y, and the f32 GEMM). Anything else is one small scale-and-convert
// kernel into a scratch buffer first; the model runs only the direct cases.
//
// Plans (descriptors, layouts and the chosen algorithm) are cached per shape. Each device has one
// handle, one workspace and one bias scratch, shared by every call on it, so the engine issues a
// device's GEMMs on one stream. The mutex only orders host-side use of those objects.

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <array>
#include <bit>
#include <cstdint>
#include <map>
#include <mutex>

#include "cuda/common.cuh"
#include "cuda/kernels.h"

namespace {

constexpr size_t kWorkspaceBytes = size_t{32} << 20;
constexpr int kMaxDevices = 16;
// Prefill lengths vary, so shapes keep arriving. Flushing a full cache is cheaper than letting it grow.
constexpr size_t kMaxPlans = 2048;

cudaError_t to_cuda(cublasStatus_t status) {
  switch (status) {
    case CUBLAS_STATUS_SUCCESS: return cudaSuccess;
    case CUBLAS_STATUS_INVALID_VALUE: return cudaErrorInvalidValue;
    case CUBLAS_STATUS_ALLOC_FAILED: return cudaErrorMemoryAllocation;
    case CUBLAS_STATUS_NOT_SUPPORTED: return cudaErrorNotSupported;
    default: return cudaErrorUnknown;
  }
}

// Everything a plan depends on: the shapes, the dtypes, the epilogue and the alignment that
// cuBLASLt must honour for the three matrices.
struct PlanKey {
  int M, N, K, ld_x, ld_y;
  bool bf16_in, bf16_out;
  bool bias;
  uint32_t align_a, align_b, align_c;
  auto operator<=>(const PlanKey&) const = default;
};

struct Plan {
  cublasLtMatmulDesc_t op = nullptr;
  cublasLtMatrixLayout_t a = nullptr;
  cublasLtMatrixLayout_t b = nullptr;
  cublasLtMatrixLayout_t c = nullptr;  // the output's layout too: D shares C's shape and type
  cublasLtMatmulAlgo_t algo{};

  void destroy() {
    if (c) cublasLtMatrixLayoutDestroy(c);
    if (b) cublasLtMatrixLayoutDestroy(b);
    if (a) cublasLtMatrixLayoutDestroy(a);
    if (op) cublasLtMatmulDescDestroy(op);
    *this = Plan{};
  }
};

struct DeviceState {
  std::once_flag ready;
  cudaError_t init_error = cudaSuccess;
  std::mutex mu;
  cublasLtHandle_t handle = nullptr;
  void* workspace = nullptr;
  void* bias_scratch = nullptr;
  size_t bias_scratch_bytes = 0;
  std::map<PlanKey, Plan> plans;
};

// Created on first use, on the device that is current then. Never destroyed: tearing handles down
// at exit would call into a CUDA runtime that may already be gone.
DeviceState* device_state() {
  int device = 0;
  if (cudaGetDevice(&device) != cudaSuccess || device < 0 || device >= kMaxDevices) return nullptr;
  static std::array<DeviceState, kMaxDevices> states;
  DeviceState& s = states[device];
  std::call_once(s.ready, [&s] {
    if (cublasLtCreate(&s.handle) != CUBLAS_STATUS_SUCCESS) {
      s.init_error = cudaErrorUnknown;
      return;
    }
    s.init_error = cudaMalloc(&s.workspace, kWorkspaceBytes);
  });
  return &s;
}

// Largest power of two (at most 256) dividing both the pointer and the row stride in bytes. The
// stride matters: an odd leading dimension leaves cuBLASLt only the narrow, slower kernels.
uint32_t alignment(const void* p, uint64_t row_bytes) {
  const uint64_t bits = reinterpret_cast<uintptr_t>(p) | row_bytes;
  const uint64_t low = bits & (~bits + 1);
  return static_cast<uint32_t>(low == 0 || low > 256 ? 256 : low);
}

template <typename T>
cublasStatus_t set_desc_attr(cublasLtMatmulDesc_t desc, cublasLtMatmulDescAttributes_t attr, const T& v) {
  return cublasLtMatmulDescSetAttribute(desc, attr, &v, sizeof(T));
}

template <typename T>
cublasStatus_t set_pref_attr(cublasLtMatmulPreference_t pref, cublasLtMatmulPreferenceAttributes_t attr,
                             const T& v) {
  return cublasLtMatmulPreferenceSetAttribute(pref, attr, &v, sizeof(T));
}

// Builds the plan for one key into p. On failure p holds whatever was created; the caller destroys it.
cublasStatus_t build_plan(DeviceState& s, const PlanKey& k, cudaDataType bias_type, Plan& p) {
  const cudaDataType in = k.bf16_in ? CUDA_R_16BF : CUDA_R_32F;
  const cudaDataType out = k.bf16_out ? CUDA_R_16BF : CUDA_R_32F;
  const uint64_t M = k.M, N = k.N, K = k.K;

  // Compute in float32 with float32 scales. The inputs are never widened to TF32 here: the
  // CUBLAS_COMPUTE_32F path is full FP32, and that is what the router and depth head need.
  if (auto st = cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F);
      st != CUBLAS_STATUS_SUCCESS)
    return st;
  // w is stored [N, K] row-major: as column-major it is [K, N], so it enters transposed.
  if (auto st = set_desc_attr(p.op, CUBLASLT_MATMUL_DESC_TRANSA, CUBLAS_OP_T); st != CUBLAS_STATUS_SUCCESS)
    return st;
  if (auto st = set_desc_attr(p.op, CUBLASLT_MATMUL_DESC_TRANSB, CUBLAS_OP_N); st != CUBLAS_STATUS_SUCCESS)
    return st;
  if (k.bias) {
    if (auto st = set_desc_attr(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, CUBLASLT_EPILOGUE_BIAS);
        st != CUBLAS_STATUS_SUCCESS)
      return st;
    if (auto st = set_desc_attr(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, bias_type);
        st != CUBLAS_STATUS_SUCCESS)
      return st;
  }

  // Column-major shapes: A = w as [K, N], B = x as [K, M], C = D = y^T as [N, M].
  if (auto st = cublasLtMatrixLayoutCreate(&p.a, in, K, N, K); st != CUBLAS_STATUS_SUCCESS) return st;
  if (auto st = cublasLtMatrixLayoutCreate(&p.b, in, K, M, k.ld_x); st != CUBLAS_STATUS_SUCCESS) return st;
  if (auto st = cublasLtMatrixLayoutCreate(&p.c, out, N, M, k.ld_y); st != CUBLAS_STATUS_SUCCESS) return st;

  cublasLtMatmulPreference_t pref = nullptr;
  if (auto st = cublasLtMatmulPreferenceCreate(&pref); st != CUBLAS_STATUS_SUCCESS) return st;
  cublasStatus_t st = set_pref_attr(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, kWorkspaceBytes);
  if (st == CUBLAS_STATUS_SUCCESS) st = set_pref_attr(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, k.align_a);
  if (st == CUBLAS_STATUS_SUCCESS) st = set_pref_attr(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, k.align_b);
  if (st == CUBLAS_STATUS_SUCCESS) st = set_pref_attr(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, k.align_c);
  if (st == CUBLAS_STATUS_SUCCESS) st = set_pref_attr(pref, CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_D_BYTES, k.align_c);

  cublasLtMatmulHeuristicResult_t best{};
  int found = 0;
  if (st == CUBLAS_STATUS_SUCCESS)
    st = cublasLtMatmulAlgoGetHeuristic(s.handle, p.op, p.a, p.b, p.c, p.c, pref, 1, &best, &found);
  cublasLtMatmulPreferenceDestroy(pref);
  if (st != CUBLAS_STATUS_SUCCESS) return st;
  if (found == 0 || best.state != CUBLAS_STATUS_SUCCESS) return CUBLAS_STATUS_NOT_SUPPORTED;
  p.algo = best.algo;
  return CUBLAS_STATUS_SUCCESS;
}

void flush_plans(DeviceState& s) {
  for (auto& entry : s.plans) entry.second.destroy();
  s.plans.clear();
}

cudaError_t ensure_scratch(DeviceState& s, size_t bytes) {
  if (s.bias_scratch_bytes >= bytes) return cudaSuccess;
  if (s.bias_scratch) cudaFree(s.bias_scratch);
  s.bias_scratch = nullptr;
  s.bias_scratch_bytes = 0;
  // Round up so a slowly growing N does not reallocate on every new shape.
  const size_t want = std::bit_ceil(bytes);
  void* p = nullptr;
  if (cudaError_t e = cudaMalloc(&p, want); e != cudaSuccess) return e;
  s.bias_scratch = p;
  s.bias_scratch_bytes = want;
  return cudaSuccess;
}

// out[i] = alpha * bias[i], rounded to TOut.
template <typename TIn, typename TOut>
__global__ void scale_bias_kernel(const TIn* __restrict__ bias, float alpha, TOut* __restrict__ out, int n) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) out[i] = lx::from_f32<TOut>(alpha * lx::to_f32(bias[i]));
}

template <typename TIn, typename TOut>
void launch_scale_bias(const void* bias, float alpha, void* out, int n, cudaStream_t stream) {
  const int blocks = (n + 255) / 256;
  scale_bias_kernel<TIn, TOut><<<blocks, 256, 0, stream>>>(static_cast<const TIn*>(bias), alpha,
                                                            static_cast<TOut*>(out), n);
}

struct Problem {
  const void* x;
  int ld_x;
  const void* w;
  const void* bias;  // bf16 when bf16_in, float32 otherwise; null for no bias
  void* y;
  int ld_y;
  bool bf16_in;   // x, w and bias are bf16; false means all float32
  bool bf16_out;  // y is bf16; false means float32
  int M, N, K;
  float alpha, beta;
  cudaStream_t stream;
};

cudaError_t run(const Problem& q) {
  if (q.M < 0 || q.N < 0 || q.K <= 0 || q.ld_x < q.K || q.ld_y < q.N) return cudaErrorInvalidValue;
  if (q.M == 0 || q.N == 0) return cudaSuccess;
  if (!q.x || !q.w || !q.y) return cudaErrorInvalidValue;

  DeviceState* s = device_state();
  if (!s) return cudaErrorInvalidDevice;
  if (s->init_error != cudaSuccess) return s->init_error;
  std::lock_guard lock(s->mu);

  // The epilogue's bias has the output's dtype. It is the caller's bias itself when that is already
  // so and alpha is 1; otherwise it is converted (and scaled) into scratch first.
  const cudaDataType bias_type = q.bf16_out ? CUDA_R_16BF : CUDA_R_32F;
  const void* epilogue_bias = nullptr;
  if (q.bias) {
    const bool direct = q.alpha == 1.0f && q.bf16_in == q.bf16_out;
    if (direct) {
      epilogue_bias = q.bias;
    } else {
      const size_t out_bytes = q.bf16_out ? 2 : 4;
      if (cudaError_t e = ensure_scratch(*s, static_cast<size_t>(q.N) * out_bytes); e != cudaSuccess) return e;
      if (q.bf16_in) {
        if (q.bf16_out) launch_scale_bias<__nv_bfloat16, __nv_bfloat16>(q.bias, q.alpha, s->bias_scratch, q.N, q.stream);
        else launch_scale_bias<__nv_bfloat16, float>(q.bias, q.alpha, s->bias_scratch, q.N, q.stream);
      } else {
        if (q.bf16_out) launch_scale_bias<float, __nv_bfloat16>(q.bias, q.alpha, s->bias_scratch, q.N, q.stream);
        else launch_scale_bias<float, float>(q.bias, q.alpha, s->bias_scratch, q.N, q.stream);
      }
      if (cudaError_t e = cudaGetLastError(); e != cudaSuccess) return e;
      epilogue_bias = s->bias_scratch;
    }
  }

  const uint64_t in_bytes = q.bf16_in ? 2 : 4;
  const uint64_t out_bytes = q.bf16_out ? 2 : 4;
  const PlanKey key{
      .M = q.M,
      .N = q.N,
      .K = q.K,
      .ld_x = q.ld_x,
      .ld_y = q.ld_y,
      .bf16_in = q.bf16_in,
      .bf16_out = q.bf16_out,
      .bias = epilogue_bias != nullptr,
      .align_a = alignment(q.w, static_cast<uint64_t>(q.K) * in_bytes),
      .align_b = alignment(q.x, static_cast<uint64_t>(q.ld_x) * in_bytes),
      .align_c = alignment(q.y, static_cast<uint64_t>(q.ld_y) * out_bytes),
  };

  auto it = s->plans.find(key);
  if (it == s->plans.end()) {
    if (s->plans.size() >= kMaxPlans) flush_plans(*s);
    Plan plan;
    if (cublasStatus_t st = build_plan(*s, key, bias_type, plan); st != CUBLAS_STATUS_SUCCESS) {
      plan.destroy();
      return to_cuda(st);
    }
    it = s->plans.emplace(key, plan).first;
  }
  Plan& plan = it->second;
  if (epilogue_bias) {
    if (cublasLtMatmulDescSetAttribute(plan.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &epilogue_bias,
                                       sizeof(epilogue_bias)) != CUBLAS_STATUS_SUCCESS)
      return cudaErrorUnknown;
  }

  // C is read only when beta != 0, and it is y itself, so the in-place update needs no copy.
  const float alpha = q.alpha;
  return to_cuda(cublasLtMatmul(s->handle, plan.op, &alpha, q.w, plan.a, q.x, plan.b, &q.beta, q.y, plan.c,
                                q.y, plan.c, &plan.algo, s->workspace, kWorkspaceBytes, q.stream));
}

}  // namespace

extern "C" int lx_gemm_bf16(const void* x, int ld_x, const void* w, const void* bias, void* y,
                            lx_dtype y_dtype, int ld_y, int M, int N, int K, float alpha, float beta,
                            void* stream) {
  if (y_dtype != LX_F32 && y_dtype != LX_BF16) return static_cast<int>(cudaErrorInvalidValue);
  return static_cast<int>(run(Problem{
      .x = x,
      .ld_x = ld_x,
      .w = w,
      .bias = bias,
      .y = y,
      .ld_y = ld_y,
      .bf16_in = true,
      .bf16_out = y_dtype == LX_BF16,
      .M = M,
      .N = N,
      .K = K,
      .alpha = alpha,
      .beta = beta,
      .stream = static_cast<cudaStream_t>(stream),
  }));
}

extern "C" int lx_gemm_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y,
                           int M, int N, int K, void* stream) {
  return static_cast<int>(run(Problem{
      .x = x,
      .ld_x = ld_x,
      .w = w,
      .bias = bias,
      .y = y,
      .ld_y = ld_y,
      .bf16_in = false,
      .bf16_out = false,
      .M = M,
      .N = N,
      .K = K,
      .alpha = 1.0f,
      .beta = 0.0f,
      .stream = static_cast<cudaStream_t>(stream),
  }));
}
