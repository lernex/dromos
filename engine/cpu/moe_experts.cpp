// Routed experts on the CPU, on host memory: lx_moe_experts_decode (kernels.h) for the CPU backend,
// and the offload runtime's CPU lane (engine/runtime/offload.cpp), which calls it only through that
// contract. Weight-only, f32 math, the intermediate rounded to bf16 before down, like every GPU
// implementation, so the CPU's share of a step matches the GPU's to float rounding.
//
// Decode reads each chosen expert once from RAM, so the kernel is bound by memory bandwidth: it is
// split into row blocks across a pool of threads that spin between calls (a call is ~100 us, a
// sleeping thread takes longer than that to wake), and the inner loops widen 16 weights at a time
// with AVX2 + FMA + F16C where the CPU has them (runtime dispatch, scalar otherwise):
//   BF16   16 bf16 -> two 8 x f32 by a zero-extend and a 16-bit shift
//   INT8   16 int8 -> two 8 x f32 by sign extension
//   FP8    16 e4m3 -> f16 bit patterns (sign | exponent-mantissa << 7, exact for every finite e4m3
//          including subnormals, the value scaled by 2^-8), then F16C; the 2^8 goes into the row scale
//   NVFP4  scalar (a 16-entry table per nibble, the e4m3 block scale, the global)
// AVX-VNNI multiplies int8 by int8; with f32 activations (weight-only math) it has nothing to do here.
//
// The stream argument is ignored: the call returns when the outputs are written.
#include <immintrin.h>
#include <pthread.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <thread>
#include <type_traits>
#include <vector>

#include "cuda/kernels.h"

namespace {

inline float bf16_to_f32(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

inline float round_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return f;  // inf / nan pass through
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

float e4m3_to_f32(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
  if (e == 15 && m == 7) v = NAN;
  return s ? -v : v;
}

const float kE2m1[16] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
                         -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f};

// ---- the thread pool --------------------------------------------------------------------------

// parallel_for(n, fn): fn(i) for i < n across the pool and the calling thread.
//
// The caller waits for the job's tasks, never for the threads: on a busy machine some workers are
// descheduled for milliseconds, and a barrier on every thread would stall each call for that long.
// Tasks are claimed from the job's own counter, so a worker that wakes late either finds tasks left
// or none; a job lives on the caller's stack, and the caller retracts it and waits for the workers
// that still hold a reference (they are running: they just took it) before returning.
// Workers spin for ~1 ms after each job (decode calls come every ~100 us while a model generates),
// then sleep. The default size is half the hardware threads: the kernel is bound by RAM bandwidth,
// which the physical cores saturate (more threads only add scheduling risk).
class Pool {
 public:
  static Pool& get() {
    static Pool pool;
    return pool;
  }
  int size() const { return static_cast<int>(threads_.size()) + 1; }

  template <typename Fn>
  void parallel_for(int64_t n, Fn&& fn) {
    using F = std::remove_reference_t<Fn>;
    if (n <= 0) return;
    if (threads_.empty() || n == 1) {
      for (int64_t i = 0; i < n; ++i) fn(i);
      return;
    }
    std::lock_guard<std::mutex> call(call_mutex_);  // one job at a time
    Job job;
    job.ctx = &fn;
    job.run = [](void* ctx, int64_t i) { (*static_cast<F*>(ctx))(i); };
    job.n = n;
    current_.store(&job, std::memory_order_seq_cst);
    {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      generation_.fetch_add(1, std::memory_order_seq_cst);
    }
    wake_.notify_all();
    work(job);
    while (job.done.load(std::memory_order_acquire) < n) _mm_pause();
    current_.store(nullptr, std::memory_order_seq_cst);
    while (job.refs.load(std::memory_order_seq_cst) > 0) _mm_pause();
  }

 private:
  struct Job {
    void* ctx = nullptr;
    void (*run)(void*, int64_t) = nullptr;
    int64_t n = 0;
    std::atomic<int64_t> next{0}, done{0};
    std::atomic<int> refs{0};
  };

  Pool() {
    int n = 0;
    if (const char* env = std::getenv("LX_CPU_THREADS")) n = std::atoi(env);
    if (n <= 0) n = static_cast<int>(std::thread::hardware_concurrency()) / 2;
    n = std::max(1, std::min(n, 64));
    for (int i = 0; i + 1 < n; ++i) threads_.emplace_back([this] { loop(); });
  }
  ~Pool() {
    stop_.store(true);
    {
      std::lock_guard<std::mutex> lock(sleep_mutex_);
      generation_.fetch_add(1);
    }
    wake_.notify_all();
    for (auto& t : threads_) t.join();
  }

  static void work(Job& job) {
    for (;;) {
      const int64_t i = job.next.fetch_add(1, std::memory_order_relaxed);
      if (i >= job.n) break;
      job.run(job.ctx, i);
      job.done.fetch_add(1, std::memory_order_release);
    }
  }

  // Takes part in the posted job, if there still is one.
  void help() {
    Job* job = current_.load(std::memory_order_seq_cst);
    if (!job) return;
    job->refs.fetch_add(1, std::memory_order_seq_cst);
    if (current_.load(std::memory_order_seq_cst) == job) work(*job);
    job->refs.fetch_sub(1, std::memory_order_seq_cst);
  }

  void loop() {
    uint64_t seen = 0;
    for (;;) {
      uint64_t g = seen;
      for (int spin = 0; spin < 8000 && (g = generation_.load(std::memory_order_acquire)) == seen; ++spin)
        for (int k = 0; k < 8; ++k) _mm_pause();
      if (g == seen) {
        std::unique_lock<std::mutex> lock(sleep_mutex_);
        wake_.wait(lock, [&] { return generation_.load(std::memory_order_acquire) != seen; });
        g = generation_.load(std::memory_order_acquire);
      }
      seen = g;
      if (stop_.load()) return;
      help();
    }
  }

  std::vector<std::thread> threads_;
  std::mutex call_mutex_, sleep_mutex_;
  std::condition_variable wake_;
  std::atomic<uint64_t> generation_{0};
  std::atomic<Job*> current_{nullptr};
  std::atomic<bool> stop_{false};
};

// ---- row dot products ---------------------------------------------------------------------------
// dot(row, x) over n weights of one row in its format (no row scale), x f32.

float dot_scalar(lx_wtype type, const void* row, const void* block_scale, const float* x, int n) {
  float acc = 0.0f;
  switch (type) {
    case LX_W_BF16: {
      const uint16_t* r = static_cast<const uint16_t*>(row);
      for (int k = 0; k < n; ++k) acc += bf16_to_f32(r[k]) * x[k];
      break;
    }
    case LX_W_INT8: {
      const int8_t* r = static_cast<const int8_t*>(row);
      for (int k = 0; k < n; ++k) acc += static_cast<float>(r[k]) * x[k];
      break;
    }
    case LX_W_FP8: {
      const uint8_t* r = static_cast<const uint8_t*>(row);
      for (int k = 0; k < n; ++k) acc += e4m3_to_f32(r[k]) * x[k];
      break;
    }
    case LX_W_NVFP4: {
      const uint8_t* r = static_cast<const uint8_t*>(row);
      const uint8_t* bs = static_cast<const uint8_t*>(block_scale);
      for (int b = 0; b < n / 16; ++b) {
        float part = 0.0f;
        for (int j = 0; j < 16; ++j) {
          const int k = b * 16 + j;
          const uint8_t byte = r[k / 2];
          part += kE2m1[(k & 1) ? byte >> 4 : byte & 15] * x[k];
        }
        acc += part * e4m3_to_f32(bs[b]);
      }
      break;
    }
  }
  return acc;
}

__attribute__((target("avx2,fma"))) inline float hsum(__m256 v) {
  __m128 s = _mm_add_ps(_mm256_castps256_ps128(v), _mm256_extractf128_ps(v, 1));
  s = _mm_add_ps(s, _mm_movehl_ps(s, s));
  s = _mm_add_ss(s, _mm_movehdup_ps(s));
  return _mm_cvtss_f32(s);
}

// 16 e4m3 bytes -> 16 floats scaled by 2^-8, as two vectors.
__attribute__((target("avx2,fma,f16c"))) inline void fp8x16(__m128i b, __m256& lo, __m256& hi) {
  const __m256i x = _mm256_cvtepu8_epi16(b);
  const __m256i mag = _mm256_slli_epi16(_mm256_and_si256(x, _mm256_set1_epi16(0x7f)), 7);
  const __m256i sgn = _mm256_slli_epi16(_mm256_and_si256(x, _mm256_set1_epi16(0x80)), 8);
  const __m256i h = _mm256_or_si256(mag, sgn);
  lo = _mm256_cvtph_ps(_mm256_castsi256_si128(h));
  hi = _mm256_cvtph_ps(_mm256_extracti128_si256(h, 1));
}

// FP8 rows come back scaled by 2^-8 (see fp8x16); the caller multiplies the 2^8 back.
__attribute__((target("avx2,fma,f16c"))) float dot_avx2(lx_wtype type, const void* row, const float* x, int n) {
  __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(), a2 = _mm256_setzero_ps(), a3 = _mm256_setzero_ps();
  int k = 0;
  switch (type) {
    case LX_W_BF16: {
      const uint16_t* r = static_cast<const uint16_t*>(row);
      for (; k + 32 <= n; k += 32) {
        const __m256i w0 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + k));
        const __m256i w1 = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + k + 16));
        const __m256 f0 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(w0)), 16));
        const __m256 f1 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(w0, 1)), 16));
        const __m256 f2 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_castsi256_si128(w1)), 16));
        const __m256 f3 = _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm256_extracti128_si256(w1, 1)), 16));
        a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + k + 8), a1);
        a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + k + 16), a2);
        a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + k + 24), a3);
      }
      float acc = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
      for (; k < n; ++k) acc += bf16_to_f32(r[k]) * x[k];
      return acc;
    }
    case LX_W_INT8: {
      const int8_t* r = static_cast<const int8_t*>(row);
      for (; k + 32 <= n; k += 32) {
        const __m256i w = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(r + k));
        const __m128i lo = _mm256_castsi256_si128(w), hi = _mm256_extracti128_si256(w, 1);
        const __m256 f0 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(lo));
        const __m256 f1 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(lo, 8)));
        const __m256 f2 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(hi));
        const __m256 f3 = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(hi, 8)));
        a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + k + 8), a1);
        a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + k + 16), a2);
        a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + k + 24), a3);
      }
      float acc = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
      for (; k < n; ++k) acc += static_cast<float>(r[k]) * x[k];
      return acc;
    }
    case LX_W_FP8: {
      const uint8_t* r = static_cast<const uint8_t*>(row);
      for (; k + 32 <= n; k += 32) {
        __m256 f0, f1, f2, f3;
        fp8x16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r + k)), f0, f1);
        fp8x16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(r + k + 16)), f2, f3);
        a0 = _mm256_fmadd_ps(f0, _mm256_loadu_ps(x + k), a0);
        a1 = _mm256_fmadd_ps(f1, _mm256_loadu_ps(x + k + 8), a1);
        a2 = _mm256_fmadd_ps(f2, _mm256_loadu_ps(x + k + 16), a2);
        a3 = _mm256_fmadd_ps(f3, _mm256_loadu_ps(x + k + 24), a3);
      }
      float acc = hsum(_mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3)));
      for (; k < n; ++k) acc += e4m3_to_f32(r[k]) * (1.0f / 256.0f) * x[k];
      return acc;
    }
    default: return 0.0f;
  }
}

std::atomic<bool> g_force_scalar{false};

bool have_avx2() {
  static const bool ok = [] {
    if (std::getenv("LX_CPU_SCALAR")) return false;
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma") && __builtin_cpu_supports("f16c");
  }();
  return ok && !g_force_scalar.load(std::memory_order_relaxed);
}

// One row of a stacked weight: the row's bytes, its NVFP4 block scales, and its full scale factor.
struct Row {
  const void* data;
  const void* block_scale;
  float factor;
};

inline Row row_of(const lx_weight& w, int64_t row, int cols, bool fp8_prescaled) {
  Row r{};
  switch (w.type) {
    case LX_W_BF16:
      r.data = static_cast<const uint16_t*>(w.data) + row * cols;
      r.factor = 1.0f;
      break;
    case LX_W_INT8:
      r.data = static_cast<const int8_t*>(w.data) + row * cols;
      r.factor = static_cast<const float*>(w.scale)[row];
      break;
    case LX_W_FP8:
      r.data = static_cast<const uint8_t*>(w.data) + row * cols;
      r.factor = static_cast<const float*>(w.scale)[row] * (fp8_prescaled ? 256.0f : 1.0f);
      break;
    case LX_W_NVFP4:
      r.data = static_cast<const uint8_t*>(w.data) + row * (cols / 2);
      r.block_scale = static_cast<const uint8_t*>(w.scale) + row * (cols / 16);
      r.factor = w.global;
      break;
  }
  return r;
}

inline float dot(const lx_weight& w, const Row& r, const float* x, int n, bool avx2) {
  if (avx2 && w.type != LX_W_NVFP4) return dot_avx2(w.type, r.data, x, n);
  return dot_scalar(w.type, r.data, r.block_scale, x, n);
}

constexpr int kRowBlock = 64;  // rows per task: 64 rows of a Metis expert are 32-128 KB

}  // namespace

extern "C" {

LX_API int64_t lx_moe_decode_workspace_bytes(int T, int I, int top_k) {
  return static_cast<int64_t>(T) * top_k * I * 4;
}

LX_API int lx_moe_experts_decode(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                                 const lx_weight* down, float* out, void* workspace, int T, int L, int I, int top_k,
                                 void* stream) {
  (void)stream;
  if (T <= 0) return 0;
  if (!latent || !ids || !weights || !gate_up || !down || !out || !workspace || top_k <= 0) return 1;  // invalid value
  if (gate_up->type != down->type) return 1;
  if (gate_up->type == LX_W_NVFP4 && (L % 16 || I % 16)) return 1;
  const bool avx2 = have_avx2();
  const bool pre = avx2;  // the AVX2 FP8 path returns values scaled by 2^-8
  const int P = T * top_k;
  float* inter = static_cast<float*>(workspace);  // [P, I], bf16-rounded values
  // the latent as f32, once
  std::vector<float> x(static_cast<size_t>(T) * L);
  const uint16_t* lb = static_cast<const uint16_t*>(latent);
  for (size_t i = 0; i < x.size(); ++i) x[i] = bf16_to_f32(lb[i]);

  Pool& pool = Pool::get();
  // Stage 1: pair p, rows [b * kRowBlock, ...) of the intermediate.
  const int blocks1 = (I + kRowBlock - 1) / kRowBlock;
  pool.parallel_for(static_cast<int64_t>(P) * blocks1, [&](int64_t task) {
    const int p = static_cast<int>(task / blocks1), b = static_cast<int>(task % blocks1);
    const int64_t e = ids[p];
    const float* xt = x.data() + static_cast<int64_t>(p / top_k) * L;
    const int end = std::min(I, (b + 1) * kRowBlock);
    for (int i = b * kRowBlock; i < end; ++i) {
      const Row g = row_of(*gate_up, e * 2 * I + i, L, pre);
      const Row u = row_of(*gate_up, e * 2 * I + I + i, L, pre);
      const float gv = g.factor * dot(*gate_up, g, xt, L, avx2);
      const float uv = u.factor * dot(*gate_up, u, xt, L, avx2);
      const float silu = gv / (1.0f + std::exp(-gv));
      inter[static_cast<int64_t>(p) * I + i] = round_bf16(silu * uv);
    }
  });
  // Stage 2: token t, output rows [b * kRowBlock, ...), summed over the token's pairs in k order.
  const int blocks2 = (L + kRowBlock - 1) / kRowBlock;
  pool.parallel_for(static_cast<int64_t>(T) * blocks2, [&](int64_t task) {
    const int t = static_cast<int>(task / blocks2), b = static_cast<int>(task % blocks2);
    const int end = std::min(L, (b + 1) * kRowBlock);
    for (int l = b * kRowBlock; l < end; ++l) {
      float acc = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const int p = t * top_k + k;
        const Row d = row_of(*down, static_cast<int64_t>(ids[p]) * L + l, I, pre);
        acc += weights[p] * (d.factor * dot(*down, d, inter + static_cast<int64_t>(p) * I, I, avx2));
      }
      out[static_cast<int64_t>(t) * L + l] = acc;
    }
  });
  return 0;
}

// The number of threads the CPU kernels use (LX_CPU_THREADS, default every hardware thread).
LX_API int lx_cpu_threads(void) { return Pool::get().size(); }

// Tests: run the portable scalar loops even where AVX2 is available.
LX_API void lx_cpu_force_scalar(int on) { g_force_scalar.store(on != 0); }

}  // extern "C"
