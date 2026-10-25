// How fast liblernex_cpu.so runs on this machine, against what the machine can do.
//
//   lx_cpu_bench [bandwidth|gemv|experts|gemm|overhead|all] [--mib N] [--reps N]
//
// bandwidth  a multithreaded read of a large buffer (the ceiling for decode, which reads every weight
//            once per token), at 1 thread, at the pool's size, and at each count in between that is a
//            power of two
// gemv       the decode GEMV at Praxis' shapes in every format (and lossless BF16X, with Gaussian bf16
//            weights so its patches come at a checkpoint's rate): time, and the weight bytes it read per
//            second as a fraction of the measured read bandwidth
// experts    Praxis' routed experts for one decode token (4 of 128) in every format, the same way
// gemm       a prefill-sized product (512 tokens) in every math, in GFLOP/s
// overhead   what a small op costs per call (decode makes hundreds of them per token)
//
// Times are the median of --reps runs after a warm-up; buffers are larger than the last-level cache, and
// rotate between calls where one copy would fit in it.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <string>
#include <thread>
#include <vector>

#include "cpu/lernex_cpu.h"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

namespace {

using Clock = std::chrono::steady_clock;
int g_reps = 9;
int64_t g_mib = 2048;
double g_bandwidth = 0;  // GB/s at the pool's thread count, once measured

double seconds_since(Clock::time_point t0) { return std::chrono::duration<double>(Clock::now() - t0).count(); }

template <class F>
double median_time(F&& f, int reps) {
  f();  // warm-up
  std::vector<double> t;
  for (int i = 0; i < reps; ++i) {
    const auto t0 = Clock::now();
    f();
    t.push_back(seconds_since(t0));
  }
  std::sort(t.begin(), t.end());
  return t[t.size() / 2];
}

// ---- Bandwidth --------------------------------------------------------------------------------------------

// Sums 64-bit words with eight independent accumulators, which the compiler vectorizes at any baseline.
uint64_t read_range(const uint64_t* p, int64_t n) {
  uint64_t a[8] = {};
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    for (int j = 0; j < 8; ++j) a[j] += p[i + j];
  for (; i < n; ++i) a[0] += p[i];
  return a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7];
}

double read_bandwidth(const uint64_t* buf, int64_t words, int threads, int reps) {
  std::atomic<uint64_t> sink{0};
  std::vector<std::thread> pool;
  std::atomic<int> go{0}, done{0};
  std::atomic<bool> quit{false};
  const int64_t per = (words + threads - 1) / threads;
  // Persistent threads that each read their slice once per round, so thread start-up is not timed.
  for (int t = 1; t < threads; ++t) {
    pool.emplace_back([&, t] {
      int seen = 0;
      for (;;) {
        int g;
        while ((g = go.load(std::memory_order_acquire)) == seen) {
          if (quit.load(std::memory_order_relaxed)) return;
        }
        seen = g;
        const int64_t b = t * per, e = std::min(words, b + per);
        if (b < e) sink.fetch_add(read_range(buf + b, e - b), std::memory_order_relaxed);
        done.fetch_add(1, std::memory_order_acq_rel);
      }
    });
  }
  auto round = [&] {
    done.store(0, std::memory_order_relaxed);
    go.fetch_add(1, std::memory_order_acq_rel);
    sink.fetch_add(read_range(buf, std::min(words, per)), std::memory_order_relaxed);
    while (done.load(std::memory_order_acquire) < threads - 1) {
    }
  };
  const double t = median_time(round, reps);
  quit.store(true);
  go.fetch_add(1, std::memory_order_acq_rel);
  for (auto& th : pool) th.join();
  if (sink.load() == 42) printf(" ");
  return static_cast<double>(words) * 8 / t / 1e9;
}

void bandwidth() {
  const int64_t bytes = g_mib << 20;
  const int64_t words = bytes / 8;
  uint64_t* buf = nullptr;
  lx_rt_alloc(reinterpret_cast<void**>(&buf), bytes);
  for (int64_t i = 0; i < words; ++i) buf[i] = static_cast<uint64_t>(i) * 0x9E3779B97F4A7C15ull;
  const int max_threads = static_cast<int>(std::thread::hardware_concurrency());
  const int pool = lx_cpu_threads();
  std::vector<int> counts = {1};
  for (int c = 2; c < max_threads; c *= 2) counts.push_back(c);
  counts.push_back(pool);
  counts.push_back(max_threads);
  std::sort(counts.begin(), counts.end());
  counts.erase(std::unique(counts.begin(), counts.end()), counts.end());
  printf("read bandwidth, %lld MiB buffer (pool: %d threads, hardware threads: %d)\n", static_cast<long long>(g_mib),
         pool, max_threads);
  double best = 0;
  for (int c : counts) {
    const double gbs = read_bandwidth(buf, words, c, g_reps);
    best = std::max(best, gbs);
    if (c == pool) g_bandwidth = gbs;
    printf("  %3d threads  %7.1f GB/s%s\n", c, gbs, c == pool ? "  (pool)" : "");
  }
  printf("  best         %7.1f GB/s\n", best);
  g_bandwidth = std::max(g_bandwidth, best);
  lx_rt_free(buf);
}

// ---- Weights ------------------------------------------------------------------------------------------------

struct Weights {
  int type;
  int64_t rows, K;
  void* data = nullptr;
  void* scale = nullptr;
  int64_t data_bytes = 0, scale_bytes = 0;
  lx_weight view(int64_t row0 = 0) const {
    lx_weight w{static_cast<lx_wtype>(type), data, scale, 0.01f, nullptr};
    if (type == LX_W_BF16) w.data = static_cast<uint16_t*>(data) + row0 * K;
    else if (type == LX_W_NVFP4) {
      w.data = static_cast<uint8_t*>(data) + row0 * K / 2;
      w.scale = static_cast<uint8_t*>(scale) + row0 * K / 16;
    } else {
      w.data = static_cast<uint8_t*>(data) + row0 * K;
      w.scale = static_cast<float*>(scale) + row0;
    }
    return w;
  }
  int64_t bytes(int64_t rows_read) const { return (data_bytes + scale_bytes) / rows * rows_read; }
};

Weights make(int type, int64_t rows, int64_t K) {
  Weights w;
  w.type = type;
  w.rows = rows;
  w.K = K;
  w.data_bytes = type == LX_W_BF16 ? 2 * rows * K : type == LX_W_NVFP4 ? rows * K / 2 : rows * K;
  w.scale_bytes = type == LX_W_BF16 ? 0 : type == LX_W_NVFP4 ? rows * K / 16 : 4 * rows;
  lx_rt_alloc(&w.data, w.data_bytes);
  uint8_t* d = static_cast<uint8_t*>(w.data);
  uint64_t s = 88172645463325252ull;
  for (int64_t i = 0; i < w.data_bytes; ++i) {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    d[i] = static_cast<uint8_t>(s);
  }
  if (type == LX_W_BF16) {
    uint16_t* b = static_cast<uint16_t*>(w.data);
    for (int64_t i = 0; i < rows * K; ++i) b[i] = static_cast<uint16_t>(0x3c00 | (b[i] & 0x80ff));  // finite, small
  } else if (type == LX_W_FP8) {
    for (int64_t i = 0; i < w.data_bytes; ++i) d[i] &= 0xb7;  // no NaN
  }
  if (w.scale_bytes) {
    lx_rt_alloc(&w.scale, w.scale_bytes);
    if (type == LX_W_NVFP4) memset(w.scale, 0x38, w.scale_bytes);  // e4m3 1.0
    else
      for (int64_t i = 0; i < rows; ++i) static_cast<float*>(w.scale)[i] = 0.01f;
  }
  return w;
}

void release(Weights& w) {
  lx_rt_free(w.data);
  if (w.scale) lx_rt_free(w.scale);
}

const char* fmt_name(int t) {
  static const char* n[] = {"BF16", "INT8", "FP8", "NVFP4"};
  return n[t];
}

// Gaussian bf16 rows (a checkpoint's spread of exponents) encoded as LX_W_BF16X: its bytes, the encoding.
struct Encoded {
  void* data = nullptr;
  int64_t bytes = 0;
};
Encoded make_bf16x(int64_t rows, int64_t K, float std) {
  std::vector<uint16_t> w(static_cast<size_t>(rows * K));
  uint64_t s = 0x2545F4914F6CDD1Dull;
  auto uniform = [&] {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return static_cast<float>((s >> 40) * (1.0 / 16777216.0));
  };
  for (auto& v : w) {
    const float g = sqrtf(-2.0f * logf(uniform() + 1e-7f)) * cosf(6.2831853f * uniform()) * std;
    uint32_t u;
    memcpy(&u, &g, 4);
    v = static_cast<uint16_t>((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
  }
  Encoded e;
  e.bytes = lx_bf16x_bytes(w.data(), rows, static_cast<int>(K), nullptr);
  lx_rt_alloc(&e.data, e.bytes);
  lx_bf16x_encode(w.data(), rows, static_cast<int>(K), e.data, e.bytes, nullptr);
  return e;
}

// ---- GEMV ---------------------------------------------------------------------------------------------------

void gemv() {
  struct Shape {
    const char* what;
    int N, K;
  };
  // Praxis decode products (hidden 2048): the Mamba in-projection, the attention qkv, a 2048-square, the
  // vocabulary head.
  const Shape shapes[] = {{"mamba in_proj", 10304, 2048}, {"attn qkv", 3072, 2048}, {"square", 2048, 2048},
                          {"lm head", 65536, 2048}};
  printf("decode GEMV (M = 1, bf16 input, f32 output)%s\n", g_bandwidth > 0 ? "" : " (run bandwidth for fractions)");
  for (const Shape& s : shapes) {
    for (int type = LX_W_BF16; type <= LX_W_NVFP4; ++type) {
      // Enough copies that each call reads from DRAM: at least 256 MiB in all.
      Weights one = make(type, s.N, s.K);
      const int copies = static_cast<int>(std::max<int64_t>(1, (256ll << 20) / one.data_bytes + 1));
      release(one);
      Weights w = make(type, int64_t{s.N} * copies, s.K);
      std::vector<uint16_t> x(s.K, 0x3f80);
      std::vector<float> y(s.N);
      int c = 0;
      const double t = median_time(
          [&] {
            const lx_weight v = w.view(int64_t{c} * s.N);
            c = (c + 1) % copies;
            lx_gemv_w(x.data(), LX_BF16, s.K, &v, nullptr, LX_F32, y.data(), LX_F32, s.N, 1, s.N, s.K, 1.0f, 0.0f,
                      nullptr);
          },
          std::max(g_reps * 5, copies * 3));
      const double gbs = w.bytes(s.N) / t / 1e9;
      printf("  %-14s %5d x %4d  %-5s  %8.1f us  %6.1f GB/s", s.what, s.N, s.K, fmt_name(type), t * 1e6, gbs);
      if (g_bandwidth > 0) printf("  %3.0f%% of read bandwidth", 100 * gbs / g_bandwidth);
      printf("\n");
      release(w);
    }
    // The same product on the lossless encoding of bf16 weights.
    const int64_t rb = lx_bf16x_row_bytes(s.K);
    const int copies = static_cast<int>(std::max<int64_t>(1, (256ll << 20) / (rb * s.N) + 1));
    Encoded e = make_bf16x(int64_t{s.N} * copies, s.K, 0.02f);
    std::vector<uint16_t> x(s.K, 0x3f80);
    std::vector<float> y(s.N);
    int c = 0;
    const double t = median_time(
        [&] {
          const char* w = static_cast<const char*>(e.data) + int64_t{c} * s.N * rb;
          c = (c + 1) % copies;
          lx_gemv_bf16x(x.data(), LX_BF16, s.K, w, nullptr, LX_F32, y.data(), LX_F32, s.N, 1, s.N, s.K, 1.0f, 0.0f,
                        nullptr);
        },
        std::max(g_reps * 5, copies * 3));
    const double gbs = static_cast<double>(e.bytes) / copies / t / 1e9;
    printf("  %-14s %5d x %4d  %-5s  %8.1f us  %6.1f GB/s", s.what, s.N, s.K, "BF16X", t * 1e6, gbs);
    if (g_bandwidth > 0) printf("  %3.0f%% of read bandwidth", 100 * gbs / g_bandwidth);
    printf("  (%.2f bits per value)\n", 8.0 * e.bytes / (static_cast<double>(s.N) * copies * s.K));
    lx_rt_free(e.data);
  }
}

// ---- Experts ------------------------------------------------------------------------------------------------

void experts() {
  const int E = 128, L = 1024, I = 512, top_k = 4;
  printf("routed experts, one decode token (top %d of %d, latent %d, intermediate %d)\n", top_k, E, L, I);
  for (int type = LX_W_BF16; type <= LX_W_NVFP4; ++type) {
    // Several layers' stacks, so consecutive calls read from DRAM, as consecutive layers would.
    Weights probe = make(type, int64_t{E} * 2 * I, L);
    const int layers = static_cast<int>(std::max<int64_t>(2, (512ll << 20) / (probe.data_bytes * 3 / 2) + 1));
    release(probe);
    std::vector<Weights> gu, dn;
    for (int l = 0; l < layers; ++l) {
      gu.push_back(make(type, int64_t{E} * 2 * I, L));
      dn.push_back(make(type, int64_t{E} * L, I));
    }
    std::vector<uint16_t> latent(L, 0x3f80);
    std::vector<float> out(L);
    std::vector<uint8_t> ws(lx_moe_decode_workspace_bytes(1, I, top_k) + 64);
    int32_t ids[top_k];
    float weights[top_k] = {0.4f, 0.3f, 0.2f, 0.1f};
    int layer = 0, step = 0;
    const double t = median_time(
        [&] {
          for (int k = 0; k < top_k; ++k) ids[k] = (step * 37 + k * 29) % E;
          ++step;
          const lx_weight g = gu[layer].view(), d = dn[layer].view();
          layer = (layer + 1) % layers;
          lx_moe_experts_decode(latent.data(), ids, weights, &g, &d, out.data(), ws.data(), 1, L, I, top_k, nullptr);
        },
        std::max(g_reps * 5, layers * 3));
    const int64_t bytes = top_k * (gu[0].bytes(2 * I) + dn[0].bytes(L));
    const double gbs = bytes / t / 1e9;
    printf("  %-5s  %8.1f us  %6.1f GB/s", fmt_name(type), t * 1e6, gbs);
    if (g_bandwidth > 0) printf("  %3.0f%% of read bandwidth", 100 * gbs / g_bandwidth);
    printf("\n");
    for (auto& w : gu) release(w);
    for (auto& w : dn) release(w);
  }
  // The lossless encoding of bf16 stacks.
  {
    const int64_t gu_rb = lx_bf16x_row_bytes(L), dn_rb = lx_bf16x_row_bytes(I);
    const int layers = static_cast<int>(std::max<int64_t>(2, (512ll << 20) / (E * (2 * I * gu_rb + L * dn_rb)) + 1));
    std::vector<Encoded> gu, dn;
    for (int l = 0; l < layers; ++l) {
      gu.push_back(make_bf16x(int64_t{E} * 2 * I, L, 0.03f));
      dn.push_back(make_bf16x(int64_t{E} * L, I, 0.04f));
    }
    std::vector<uint16_t> latent(L, 0x3f80);
    std::vector<float> out(L);
    std::vector<uint8_t> ws(lx_moe_bf16x_workspace_bytes(1, I, top_k) + 64);
    int32_t ids[top_k];
    float weights[top_k] = {0.4f, 0.3f, 0.2f, 0.1f};
    int layer = 0, step = 0;
    const double t = median_time(
        [&] {
          for (int k = 0; k < top_k; ++k) ids[k] = (step * 37 + k * 29) % E;
          ++step;
          lx_moe_experts_bf16x(latent.data(), ids, weights, gu[layer].data, dn[layer].data, out.data(), ws.data(), 1, L,
                               I, top_k, nullptr);
          layer = (layer + 1) % layers;
        },
        std::max(g_reps * 5, layers * 3));
    const int64_t bytes = top_k * (2 * I * gu_rb + L * dn_rb);
    const double gbs = bytes / t / 1e9;
    printf("  %-5s  %8.1f us  %6.1f GB/s", "BF16X", t * 1e6, gbs);
    if (g_bandwidth > 0) printf("  %3.0f%% of read bandwidth", 100 * gbs / g_bandwidth);
    printf("\n");
    for (auto& e : gu) lx_rt_free(e.data);
    for (auto& e : dn) lx_rt_free(e.data);
  }
}

// ---- GEMM ---------------------------------------------------------------------------------------------------

void gemm() {
  const int M = 512, N = 2048, K = 2048;
  printf("prefill GEMM (M = %d, N = %d, K = %d, f32 input and output)\n", M, N, K);
  struct MathCase {
    int math, type;
  };
  const MathCase cases[] = {{LX_MATH_BF16, LX_W_BF16}, {LX_MATH_BF16, LX_W_INT8}, {LX_MATH_INT8, LX_W_INT8},
                            {LX_MATH_FP8, LX_W_FP8},   {LX_MATH_FP4, LX_W_NVFP4}};
  static const char* math_names[] = {"BF16", "INT8", "FP8", "FP4"};
  std::vector<float> x(static_cast<size_t>(M) * K);
  for (size_t i = 0; i < x.size(); ++i) x[i] = static_cast<float>((i * 2654435761u) % 1000) / 1000.0f - 0.5f;
  std::vector<float> y(static_cast<size_t>(M) * N);
  for (const MathCase& c : cases) {
    Weights w = make(c.type, N, K);
    std::vector<uint8_t> ws(lx_gemm_workspace_bytes(M, K, static_cast<lx_math>(c.math)));
    const lx_weight v = w.view();
    const double t = median_time(
        [&] {
          lx_gemm_w(x.data(), LX_F32, K, &v, static_cast<lx_math>(c.math), nullptr, LX_F32, y.data(), LX_F32, N, M, N,
                    K, 1.0f, 0.0f, ws.data(), nullptr);
        },
        g_reps);
    printf("  math %-4s weights %-5s  %8.2f ms  %7.1f GFLOP/s\n", math_names[c.math], fmt_name(c.type), t * 1e3,
           2.0 * M * N * K / t / 1e9);
    release(w);
  }
}

// ---- Per-call overhead ------------------------------------------------------------------------------------

// Decode makes a few hundred calls per token, most of them small: what one costs when the work is tiny.
void overhead() {
  printf("per-call cost of small ops (median, us)\n");
  std::vector<float> f(1 << 16, 0.5f), g(1 << 16);
  auto report = [](const char* what, double t) { printf("  %-44s %8.2f us\n", what, t * 1e6); };
  report("activate_f32, 256 values", median_time([&] { lx_activate_f32(f.data(), 256, 0, nullptr); }, 2001));
  report("stream_mean, 1 token x 4 streams x 2048", median_time([&] { lx_stream_mean(f.data(), g.data(), 1, 4, 2048, nullptr); }, 2001));
  for (int type : {LX_W_BF16, LX_W_INT8}) {
    for (int N : {64, 256, 1024}) {
      const int K = 2048;
      Weights w = make(type, N, K);
      std::vector<uint16_t> x(K, 0x3f80);
      std::vector<float> y(N);
      const lx_weight v = w.view();
      char what[96];
      snprintf(what, sizeof(what), "gemv %s, %d x %d (%.0f KB)", fmt_name(type), N, K, w.bytes(N) / 1024.0);
      report(what, median_time([&] {
               lx_gemv_w(x.data(), LX_BF16, K, &v, nullptr, LX_F32, y.data(), LX_F32, N, 1, N, K, 1.0f, 0.0f, nullptr);
             }, 2001));
      release(w);
    }
  }
}

}  // namespace

int main(int argc, char** argv) {
  std::string what = "all";
  for (int i = 1; i < argc; ++i) {
    if (!strcmp(argv[i], "--mib") && i + 1 < argc) g_mib = atoll(argv[++i]);
    else if (!strcmp(argv[i], "--reps") && i + 1 < argc) g_reps = atoi(argv[++i]);
    else what = argv[i];
  }
  printf("lernex_cpu bench: level %d, %d threads\n", lx_arch(), lx_cpu_threads());
  if (what == "bandwidth" || what == "all" || what == "gemv" || what == "experts") bandwidth();
  if (what == "gemv" || what == "all") gemv();
  if (what == "experts" || what == "all") experts();
  if (what == "gemm" || what == "all") gemm();
  if (what == "overhead" || what == "all") overhead();
  return 0;
}
