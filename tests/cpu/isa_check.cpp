// libdromos_cpu.so's kernels at every ISA level this machine runs, without PyTorch.
//
// Each case runs at every level (lx_set_arch_override) and is checked two ways: against a reference
// written here in double precision (the products and experts, whose maths are simple), and against the
// generic level (every case, so the vector code is held to the portable code, which the PyTorch suite in
// tests/kernels checks on the x86 server). It also checks the invariants the model relies on at each
// level: a row's result does not depend on the batch it is in, and repeated calls are bitwise equal.
//
//   lx_cpu_isa_check [filter]    exit status 0 when everything passes
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "cpu/dromos_cpu.h"
#include "cuda/kernels.h"
#include "cuda/tunable.h"

namespace {

// ---- Data ---------------------------------------------------------------------------------------------

struct Rng {
  uint64_t s;
  explicit Rng(uint64_t seed) : s(seed * 0x9E3779B97F4A7C15ull + 1) {}
  uint64_t next() {
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return s;
  }
  float uniform() { return static_cast<float>((next() >> 40) * (1.0 / 16777216.0)); }  // [0, 1)
  float normal() {
    float u = uniform() + 1e-7f, v = uniform();
    return sqrtf(-2.0f * logf(u)) * cosf(6.2831853f * v);
  }
  int range(int n) { return static_cast<int>(next() % static_cast<uint64_t>(n)); }
};

float bf(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  memcpy(&f, &u, 4);
  return f;
}
uint16_t to_bf(float f) {
  uint32_t u;
  memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}
// e4m3fn and e2m1, decoded independently of the library.
double e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  const double v = e == 0 ? ldexp(m, -9) : ldexp(1.0 + m / 8.0, e - 7);
  return s ? -v : v;
}
double e2m1(int c) {
  static const double t[8] = {0, 0.5, 1, 1.5, 2, 3, 4, 6};
  return (c & 8) ? -t[c & 7] : t[c & 7];
}

std::vector<uint16_t> bf16_vec(Rng& r, size_t n, float scale) {
  std::vector<uint16_t> v(n);
  for (auto& x : v) x = to_bf(r.normal() * scale);
  return v;
}
std::vector<float> f32_vec(Rng& r, size_t n, float scale) {
  std::vector<float> v(n);
  for (auto& x : v) x = r.normal() * scale;
  return v;
}

// A weight [rows, K] (rows = E * N for a stack) in one format, with its dequantized values in double
// (row factors applied) for the references.
struct Weight {
  int type = LX_W_BF16;
  int64_t rows = 0, K = 0;
  std::vector<uint8_t> data, scale;
  float global = 1.0f;
  std::vector<float> expert_global;
  std::vector<double> deq;
  lx_weight view() const {
    return lx_weight{static_cast<lx_wtype>(type), data.data(), scale.empty() ? nullptr : scale.data(), global,
                     expert_global.empty() ? nullptr : expert_global.data()};
  }
};

Weight make_weight(Rng& r, int type, int64_t rows, int64_t K, float std, int experts = 1) {
  Weight w;
  w.type = type;
  w.rows = rows;
  w.K = K;
  w.deq.resize(rows * K);
  if (type == LX_W_BF16) {
    w.data.resize(2 * rows * K);
    uint16_t* d = reinterpret_cast<uint16_t*>(w.data.data());
    for (int64_t i = 0; i < rows * K; ++i) {
      d[i] = to_bf(r.normal() * std);
      w.deq[i] = bf(d[i]);
    }
  } else if (type == LX_W_INT8 || type == LX_W_FP8) {
    w.data.resize(rows * K);
    w.scale.resize(4 * rows);
    float* s = reinterpret_cast<float*>(w.scale.data());
    for (int64_t n = 0; n < rows; ++n) {
      s[n] = std * (0.5f + r.uniform()) / (type == LX_W_INT8 ? 64.0f : 2.0f);
      for (int64_t k = 0; k < K; ++k) {
        uint8_t b;
        double v;
        if (type == LX_W_INT8) {
          const int q = r.range(255) - 127;
          b = static_cast<uint8_t>(static_cast<int8_t>(q));
          v = q;
        } else {
          do b = static_cast<uint8_t>(r.next()); while ((b & 0x7f) == 0x7f || ((b >> 3) & 15) > 9);
          v = e4m3(b);
        }
        w.data[n * K + k] = b;
        w.deq[n * K + k] = v * s[n];
      }
    }
  } else {
    w.data.resize(rows * K / 2);
    w.scale.resize(rows * K / 16);
    w.global = std / 3.0f;
    if (experts > 1) {
      w.expert_global.resize(experts);
      for (auto& g : w.expert_global) g = std * (0.2f + r.uniform() * 0.4f);
    }
    const int64_t per_expert = rows / experts;
    for (int64_t n = 0; n < rows; ++n) {
      const double g = w.expert_global.empty() ? w.global : w.expert_global[n / per_expert];
      for (int64_t b = 0; b < K / 16; ++b) {
        uint8_t sb;
        do sb = static_cast<uint8_t>(0x28 + r.range(0x20)); while (false);  // e4m3 in [0.125, 1.75]
        w.scale[n * (K / 16) + b] = sb;
      }
      for (int64_t k = 0; k < K; k += 2) {
        const int c0 = r.range(16), c1 = r.range(16);
        w.data[n * (K / 2) + k / 2] = static_cast<uint8_t>(c0 | (c1 << 4));
        const double sb = e4m3(w.scale[n * (K / 16) + k / 16]);
        w.deq[n * K + k] = e2m1(c0) * sb * g;
        w.deq[n * K + k + 1] = e2m1(c1) * sb * g;
      }
    }
  }
  return w;
}

const char* format_name(int t) {
  static const char* n[] = {"bf16", "int8", "fp8", "nvfp4"};
  return n[t];
}
const char* math_name(int m) {
  static const char* n[] = {"bf16", "int8", "fp8", "fp4"};
  return n[m];
}

// ---- Harness -------------------------------------------------------------------------------------------

struct Level {
  int code;
  const char* name;
};
const Level kAll[] = {{1, "generic"},   {10, "avx2"},       {11, "avxvnni"}, {20, "avx512"},  {21, "avx512vnni"},
                      {22, "avx512bf16"}, {50, "neon"},     {51, "neondot"}, {52, "neoni8mm"}};

std::vector<Level> g_levels;
int g_failures = 0, g_checks = 0;
std::string g_filter;

bool set_level(int code) { return lx_set_arch_override(code) == code; }

// max |a - b| <= atol + rtol * max |b|, over the whole output.
bool near(const std::vector<float>& a, const std::vector<double>& b, double rtol, double atol, double* worst) {
  double mx = 0, err = 0;
  for (size_t i = 0; i < b.size(); ++i) mx = fmax(mx, fabs(b[i]));
  for (size_t i = 0; i < b.size(); ++i) {
    const double e = fabs(static_cast<double>(a[i]) - b[i]);
    if (!(e <= err)) err = e;  // NaN propagates
  }
  *worst = mx > 0 ? err / mx : err;
  return err <= atol + rtol * mx;
}

std::vector<double> widen(const std::vector<float>& v) { return std::vector<double>(v.begin(), v.end()); }

void report(bool ok, const std::string& what, const char* level, double worst) {
  ++g_checks;
  if (!ok) {
    ++g_failures;
    printf("FAIL  %-11s %-58s rel err %.3g\n", level, what.c_str(), worst);
  }
}

// One case: run(level) gives the output; ref (optional) the double reference; rtol the tolerance for
// both checks (the reference and the generic level).
struct Case {
  std::string name;
  std::function<std::vector<float>()> run;
  std::function<std::vector<double>()> ref;
  double rtol;
};

void check(const Case& c) {
  if (!g_filter.empty() && c.name.find(g_filter) == std::string::npos) return;
  set_level(1);
  const std::vector<float> generic = c.run();
  std::vector<double> ref;
  if (c.ref) ref = c.ref();
  for (const Level& l : g_levels) {
    set_level(l.code);
    const std::vector<float> out = c.run();
    double worst = 0;
    if (c.ref) report(near(out, ref, c.rtol, 0.0, &worst), c.name + " vs reference", l.name, worst);
    if (l.code != 1) report(near(out, widen(generic), c.rtol, 0.0, &worst), c.name + " vs generic", l.name, worst);
    const std::vector<float> again = c.run();
    report(memcmp(out.data(), again.data(), 4 * out.size()) == 0, c.name + " repeats bitwise", l.name, 0);
  }
  set_level(0);
}

// A per-level invariant, true when it holds.
void invariant(const std::string& name, const std::function<bool()> f) {
  if (!g_filter.empty() && name.find(g_filter) == std::string::npos) return;
  for (const Level& l : g_levels) {
    set_level(l.code);
    report(f(), name, l.name, 0);
  }
  set_level(0);
}

std::vector<float> from_bf16(const std::vector<uint16_t>& v) {
  std::vector<float> o(v.size());
  for (size_t i = 0; i < v.size(); ++i) o[i] = bf(v[i]);
  return o;
}

#define CALL(expr)                                                    \
  do {                                                                \
    const int st_ = (expr);                                           \
    if (st_ != 0) {                                                   \
      printf("FAIL  %s returned %d (%s:%d)\n", #expr, st_, __FILE__, __LINE__); \
      ++g_failures;                                                   \
    }                                                                 \
  } while (0)

// ---- Products ------------------------------------------------------------------------------------------

void products() {
  // Decode GEMV, every format, f32 and bf16 inputs, bias and beta, and odd K where the format allows.
  for (int type = LX_W_BF16; type <= LX_W_NVFP4; ++type) {
    for (int M : {1, 3, 8}) {
      for (int K : {1040, 1037}) {
        if (type == LX_W_NVFP4 && K % 16) continue;
        const int N = 67;
        Rng r(type * 100 + M * 10 + K);
        auto w = std::make_shared<Weight>(make_weight(r, type, N, K, 1.0f / sqrtf(K)));
        auto x = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(M) * K, 1.0f));
        auto bias = std::make_shared<std::vector<float>>(f32_vec(r, N, 0.1f));
        auto y0 = std::make_shared<std::vector<float>>(f32_vec(r, static_cast<size_t>(M) * N, 1.0f));
        const float alpha = 0.75f, beta = 0.5f;
        Case c;
        c.name = std::string("gemv_w ") + format_name(type) + " M=" + std::to_string(M) + " K=" + std::to_string(K);
        c.rtol = 2e-5;
        c.run = [=] {
          std::vector<float> y = *y0;
          const lx_weight v = w->view();
          CALL(lx_gemv_w(x->data(), LX_BF16, K, &v, bias->data(), LX_F32, y.data(), LX_F32, N, M, N, K, alpha, beta,
                         nullptr));
          return y;
        };
        c.ref = [=] {
          std::vector<double> y(static_cast<size_t>(M) * N);
          for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
              double s = 0;
              for (int k = 0; k < K; ++k) s += bf((*x)[m * K + k]) * w->deq[static_cast<size_t>(n) * K + k];
              y[m * N + n] = alpha * (s + (*bias)[n]) + beta * (*y0)[m * N + n];
            }
          return y;
        };
        check(c);
      }
    }
  }

  // The GEMM of every math, at decode, small and prefill sizes. BF16 math has a double reference; the
  // quantized maths are held to the generic level (and to PyTorch by tests/kernels on the server).
  struct MathCase {
    int math, type;
  };
  const MathCase maths[] = {{LX_MATH_BF16, LX_W_BF16}, {LX_MATH_BF16, LX_W_INT8},  {LX_MATH_BF16, LX_W_NVFP4},
                            {LX_MATH_INT8, LX_W_INT8}, {LX_MATH_FP8, LX_W_FP8},    {LX_MATH_FP8, LX_W_NVFP4},
                            {LX_MATH_FP4, LX_W_NVFP4}};
  for (const MathCase& mc : maths) {
    for (int M : {1, 9, 70}) {
      const int N = 50, K = 1056;
      Rng r(mc.math * 1000 + mc.type * 100 + M);
      auto w = std::make_shared<Weight>(make_weight(r, mc.type, N, K, 1.0f / sqrtf(K)));
      auto x = std::make_shared<std::vector<float>>(f32_vec(r, static_cast<size_t>(M) * K, 1.0f));
      Case c;
      c.name = std::string("gemm_w ") + math_name(mc.math) + "/" + format_name(mc.type) + " M=" + std::to_string(M);
      c.rtol = mc.math == LX_MATH_BF16 ? 2e-5 : 1e-4;
      c.run = [=] {
        std::vector<float> y(static_cast<size_t>(M) * N);
        const lx_weight v = w->view();
        CALL(lx_gemm_w(x->data(), LX_F32, K, &v, static_cast<lx_math>(mc.math), nullptr, LX_F32, y.data(), LX_F32, N, M,
                       N, K, 1.0f, 0.0f, nullptr, nullptr));
        return y;
      };
      if (mc.math == LX_MATH_BF16) {
        c.ref = [=] {
          std::vector<double> y(static_cast<size_t>(M) * N);
          for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
              double s = 0;
              for (int k = 0; k < K; ++k) s += bf(to_bf((*x)[m * K + k])) * w->deq[static_cast<size_t>(n) * K + k];
              y[m * N + n] = s;
            }
          return y;
        };
      }
      check(c);
    }
  }

  // lx_gemm_bf16 (bf16 output, bias, beta) and lx_gemm_f32.
  for (int M : {2, 33}) {
    const int N = 45, K = 300;
    Rng r(M);
    auto w = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(N) * K, 0.05f));
    auto x = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(M) * K, 1.0f));
    auto bias = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, N, 0.1f));
    Case c;
    c.name = "gemm_bf16 M=" + std::to_string(M);
    c.rtol = 8e-3;  // the output is bf16
    c.run = [=] {
      std::vector<uint16_t> y(static_cast<size_t>(M) * N);
      CALL(lx_gemm_bf16(x->data(), K, w->data(), bias->data(), y.data(), LX_BF16, N, M, N, K, 1.0f, 0.0f, nullptr));
      return from_bf16(y);
    };
    c.ref = [=] {
      std::vector<double> y(static_cast<size_t>(M) * N);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double s = bf((*bias)[n]);
          for (int k = 0; k < K; ++k) s += static_cast<double>(bf((*x)[m * K + k])) * bf((*w)[n * K + k]);
          y[m * N + n] = s;
        }
      return y;
    };
    check(c);

    auto wf = std::make_shared<std::vector<float>>(f32_vec(r, static_cast<size_t>(N) * K, 0.05f));
    auto xf = std::make_shared<std::vector<float>>(f32_vec(r, static_cast<size_t>(M) * K, 1.0f));
    Case f;
    f.name = "gemm_f32 M=" + std::to_string(M);
    f.rtol = 2e-5;
    f.run = [=] {
      std::vector<float> y(static_cast<size_t>(M) * N);
      CALL(lx_gemm_f32(xf->data(), K, wf->data(), nullptr, y.data(), N, M, N, K, nullptr));
      return y;
    };
    f.ref = [=] {
      std::vector<double> y(static_cast<size_t>(M) * N);
      for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
          double s = 0;
          for (int k = 0; k < K; ++k) s += static_cast<double>((*xf)[m * K + k]) * (*wf)[n * K + k];
          y[m * N + n] = s;
        }
      return y;
    };
    check(f);
  }

  // A row's GEMV result does not depend on the other rows of the call.
  invariant("gemv rows are batch-invariant", [] {
    const int N = 40, K = 512;
    Rng r(7);
    Weight w = make_weight(r, LX_W_INT8, N, K, 0.05f);
    std::vector<uint16_t> x = bf16_vec(r, 8 * K, 1.0f);
    std::vector<float> y8(8 * N), y1(N);
    const lx_weight v = w.view();
    lx_gemv_w(x.data(), LX_BF16, K, &v, nullptr, LX_F32, y8.data(), LX_F32, N, 8, N, K, 1.0f, 0.0f, nullptr);
    lx_gemv_w(x.data() + 5 * K, LX_BF16, K, &v, nullptr, LX_F32, y1.data(), LX_F32, N, 1, N, K, 1.0f, 0.0f, nullptr);
    return memcmp(y1.data(), y8.data() + 5 * N, 4 * N) == 0;
  });
  invariant("gemm_w int8 rows are batch-invariant", [] {
    const int N = 40, K = 512;
    Rng r(8);
    Weight w = make_weight(r, LX_W_INT8, N, K, 0.05f);
    std::vector<float> x = f32_vec(r, 70 * K, 1.0f);
    std::vector<float> y70(70 * N), y1(N);
    const lx_weight v = w.view();
    lx_gemm_w(x.data(), LX_F32, K, &v, LX_MATH_INT8, nullptr, LX_F32, y70.data(), LX_F32, N, 70, N, K, 1, 0, nullptr,
              nullptr);
    lx_gemm_w(x.data() + 33 * K, LX_F32, K, &v, LX_MATH_INT8, nullptr, LX_F32, y1.data(), LX_F32, N, 1, N, K, 1, 0,
              nullptr, nullptr);
    return memcmp(y1.data(), y70.data() + 33 * N, 4 * N) == 0;
  });
}

// ---- Experts -------------------------------------------------------------------------------------------

struct Routing {
  std::vector<int32_t> ids;
  std::vector<float> weights;
};
Routing route(Rng& r, int T, int E, int top_k, bool repeat = false) {
  Routing o;
  for (int t = 0; t < T; ++t) {
    std::vector<int> picks;
    while (static_cast<int>(picks.size()) < top_k) {
      const int e = r.range(E);
      bool dup = false;
      for (int p : picks) dup |= p == e;
      if (!dup || repeat) picks.push_back(e);
    }
    float sum = 0;
    std::vector<float> w(top_k);
    for (auto& v : w) sum += (v = 0.1f + r.uniform());
    for (int k = 0; k < top_k; ++k) {
      o.ids.push_back(picks[k]);
      o.weights.push_back(w[k] / sum);
    }
  }
  return o;
}

// out[t] = sum_k w * down_e(bf16(silu(gate_e x) * up_e x)), from dequantized weights, in double.
std::vector<double> experts_ref(const std::vector<uint16_t>& latent, const Routing& rt, const Weight& gu,
                                const Weight& dn, int T, int L, int I, int top_k) {
  std::vector<double> out(static_cast<size_t>(T) * L, 0.0);
  std::vector<double> mid(I);
  for (int t = 0; t < T; ++t) {
    for (int k = 0; k < top_k; ++k) {
      const int64_t e = rt.ids[t * top_k + k];
      for (int i = 0; i < I; ++i) {
        double g = 0, u = 0;
        const double* gr = gu.deq.data() + (e * 2 * I + i) * L;
        const double* ur = gu.deq.data() + (e * 2 * I + I + i) * L;
        for (int l = 0; l < L; ++l) {
          g += bf(latent[t * L + l]) * gr[l];
          u += bf(latent[t * L + l]) * ur[l];
        }
        mid[i] = bf(to_bf(static_cast<float>(g / (1.0 + exp(-g)) * u)));
      }
      for (int l = 0; l < L; ++l) {
        double s = 0;
        const double* dr = dn.deq.data() + (e * L + l) * I;
        for (int i = 0; i < I; ++i) s += mid[i] * dr[i];
        out[t * L + l] += rt.weights[t * top_k + k] * s;
      }
    }
  }
  return out;
}

void experts() {
  struct Shape {
    int T, E, L, I, top_k;
  };
  // Praxis' experts (L 1024, I 512, top 4 of 128 at decode), and small odd shapes.
  const Shape decode_shapes[] = {{1, 128, 1024, 512, 4}, {4, 16, 256, 96, 8}, {7, 5, 160, 48, 2}, {8, 3, 96, 32, 3}};
  for (int type = LX_W_BF16; type <= LX_W_NVFP4; ++type) {
    for (const Shape& s : decode_shapes) {
      Rng r(type * 7 + s.T * 3 + s.E);
      auto gu = std::make_shared<Weight>(make_weight(r, type, int64_t{s.E} * 2 * s.I, s.L, 1.0f / sqrtf(s.L), s.E));
      auto dn = std::make_shared<Weight>(make_weight(r, type, int64_t{s.E} * s.L, s.I, 1.0f / sqrtf(s.I), s.E));
      auto rt = std::make_shared<Routing>(route(r, s.T, s.E, s.top_k));
      auto latent = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(s.T) * s.L, 1.0f));
      Case c;
      c.name = std::string("moe_experts_decode ") + format_name(type) + " T=" + std::to_string(s.T) + " L=" +
               std::to_string(s.L);
      c.rtol = 2e-3;  // the bf16 intermediate can round either way between float and double sums
      c.run = [=] {
        std::vector<float> out(static_cast<size_t>(s.T) * s.L);
        std::vector<uint8_t> ws(lx_moe_decode_workspace_bytes(s.T, s.I, s.top_k));
        const lx_weight g = gu->view(), d = dn->view();
        CALL(lx_moe_experts_decode(latent->data(), rt->ids.data(), rt->weights.data(), &g, &d, out.data(), ws.data(),
                                   s.T, s.L, s.I, s.top_k, nullptr));
        return out;
      };
      c.ref = [=] { return experts_ref(*latent, *rt, *gu, *dn, s.T, s.L, s.I, s.top_k); };
      check(c);
    }
  }

  // lx_moe_experts (bf16) and lx_moe_experts_w8 at prefill sizes: many tokens per expert.
  const Shape prefill_shapes[] = {{40, 8, 256, 128, 4}, {300, 16, 128, 64, 4}, {9, 4, 96, 48, 2}};
  for (const Shape& s : prefill_shapes) {
    for (int type : {LX_W_BF16, LX_W_INT8}) {
      Rng r(s.T + type);
      auto gu = std::make_shared<Weight>(make_weight(r, type, int64_t{s.E} * 2 * s.I, s.L, 1.0f / sqrtf(s.L)));
      auto dn = std::make_shared<Weight>(make_weight(r, type, int64_t{s.E} * s.L, s.I, 1.0f / sqrtf(s.I)));
      auto rt = std::make_shared<Routing>(route(r, s.T, s.E, s.top_k));
      auto latent = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(s.T) * s.L, 1.0f));
      Case c;
      c.name = std::string(type == LX_W_BF16 ? "moe_experts" : "moe_experts_w8") + " T=" + std::to_string(s.T);
      c.rtol = 2e-3;
      c.run = [=] {
        std::vector<float> out(static_cast<size_t>(s.T) * s.L);
        if (type == LX_W_BF16) {
          CALL(lx_moe_experts(latent->data(), rt->ids.data(), rt->weights.data(), gu->data.data(), dn->data.data(),
                              out.data(), s.T, s.L, s.I, s.top_k, nullptr));
        } else {
          std::vector<uint8_t> ws(lx_moe_workspace_bytes(s.T, s.I, s.top_k));
          CALL(lx_moe_experts_w8(latent->data(), rt->ids.data(), rt->weights.data(),
                                 reinterpret_cast<const int8_t*>(gu->data.data()),
                                 reinterpret_cast<const float*>(gu->scale.data()),
                                 reinterpret_cast<const int8_t*>(dn->data.data()),
                                 reinterpret_cast<const float*>(dn->scale.data()), out.data(), ws.data(), s.T, s.L, s.I,
                                 s.top_k, nullptr));
        }
        return out;
      };
      c.ref = [=] { return experts_ref(*latent, *rt, *gu, *dn, s.T, s.L, s.I, s.top_k); };
      check(c);
    }
  }

  // The grouped kernel in every math, against the generic level; BF16 math against the reference.
  struct MathCase {
    int math, type;
  };
  const MathCase maths[] = {{LX_MATH_BF16, LX_W_BF16}, {LX_MATH_BF16, LX_W_FP8}, {LX_MATH_INT8, LX_W_INT8},
                            {LX_MATH_FP8, LX_W_FP8},   {LX_MATH_FP8, LX_W_NVFP4}, {LX_MATH_FP4, LX_W_NVFP4}};
  for (const MathCase& mc : maths) {
    for (const Shape& s : {Shape{3, 6, 128, 64, 2}, Shape{100, 8, 256, 128, 4}}) {
      Rng r(mc.math * 31 + mc.type + s.T);
      auto gu = std::make_shared<Weight>(make_weight(r, mc.type, int64_t{s.E} * 2 * s.I, s.L, 1.0f / sqrtf(s.L), s.E));
      auto dn = std::make_shared<Weight>(make_weight(r, mc.type, int64_t{s.E} * s.L, s.I, 1.0f / sqrtf(s.I), s.E));
      auto rt = std::make_shared<Routing>(route(r, s.T, s.E, s.top_k));
      auto latent = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(s.T) * s.L, 1.0f));
      Case c;
      c.name = std::string("moe_experts_grouped ") + math_name(mc.math) + "/" + format_name(mc.type) +
               " T=" + std::to_string(s.T);
      c.rtol = mc.math == LX_MATH_BF16 ? 2e-3 : 5e-3;
      c.run = [=] {
        std::vector<float> out(static_cast<size_t>(s.T) * s.L);
        std::vector<uint8_t> ws(
            lx_moe_grouped_workspace_bytes(s.T, s.L, s.I, s.E, s.top_k, static_cast<lx_math>(mc.math)));
        const lx_weight g = gu->view(), d = dn->view();
        CALL(lx_moe_experts_grouped(latent->data(), rt->ids.data(), rt->weights.data(), &g, &d,
                                    static_cast<lx_math>(mc.math), out.data(), ws.data(), s.T, s.L, s.I, s.E, s.top_k,
                                    nullptr));
        return out;
      };
      if (mc.math == LX_MATH_BF16) c.ref = [=] { return experts_ref(*latent, *rt, *gu, *dn, s.T, s.L, s.I, s.top_k); };
      check(c);
    }
  }

  // The int8 experts' first 8 tokens give the same bits alone as at the head of 300 (the model runs
  // decode and prefill through the same kernel).
  invariant("moe_experts_w8 is batch-invariant", [] {
    const int E = 16, L = 128, I = 64, top_k = 4;
    Rng r(11);
    Weight gu = make_weight(r, LX_W_INT8, int64_t{E} * 2 * I, L, 0.08f);
    Weight dn = make_weight(r, LX_W_INT8, int64_t{E} * L, I, 0.1f);
    Routing rt = route(r, 300, E, top_k);
    std::vector<uint16_t> latent = bf16_vec(r, 300 * L, 1.0f);
    std::vector<float> all(300 * L), head(8 * L);
    std::vector<uint8_t> ws(lx_moe_workspace_bytes(300, I, top_k));
    auto run = [&](int T, float* out) {
      return lx_moe_experts_w8(latent.data(), rt.ids.data(), rt.weights.data(),
                               reinterpret_cast<const int8_t*>(gu.data.data()),
                               reinterpret_cast<const float*>(gu.scale.data()),
                               reinterpret_cast<const int8_t*>(dn.data.data()),
                               reinterpret_cast<const float*>(dn.scale.data()), out, ws.data(), T, L, I, top_k, nullptr);
    };
    return run(300, all.data()) == 0 && run(8, head.data()) == 0 && memcmp(all.data(), head.data(), 4 * 8 * L) == 0;
  });
}

// ---- Sequence mixers and the rest ------------------------------------------------------------------------

void mixers() {
  // Attention: a prefill of two sequences into slots, then a decode step over the cache.
  {
    const int H = 32, KVH = 8, Dh = 64, cap = 600;
    const int width = (H + 2 * KVH) * Dh;
    Case c;
    c.name = "attention_varlen prefill + decode";
    c.rtol = 2e-3;  // bf16 output
    c.run = [=] {
      Rng r(3);
      std::vector<uint16_t> kc(static_cast<size_t>(2) * cap * KVH * Dh), vc(kc.size());
      const int32_t slots[2] = {1, 0}, start[2] = {0, 0}, cu[3] = {0, 37, 37 + 300};
      std::vector<float> qkv = f32_vec(r, static_cast<size_t>(337) * width, 1.0f);
      std::vector<uint16_t> out(static_cast<size_t>(337) * H * Dh);
      CALL(lx_attention_varlen(qkv.data(), width, kc.data(), vc.data(), slots, start, cu, 2, cap, out.data(), H, KVH,
                               Dh, 0.125f, nullptr));
      const int32_t start2[2] = {37, 300}, cu2[3] = {0, 1, 2};
      std::vector<float> q2 = f32_vec(r, static_cast<size_t>(2) * width, 1.0f);
      std::vector<uint16_t> out2(static_cast<size_t>(2) * H * Dh);
      CALL(lx_attention_varlen(q2.data(), width, kc.data(), vc.data(), slots, start2, cu2, 2, cap, out2.data(), H,
                               KVH, Dh, 0.125f, nullptr));
      std::vector<float> all = from_bf16(out);
      for (uint16_t v : out2) all.push_back(bf(v));
      return all;
    };
    check(c);
  }

  // Mamba: the causal conv and the selective scan over two sequences (one continuing from state).
  {
    const int H = 8, P = 64, G = 2, N = 128, C = 384, Wd = 4;
    Case c;
    c.name = "conv1d_varlen + ssm_scan_varlen";
    c.rtol = 1e-4;
    c.run = [=] {
      Rng r(5);
      const int T = 70;
      const int32_t slots[2] = {0, 2}, has[2] = {0, 1}, cu[3] = {0, 20, T};
      std::vector<float> x = f32_vec(r, static_cast<size_t>(T) * C, 1.0f);
      std::vector<uint16_t> w = bf16_vec(r, static_cast<size_t>(C) * Wd, 0.5f), b = bf16_vec(r, C, 0.1f);
      std::vector<float> conv_state = f32_vec(r, static_cast<size_t>(3) * C * (Wd - 1), 1.0f);
      std::vector<float> conv(static_cast<size_t>(T) * C);
      CALL(lx_conv1d_varlen(x.data(), C, w.data(), b.data(), conv_state.data(), slots, has, cu, 2, conv.data(), C, Wd,
                            nullptr));
      std::vector<float> xs = f32_vec(r, static_cast<size_t>(T) * H * P, 1.0f);
      std::vector<float> dt = f32_vec(r, static_cast<size_t>(T) * H, 1.0f);
      std::vector<float> B = f32_vec(r, static_cast<size_t>(T) * G * N, 0.3f), Cm = f32_vec(r, B.size(), 0.3f);
      std::vector<float> A_log = f32_vec(r, H, 0.5f), D = f32_vec(r, H, 1.0f), dtb = f32_vec(r, H, 0.5f);
      std::vector<float> state = f32_vec(r, static_cast<size_t>(3) * H * P * N, 0.1f);
      std::vector<float> y(static_cast<size_t>(T) * H * P);
      CALL(lx_ssm_scan_varlen(xs.data(), H * P, dt.data(), H, B.data(), Cm.data(), G * N, A_log.data(), D.data(),
                              dtb.data(), state.data(), slots, has, cu, 2, y.data(), H, P, G, N, nullptr));
      std::vector<float> all = conv;
      all.insert(all.end(), conv_state.begin(), conv_state.end());
      all.insert(all.end(), y.begin(), y.end());
      all.insert(all.end(), state.begin(), state.end());
      return all;
    };
    check(c);
  }

  // The element-wise ops that go through the level's exp.
  {
    Case c;
    c.name = "activate_f32 + silu_mul";
    c.rtol = 1e-5;
    c.run = [] {
      Rng r(9);
      std::vector<float> y = f32_vec(r, 4099, 4.0f), out;
      for (int act = 0; act < 3; ++act) {
        std::vector<float> v = y;
        if (lx_activate_f32(v.data(), static_cast<int64_t>(v.size()), act, nullptr) == 0)
          out.insert(out.end(), v.begin(), v.end());
      }
      std::vector<float> x = f32_vec(r, static_cast<size_t>(5) * 2 * 300, 3.0f), m(static_cast<size_t>(5) * 300);
      CALL(lx_silu_mul(x.data(), 600, m.data(), LX_F32, 5, 300, nullptr));
      out.insert(out.end(), m.begin(), m.end());
      return out;
    };
    check(c);
  }
}

// ---- Lossless BF16 -------------------------------------------------------------------------------------

// bf16 rows with values far below their block's largest (zeros and tiny values): the encoding's patches.
// Row 3 has more patches than slots, so its patches live in the overflow record.
std::vector<uint16_t> patchy_bf16(Rng& r, int64_t rows, int64_t K, float std) {
  std::vector<uint16_t> w = bf16_vec(r, static_cast<size_t>(rows * K), std);
  for (int64_t n = 0; n < rows; ++n)
    for (int64_t k = 0; k < K; ++k) {
      const uint64_t u = r.next() % 1000;
      const bool dense = n == 3 && k % 7 == 0;
      if (u < 3 || dense) w[n * K + k] = u & 1 ? 0 : to_bf(std * 1e-7f * (r.uniform() - 0.5f));
    }
  return w;
}

struct Bf16x {
  std::vector<uint16_t> w;  // the bf16 values
  std::vector<uint64_t> enc;  // their encoding (8-byte aligned)
};
std::shared_ptr<Bf16x> make_bf16x(Rng& r, int64_t rows, int64_t K, float std) {
  auto b = std::make_shared<Bf16x>();
  b->w = patchy_bf16(r, rows, K, std);
  const int64_t bytes = lx_bf16x_bytes(b->w.data(), rows, static_cast<int>(K), nullptr);
  b->enc.resize(static_cast<size_t>((bytes + 7) / 8));
  CALL(lx_bf16x_encode(b->w.data(), rows, static_cast<int>(K), b->enc.data(), bytes, nullptr));
  return b;
}

void bf16x() {
  // The decode GEMV on the encoding equals the bf16 GEMV bit for bit at every level: M = 1 and 2 (the
  // spans decoded in registers), 3 and 8 (the buffered reads), K with partial spans and partial vectors.
  for (int K : {2048, 1040, 1037, 96}) {
    const int N = 37;
    Rng r(K);
    auto b = make_bf16x(r, N, K, 1.0f / sqrtf(K));
    for (int M : {1, 2, 3, 8}) {
      auto x = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(M) * K, 1.0f));
      invariant("gemv_bf16x = gemv bf16 M=" + std::to_string(M) + " K=" + std::to_string(K), [=] {
        std::vector<float> a(static_cast<size_t>(M) * N, 0.25f), c = a;
        lx_gemv_bf16x(x->data(), LX_BF16, K, b->enc.data(), nullptr, LX_F32, a.data(), LX_F32, N, M, N, K, 1.0f, 0.5f,
                      nullptr);
        const lx_weight v{LX_W_BF16, b->w.data(), nullptr, 1.0f, nullptr};
        lx_gemv_w(x->data(), LX_BF16, K, &v, nullptr, LX_F32, c.data(), LX_F32, N, M, N, K, 1.0f, 0.5f, nullptr);
        return memcmp(a.data(), c.data(), 4 * a.size()) == 0;
      });
    }
  }

  // The encoding decodes back to its values.
  invariant("bf16x_decode round trip", [] {
    Rng r(11);
    const int64_t rows = 9, K = 1037;
    auto b = make_bf16x(r, rows, K, 0.02f);
    std::vector<uint16_t> back(static_cast<size_t>(rows * K));
    lx_bf16x_decode(b->enc.data(), rows, static_cast<int>(K), back.data(), nullptr);
    return back == b->w;
  });

  // Routed experts on encoded stacks equal the bf16 experts bit for bit.
  for (int T : {1, 2, 4}) {
    const int E = 16, L = 256, I = 128, top_k = 4;
    Rng r(300 + T);
    auto gu = make_bf16x(r, int64_t{E} * 2 * I, L, 1.0f / sqrtf(L));
    auto dn = make_bf16x(r, int64_t{E} * L, I, 1.0f / sqrtf(I));
    auto latent = std::make_shared<std::vector<uint16_t>>(bf16_vec(r, static_cast<size_t>(T) * L, 1.0f));
    auto rt = std::make_shared<Routing>(route(r, T, E, top_k));
    invariant("moe_experts_bf16x = moe_experts bf16 T=" + std::to_string(T), [=] {
      std::vector<float> a(static_cast<size_t>(T) * L), c(a.size());
      std::vector<uint8_t> ws(static_cast<size_t>(lx_moe_bf16x_workspace_bytes(T, I, top_k)) + 64);
      lx_moe_experts_bf16x(latent->data(), rt->ids.data(), rt->weights.data(), gu->enc.data(), dn->enc.data(), a.data(),
                           ws.data(), T, L, I, top_k, nullptr);
      lx_moe_experts(latent->data(), rt->ids.data(), rt->weights.data(), gu->w.data(), dn->w.data(), c.data(), T, L, I,
                     top_k, nullptr);
      return memcmp(a.data(), c.data(), 4 * a.size()) == 0;
    });
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc > 1) g_filter = argv[1];
  lx_set_arch_override(0);
  const int best = lx_arch();
  for (const Level& l : kAll)
    if (l.code <= best && set_level(l.code)) g_levels.push_back(l);
  set_level(0);
  printf("dromos_cpu: best level %d, %d threads; checking", best, lx_cpu_threads());
  for (const Level& l : g_levels) printf(" %s", l.name);
  printf("\n");
  products();
  experts();
  mixers();
  bf16x();
  printf("%d checks, %d failed\n", g_checks, g_failures);
  return g_failures ? 1 : 0;
}
