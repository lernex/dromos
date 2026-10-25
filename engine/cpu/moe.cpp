// The latent mixture of experts (kernels.h): routing, the expert summary, and the routed experts in every
// flavour (lx_moe_experts, lx_moe_experts_w8, lx_moe_experts_decode, lx_moe_experts_grouped) on one engine.
//
// The (token, k) pairs are grouped by expert, so each chosen expert's rows are read once for all the
// tokens that picked it. Then two passes, each split over (expert, block of rows) items:
//   gate/up  inter[pair, i] = bf16(silu(s_g * gate_i . x) * (s_u * up_i . x))
//   down     contrib[pair, l] = s_l * down_l . inter[pair]
// and out[t, l] = sum over k, in order, of weights[t, k] * contrib[(t, k), l]. Every output is written
// once, by one thread, in a fixed order, so results are deterministic; and since the dot kernels give a
// row the same arithmetic whatever else is in the batch, a token's output does not depend on the others.
//
// The decode and int8 contracts are weight-only (f32 math on the expanded weights, the same arithmetic
// for any batch, which the int8 kernel's batch-invariance check needs); lx_moe_experts, whose operands
// are all bf16, runs bf16 math (exact products) so a prefill can use the bf16 dot instructions; the
// grouped kernel takes a math (kernels.h, lx_math) and runs it as gemm.cpp does.
#include <algorithm>
#include <vector>

#include "cpu/cpu.h"
#include "cpu/product.h"

namespace lx::cpu {
namespace {

constexpr int kMathWeightOnly = -1;

// Expert e of a stack [E, N, K]: its rows, its per-row factors (or a scalar), in the stack's format.
struct Slice {
  WeightRef w;
  const float* rf;  // per-row factor [N] (INT8 / FP8), or null
  float rf_scalar;  // the NVFP4 global (the expert's own when the stack has them), else 1
};

Slice slice(const lx_weight& s, int64_t e, int64_t N, int64_t K) {
  Slice out{};
  out.w.type = s.type;
  out.rf = nullptr;
  out.rf_scalar = 1.0f;
  const int64_t rows = e * N;
  switch (s.type) {
    case LX_W_BF16:
      out.w.data = static_cast<const uint16_t*>(s.data) + rows * K;
      break;
    case LX_W_BF16X:  // a stack is E * N encoded rows
      out.w.data = static_cast<const uint8_t*>(s.data) + rows * bf16x::row_bytes(K);
      break;
    case LX_W_INT8:
    case LX_W_FP8:
      out.w.data = static_cast<const uint8_t*>(s.data) + rows * K;
      out.w.scale = static_cast<const float*>(s.scale) + rows;
      out.rf = static_cast<const float*>(out.w.scale);
      break;
    default:
      out.w.data = static_cast<const uint8_t*>(s.data) + rows * (K / 2);
      out.w.scale = static_cast<const uint8_t*>(s.scale) + rows * (K / 16);
      out.rf_scalar = s.expert_global ? s.expert_global[e] : s.global;
      break;
  }
  return out;
}

bool valid_stack(const lx_weight& w, int64_t K) {
  if (!w.data) return false;
  switch (w.type) {
    case LX_W_BF16: return true;
    case LX_W_INT8:
    case LX_W_FP8: return w.scale != nullptr;
    case LX_W_NVFP4: return w.scale != nullptr && K % 16 == 0;
    default: return false;
  }
}

struct Job {
  const uint16_t* latent;  // bf16 [T, L]
  const int32_t* ids;
  const float* weights;
  lx_weight gu, dn;
  int T, L, I, top_k, E;  // E = 0: not given (found from the ids)
  int math;
  float* out;
};

// Rows of activations for one pass, in grouped order, as the path needs them.
struct Acts {
  int path = kPathF32;
  float* f = nullptr;
  int8_t* i = nullptr;
  uint16_t* b = nullptr;
  float* scale = nullptr;  // per row (INT8 / FP8 / FP4 maths), or null
  int64_t K = 0;
  bool exact = false;  // FP8 / FP4 maths: quantized operands, so the dots are summed exactly (double sums)
};

// Fills row r of acts from a bf16 source row, for the math.
void fill_row(Acts& a, int64_t r, const uint16_t* src, int math, uint8_t* tmp) {
  const int64_t K = a.K;
  if (math == kMathWeightOnly || math == LX_MATH_BF16) {
    if (a.path == kPathBF16) memcpy(a.b + r * K, src, 2 * K);
    else
      for (int64_t k = 0; k < K; ++k) a.f[r * K + k] = bf16_to_f32(src[k]);
    return;
  }
  if (math == LX_MATH_INT8) {
    quantize_act_row(src, LX_BF16, 0, K, LX_MATH_INT8, static_cast<int>(K), reinterpret_cast<uint8_t*>(a.i + r * K),
                     a.scale + r, nullptr);
    return;
  }
  quantize_act_row(src, LX_BF16, 0, K, math, static_cast<int>(K), tmp, a.scale + r, tmp + K);
  for (int64_t k = 0; k < K; ++k) {
    float v;
    if (math == LX_MATH_FP8) {
      v = e4m3_to_f32(tmp[k]);
    } else {
      const uint8_t byte = tmp[k >> 1];
      v = e2m1_to_f32((k & 1) ? (byte >> 4) : (byte & 15u)) * e4m3_to_f32(tmp[K + (k >> 4)]);
    }
    if (a.path == kPathBF16) a.b[r * K + k] = f32_to_bf16(v);
    else a.f[r * K + k] = v;
  }
}

// sums[j, n - n0] for rows [n0, n1) of `w` against acts rows [r0, r0 + count): f32, or double when the
// acts are exact (then `sums` points at doubles).
void dot_rows(const Kern& k, const Acts& a, int64_t r0, int64_t count, const WeightRef& w, int64_t n0, int64_t n1,
              void* sums, int64_t ld_s) {
  const int64_t K = a.K;
  switch (a.path) {
    case kPathF32: {
      DotF32 d{a.f + r0 * K, K, w, K, static_cast<float*>(sums), ld_s, nullptr, 0};
      if (a.exact) d.out_d = static_cast<double*>(sums);
      if (count > 8) {
        const int64_t floats = std::min<int64_t>((n1 - n0) * K, std::max<int64_t>(128 * 1024, 4 * K));
        d.panel = static_cast<float*>(scratch(sizeof(float) * floats, 3));
        d.panel_floats = floats;
      }
      k.dot_f32(d, 0, count, n0, n1);
      break;
    }
    case kPathI8: {
      DotI8 d{a.i + r0 * K, K, static_cast<const int8_t*>(w.data), K, static_cast<float*>(sums), ld_s, nullptr, 0};
      k.dot_i8(d, 0, count, n0, n1);
      break;
    }
    default: {
      DotBF16 d{a.b + r0 * K, K, w, K, static_cast<float*>(sums), ld_s, nullptr, 0};
      if (w.type != kW_BF16) {
        const int64_t bytes = 2 * std::min<int64_t>((n1 - n0) * K, std::max<int64_t>(128 * 1024, 4 * K));
        d.panel = scratch(bytes, 3);
        d.panel_bytes = bytes;
      }
      k.dot_bf16(d, 0, count, n0, n1);
      break;
    }
  }
}

int path_for(const Kern& k, int math, int max_group) {
  if (math == LX_MATH_INT8) return kPathI8;
  if (math == LX_MATH_BF16 && k.dot_bf16 && max_group > 8) return kPathBF16;
  return kPathF32;  // weight-only, and the exact dots of FP8 / FP4
}

int run(const Job& j) {
  const int64_t T = j.T, L = j.L, I = j.I, top_k = j.top_k;
  const int64_t P = T * top_k;
  const Kern& k = kern();

  // Group the pairs by expert: order[g] is the pair at grouped position g, pos[p] its position, and
  // expert e owns positions [start[e], start[e + 1]).
  int E = j.E;
  if (E <= 0) {
    E = 0;
    for (int64_t p = 0; p < P; ++p) E = std::max(E, j.ids[p] + 1);
  }
  for (int64_t p = 0; p < P; ++p)
    if (j.ids[p] < 0 || j.ids[p] >= E) return fail(kInvalid);
  std::vector<int32_t> start(E + 1, 0), order(P), pos(P);
  for (int64_t p = 0; p < P; ++p) ++start[j.ids[p] + 1];
  for (int e = 0; e < E; ++e) start[e + 1] += start[e];
  {
    std::vector<int32_t> fill(start.begin(), start.end() - 1);
    for (int64_t p = 0; p < P; ++p) {
      const int32_t g = fill[j.ids[p]]++;
      order[g] = static_cast<int32_t>(p);
      pos[p] = g;
    }
  }
  std::vector<int32_t> experts;
  int max_group = 0;
  for (int e = 0; e < E; ++e) {
    if (start[e + 1] > start[e]) experts.push_back(e);
    max_group = std::max(max_group, start[e + 1] - start[e]);
  }
  const int path = path_for(k, j.math, max_group);
  const int64_t elem = path == kPathF32 ? 4 : path == kPathBF16 ? 2 : 1;

  // Shared buffers, from the calling thread's scratch: the gathered latents, the intermediates (as
  // activations for the down pass), and every pair's down contribution.
  Acts x1, x2;
  x1.path = x2.path = path;
  const bool exact = j.math == LX_MATH_FP8 || j.math == LX_MATH_FP4;
  x1.exact = x2.exact = exact;
  x1.K = L;
  x2.K = I;
  char* b1 = static_cast<char*>(scratch(static_cast<size_t>(P * L * elem + P * 4 + 64), 4));
  char* b2 = static_cast<char*>(scratch(static_cast<size_t>(P * I * elem + P * 4 + 64), 5));
  float* contrib = static_cast<float*>(scratch(sizeof(float) * P * L, 6));
  const bool quantized = j.math == LX_MATH_INT8 || j.math == LX_MATH_FP8 || j.math == LX_MATH_FP4;
  uint16_t* staged = quantized ? static_cast<uint16_t*>(scratch(static_cast<size_t>(2 * P * I), 7)) : nullptr;
  auto bind = [&](Acts& a, char* base, int64_t K) {
    a.f = reinterpret_cast<float*>(base);
    a.i = reinterpret_cast<int8_t*>(base);
    a.b = reinterpret_cast<uint16_t*>(base);
    a.scale = (j.math == kMathWeightOnly || j.math == LX_MATH_BF16)
                  ? nullptr
                  : reinterpret_cast<float*>(base + (P * K * elem + 63) / 64 * 64);
  };
  bind(x1, b1, L);
  bind(x2, b2, I);

  // Latent rows in grouped order. A token's row is the same for each of its experts.
  parallel_for(P, grain_for(P, 8), [&](int64_t g0, int64_t g1) {
    uint8_t* tmp = static_cast<uint8_t*>(scratch(static_cast<size_t>(L + L / 16 + 64), 0));
    for (int64_t g = g0; g < g1; ++g) fill_row(x1, g, j.latent + (order[g] / top_k) * L, j.math, tmp);
  });

  // Gate/up: items are (expert, block of intermediate rows); a block covers gate rows [i0, i1) and the
  // matching up rows [I + i0, I + i1).
  // Blocks small enough that the pool gets several items per thread even when a decode step picks only
  // a few experts (4 experts x 64-row blocks is fewer items than threads on a desktop).
  const int64_t n_exp = static_cast<int64_t>(experts.size());
  const int64_t want = static_cast<int64_t>(threads()) * 4;
  auto block_rows = [&](int64_t rows) {
    int64_t b = (rows * n_exp + want - 1) / want;
    b = (b + 7) / 8 * 8;
    return std::max<int64_t>(8, std::min<int64_t>({64, b, rows}));
  };
  const int64_t rb = block_rows(I);
  const int64_t blocks1 = (I + rb - 1) / rb;
  const int64_t items1 = n_exp * blocks1;
  parallel_for(items1, 1, [&](int64_t a0, int64_t a1) {
    for (int64_t it = a0; it < a1; ++it) {
      const int e = experts[it / blocks1];
      const int64_t i0 = (it % blocks1) * rb, i1 = std::min(I, i0 + rb);
      const int64_t g0 = start[e], cnt = start[e + 1] - g0, ni = i1 - i0;
      const Slice s = slice(j.gu, e, 2 * I, L);
      void* buf = scratch(sizeof(double) * 2 * cnt * ni, 2);
      float* sums = static_cast<float*>(buf);
      double* sums_d = static_cast<double*>(buf);
      const int64_t esz = exact ? 2 : 1;  // the up sums follow the gate sums, in floats
      dot_rows(k, x1, g0, cnt, s.w, i0, i1, buf, ni);
      dot_rows(k, x1, g0, cnt, s.w, I + i0, I + i1, sums + esz * cnt * ni, ni);
      uint16_t* row = static_cast<uint16_t*>(scratch(2 * ni, 0));
      for (int64_t r = 0; r < cnt; ++r) {
        const float xs = x1.scale ? x1.scale[g0 + r] : 1.0f;
        for (int64_t i = 0; i < ni; ++i) {
          const float sg = s.rf ? s.rf[i0 + i] : s.rf_scalar;
          const float su = s.rf ? s.rf[I + i0 + i] : s.rf_scalar;
          if (exact) {
            // From the exact dots, the whole SwiGLU in double: the intermediate is quantized again, and in a
            // format this coarse a last-bit difference here can move a value by a whole code.
            const double g = sg * (static_cast<double>(xs) * sums_d[r * ni + i]);
            const double u = su * (static_cast<double>(xs) * sums_d[(cnt + r) * ni + i]);
            row[i] = f32_to_bf16(static_cast<float>(g / (1.0 + std::exp(-g)) * u));
            continue;
          }
          const float g = sg * (xs * sums[r * ni + i]);
          const float u = su * (xs * sums[(cnt + r) * ni + i]);
          // The intermediate is rounded to bf16 before the down projection, as the reference does.
          row[i] = f32_to_bf16(silu(g) * u);
        }
        const int64_t at = (g0 + r) * I + i0;
        if (staged) memcpy(staged + at, row, 2 * ni);
        else if (x2.path == kPathBF16) memcpy(x2.b + at, row, 2 * ni);
        else
          for (int64_t i = 0; i < ni; ++i) x2.f[at + i] = bf16_to_f32(row[i]);
      }
    }
  });
  // The maths that quantize the intermediate do it per row, once every row is complete.
  if (staged) {
    parallel_for(P, grain_for(P, 8), [&](int64_t g0, int64_t g1) {
      uint8_t* tmp = static_cast<uint8_t*>(scratch(static_cast<size_t>(I + I / 16 + 64), 0));
      for (int64_t g = g0; g < g1; ++g) fill_row(x2, g, staged + g * I, j.math, tmp);
    });
  }

  // Down: items are (expert, block of output rows).
  const int64_t lb = block_rows(L);
  const int64_t blocks2 = (L + lb - 1) / lb;
  const int64_t items2 = n_exp * blocks2;
  parallel_for(items2, 1, [&](int64_t a0, int64_t a1) {
    for (int64_t it = a0; it < a1; ++it) {
      const int e = experts[it / blocks2];
      const int64_t l0 = (it % blocks2) * lb, l1 = std::min(L, l0 + lb);
      const int64_t g0 = start[e], cnt = start[e + 1] - g0, nl = l1 - l0;
      const Slice s = slice(j.dn, e, L, I);
      void* buf = scratch(sizeof(double) * cnt * nl, 2);
      dot_rows(k, x2, g0, cnt, s.w, l0, l1, buf, nl);
      const float* sums = static_cast<const float*>(buf);
      const double* sums_d = static_cast<const double*>(buf);
      for (int64_t r = 0; r < cnt; ++r) {
        const float xs = x2.scale ? x2.scale[g0 + r] : 1.0f;
        float* c = contrib + (g0 + r) * L;
        for (int64_t l = 0; l < nl; ++l) {
          const float rf = s.rf ? s.rf[l0 + l] : s.rf_scalar;
          c[l0 + l] = exact ? static_cast<float>(rf * (static_cast<double>(xs) * sums_d[r * nl + l]))
                            : rf * (xs * sums[r * nl + l]);
        }
      }
    }
  });

  // out[t, l] = sum_k weights[t, k] * contrib[(t, k), l], k in order.
  parallel_for(T, grain_for(T, std::max<int64_t>(1, 16384 / L)), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      float* o = j.out + t * L;
      const float* c0 = contrib + static_cast<int64_t>(pos[t * top_k]) * L;
      const float w0 = j.weights[t * top_k];
      for (int64_t l = 0; l < L; ++l) o[l] = w0 * c0[l];
      for (int64_t kk = 1; kk < top_k; ++kk) {
        const float* c = contrib + static_cast<int64_t>(pos[t * top_k + kk]) * L;
        const float w = j.weights[t * top_k + kk];
        for (int64_t l = 0; l < L; ++l) o[l] += w * c[l];
      }
    }
  });
  return kOk;
}

int check_common(const void* latent, const int32_t* ids, const float* weights, float* out, int L, int I, int top_k) {
  if (!latent || !ids || !weights || !out || L <= 0 || I <= 0 || top_k < 1) return fail(kInvalid);
  return kOk;
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_moe_route(const float* logits, const float* selection_bias, int32_t* ids, float* weights, float* entropy,
                 float* confidence, int T, int E, int top_k, void* stream) {
  LX_PROFILE("moe_route");
  (void)stream;
  if (T <= 0) return kOk;
  if (top_k < 1 || top_k > 64 || top_k > E || !logits || !ids || !weights || !entropy || !confidence)
    return fail(kInvalid);
  parallel_for(T, grain_for(T, std::max(1, 8192 / E)), [&](int64_t t0, int64_t t1) {
    int picked[64];
    for (int64_t t = t0; t < t1; ++t) {
      const float* row = logits + t * E;
      // Softmax statistics. The largest probability is exp(0) / z, so confidence is 1 / z.
      float m = -INFINITY;
      for (int i = 0; i < E; ++i) m = row[i] > m ? row[i] : m;
      float z = 0.0f;
      for (int i = 0; i < E; ++i) z += std::exp(row[i] - m);
      float h = 0.0f;
      for (int i = 0; i < E; ++i) {
        const float p = std::exp(row[i] - m) / z;
        h -= p * std::log(p > 1e-9f ? p : 1e-9f);
      }
      entropy[t] = h;
      confidence[t] = 1.0f / z;
      // Top-k by biased logit (an f32 add), ties to the lower index.
      for (int r = 0; r < top_k; ++r) {
        float best = -INFINITY;
        int best_i = E;
        for (int i = 0; i < E; ++i) {
          bool taken = false;
          for (int q = 0; q < r; ++q) taken |= picked[q] == i;
          if (taken) continue;
          const float v = row[i] + (selection_bias ? selection_bias[i] : 0.0f);
          if (v > best || (v == best && i < best_i)) {
            best = v;
            best_i = i;
          }
        }
        picked[r] = best_i;
      }
      // Weights: softmax over the picks of their unbiased logits.
      float lmax = -INFINITY;
      for (int r = 0; r < top_k; ++r) lmax = row[picked[r]] > lmax ? row[picked[r]] : lmax;
      float e[64], esum = 0.0f;
      for (int r = 0; r < top_k; ++r) {
        e[r] = std::exp(row[picked[r]] - lmax);
        esum += e[r];
      }
      for (int r = 0; r < top_k; ++r) {
        ids[t * top_k + r] = picked[r];
        weights[t * top_k + r] = e[r] / esum;
      }
    }
  });
  return kOk;
}

int lx_moe_summary(const int32_t* ids, const float* weights, const void* expert_emb, float* summary, int T, int R,
                   int top_k, void* stream) {
  LX_PROFILE("moe_summary");
  (void)stream;
  if (T <= 0 || R <= 0) return kOk;
  if (!ids || !weights || !expert_emb || !summary || top_k < 1) return fail(kInvalid);
  const uint16_t* emb = static_cast<const uint16_t*>(expert_emb);
  parallel_for(T, grain_for(T, std::max(1, 16384 / R)), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      float* s = summary + t * R;
      for (int r = 0; r < R; ++r) s[r] = 0.0f;
      for (int k = 0; k < top_k; ++k) {
        const float w = weights[t * top_k + k];
        const uint16_t* row = emb + static_cast<int64_t>(ids[t * top_k + k]) * R;
        for (int r = 0; r < R; ++r) s[r] += w * bf16_to_f32(row[r]);
      }
    }
  });
  return kOk;
}

int lx_moe_experts(const void* latent, const int32_t* ids, const float* weights, const void* gate_up, const void* down,
                   float* out, int T, int L, int I, int top_k, void* stream) {
  LX_PROFILE("moe_experts");
  (void)stream;
  if (T <= 0) return kOk;
  if (int e = check_common(latent, ids, weights, out, L, I, top_k)) return e;
  if (!gate_up || !down) return fail(kInvalid);
  const Job j{static_cast<const uint16_t*>(latent),
              ids,
              weights,
              lx_weight{LX_W_BF16, gate_up, nullptr, 1.0f, nullptr},
              lx_weight{LX_W_BF16, down, nullptr, 1.0f, nullptr},
              T, L, I, top_k, 0, LX_MATH_BF16, out};
  return run(j);
}

int lx_moe_experts_w8(const void* latent, const int32_t* ids, const float* weights, const int8_t* gate_up,
                      const float* gate_up_scale, const int8_t* down, const float* down_scale, float* out,
                      void* workspace, int T, int L, int I, int top_k, void* stream) {
  LX_PROFILE("moe_experts_w8");
  (void)stream;
  (void)workspace;  // the intermediates live in internal scratch, sized with L, which the size query lacks
  if (T <= 0) return kOk;
  if (int e = check_common(latent, ids, weights, out, L, I, top_k)) return e;
  if (!gate_up || !gate_up_scale || !down || !down_scale) return fail(kInvalid);
  const Job j{static_cast<const uint16_t*>(latent),
              ids,
              weights,
              lx_weight{LX_W_INT8, gate_up, gate_up_scale, 1.0f, nullptr},
              lx_weight{LX_W_INT8, down, down_scale, 1.0f, nullptr},
              T, L, I, top_k, 0, kMathWeightOnly, out};
  return run(j);
}

// Routed experts on BF16X stacks, and the plain-bf16 baseline of the same contract. Both are lx_moe_experts
// (bf16 math, the same grouping and order) on another weight row reader, so out equals it bit for bit.
int lx_moe_experts_bf16x(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                         const void* down, float* out, void* workspace, int T, int L, int I, int top_k, void* stream) {
  LX_PROFILE("moe_experts_bf16x");
  (void)stream;
  if (T <= 0) return kOk;
  if (int e = check_common(latent, ids, weights, out, L, I, top_k)) return e;
  if (top_k > 8 || !gate_up || !down || !workspace || L >= bf16x::kMaxK || I >= bf16x::kMaxK) return fail(kInvalid);
  const Job j{static_cast<const uint16_t*>(latent),
              ids,
              weights,
              lx_weight{LX_W_BF16X, gate_up, nullptr, 1.0f, nullptr},
              lx_weight{LX_W_BF16X, down, nullptr, 1.0f, nullptr},
              T, L, I, top_k, 0, LX_MATH_BF16, out};
  return run(j);
}

int lx_moe_experts_decode_bf16(const void* latent, const int32_t* ids, const float* weights, const void* gate_up,
                               const void* down, float* out, void* workspace, int T, int L, int I, int top_k,
                               void* stream) {
  // Profiled as moe_experts, which it is.
  if (T <= 0) return kOk;
  if (top_k > 8 || !workspace) return fail(kInvalid);
  return lx_moe_experts(latent, ids, weights, gate_up, down, out, T, L, I, top_k, stream);
}

// The CUDA size (the intermediates), so a caller that allocates by it works on every backend.
int64_t lx_moe_bf16x_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  return (2 * static_cast<int64_t>(T) * top_k * I + 255) / 256 * 256;
}

// The CUDA sizes, so callers that size buffers from them work on every backend.
int64_t lx_moe_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t pairs = static_cast<int64_t>(T) * top_k;
  const int64_t bytes = (2 * pairs * I + 15) / 16 * 16 + 8 * pairs;
  return (bytes + 255) / 256 * 256;
}

int lx_moe_experts_decode(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                          const lx_weight* down, float* out, void* workspace, int T, int L, int I, int top_k,
                          void* stream) {
  LX_PROFILE("moe_experts_decode");
  (void)stream;
  if (T <= 0) return kOk;
  if (T > 8 || top_k > 8) return fail(kInvalid);
  if (int e = check_common(latent, ids, weights, out, L, I, top_k)) return e;
  if (!gate_up || !down) return fail(kInvalid);
  if (gate_up->type == LX_W_BF16X || down->type == LX_W_BF16X) {
    // Lossless stacks have their own entry point; both must be encoded, a mixed pair has no kernel.
    if (gate_up->type != LX_W_BF16X || down->type != LX_W_BF16X) return fail(kNotSupported);
    return lx_moe_experts_bf16x(latent, ids, weights, gate_up->data, down->data, out, workspace, T, L, I, top_k,
                                stream);
  }
  if (!valid_stack(*gate_up, L) || !valid_stack(*down, I)) return fail(kInvalid);
  const Job j{static_cast<const uint16_t*>(latent), ids, weights, *gate_up, *down, T, L, I, top_k, 0, kMathWeightOnly,
              out};
  return run(j);
}

int64_t lx_moe_decode_workspace_bytes(int T, int I, int top_k) {
  if (T <= 0 || I <= 0 || top_k <= 0) return 0;
  const int64_t bytes = 2 * static_cast<int64_t>(T) * top_k * I;
  return (bytes + 255) / 256 * 256;
}

int lx_moe_experts_grouped(const void* latent, const int32_t* ids, const float* weights, const lx_weight* gate_up,
                           const lx_weight* down, lx_math math, float* out, void* workspace, int T, int L, int I, int E,
                           int top_k, void* stream) {
  LX_PROFILE("moe_experts_grouped");
  (void)stream;
  (void)workspace;
  if (T <= 0) return kOk;
  if (int e = check_common(latent, ids, weights, out, L, I, top_k)) return e;
  if (E <= 0 || !gate_up || !down || !valid_stack(*gate_up, L) || !valid_stack(*down, I)) return fail(kInvalid);
  switch (math) {
    case LX_MATH_BF16: break;
    case LX_MATH_INT8:
      if (gate_up->type != LX_W_INT8 || down->type != LX_W_INT8) return fail(kNotSupported);
      break;
    case LX_MATH_FP8:
      if ((gate_up->type != LX_W_FP8 && gate_up->type != LX_W_NVFP4) || (down->type != LX_W_FP8 && down->type != LX_W_NVFP4))
        return fail(kNotSupported);
      break;
    case LX_MATH_FP4:
      if (gate_up->type != LX_W_NVFP4 || down->type != LX_W_NVFP4 || L % 16 || I % 16) return fail(kNotSupported);
      break;
    default: return fail(kInvalid);
  }
  const Job j{static_cast<const uint16_t*>(latent), ids, weights, *gate_up, *down, T, L, I, top_k, E, math, out};
  return run(j);
}

int64_t lx_moe_grouped_workspace_bytes(int T, int L, int I, int E, int top_k, lx_math math) {
  (void)E;
  (void)math;
  if (T <= 0 || L <= 0 || I <= 0 || top_k <= 0) return 0;
  // The CPU keeps its working buffers in internal scratch; this reports the intermediates, so a caller
  // that allocates by it has room for them on every backend.
  const int64_t bytes = 2 * static_cast<int64_t>(T) * top_k * I;
  return (bytes + 255) / 256 * 256;
}

}  // extern "C"
