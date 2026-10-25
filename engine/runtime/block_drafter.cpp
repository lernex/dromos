#include "runtime/block_drafter.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>

#include "core/device.h"
#include "core/safetensors.h"

namespace lx::runtime {
namespace {

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed: " + lx_rt_error_string(status));
}
#define LX_CHECK(expr) check(static_cast<int>(expr), #expr)

// Philox domain of a block drafter's proposals ("BLKD"), apart from the acceptance draws.
constexpr uint32_t kDomainBlock = 0x424C4B44u;
constexpr int kRows = 8;  // lx_gemv's row limit: the block, and the chunks of a context update
// Calibration of the confidence head: the prior trust (in judged drafts), how much of it each step
// keeps (so the counts reflect the last ~50 steps and a change of content shows quickly), and the
// longest run of steps sat out.
constexpr double kTrust = 2.0;
constexpr double kDecay = 0.98;
constexpr int kMaxBackoff = 15;
// A proposal's own time in plain decode steps: 0.55 to 0.69 ms against 3.4 to 3.5 ms steps, measured
// with Praxis on the RTX 5070 Ti (a larger target makes it relatively cheaper).
constexpr double kProposeCost = 0.17;
constexpr double kSurplusWeight = 0.2;  // of the newest proposal in the running average

template <typename T>
T* device_alloc(size_t count) {
  void* p = nullptr;
  check(lx_rt_alloc(&p, static_cast<int64_t>(count * sizeof(T))),
        "device allocation of " + std::to_string(count * sizeof(T)) + " bytes");  // zeroed
  return static_cast<T*>(p);
}

template <typename T>
T* host_alloc(size_t count) {
  void* p = nullptr;
  check(lx_rt_host_alloc(&p, static_cast<int64_t>(count * sizeof(T))),
        "host allocation of " + std::to_string(count * sizeof(T)) + " bytes");
  return static_cast<T*>(p);
}

}  // namespace

struct BlockDrafter::Impl {
  // config
  int D = 0, H = 0, KVH = 0, Dh = 0, I = 0, B = 0, W = 0, L = 0, F = 0, V = 0, qkv_w = 0;
  float eps = 1e-6f;
  double theta = 10000.0;
  std::vector<FeatureTap> taps;

  // weights: the drafter's (bf16, in `weights`), q | k | v of each layer packed into one matrix,
  // and the target's head
  Weights weights;
  // A product's weight: bf16 [N, K], and its int8 copy (per-row scales) unless the drafter runs bf16.
  struct Lin {
    const void* w = nullptr;
    const int8_t* q = nullptr;
    const float* scale = nullptr;
    int N = 0, K = 0;
    // rows [r0, r0 + n) of it
    Lin rows(int r0, int n) const {
      Lin l = *this;
      l.w = static_cast<const uint16_t*>(w) + static_cast<int64_t>(r0) * K;
      if (q) l.q = q + static_cast<int64_t>(r0) * K, l.scale = scale + r0;
      l.N = n;
      return l;
    }
  };
  struct Layer {
    const void *in_norm, *q_norm, *k_norm, *post_norm;
    Lin qkv, o, gate_up, down;  // qkv: q | k | v packed, [(H + 2 KVH) Dh, D], owned
  };
  std::vector<Layer> layers;
  Lin fc, conf;
  const void *fc_norm, *mask, *norm, *conf_b;
  bool int8 = true;
  // The target's tied embedding: its bf16 gather table (the anchor's row), and the LM head the logits read,
  // as the target holds it (or its BF16X decode copy), or the bf16 table itself for a bf16 drafter.
  const void* embedding = nullptr;
  lx_weight head{LX_W_BF16, nullptr, nullptr, 1.0f, nullptr};
  const void* head_bf16x = nullptr;

  // context: per layer, keys and values of the last W committed positions, f32 [W, KVH Dh], slot
  // = position mod W (attention does not care about their order)
  std::vector<float*> ring_k, ring_v;
  int64_t committed = 0;

  // scratch. Every product reads bf16 activations (the *_b buffers), as the drafter trained under
  // bf16 autocast: at 7 rows a product reading f32 rows ran at 300 to 390 GB/s of weights, bound by
  // re-reading the rows, against 420 to 630 GB/s reading bf16 rows.
  float *fused = nullptr, *kv = nullptr;                                         // [kRows, ...]
  uint16_t *feat_b = nullptr, *fused_b = nullptr;                                // [kRows, ...]
  float *x = nullptr, *x_init = nullptr, *qkv = nullptr, *att = nullptr, *gu = nullptr, *logits = nullptr,
        *probs = nullptr, *conf_out = nullptr;
  uint16_t *h_b = nullptr, *att_b = nullptr, *act_b = nullptr, *out_b = nullptr;
  int32_t *d_tokens = nullptr, *d_dummy = nullptr, *d_cu = nullptr, *d_acc = nullptr;
  lx_spec_filter* d_filters = nullptr;
  lx_spec_rng* d_rng = nullptr;
  int32_t* h_tokens = nullptr;
  float* h_conf = nullptr;
  std::vector<void*> owned;

  template <typename T>
  T* alloc(size_t count) {
    T* p = device_alloc<T>(count);
    owned.push_back(p);
    return p;
  }

  const void* need(const std::string& name, std::initializer_list<int64_t> shape) {
    return weights.get(name, DType::BF16, shape).data;
  }

  Lin lin(const void* w, int N, int K, bool quantize) {
    Lin l{w, nullptr, nullptr, N, K};
    if (quantize && int8) {
      int8_t* q = alloc<int8_t>(static_cast<size_t>(N) * K);
      float* scale = alloc<float>(N);
      LX_CHECK(lx_quantize_w8(w, q, scale, N, K, nullptr));
      l.q = q, l.scale = scale;
    }
    return l;
  }

  // y[M, N] = x[M, K] @ w^T (+ beta * y): x bf16, y f32
  void gemv(const uint16_t* x_, int ld_x, const Lin& w, float* y, int ld_y, int M, float beta, void* s,
            const void* bias = nullptr) {
    LX_CHECK(lx_gemv(x_, LX_BF16, ld_x, w.q ? static_cast<const void*>(w.q) : w.w, w.q ? LX_W_INT8 : LX_W_BF16, w.scale,
                     bias, LX_BF16, y, LX_F32, ld_y, M, w.N, w.K, 1.0f, beta, s));
  }
};

BlockDrafter::BlockDrafter(const std::string& dir, const MetisModel& target, lx_spec_filter filter, uint64_t seed,
                           DraftCost cost, bool bf16)
    : m_(std::make_unique<Impl>()), cost_(std::move(cost)), filter_(filter), seed_(seed) {
  Impl& m = *m_;
  nlohmann::json c;
  std::ifstream(std::filesystem::path(dir) / "config.json") >> c;
  if (c.value("format", "") != "lernex.block/1") throw std::runtime_error(dir + ": not a lernex.block/1 drafter");
  m.D = c.at("hidden_size");
  m.H = c.at("num_heads");
  m.KVH = c.at("num_kv_heads");
  m.Dh = c.at("head_dim");
  m.I = c.at("intermediate_size");
  m.B = c.at("block_size");
  m.W = c.at("window");
  m.L = c.at("num_layers");
  m.V = c.at("vocab_size");
  m.eps = c.at("rms_norm_eps");
  m.theta = c.at("rope_theta");
  for (const auto& t : c.at("taps")) m.taps.push_back({t.at(0).get<int>(), t.at(1).get<int>()});
  m.F = static_cast<int>(m.taps.size()) * m.D;
  m.qkv_w = (m.H + 2 * m.KVH) * m.Dh;
  if (m.D != target.config().hidden_size || m.V != target.vocab_size())
    throw std::runtime_error(dir + ": drafter width or vocabulary differs from the target's");
  if (m.B < 2 || m.B > kRows || m.H % m.KVH || m.W < 1)
    throw std::runtime_error(dir + ": unsupported block drafter shape");

  m.weights = Weights::upload(SafetensorsSet::open(dir));
  m.int8 = !bf16;
  const int64_t D = m.D, Dh = m.Dh, H = m.H, KVH = m.KVH, I = m.I;
  m.fc = m.lin(m.need("fc.weight", {D, m.F}), m.D, m.F, true);
  m.fc_norm = m.need("fc_norm", {D});
  m.mask = m.need("mask", {D});
  m.norm = m.need("norm", {D});
  m.conf = m.lin(m.need("conf.weight", {2, D}), 2, m.D, false);
  m.conf_b = m.need("conf.bias", {2});
  for (int l = 0; l < m.L; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    Impl::Layer layer{};
    layer.in_norm = m.need(p + "in_norm", {D});
    layer.o = m.lin(m.need(p + "o.weight", {D, H * Dh}), m.D, m.H * m.Dh, true);
    layer.q_norm = m.need(p + "q_norm", {Dh});
    layer.k_norm = m.need(p + "k_norm", {Dh});
    layer.post_norm = m.need(p + "post_norm", {D});
    layer.gate_up = m.lin(m.need(p + "gate_up.weight", {2 * I, D}), 2 * m.I, m.D, true);
    layer.down = m.lin(m.need(p + "down.weight", {D, I}), m.D, m.I, true);
    auto* qkv = m.alloc<uint16_t>(static_cast<size_t>(m.qkv_w) * D);
    const size_t qb = sizeof(uint16_t) * H * Dh * D, kb = sizeof(uint16_t) * KVH * Dh * D;
    LX_CHECK(lx_rt_copy(qkv, m.need(p + "q.weight", {H * Dh, D}), static_cast<int64_t>(qb), LX_COPY_D2D, nullptr));
    LX_CHECK(lx_rt_stream_sync(nullptr));
    LX_CHECK(lx_rt_copy(reinterpret_cast<char*>(qkv) + qb, m.need(p + "k.weight", {KVH * Dh, D}),
                        static_cast<int64_t>(kb), LX_COPY_D2D, nullptr));
    LX_CHECK(lx_rt_stream_sync(nullptr));
    LX_CHECK(lx_rt_copy(reinterpret_cast<char*>(qkv) + qb + kb, m.need(p + "v.weight", {KVH * Dh, D}),
                        static_cast<int64_t>(kb), LX_COPY_D2D, nullptr));
    LX_CHECK(lx_rt_stream_sync(nullptr));
    layer.qkv = m.lin(qkv, m.qkv_w, m.D, true);
    m.layers.push_back(layer);
    m.ring_k.push_back(m.alloc<float>(static_cast<size_t>(m.W) * KVH * Dh));
    m.ring_v.push_back(m.alloc<float>(static_cast<size_t>(m.W) * KVH * Dh));
  }
  const MetisModel::HeadWeights head = target.head_weights();
  m.embedding = head.embedding;
  if (bf16) {
    m.head = lx_weight{LX_W_BF16, head.embedding, nullptr, 1.0f, nullptr};
  } else {
    m.head = head.head;
    m.head_bf16x = head.head_bf16x;
  }

  const size_t B = m.B, V = m.V;
  m.fused = m.alloc<float>(kRows * D);
  m.kv = m.alloc<float>(kRows * 2 * KVH * Dh);
  m.feat_b = m.alloc<uint16_t>(static_cast<size_t>(kRows) * m.F);
  m.fused_b = m.alloc<uint16_t>(kRows * D);
  m.x = m.alloc<float>(B * D);
  m.x_init = m.alloc<float>(B * D);
  m.qkv = m.alloc<float>(B * m.qkv_w);
  m.att = m.alloc<float>(B * H * Dh);
  m.gu = m.alloc<float>(B * 2 * I);
  m.h_b = m.alloc<uint16_t>(B * D);
  m.att_b = m.alloc<uint16_t>(B * H * Dh);
  m.act_b = m.alloc<uint16_t>(B * I);
  m.out_b = m.alloc<uint16_t>(B * D);
  m.logits = m.alloc<float>(B * V);
  m.probs = m.alloc<float>(B * V);
  m.conf_out = m.alloc<float>(B * 2);
  m.d_tokens = m.alloc<int32_t>(B);
  m.d_dummy = m.alloc<int32_t>(B);
  m.d_cu = m.alloc<int32_t>(2);
  m.d_acc = m.alloc<int32_t>(2);
  m.d_filters = m.alloc<lx_spec_filter>(B);
  m.d_rng = m.alloc<lx_spec_rng>(B);
  m.h_tokens = host_alloc<int32_t>(B);
  m.h_conf = host_alloc<float>(2 * B);
  // rows 1 .. B-1 of every block start as the mask embedding
  for (int j = 1; j < m.B; ++j) LX_CHECK(lx_copy2d(m.mask, LX_BF16, m.D, m.x_init + j * D, LX_F32, m.D, 1, m.D, nullptr));
  const int32_t cu[2] = {0, m.B - 1};
  LX_CHECK(lx_rt_copy(m.d_cu, cu, static_cast<int64_t>(sizeof(cu)), LX_COPY_H2D, nullptr));
  LX_CHECK(lx_rt_stream_sync(nullptr));
  const std::vector<lx_spec_filter> filters(B, filter_);
  LX_CHECK(lx_rt_copy(m.d_filters, filters.data(), static_cast<int64_t>(sizeof(lx_spec_filter) * B), LX_COPY_H2D,
                      nullptr));
  LX_CHECK(lx_rt_stream_sync(nullptr));
  LX_CHECK(lx_rt_device_sync());
}

BlockDrafter::~BlockDrafter() {
  for (void* p : m_->owned) lx_rt_free(p);
  lx_rt_host_free(m_->h_tokens);
  lx_rt_host_free(m_->h_conf);
}

std::vector<FeatureTap> BlockDrafter::feature_taps() const { return m_->taps; }
int BlockDrafter::block_size() const { return m_->B; }
const float* BlockDrafter::last_logits() const { return m_->logits; }

void BlockDrafter::begin(const std::vector<int32_t>&, void*) {
  m_->committed = 0;
  lambda_ = 1.0;
  last_k_ = 0;
  propose_ms_ = 0;
  proposals_ = 0;
  predicted_.assign(m_->B, 0.0);
  held_.assign(m_->B, 0.0);
  last_p_.clear();
  surplus_ = 1.0;
  proposed_ = false;
  skip_ = backoff_ = 0;
}

void BlockDrafter::observe_features(const float* rows, int n, int feature_dim, void* s) {
  Impl& m = *m_;
  if (feature_dim != m.F) throw std::runtime_error("block drafter: feature rows of the wrong width");
  // only the last W positions can ever be read again
  const int skip = std::max(0, n - m.W);
  const int kvw = m.KVH * m.Dh;
  for (int r0 = skip; r0 < n; r0 += kRows) {
    const int M = std::min(kRows, n - r0);
    const int64_t pos0 = m.committed + r0;
    LX_CHECK(lx_copy2d(rows + static_cast<int64_t>(r0) * m.F, LX_F32, m.F, m.feat_b, LX_BF16, m.F, M, m.F, s));
    m.gemv(m.feat_b, m.F, m.fc, m.fused, m.D, M, 0.0f, s);
    LX_CHECK(lx_rmsnorm(m.fused, LX_F32, m.fc_norm, m.fused_b, LX_BF16, M, m.D, m.eps, s));
    for (int l = 0; l < m.L; ++l) {
      const auto& layer = m.layers[l];
      m.gemv(m.fused_b, m.D, layer.qkv.rows(m.H * m.Dh, 2 * kvw), m.kv, 2 * kvw, M, 0.0f, s);
      LX_CHECK(lx_spec_head_norm_rope(m.kv, 2 * kvw, M, m.KVH, m.Dh, layer.k_norm, m.eps, pos0, m.theta, s));
      // into the ring: at most two runs of slots
      for (int i = 0; i < M;) {
        const int slot = static_cast<int>((pos0 + i) % m.W);
        const int run = std::min(M - i, m.W - slot);
        const float* src = m.kv + static_cast<int64_t>(i) * 2 * kvw;
        LX_CHECK(lx_copy2d(src, LX_F32, 2 * kvw, m.ring_k[l] + static_cast<int64_t>(slot) * kvw, LX_F32, kvw, run, kvw, s));
        LX_CHECK(lx_copy2d(src + kvw, LX_F32, 2 * kvw, m.ring_v[l] + static_cast<int64_t>(slot) * kvw, LX_F32, kvw, run,
                           kvw, s));
        i += run;
      }
    }
  }
  m.committed += n;
}

Draft BlockDrafter::propose(const DraftRequest& r) {
  Impl& m = *m_;
  const auto t0 = std::chrono::steady_clock::now();
  const std::vector<int32_t>& ctx = *r.context;
  if (static_cast<int64_t>(ctx.size()) - 1 != m.committed)
    throw std::runtime_error("block drafter: the target's features are out of step with the context");
  last_k_ = 0;
  last_p_.clear();
  proposed_ = false;
  if (adaptive_ && skip_ > 0) {  // sitting out: a plain step, and no device work
    --skip_;
    return Draft{};
  }
  const int B = m.B, D = m.D, rows = B - 1;
  const int64_t P = m.committed;  // the anchor's position
  const int32_t anchor = ctx.back();
  void* s = r.stream;
  LX_CHECK(lx_rt_copy(m.x, m.x_init, static_cast<int64_t>(sizeof(float) * B * D), LX_COPY_D2D, s));
  LX_CHECK(lx_copy2d(static_cast<const uint16_t*>(m.embedding) + static_cast<int64_t>(anchor) * D, LX_BF16, D, m.x,
                     LX_F32, D, 1, D, s));
  const int n_ctx = static_cast<int>(std::min<int64_t>(m.committed, m.W));
  const int qw = m.H * m.Dh, kvw = m.KVH * m.Dh;
  for (int l = 0; l < m.L; ++l) {
    const auto& layer = m.layers[l];
    LX_CHECK(lx_rmsnorm(m.x, LX_F32, layer.in_norm, m.h_b, LX_BF16, B, D, m.eps, s));
    m.gemv(m.h_b, D, layer.qkv, m.qkv, m.qkv_w, B, 0.0f, s);
    LX_CHECK(lx_spec_head_norm_rope(m.qkv, m.qkv_w, B, m.H, m.Dh, layer.q_norm, m.eps, P, m.theta, s));
    LX_CHECK(lx_spec_head_norm_rope(m.qkv + qw, m.qkv_w, B, m.KVH, m.Dh, layer.k_norm, m.eps, P, m.theta, s));
    LX_CHECK(lx_spec_block_attention(m.qkv, m.qkv_w, m.ring_k[l], m.ring_v[l], n_ctx, m.qkv + qw, m.qkv + qw + kvw,
                                     m.qkv_w, m.att, qw, B, m.H, m.KVH, m.Dh, 1.0f / std::sqrt(float(m.Dh)), s));
    LX_CHECK(lx_copy2d(m.att, LX_F32, qw, m.att_b, LX_BF16, qw, B, qw, s));
    m.gemv(m.att_b, qw, layer.o, m.x, D, B, 1.0f, s);
    LX_CHECK(lx_rmsnorm(m.x, LX_F32, layer.post_norm, m.h_b, LX_BF16, B, D, m.eps, s));
    m.gemv(m.h_b, D, layer.gate_up, m.gu, 2 * m.I, B, 0.0f, s);
    LX_CHECK(lx_silu_mul(m.gu, 2 * m.I, m.act_b, LX_BF16, B, m.I, s));
    m.gemv(m.act_b, m.I, layer.down, m.x, D, B, 1.0f, s);
  }
  LX_CHECK(lx_rmsnorm(m.x + D, LX_F32, m.norm, m.out_b, LX_BF16, rows, D, m.eps, s));
  if (m.head_bf16x)
    LX_CHECK(lx_gemv_bf16x(m.out_b, LX_BF16, D, m.head_bf16x, nullptr, LX_BF16, m.logits, LX_F32, m.V, rows, m.V, m.D,
                           1.0f, 0.0f, s));
  else
    LX_CHECK(lx_gemv_w(m.out_b, LX_BF16, D, &m.head, nullptr, LX_BF16, m.logits, LX_F32, m.V, rows, m.V, m.D, 1.0f,
                       0.0f, s));
  m.gemv(m.out_b, D, m.conf, m.conf_out, 2, rows, 0.0f, s, m.conf_b);

  const bool greedy = !(filter_.temperature > 0.0f);
  if (greedy) {
    LX_CHECK(lx_spec_verify_greedy(m.logits, m.V, m.d_dummy, m.d_cu, 1, rows, m.d_tokens, m.d_acc, m.d_acc + 1, s));
  } else {
    std::vector<lx_spec_rng> rng(rows);
    for (int j = 0; j < rows; ++j) rng[j] = {seed_ ^ 0xB10C0B10C0ull, r.position + j};
    LX_CHECK(lx_rt_copy(m.d_rng, rng.data(), static_cast<int64_t>(sizeof(lx_spec_rng) * rows), LX_COPY_H2D, s));
    LX_CHECK(lx_spec_probs(m.logits, m.V, m.d_filters, rows, m.probs, s));
    LX_CHECK(lx_spec_sample(m.probs, m.V, nullptr, m.d_rng, kDomainBlock, rows, m.d_tokens, s));
  }
  LX_CHECK(lx_rt_copy(m.h_tokens, m.d_tokens, static_cast<int64_t>(sizeof(int32_t) * rows), LX_COPY_D2H, s));
  LX_CHECK(lx_rt_copy(m.h_conf, m.conf_out, static_cast<int64_t>(sizeof(float) * 2 * rows), LX_COPY_D2H, s));
  LX_CHECK(lx_rt_stream_sync(s));

  tokens_.assign(m.h_tokens, m.h_tokens + rows);
  conf_.resize(2 * rows);
  std::vector<double> p_hold(rows);
  for (int j = 0; j < rows; ++j) {
    for (int c = 0; c < 2; ++c) conf_[2 * j + c] = 1.0f / (1.0f + std::exp(-m.h_conf[2 * j + c]));
    p_hold[j] = conf_[2 * j + (greedy ? 0 : 1)];
  }
  int k = std::min(r.max_drafts, rows);
  if (adaptive_) {
    std::vector<double> p(rows);
    for (int j = 0; j < rows; ++j) p[j] = std::min(1.0, p_hold[j] * (held_[j] + kTrust) / (predicted_[j] + kTrust));
    k = cost_.choose(p, k, lambda_);
  }
  proposed_ = true;
  last_k_ = k;
  last_p_ = std::move(p_hold);
  Draft d;
  d.tokens.assign(tokens_.begin(), tokens_.begin() + k);
  if (!greedy) d.q = m.probs;
  propose_ms_ += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
  ++proposals_;
  return d;
}

void BlockDrafter::observe(int accepted, int32_t, void*) {
  const double c = cost_.cost[std::min<size_t>(last_k_ + 1, cost_.cost.size() - 1)];
  lambda_ = 0.9 * lambda_ + 0.1 * (accepted + 1) / c;
  for (size_t j = 0; j < predicted_.size(); ++j) predicted_[j] *= kDecay, held_[j] *= kDecay;
  // draft j was judged when the ones before it held, and held when j < accepted
  for (int j = 0; j < std::min(last_k_, accepted + 1); ++j) {
    predicted_[j] += last_p_[j];
    if (j < accepted) held_[j] += 1.0;
  }
  if (!proposed_ || !adaptive_) return;
  const double extra = cost_.cost[std::min<size_t>(last_k_ + 1, cost_.cost.size() - 1)] - cost_.cost[1];
  surplus_ += kSurplusWeight * (accepted - lambda_ * (extra + kProposeCost) - surplus_);
  if (surplus_ < 0.0) {
    backoff_ = std::min(2 * backoff_ + 1, kMaxBackoff);
    skip_ = backoff_;
  } else {
    backoff_ = 0;
  }
}

}  // namespace lx::runtime
