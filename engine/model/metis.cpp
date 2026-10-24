#include "model/metis.h"

#include <cmath>
#include <cstdlib>
#include <map>
#include <set>
#include <tuple>
#include <cstring>
#include <stdexcept>

#include "cuda/kernels.h"
#include "model/glue.h"

namespace lx {
namespace {

void check(int status, const char* what) {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed with CUDA error " + std::to_string(status));
}
#define LX_CHECK(expr) check((expr), #expr)

float bf16_to_float(uint16_t v) {
  uint32_t u = static_cast<uint32_t>(v) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

// Round to the nearest bf16 (ties to even), as the reference does when it computes in bf16.
float round_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  u &= 0xFFFF0000u;
  std::memcpy(&f, &u, 4);
  return f;
}

// A weight's values on the host as float (for the per-pass constants computed at load).
std::vector<float> host_values(const WeightView& w) {
  int64_t n = 1;
  for (auto d : w.shape) n *= d;
  std::vector<float> out(n);
  if (w.dtype == LX_F32) {
    check(static_cast<int>(cudaMemcpy(out.data(), w.data, n * 4, cudaMemcpyDeviceToHost)), "cudaMemcpy");
  } else {
    std::vector<uint16_t> raw(n);
    check(static_cast<int>(cudaMemcpy(raw.data(), w.data, n * 2, cudaMemcpyDeviceToHost)), "cudaMemcpy");
    for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_float(raw[i]);
  }
  return out;
}

}  // namespace

struct MetisModel::Impl {
  // ---- weights -----------------------------------------------------------------------------
  struct Connection {
    const void* norm;
    float* params;  // device: matrix [S*S], read [S], write [S] for each pass
  };
  struct Layer {
    bool attention;
    // Mamba-2
    const void *in_proj, *conv_w, *conv_b, *norm_w, *out_proj;
    const float *A_log, *D, *dt_bias;
    // attention
    const void *qkv, *out, *lora_down, *lora_up;
    std::vector<float> pass_gate;  // tanh(pass_gates[p]) rounded to bf16, per pass
    // MoE
    const void *latent_down, *latent_up, *expert_emb, *gate_up, *down, *shared_gate_up, *shared_down;
    const float *router_w, *router_b, *selection_bias;
    Connection mixer_conn, moe_conn;
  };
  std::vector<Layer> layers;
  const void *embedding, *stream_emb, *final_norm;
  // depth memory
  const void *m_state_w, *m_state_b, *m_meta_w, *m_meta_b, *m_pass_emb, *m_anchor_emb, *m_query_w, *m_query_b,
      *m_key_w, *m_key_b, *m_value_w, *m_value_b, *m_out_w, *m_out_b, *m_route_w, *m_route_b, *m_gate, *m_gate_b;
  const float *c_hidden_w, *c_hidden_b, *c_out_w, *c_out_b;

  // ---- dimensions --------------------------------------------------------------------------
  int S, D, L_layers, P_passes, M, R, E, K, Lat, Iexp, Ishared, Emax;
  int I, H, Pd, G, N, W, conv_dim, proj;  // Mamba
  int AH, KVH, Dh, qkv_width, rank;       // attention
  int max_tokens, max_slots, kv_capacity;

  // ---- persistent state, per (pass, layer) ---------------------------------------------------
  std::vector<float*> conv_state, ssm_state;
  std::vector<void*> k_cache, v_cache;

  // ---- per-step scratch --------------------------------------------------------------------
  std::vector<void*> allocations;
  float *streams, *mixed, *q, *projected, *mem_summary, *state, *previous, *difference, *history, *features,
      *cont_in, *cont_hidden, *cont_logit, *continuation, *update, *zxbcdt, *xbc, *ssm_y, *qkv_buf, *latent,
      *router_in, *router_logits, *route_w, *entropy, *confidence, *routed, *shared_gu, *summary, *bank_k,
      *bank_v, *entry;
  int32_t *route_ids, *d_meta, *d_tokens, *d_cu, *d_slots, *d_has_state, *d_start_pos;
  int32_t* h_meta = nullptr;  // pinned staging for d_meta, so the upload is one async copy
  size_t meta_count = 0;
  cudaEvent_t meta_copied = nullptr;
  float* last_rows = nullptr;
  // Decode steps (one token per sequence) replay a captured graph: the first step of a shape runs
  // eagerly (which also warms cuBLASLt up), the second captures, every later one replays.
  using GraphKey = std::tuple<int, int, bool, float*>;
  std::set<GraphKey> seen;
  std::map<GraphKey, cudaGraphExec_t> graphs;
  bool use_graphs = true;
  void *streams_bf, *retrieved_bf, *feat_in_bf, *h_bf, *normed_bf, *lora_bf, *att_bf, *latent_bf, *shared_bf,
      *routed_bf, *entry_bf, *meta_bf, *out_bf;

  template <typename T>
  T* alloc(size_t count, size_t elem = sizeof(T)) {
    void* p = nullptr;
    check(static_cast<int>(cudaMalloc(&p, count * elem)), "cudaMalloc");
    check(static_cast<int>(cudaMemset(p, 0, count * elem)), "cudaMemset");
    allocations.push_back(p);
    return static_cast<T*>(p);
  }
  ~Impl() {
    for (auto& [key, exec] : graphs) cudaGraphExecDestroy(exec);
    if (meta_copied) cudaEventDestroy(meta_copied);
    if (h_meta) cudaFreeHost(h_meta);
    for (void* p : allocations) cudaFree(p);
  }
};

namespace {

using Impl = MetisModel::Impl;

const void* need(const WeightLookup& lookup, const std::string& name, int dtype, std::vector<int64_t> shape) {
  WeightView w = lookup(name);
  if (!w.data) throw std::runtime_error("missing weight " + name);
  if (w.dtype != dtype) throw std::runtime_error("weight " + name + " has the wrong dtype");
  if (w.shape != shape) {
    std::string got, want;
    for (auto d : w.shape) got += std::to_string(d) + " ";
    for (auto d : shape) want += std::to_string(d) + " ";
    throw std::runtime_error("weight " + name + " has shape [ " + got + "], expected [ " + want + "]");
  }
  return w.data;
}

// The Sinkhorn mix, read and write weights of one hyper-connection for every pass. They depend
// only on the weights and the pass, so they are computed once, here, as the reference computes
// them: the controller in bf16, everything after it in f32.
float* connection_params(Impl& m, const WeightLookup& lookup, const std::string& prefix, const MetisConfig& c,
                         const std::vector<float>& pass_embeddings) {
  const int S = c.n_streams, E = c.mhc_pass_embedding_dim, out = S * S + 2 * S;
  auto W = host_values(lookup(prefix + ".controller.weight"));
  auto b = host_values(lookup(prefix + ".controller.bias"));
  auto mix = host_values(lookup(prefix + ".mix_logits"));
  auto read_logits = host_values(lookup(prefix + ".read_logits"));
  auto write_logits = host_values(lookup(prefix + ".write_logits"));
  need(lookup, prefix + ".controller.weight", LX_BF16, {out, E});
  std::vector<float> params(static_cast<size_t>(c.num_passes) * out);
  for (int p = 0; p < c.num_passes; ++p) {
    std::vector<double> delta(out);
    for (int j = 0; j < out; ++j) {
      float acc = 0.0f;
      for (int k = 0; k < E; ++k) acc += pass_embeddings[p * E + k] * W[j * E + k];
      delta[j] = round_bf16(round_bf16(acc) + b[j]);
    }
    std::vector<double> lm(S * S);
    for (int i = 0; i < S * S; ++i) lm[i] = mix[i] + delta[i];
    for (int it = 0; it < c.mhc_sinkhorn_iterations; ++it) {
      for (int r = 0; r < S; ++r) {  // normalize rows (last axis)
        double mx = -1e300, sum = 0;
        for (int k = 0; k < S; ++k) mx = std::max(mx, lm[r * S + k]);
        for (int k = 0; k < S; ++k) sum += std::exp(lm[r * S + k] - mx);
        const double lse = mx + std::log(sum);
        for (int k = 0; k < S; ++k) lm[r * S + k] -= lse;
      }
      for (int col = 0; col < S; ++col) {  // then columns
        double mx = -1e300, sum = 0;
        for (int k = 0; k < S; ++k) mx = std::max(mx, lm[k * S + col]);
        for (int k = 0; k < S; ++k) sum += std::exp(lm[k * S + col] - mx);
        const double lse = mx + std::log(sum);
        for (int k = 0; k < S; ++k) lm[k * S + col] -= lse;
      }
    }
    float* dst = params.data() + static_cast<size_t>(p) * out;
    for (int i = 0; i < S * S; ++i) dst[i] = static_cast<float>(std::exp(lm[i]));
    auto softmax = [&](const std::vector<float>& logits, int offset, float* o) {
      double mx = -1e300, sum = 0;
      for (int i = 0; i < S; ++i) mx = std::max(mx, logits[i] + delta[offset + i]);
      for (int i = 0; i < S; ++i) sum += std::exp(logits[i] + delta[offset + i] - mx);
      for (int i = 0; i < S; ++i) o[i] = static_cast<float>(std::exp(logits[i] + delta[offset + i] - mx) / sum);
    };
    softmax(read_logits, S * S, dst + S * S);
    softmax(write_logits, S * S + S, dst + S * S + S);
  }
  float* d = m.alloc<float>(params.size());
  check(static_cast<int>(cudaMemcpy(d, params.data(), params.size() * 4, cudaMemcpyHostToDevice)), "cudaMemcpy");
  return d;
}

}  // namespace

MetisModel::MetisModel(const MetisConfig& c, const WeightLookup& lw, int max_slots, int kv_capacity,
                       int max_tokens)
    : cfg_(c), slot_len_(max_slots, 0), impl_(new Impl) {
  Impl& m = *impl_;
  m.S = c.n_streams;
  m.D = c.hidden_size;
  m.L_layers = c.num_hidden_layers;
  m.P_passes = c.num_passes;
  m.M = c.memory_dim;
  m.R = c.route_feature_dim;
  m.E = c.n_routed_experts;
  m.K = c.num_experts_per_tok;
  m.Lat = c.latent_dim;
  m.Iexp = c.expert_intermediate_size;
  m.Ishared = c.shared_expert_intermediate_size;
  m.Emax = static_cast<int>((c.attention_layer_indices.size() + 1) * c.num_passes);
  m.I = c.hidden_size * c.mamba_expand;
  m.Pd = c.mamba_head_dim;
  m.H = m.I / m.Pd;
  m.G = c.mamba_n_groups;
  m.N = c.mamba_d_state;
  m.W = c.mamba_d_conv;
  m.conv_dim = m.I + 2 * m.G * m.N;
  m.proj = 2 * m.I + 2 * m.G * m.N + m.H;
  m.AH = c.num_attention_heads;
  m.KVH = c.num_key_value_heads;
  m.Dh = c.head_dim;
  m.qkv_width = (m.AH + 2 * m.KVH) * m.Dh;
  m.rank = c.attention_pass_lora_rank;
  m.max_tokens = max_tokens;
  m.max_slots = max_slots;
  m.kv_capacity = kv_capacity;
  const int D = m.D, S = m.S, V = c.vocab_size, R = m.R, Mm = m.M, E = m.E, Lat = m.Lat;

  m.embedding = need(lw, "embedding.weight", LX_BF16, {V, D});
  m.stream_emb = need(lw, "stream_embeddings", LX_BF16, {S, D});
  m.final_norm = need(lw, "final_norm.weight", LX_BF16, {D});
  const auto pass_embeddings = host_values(lw("pass_embeddings"));
  need(lw, "pass_embeddings", LX_BF16, {c.max_passes, c.mhc_pass_embedding_dim});

  const std::string dm = "depth_memory.";
  m.m_state_w = need(lw, dm + "state_write.weight", LX_BF16, {Mm, D});
  m.m_state_b = need(lw, dm + "state_write.bias", LX_BF16, {Mm});
  m.m_meta_w = need(lw, dm + "metadata_write.weight", LX_BF16, {Mm, R + 4});
  m.m_meta_b = need(lw, dm + "metadata_write.bias", LX_BF16, {Mm});
  m.m_pass_emb = need(lw, dm + "pass_embeddings", LX_BF16, {c.max_passes, Mm});
  m.m_anchor_emb =
      need(lw, dm + "anchor_embeddings", LX_BF16, {static_cast<int64_t>(c.attention_layer_indices.size() + 1), Mm});
  m.m_query_w = need(lw, dm + "query.weight", LX_BF16, {Mm, D});
  m.m_query_b = need(lw, dm + "query.bias", LX_BF16, {Mm});
  m.m_key_w = need(lw, dm + "key.weight", LX_BF16, {Mm, Mm});
  m.m_key_b = need(lw, dm + "key.bias", LX_BF16, {Mm});
  m.m_value_w = need(lw, dm + "value.weight", LX_BF16, {Mm, Mm});
  m.m_value_b = need(lw, dm + "value.bias", LX_BF16, {Mm});
  m.m_out_w = need(lw, dm + "output.weight", LX_BF16, {D, Mm});
  m.m_out_b = need(lw, dm + "output.bias", LX_BF16, {D});
  m.m_route_w = need(lw, dm + "route_projection.weight", LX_BF16, {R, 3 * D + R});
  m.m_route_b = need(lw, dm + "route_projection.bias", LX_BF16, {R});
  m.m_gate = need(lw, dm + "stream_gate", LX_BF16, {S, D});
  m.m_gate_b = need(lw, dm + "stream_gate_bias", LX_BF16, {S});
  m.c_hidden_w = static_cast<const float*>(need(lw, "continuation.hidden.weight", LX_F32, {R, 3 * D + R}));
  m.c_hidden_b = static_cast<const float*>(need(lw, "continuation.hidden.bias", LX_F32, {R}));
  m.c_out_w = static_cast<const float*>(need(lw, "continuation.output.weight", LX_F32, {1, R}));
  m.c_out_b = static_cast<const float*>(need(lw, "continuation.output.bias", LX_F32, {1}));

  for (int i = 0; i < m.L_layers; ++i) {
    Impl::Layer layer{};
    const std::string p = "layers." + std::to_string(i) + ".";
    layer.attention = c.is_attention(i);
    if (layer.attention) {
      const std::string a = p + "mixer.";
      layer.qkv = need(lw, a + "qkv.weight", LX_BF16, {m.qkv_width, D});
      layer.out = need(lw, a + "out.weight", LX_BF16, {D, m.AH * m.Dh});
      layer.lora_down = need(lw, a + "pass_lora_down.weight", LX_BF16, {m.rank, D});
      layer.lora_up = need(lw, a + "pass_lora_up.weight", LX_BF16, {m.qkv_width, m.rank});
      auto gates = host_values(lw(a + "pass_gates"));
      need(lw, a + "pass_gates", LX_BF16, {c.max_passes});
      for (int pass = 0; pass < m.P_passes; ++pass) layer.pass_gate.push_back(round_bf16(std::tanh(gates[pass])));
    } else {
      const std::string a = p + "mixer.impl.mixer.";
      layer.in_proj = need(lw, a + "in_proj.weight", LX_BF16, {m.proj, D});
      layer.conv_w = need(lw, a + "conv1d.weight", LX_BF16, {m.conv_dim, 1, m.W});
      layer.conv_b = need(lw, a + "conv1d.bias", LX_BF16, {m.conv_dim});
      layer.norm_w = need(lw, a + "norm.weight", LX_BF16, {m.I});
      layer.out_proj = need(lw, a + "out_proj.weight", LX_BF16, {D, m.I});
      layer.A_log = static_cast<const float*>(need(lw, a + "A_log", LX_F32, {m.H}));
      layer.D = static_cast<const float*>(need(lw, a + "D", LX_F32, {m.H}));
      layer.dt_bias = static_cast<const float*>(need(lw, a + "dt_bias", LX_F32, {m.H}));
    }
    const std::string e = p + "moe.";
    layer.latent_down = need(lw, e + "latent_down.weight", LX_BF16, {Lat, D});
    layer.latent_up = need(lw, e + "latent_up.weight", LX_BF16, {D, Lat});
    layer.expert_emb = need(lw, e + "expert_embeddings", LX_BF16, {E, R});
    layer.gate_up = need(lw, e + "local_experts.gate_up.weight", LX_BF16, {E, 2 * m.Iexp, Lat});
    layer.down = need(lw, e + "local_experts.down.weight", LX_BF16, {E, Lat, m.Iexp});
    layer.shared_gate_up = need(lw, e + "shared_expert.gate_up.weight", LX_BF16, {2 * m.Ishared, Lat});
    layer.shared_down = need(lw, e + "shared_expert.down.weight", LX_BF16, {Lat, m.Ishared});
    layer.router_w = static_cast<const float*>(need(lw, e + "expert_router.weight", LX_F32, {E, Lat + R}));
    layer.router_b = static_cast<const float*>(need(lw, e + "expert_router.bias", LX_F32, {E}));
    layer.selection_bias = static_cast<const float*>(need(lw, e + "selection_bias", LX_F32, {E}));
    layer.mixer_conn.norm = need(lw, p + "mixer_connection.norm.weight", LX_BF16, {D});
    layer.mixer_conn.params = connection_params(m, lw, p + "mixer_connection", c, pass_embeddings);
    layer.moe_conn.norm = need(lw, p + "moe_connection.norm.weight", LX_BF16, {D});
    layer.moe_conn.params = connection_params(m, lw, p + "moe_connection", c, pass_embeddings);
    m.layers.push_back(std::move(layer));
  }

  // persistent state, one per (pass, layer)
  for (int pass = 0; pass < m.P_passes; ++pass) {
    for (int i = 0; i < m.L_layers; ++i) {
      if (m.layers[i].attention) {
        const size_t n = static_cast<size_t>(max_slots) * kv_capacity * m.KVH * m.Dh;
        m.k_cache.push_back(m.alloc<uint16_t>(n));
        m.v_cache.push_back(m.alloc<uint16_t>(n));
        m.conv_state.push_back(nullptr);
        m.ssm_state.push_back(nullptr);
      } else {
        m.conv_state.push_back(m.alloc<float>(static_cast<size_t>(max_slots) * (m.W - 1) * m.conv_dim));
        m.ssm_state.push_back(m.alloc<float>(static_cast<size_t>(max_slots) * m.H * m.Pd * m.N));
        m.k_cache.push_back(nullptr);
        m.v_cache.push_back(nullptr);
      }
    }
  }

  // per-step scratch, for up to max_tokens tokens
  const size_t T = max_tokens, TS = T * S;
  m.streams = m.alloc<float>(TS * D);
  m.mixed = m.alloc<float>(TS * D);
  m.streams_bf = m.alloc<uint16_t>(TS * D);
  m.q = m.alloc<float>(TS * Mm);
  m.retrieved_bf = m.alloc<uint16_t>(TS * Mm);
  m.projected = m.alloc<float>(TS * D);
  m.mem_summary = m.alloc<float>(T * D);
  m.state = m.alloc<float>(T * D);
  m.previous = m.alloc<float>(T * D);
  m.difference = m.alloc<float>(T * D);
  m.history = m.alloc<float>(T * R);
  m.features = m.alloc<float>(T * R);
  m.feat_in_bf = m.alloc<uint16_t>(T * (3 * D + R));
  m.cont_in = m.alloc<float>(T * (3 * D + R));
  m.cont_hidden = m.alloc<float>(T * R);
  m.cont_logit = m.alloc<float>(T);
  m.continuation = m.alloc<float>(T);
  m.h_bf = m.alloc<uint16_t>(T * D);
  m.update = m.alloc<float>(T * D);
  m.zxbcdt = m.alloc<float>(T * m.proj);
  m.xbc = m.alloc<float>(T * m.conv_dim);
  m.ssm_y = m.alloc<float>(T * m.I);
  m.normed_bf = m.alloc<uint16_t>(T * m.I);
  m.qkv_buf = m.alloc<float>(T * m.qkv_width);
  m.lora_bf = m.alloc<uint16_t>(T * m.rank);
  m.att_bf = m.alloc<uint16_t>(T * m.AH * m.Dh);
  m.latent = m.alloc<float>(T * Lat);
  m.latent_bf = m.alloc<uint16_t>(T * Lat);
  m.router_in = m.alloc<float>(T * (Lat + R));
  m.router_logits = m.alloc<float>(T * E);
  m.route_ids = m.alloc<int32_t>(T * m.K);
  m.route_w = m.alloc<float>(T * m.K);
  m.entropy = m.alloc<float>(T);
  m.confidence = m.alloc<float>(T);
  m.routed = m.alloc<float>(T * Lat);
  m.shared_gu = m.alloc<float>(T * 2 * m.Ishared);
  m.shared_bf = m.alloc<uint16_t>(T * m.Ishared);
  m.routed_bf = m.alloc<uint16_t>(T * Lat);
  m.summary = m.alloc<float>(T * R);
  m.bank_k = m.alloc<float>(T * m.Emax * Mm);
  m.bank_v = m.alloc<float>(T * m.Emax * Mm);
  m.entry = m.alloc<float>(T * Mm);
  m.entry_bf = m.alloc<uint16_t>(T * Mm);
  m.meta_bf = m.alloc<uint16_t>(T * (R + 4));
  m.out_bf = m.alloc<uint16_t>(T * D);
  m.last_rows = m.alloc<float>(static_cast<size_t>(max_slots) * D);
  m.meta_count = T + (max_slots + 1) + 3 * static_cast<size_t>(max_slots);
  m.d_meta = m.alloc<int32_t>(m.meta_count);
  m.d_tokens = m.d_meta;
  m.d_cu = m.d_tokens + T;
  m.d_slots = m.d_cu + max_slots + 1;
  m.d_has_state = m.d_slots + max_slots;
  m.d_start_pos = m.d_has_state + max_slots;
  check(static_cast<int>(cudaMallocHost(&m.h_meta, m.meta_count * 4)), "cudaMallocHost");
  check(static_cast<int>(cudaEventCreateWithFlags(&m.meta_copied, cudaEventDisableTiming)), "cudaEventCreate");
  m.use_graphs = std::getenv("LX_NO_GRAPHS") == nullptr;
}

MetisModel::~MetisModel() { delete impl_; }

void MetisModel::reset_slot(int slot) { slot_len_.at(slot) = 0; }

namespace {

// y = alpha * (x @ w^T + bias) + beta * y with bf16 inputs.
void gemm(const void* x, int ld_x, const void* w, const void* bias, void* y, lx_dtype y_dtype, int ld_y, int M,
          int N, int K, cudaStream_t s, float alpha = 1.0f, float beta = 0.0f) {
  LX_CHECK(lx_gemm_bf16(x, ld_x, w, bias, y, y_dtype, ld_y, M, N, K, alpha, beta, s));
}

void to_bf16(const float* src, int ld_src, void* dst, int ld_dst, int rows, int cols, cudaStream_t s) {
  LX_CHECK(lx_copy2d(src, LX_F32, ld_src, dst, LX_BF16, ld_dst, rows, cols, s));
}

void copy_f32(const float* src, int ld_src, float* dst, int ld_dst, int rows, int cols, cudaStream_t s) {
  LX_CHECK(lx_copy2d(src, LX_F32, ld_src, dst, LX_F32, ld_dst, rows, cols, s));
}

// Depth-memory read: attend over the token's bank and gate the result into the streams. Leaves
// the mean retrieved vector in mem_summary (zeros while the bank is empty).
void memory_retrieve(Impl& m, int T, int n_entries, cudaStream_t s) {
  const int S = m.S, D = m.D, Mm = m.M;
  if (n_entries == 0) {
    cudaMemsetAsync(m.mem_summary, 0, sizeof(float) * T * D, s);
    return;
  }
  to_bf16(m.streams, D, m.streams_bf, D, T * S, D, s);
  gemm(m.streams_bf, D, m.m_query_w, m.m_query_b, m.q, LX_F32, Mm, T * S, Mm, D, s);
  LX_CHECK(lx_memory_attend(m.q, m.bank_k, m.bank_v, n_entries, m.Emax, m.retrieved_bf, T, S, Mm, s));
  gemm(m.retrieved_bf, Mm, m.m_out_w, m.m_out_b, m.projected, LX_F32, D, T * S, D, Mm, s);
  LX_CHECK(lx_memory_fuse(m.streams, m.projected, m.m_gate, m.m_gate_b, m.mem_summary, T, S, D, s));
}

// [a | b | c | d] into one row per token, as bf16 (for a GEMM) or f32.
void concat4(Impl& m, const float* a, const float* b, const float* c, const float* d, void* out, lx_dtype dtype,
             int T, cudaStream_t s) {
  const int D = m.D, R = m.R, ld = 3 * D + R;
  const size_t es = dtype == LX_F32 ? 4 : 2;
  char* o = static_cast<char*>(out);
  LX_CHECK(lx_copy2d(a, LX_F32, D, o, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(b, LX_F32, D, o + es * D, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(c, LX_F32, D, o + es * 2 * D, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(d, LX_F32, R, o + es * 3 * D, dtype, ld, T, R, s));
}

// Routing features: tanh(route_projection([state, retrieved, difference, history])).
void routing_features(Impl& m, int T, cudaStream_t s) {
  concat4(m, m.state, m.mem_summary, m.difference, m.history, m.feat_in_bf, LX_BF16, T, s);
  gemm(m.feat_in_bf, 3 * m.D + m.R, m.m_route_w, m.m_route_b, m.features, LX_F32, m.R, T, m.R, 3 * m.D + m.R, s);
  LX_CHECK(lx_activate_f32(m.features, static_cast<int64_t>(T) * m.R, 0, s));
}

// A new memory entry from the streams and the last MoE's routing; stores its key and value.
void memory_write(Impl& m, const MetisConfig& c, int T, int pass, int anchor, int& n_entries, cudaStream_t s) {
  const int D = m.D, R = m.R, Mm = m.M;
  LX_CHECK(lx_stream_mean(m.streams, m.state, T, m.S, D, s));
  to_bf16(m.state, D, m.out_bf, D, T, D, s);
  gemm(m.out_bf, D, m.m_state_w, m.m_state_b, m.entry, LX_F32, Mm, T, Mm, D, s);
  LX_CHECK(lx_copy2d(m.summary, LX_F32, R, m.meta_bf, LX_BF16, R + 4, T, R, s));
  const float mean_k = static_cast<float>(c.num_experts_per_tok) / static_cast<float>(c.max_routed_k);
  const float inv_log_e = c.n_routed_experts > 1 ? 1.0f / std::log(static_cast<float>(c.n_routed_experts)) : 1.0f;
  LX_CHECK(glue::meta_scalars(m.entropy, m.confidence, m.continuation, mean_k, inv_log_e, m.meta_bf, R + 4, R, T, s));
  gemm(m.meta_bf, R + 4, m.m_meta_w, m.m_meta_b, m.entry, LX_F32, Mm, T, Mm, R + 4, s, 1.0f, 1.0f);
  LX_CHECK(lx_add_row_bf16(m.entry, static_cast<const uint16_t*>(m.m_pass_emb) + pass * Mm, T, Mm, s));
  LX_CHECK(lx_add_row_bf16(m.entry, static_cast<const uint16_t*>(m.m_anchor_emb) + anchor * Mm, T, Mm, s));
  to_bf16(m.entry, Mm, m.entry_bf, Mm, T, Mm, s);
  const int ld_bank = m.Emax * Mm;
  gemm(m.entry_bf, Mm, m.m_key_w, m.m_key_b, m.bank_k + n_entries * Mm, LX_F32, ld_bank, T, Mm, Mm, s);
  gemm(m.entry_bf, Mm, m.m_value_w, m.m_value_b, m.bank_v + n_entries * Mm, LX_F32, ld_bank, T, Mm, Mm, s);
  ++n_entries;
}

}  // namespace

void MetisModel::forward(const StepBatch& b, float* logits, bool all_rows, cudaStream_t s) {
  Impl& m = *impl_;
  const int T = static_cast<int>(b.tokens.size());
  const int n_seq = static_cast<int>(b.slots.size());
  if (T == 0) return;
  if (T > m.max_tokens) throw std::runtime_error("step has more tokens than max_tokens");
  if (n_seq > m.max_slots || static_cast<int>(b.cu_seqlens.size()) != n_seq + 1)
    throw std::runtime_error("bad step batch");

  // stage the step's metadata (wait until the previous step's copy has left the staging buffer)
  cudaEventSynchronize(m.meta_copied);
  int32_t* tokens = m.h_meta;
  int32_t* cu = tokens + (m.d_cu - m.d_tokens);
  int32_t* slots = tokens + (m.d_slots - m.d_tokens);
  int32_t* has_state = tokens + (m.d_has_state - m.d_tokens);
  int32_t* start_pos = tokens + (m.d_start_pos - m.d_tokens);
  std::memcpy(tokens, b.tokens.data(), 4 * T);
  std::memcpy(cu, b.cu_seqlens.data(), 4 * (n_seq + 1));
  std::memcpy(slots, b.slots.data(), 4 * n_seq);
  for (int i = 0; i < n_seq; ++i) {
    const int slot = b.slots[i];
    has_state[i] = slot_len_[slot] > 0;
    start_pos[i] = slot_len_[slot];
    if (start_pos[i] + b.cu_seqlens[i + 1] - b.cu_seqlens[i] > m.kv_capacity)
      throw std::runtime_error("sequence longer than the KV capacity");
  }
  cudaMemcpyAsync(m.d_meta, m.h_meta, m.meta_count * 4, cudaMemcpyHostToDevice, s);
  cudaEventRecord(m.meta_copied, s);

  const bool decode = T == n_seq;
  const Impl::GraphKey key{T, n_seq, all_rows, logits};
  if (m.use_graphs && decode) {
    auto it = m.graphs.find(key);
    if (it == m.graphs.end() && m.seen.count(key)) {
      cudaGraph_t graph;
      check(static_cast<int>(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)), "cudaStreamBeginCapture");
      run(T, n_seq, all_rows, logits, s);
      check(static_cast<int>(cudaStreamEndCapture(s, &graph)), "cudaStreamEndCapture");
      cudaGraphExec_t exec;
      check(static_cast<int>(cudaGraphInstantiate(&exec, graph, 0)), "cudaGraphInstantiate");
      cudaGraphDestroy(graph);
      it = m.graphs.emplace(key, exec).first;
    }
    if (it != m.graphs.end()) {
      check(static_cast<int>(cudaGraphLaunch(it->second, s)), "cudaGraphLaunch");
    } else {
      run(T, n_seq, all_rows, logits, s);
      m.seen.insert(key);
    }
  } else {
    run(T, n_seq, all_rows, logits, s);
  }
  for (int i = 0; i < n_seq; ++i) slot_len_[b.slots[i]] += b.cu_seqlens[i + 1] - b.cu_seqlens[i];
}

// The step's device work, from the token ids in d_meta to the logits. Everything it decides on the
// host depends only on (T, n_seq, all_rows), which is what makes it capturable.
void MetisModel::run(int T, int n_seq, bool all_rows, float* logits, cudaStream_t s) {
  Impl& m = *impl_;
  const MetisConfig& c = cfg_;
  const int S = m.S, D = m.D, R = m.R, Lat = m.Lat, E = m.E;

  LX_CHECK(lx_embed_streams(m.embedding, m.d_tokens, m.stream_emb, m.streams, T, S, D, s));
  LX_CHECK(lx_stream_mean(m.streams, m.previous, T, S, D, s));
  cudaMemsetAsync(m.history, 0, sizeof(float) * T * R, s);
  cudaMemsetAsync(m.continuation, 0, sizeof(float) * T, s);
  int n_entries = 0;

  for (int pass = 0; pass < m.P_passes; ++pass) {
    LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
    LX_CHECK(glue::sub(m.state, m.previous, m.difference, static_cast<int64_t>(T) * D, s));
    int anchor = 0;
    for (int i = 0; i < m.L_layers; ++i) {
      Impl::Layer& layer = m.layers[i];
      const int slot_index = pass * m.L_layers + i;
      const int params = S * S + 2 * S;

      // depth-memory read and routing features
      memory_retrieve(m, T, n_entries, s);
      LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
      routing_features(m, T, s);

      // mixer
      const float* mc = layer.mixer_conn.params + pass * params;
      LX_CHECK(lx_mhc_read(m.streams, mc, mc + S * S, layer.mixer_conn.norm, c.rms_norm_eps, m.mixed, m.h_bf, T, S, D, s));
      if (layer.attention) {
        gemm(m.h_bf, D, layer.qkv, nullptr, m.qkv_buf, LX_F32, m.qkv_width, T, m.qkv_width, D, s);
        gemm(m.h_bf, D, layer.lora_down, nullptr, m.lora_bf, LX_BF16, m.rank, T, m.rank, D, s);
        gemm(m.lora_bf, m.rank, layer.lora_up, nullptr, m.qkv_buf, LX_F32, m.qkv_width, T, m.qkv_width, m.rank, s,
             layer.pass_gate[pass], 1.0f);
        LX_CHECK(lx_attention_varlen(m.qkv_buf, m.qkv_width, m.k_cache[slot_index], m.v_cache[slot_index], m.d_slots,
                                     m.d_start_pos, m.d_cu, n_seq, m.kv_capacity, m.att_bf, m.AH, m.KVH, m.Dh,
                                     1.0f / std::sqrt(static_cast<float>(m.Dh)), s));
        gemm(m.att_bf, m.AH * m.Dh, layer.out, nullptr, m.update, LX_F32, D, T, D, m.AH * m.Dh, s);
      } else {
        const int gn = m.G * m.N;
        gemm(m.h_bf, D, layer.in_proj, nullptr, m.zxbcdt, LX_F32, m.proj, T, m.proj, D, s);
        LX_CHECK(lx_conv1d_varlen(m.zxbcdt + m.I, m.proj, layer.conv_w, layer.conv_b, m.conv_state[slot_index],
                                  m.d_slots, m.d_has_state, m.d_cu, n_seq, m.xbc, m.conv_dim, m.W, s));
        LX_CHECK(lx_ssm_scan_varlen(m.xbc, m.conv_dim, m.zxbcdt + m.I + m.conv_dim, m.proj, m.xbc + m.I,
                                    m.xbc + m.I + gn, m.conv_dim, layer.A_log, layer.D, layer.dt_bias,
                                    m.ssm_state[slot_index], m.d_slots, m.d_has_state, m.d_cu, n_seq, m.ssm_y, m.H,
                                    m.Pd, m.G, m.N, s));
        LX_CHECK(lx_gated_rmsnorm(m.ssm_y, m.zxbcdt, m.proj, layer.norm_w, m.normed_bf, T, m.I, m.G,
                                  c.mamba_norm_eps, s));
        gemm(m.normed_bf, m.I, layer.out_proj, nullptr, m.update, LX_F32, D, T, D, m.I, s);
      }
      LX_CHECK(lx_mhc_write(m.mixed, mc + S * S + S, m.update, m.streams, T, S, D, s));

      // latent mixture of experts
      const float* ec = layer.moe_conn.params + pass * params;
      LX_CHECK(lx_mhc_read(m.streams, ec, ec + S * S, layer.moe_conn.norm, c.rms_norm_eps, m.mixed, m.h_bf, T, S, D, s));
      gemm(m.h_bf, D, layer.latent_down, nullptr, m.latent, LX_F32, Lat, T, Lat, D, s);
      to_bf16(m.latent, Lat, m.latent_bf, Lat, T, Lat, s);
      copy_f32(m.latent, Lat, m.router_in, Lat + R, T, Lat, s);
      copy_f32(m.features, R, m.router_in + Lat, Lat + R, T, R, s);
      LX_CHECK(lx_gemm_f32(m.router_in, Lat + R, layer.router_w, layer.router_b, m.router_logits, E, T, E, Lat + R, s));
      LX_CHECK(lx_moe_route(m.router_logits, layer.selection_bias, m.route_ids, m.route_w, m.entropy, m.confidence, T,
                            E, m.K, s));
      LX_CHECK(lx_moe_experts(m.latent_bf, m.route_ids, m.route_w, layer.gate_up, layer.down, m.routed, T, Lat, m.Iexp,
                              m.K, s));
      gemm(m.latent_bf, Lat, layer.shared_gate_up, nullptr, m.shared_gu, LX_F32, 2 * m.Ishared, T, 2 * m.Ishared, Lat, s);
      LX_CHECK(lx_silu_mul(m.shared_gu, 2 * m.Ishared, m.shared_bf, LX_BF16, T, m.Ishared, s));
      gemm(m.shared_bf, m.Ishared, layer.shared_down, nullptr, m.routed, LX_F32, Lat, T, Lat, m.Ishared, s, 1.0f, 1.0f);
      to_bf16(m.routed, Lat, m.routed_bf, Lat, T, Lat, s);
      gemm(m.routed_bf, Lat, layer.latent_up, nullptr, m.update, LX_F32, D, T, D, Lat, s);
      LX_CHECK(lx_moe_summary(m.route_ids, m.route_w, layer.expert_emb, m.summary, T, R, m.K, s));
      LX_CHECK(lx_mhc_write(m.mixed, ec + S * S + S, m.update, m.streams, T, S, D, s));

      LX_CHECK(glue::axpby(m.history, m.summary, 0.5f, 0.5f, static_cast<int64_t>(T) * R, s));
      if (layer.attention) memory_write(m, c, T, pass, anchor++, n_entries, s);
    }

    // the end of a pass: one more read, the depth head, and an entry for where the pass left the token
    memory_retrieve(m, T, n_entries, s);
    LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
    LX_CHECK(glue::sub(m.state, m.previous, m.difference, static_cast<int64_t>(T) * D, s));
    routing_features(m, T, s);
    concat4(m, m.state, m.mem_summary, m.difference, m.features, m.cont_in, LX_F32, T, s);
    LX_CHECK(lx_gemm_f32(m.cont_in, 3 * D + R, m.c_hidden_w, m.c_hidden_b, m.cont_hidden, R, T, R, 3 * D + R, s));
    LX_CHECK(lx_activate_f32(m.cont_hidden, static_cast<int64_t>(T) * R, 1, s));
    LX_CHECK(lx_gemm_f32(m.cont_hidden, R, m.c_out_w, m.c_out_b, m.cont_logit, 1, T, 1, R, s));
    LX_CHECK(lx_continuation_finish(m.cont_logit, m.state, m.difference, c.continuation_logit_limit, m.continuation, T,
                                    D, s));
    memory_write(m, c, T, pass, anchor, n_entries, s);
    copy_f32(m.state, D, m.previous, D, T, D, s);
  }

  // mean over streams, the final norm and the tied LM head
  LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
  if (all_rows) {
    LX_CHECK(lx_rmsnorm(m.state, LX_F32, m.final_norm, m.out_bf, LX_BF16, T, D, c.rms_norm_eps, s));
    gemm(m.out_bf, D, m.embedding, nullptr, logits, LX_F32, c.vocab_size, T, c.vocab_size, D, s);
  } else {
    LX_CHECK(glue::gather_last(m.state, m.d_cu, m.last_rows, n_seq, D, s));
    LX_CHECK(lx_rmsnorm(m.last_rows, LX_F32, m.final_norm, m.out_bf, LX_BF16, n_seq, D, c.rms_norm_eps, s));
    gemm(m.out_bf, D, m.embedding, nullptr, logits, LX_F32, c.vocab_size, n_seq, c.vocab_size, D, s);
  }
}

}  // namespace lx
