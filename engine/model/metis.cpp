#include "model/metis.h"

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <initializer_list>
#include <map>
#include <set>
#include <tuple>
#include <cstring>
#include <stdexcept>

#include "cuda/kernels.h"
#include "model/glue.h"
#include "runtime/offload.h"

namespace lx {
namespace {

void check(int status, const char* what) {
  if (status != 0)
    throw std::runtime_error(std::string(what) + " failed with error " + std::to_string(status) + ": " +
                             lx_rt_error_string(status));
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
    check(lx_rt_copy(out.data(), w.data, n * 4, LX_COPY_D2H, nullptr), "device to host copy");
    check(lx_rt_stream_sync(nullptr), "device to host copy");
  } else {
    std::vector<uint16_t> raw(n);
    check(lx_rt_copy(raw.data(), w.data, n * 2, LX_COPY_D2H, nullptr), "device to host copy");
    check(lx_rt_stream_sync(nullptr), "device to host copy");
    for (int64_t i = 0; i < n; ++i) out[i] = bf16_to_float(raw[i]);
  }
  return out;
}

}  // namespace

// A projection: its bf16 weight (prefill, through the dense GEMM) and, when quantized, an int8 copy with
// per-row scales that decode steps read instead, at half the bytes. With LX_BF16X, decode reads a
// lossless compressed copy instead (LX_W_BF16X: the same bf16 values in ~12.2 bits, so the same
// logits to the bit at ~1.3x fewer bytes).
struct Linear {
  const void* w = nullptr;     // bf16 [N, K]
  const void* bias = nullptr;  // bf16 [N] or null
  int8_t* q = nullptr;         // int8 [N, K], or null to decode from w
  float* scale = nullptr;      // f32 [N]
  void* x16 = nullptr;         // LX_W_BF16X encoding of w, or null
  int N = 0, K = 0;
  bool host = false;           // w is in pinned host memory (offload): prefill stages it into VRAM
  lx_weight pw{LX_W_BF16, nullptr, nullptr, 1.0f};  // what prefill multiplies by (LX_PREFILL): w, or a copy
};

struct MetisModel::Impl {
  // ---- weights -----------------------------------------------------------------------------
  struct Connection {
    const void* norm;
    float* params;  // device: matrix [S*S], read [S], write [S] for each pass
  };
  struct Layer {
    bool attention;
    // Mamba-2
    Linear in_proj, out_proj;
    const void *conv_w, *conv_b, *norm_w;
    const float *A_log, *D, *dt_bias;
    // attention
    Linear qkv, out, lora_down, lora_up;
    std::vector<float> pass_gate;  // tanh(pass_gates[p]) rounded to bf16, per pass
    // MoE
    Linear latent_down, latent_up, shared_gate_up, shared_down;
    const void *expert_emb, *gate_up, *down;      // bf16 experts, for prefill
    int8_t *gate_up_q = nullptr, *down_q = nullptr;  // int8 experts, for decode
    float *gate_up_scale = nullptr, *down_scale = nullptr;
    lx_weight gate_up_pw{}, down_pw{};  // the experts as prefill multiplies them (LX_PREFILL)
    void *gate_up_x16 = nullptr, *down_x16 = nullptr;  // LX_W_BF16X experts, for decode (LX_BF16X)
    const float *router_w, *router_b, *selection_bias;
    Connection mixer_conn, moe_conn;
  };
  std::vector<Layer> layers;
  const void *embedding, *stream_emb, *final_norm;
  Linear head;  // the tied LM head (the embedding)
  // depth memory
  Linear m_state, m_meta, m_query, m_key, m_value, m_out, m_route;
  const void *m_pass_emb, *m_anchor_emb, *m_gate, *m_gate_b;
  const float *c_hidden_w, *c_hidden_b, *c_out_w, *c_out_b;

  // ---- dimensions --------------------------------------------------------------------------
  int S, D, L_layers, P_passes, M, R, E, K, Lat, Iexp, Ishared, Emax;
  int I, H, Pd, G, N, W, conv_dim, proj;  // Mamba
  int AH, KVH, Dh, qkv_width, rank;       // attention
  int max_tokens, max_slots, kv_capacity;

  // ---- persistent state, per (pass, layer) ---------------------------------------------------
  std::vector<float*> conv_state, ssm_state;
  std::vector<void*> k_cache, v_cache;
  // speculative mode: each Mamba layer's pending positions (kernels.h, "Speculative decoding")
  bool speculative = false;
  std::vector<float*> conv_hist, ssm_hist;
  // feature taps: tap_col[pass * (L_layers + 1) + layer + 1] is the tap's first column in tap_out,
  // or -1 (layer -1, slot 0, is the end of the pass)
  std::vector<FeatureTap> taps;
  std::vector<int> tap_col;
  float* tap_out = nullptr;   // f32 [max_tokens, tap_dim]
  float* tap_mean = nullptr;  // scratch [max_tokens, D]
  int tap_dim = 0;

  // ---- per-step scratch --------------------------------------------------------------------
  std::vector<void*> allocations;
  float *streams, *mixed, *q, *projected, *mem_summary, *state, *previous, *difference, *history, *features,
      *cont_in, *cont_hidden, *cont_logit, *continuation, *update, *zxbcdt, *xbc, *ssm_y, *qkv_buf, *latent,
      *router_in, *router_logits, *route_w, *entropy, *confidence, *routed, *shared_gu, *summary, *bank_k,
      *bank_v, *entry;
  int32_t *route_ids, *d_meta, *d_tokens, *d_cu, *d_slots, *d_has_state, *d_start_pos, *d_fold, *d_keep;
  int32_t* h_meta = nullptr;  // pinned staging for d_meta, so the upload is one async copy
  size_t meta_count = 0;
  void* meta_copied = nullptr;
  float* last_rows = nullptr;
  // Decode steps (one token per sequence) and verify steps of up to kGemvRows tokens replay a captured
  // graph: the first step of a shape runs eagerly (which also warms the dense GEMM up), the second
  // captures, every later one replays.
  using GraphKey = std::tuple<int, int, bool, float*, bool>;
  std::set<GraphKey> seen;
  std::map<GraphKey, void*> graphs;
  bool use_graphs = true;
  // decode-sized products (at most this many rows) run as GEMVs; larger ones on the dense GEMM
  static constexpr int kGemvRows = 8;
  void* cast_bf = nullptr;  // bf16 copy of an f32 input for a dense GEMM product
  size_t cast_capacity = 0;
  void* moe_workspace = nullptr;
  offload::ExpertOffload* offload = nullptr;  // the routed experts, when they run offloaded
  FILE* route_trace = nullptr;                // LX_ROUTE_TRACE: every layer's routing, for the placement stats
  // With the trace: the next layer's routing predicted from the streams as they leave this layer
  // (the next layer's MoE read, latent_down and router applied early, with this layer's features).
  float *pred_mixed = nullptr, *pred_router_in = nullptr, *pred_logits = nullptr,
        *pred_w = nullptr, *pred_entropy = nullptr, *pred_confidence = nullptr;
  void* pred_h = nullptr;
  int32_t* pred_ids = nullptr;
  static constexpr int kPredicted = 8;
  // LX_OFFLOAD_PREFETCH=n: in offloaded decode steps, a graph branch predicts the next layer's routing
  // the same way and copies the top n predicted experts that are not in VRAM ahead of the layer.
  int prefetch = 0;
  void* side = nullptr;
  // Decode steps: the routing features run on route_side, beside the mixer (see run)
  void *route_side = nullptr, *route_fork = nullptr, *route_join = nullptr;
  std::vector<void*> fork_events, join_events;  // per (pass, layer)
  // Projections kept in host memory (offload, LX_OFFLOAD_DENSE_HOST): a prefill product copies its
  // weight into one of two VRAM buffers on a copy stream, so the next weight crosses PCIe while this
  // product runs, instead of the product reading host memory tile by tile.
  size_t stage_bytes = 0;
  void* stage[2] = {nullptr, nullptr};
  int stage_next = 0;
  void* stage_stream = nullptr;
  void *stage_copied[2] = {nullptr, nullptr}, *stage_free[2] = {nullptr, nullptr};
  // Prefill products (more than kGemvRows rows): lx_gemm_w and lx_moe_experts_grouped on the weight
  // format and math LX_PREFILL names (bf16, w8a16, w4a16, int8, fp8, w4a8, fp4); LX_PREFILL_KEEP lists
  // name fragments that stay bf16. The default keeps what the FP8 release keeps (the Mamba-2 output
  // projection, the tied LM head, the depth memory, the latent projections and the shared expert), so
  // only the experts, the Mamba-2 input projections and the attention projections are quantized: the
  // measured best trade (W8A16 then sits at the BF16 noise floor; INT8 at 0.010 / 0.024 / 0.0058).
  lx_wtype prefill_w = LX_W_BF16;
  lx_math prefill_math = LX_MATH_BF16;
  std::string prefill_keep;
  bool grouped_experts = true;  // LX_OLD_EXPERTS=1: lx_moe_experts for prefill, as before
  void* gemm_ws = nullptr;
  void* moe_grouped_ws = nullptr;
  std::string w8_keep;  // LX_W8_KEEP: comma-separated name fragments that stay bf16 for decode
  bool bf16x = false;        // LX_BF16X: decode reads lossless compressed bf16 copies (no int8)
  bool keep_bf16x = false;   // LX_KEEP_BF16X: what decode would read as plain bf16 it reads as LX_W_BF16X
  bool decode_bf16 = false;  // LX_DECODE_BF16: plain bf16 decode experts run on bf16x.cu's expert kernels
  void *streams_bf, *retrieved_bf, *feat_in_bf, *h_bf, *normed_bf, *lora_bf, *att_bf, *latent_bf, *shared_bf,
      *routed_bf, *entry_bf, *meta_bf, *out_bf;

  // ---- decode programs (decode_*.cu): a decode step as one persistent kernel ------------------
  // Built on a shape's first decode step and run from then on, with LX_PROGRAM set and nothing the
  // program does not cover (decode_program_covers); otherwise decode goes through the kernel-per-op
  // path above, captured in graphs.
  bool use_program = false;
  std::map<std::pair<int, float*>, lx_program*> programs;
  // route_projection without its difference columns, for the routing inside a pass (where the
  // difference is zero); the same rows and scales, so it multiplies exactly as the full weight
  Linear route_in;
  // The depth memory's composite bank projection: an entry e becomes U | P | c (2D + 4 rows), with
  // U = Wq^T k_e, P = Wout v_e + out bias and c = bq . k_e, so a read needs no projections of its own.
  lx_weight bank_w{};
  float* bank_b = nullptr;  // its bias f32 [2D + 4]
  float *pbank = nullptr, *lora_f = nullptr, *head_ss = nullptr, *next_src = nullptr, *next_parts = nullptr,
        *rms = nullptr, *wstate = nullptr;
  int next_parts_cap = 0;
  std::map<lx_program*, std::vector<std::pair<int, std::string>>> program_names;
  int traced_steps = 0;

  template <typename T>
  T* alloc(size_t count, size_t elem = sizeof(T)) {
    void* p = nullptr;
    const int64_t bytes = static_cast<int64_t>(count * elem);
    const std::string what = "device allocation of " + std::to_string(bytes) + " bytes";
    check(lx_rt_alloc(&p, bytes), what.c_str());  // zeroed by the runtime
    allocations.push_back(p);
    return static_cast<T*>(p);
  }
  ~Impl() {
    for (auto& [key, prog] : programs) lx_program_destroy(prog);
    if (route_trace) std::fclose(route_trace);
    for (void* e : fork_events) lx_rt_event_destroy(e);
    for (void* e : join_events) lx_rt_event_destroy(e);
    if (side) lx_rt_stream_destroy(side);
    if (route_side) lx_rt_stream_destroy(route_side);
    if (route_fork) lx_rt_event_destroy(route_fork);
    if (route_join) lx_rt_event_destroy(route_join);
    for (int b = 0; b < 2; ++b) {
      if (stage_copied[b]) lx_rt_event_destroy(stage_copied[b]);
      if (stage_free[b]) lx_rt_event_destroy(stage_free[b]);
    }
    if (stage_stream) lx_rt_stream_destroy(stage_stream);
    for (auto& [key, exec] : graphs) lx_rt_graph_destroy(exec);
    if (meta_copied) lx_rt_event_destroy(meta_copied);
    if (h_meta) lx_rt_host_free(h_meta);
    for (void* p : allocations) lx_rt_free(p);
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

// Whether a weight stays bf16 for decode: LX_W8_KEEP lists name fragments to keep, for measuring
// what quantizing each part of the model costs.
bool listed(const std::string& list, const std::string& name) {
  for (size_t pos = 0; pos < list.size();) {
    const size_t end = list.find(',', pos);
    const std::string frag = list.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    if (!frag.empty() && name.find(frag) != std::string::npos) return true;
    pos = end == std::string::npos ? list.size() : end + 1;
  }
  return false;
}
bool kept_bf16(const Impl& m, const std::string& name) { return listed(m.w8_keep, name); }

// The prefill copy of a bf16 [rows, cols] weight (rows may be a whole expert stack) in m.prefill_w;
// an int8 decode copy is shared when the formats agree.
lx_weight prefill_weight(Impl& m, const void* w, int8_t* q, float* scale, int64_t rows, int64_t cols) {
  lx_weight out{LX_W_BF16, w, nullptr, 1.0f};
  if (m.prefill_w == LX_W_BF16) return out;
  if (m.prefill_w == LX_W_INT8 && q) return lx_weight{LX_W_INT8, q, scale, 1.0f};
  out.type = m.prefill_w;
  if (m.prefill_w == LX_W_NVFP4) {
    out.data = m.alloc<uint8_t>(static_cast<size_t>(rows) * cols / 2);
    out.scale = m.alloc<uint8_t>(static_cast<size_t>(rows) * cols / 16);
  } else {
    out.data = m.alloc<uint8_t>(static_cast<size_t>(rows) * cols);
    out.scale = m.alloc<float>(rows);
  }
  LX_CHECK(lx_quantize_weight(w, &out, rows, static_cast<int>(cols), nullptr));
  return out;
}

// The lossless compressed copy of a bf16 tensor ([rows, K]), for decode.
void* encode_bf16x(Impl& m, const void* w, int64_t rows, int64_t K) {
  const int64_t bytes = lx_bf16x_bytes(w, rows, static_cast<int>(K), nullptr);
  if (bytes <= 0) throw std::runtime_error("lx_bf16x_bytes failed");
  void* out = m.alloc<uint8_t>(static_cast<size_t>(bytes));
  LX_CHECK(lx_bf16x_encode(w, rows, static_cast<int>(K), out, bytes, nullptr));
  return out;
}

// A projection from the checkpoint; with `quantize`, also its int8 decode copy, unless its name
// contains a fragment of LX_W8_KEEP (for measuring what quantizing each part costs). With LX_BF16X,
// every projection whose rows the format makes smaller gets a lossless copy instead, whatever
// `quantize` says (lossless, so the Mamba out_proj too).
Linear make_linear(Impl& m, const WeightLookup& lw, const std::string& weight, const std::string& bias, int64_t N,
                   int64_t K, bool quantize) {
  Linear l;
  l.w = need(lw, weight, LX_BF16, {N, K});
  if (!bias.empty()) l.bias = need(lw, bias, LX_BF16, {N});
  l.N = static_cast<int>(N);
  l.K = static_cast<int>(K);
  // A host-resident weight up to kStageMaxBytes is staged for prefill; a larger one (the tied head's
  // embedding table, 335 MB on Logos, read by prefill only when every row's logits are wanted) is
  // read across PCIe where it lies rather than doubling its size in VRAM buffers.
  constexpr size_t kStageMaxBytes = 128ull << 20;
  const size_t bytes = static_cast<size_t>(N) * static_cast<size_t>(K) * 2;
  if (lx_rt_is_host_memory(l.w) && bytes <= kStageMaxBytes) {
    l.host = true;
    m.stage_bytes = std::max(m.stage_bytes, bytes);
  }
  if (m.bf16x) {
    if (lx_bf16x_row_bytes(static_cast<int>(K)) < 2 * K) l.x16 = encode_bf16x(m, l.w, N, K);
  } else if (quantize && !kept_bf16(m, weight)) {
    l.q = m.alloc<int8_t>(static_cast<size_t>(N) * K);
    l.scale = m.alloc<float>(N);
    LX_CHECK(lx_quantize_w8(l.w, l.q, l.scale, N, static_cast<int>(K), nullptr));
  } else if (m.keep_bf16x && lx_bf16x_row_bytes(static_cast<int>(K)) < 2 * K) {
    l.x16 = encode_bf16x(m, l.w, N, K);
  }
  l.pw = lx_weight{LX_W_BF16, l.w, nullptr, 1.0f};
  if (quantize && !listed(m.prefill_keep, weight)) l.pw = prefill_weight(m, l.w, l.q, l.scale, N, K);
  return l;
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
  check(lx_rt_copy(d, params.data(), static_cast<int64_t>(params.size() * 4), LX_COPY_H2D, nullptr),
        "host to device copy");
  check(lx_rt_stream_sync(nullptr), "host to device copy");
  return d;
}

// A projection as the program products read it: the int8 decode copy when there is one.
lx_weight wview(const Linear& l) {
  lx_weight w{};
  w.type = l.q ? LX_W_INT8 : LX_W_BF16;
  w.data = l.q ? static_cast<const void*>(l.q) : l.w;
  w.scale = l.q ? l.scale : nullptr;
  w.global = 1.0f;
  return w;
}

// route_projection [R, 3D + R] without the difference columns [2D, 3D): [R, 2D + R], in the
// decode format of the full weight (an int8 copy keeps its rows' scales).
Linear route_without_difference(Impl& m, const Linear& full, int D, int R) {
  Linear l = full;
  const int K = 3 * D + R, Kin = 2 * D + R, elem = full.q ? 1 : 2;
  const void* src = full.q ? static_cast<const void*>(full.q) : full.w;
  std::vector<char> h(static_cast<size_t>(R) * K * elem), out(static_cast<size_t>(R) * Kin * elem);
  check(lx_rt_copy(h.data(), src, static_cast<int64_t>(h.size()), LX_COPY_D2H, nullptr), "device to host copy");
  check(lx_rt_stream_sync(nullptr), "device to host copy");
  for (int r = 0; r < R; ++r) {
    const char* row = h.data() + static_cast<size_t>(r) * K * elem;
    char* dst = out.data() + static_cast<size_t>(r) * Kin * elem;
    std::memcpy(dst, row, static_cast<size_t>(2 * D) * elem);
    std::memcpy(dst + static_cast<size_t>(2 * D) * elem, row + static_cast<size_t>(3 * D) * elem,
                static_cast<size_t>(R) * elem);
  }
  void* d = m.alloc<char>(out.size());
  check(lx_rt_copy(d, out.data(), static_cast<int64_t>(out.size()), LX_COPY_H2D, nullptr), "host to device copy");
  check(lx_rt_stream_sync(nullptr), "host to device copy");
  if (full.q) {
    l.q = static_cast<int8_t*>(d);
    l.scale = full.scale;
  } else {
    l.w = d;
  }
  l.K = Kin;
  return l;
}

// The composite bank projection (see Impl::bank_w), computed in double from the bf16 checkpoint
// weights of the query, key, value and output projections. It stays bf16 in every format: an int8
// copy of the folded matrix costs more than int8 copies of its four factors (INT8 top-20 KL
// 0.0046 / 0.0143 / 0.0029 against 0.0038 / 0.0136 / 0.0024), for 5 MB a token.
void build_bank_projection(Impl& m, const WeightLookup& lw, const std::string& dm) {
  const int D = m.D, Mm = m.M, rows = 2 * D + 4;
  const auto Wq = host_values(lw(dm + "query.weight")), bq = host_values(lw(dm + "query.bias"));
  const auto Wk = host_values(lw(dm + "key.weight")), bk = host_values(lw(dm + "key.bias"));
  const auto Wv = host_values(lw(dm + "value.weight")), bv = host_values(lw(dm + "value.bias"));
  const auto Wo = host_values(lw(dm + "output.weight")), bo = host_values(lw(dm + "output.bias"));
  std::vector<float> W(static_cast<size_t>(rows) * Mm, 0.0f), b(rows, 0.0f);
  std::vector<double> acc(Mm);
  // U rows: A = Wq^T Wk (row d: sum_i Wq[i, d] Wk[i, :]), a = Wq^T bk
  for (int d = 0; d < D; ++d) {
    std::fill(acc.begin(), acc.end(), 0.0);
    double ad = 0.0;
    for (int i = 0; i < Mm; ++i) {
      const double q = Wq[static_cast<size_t>(i) * D + d];
      const float* wk = &Wk[static_cast<size_t>(i) * Mm];
      for (int j = 0; j < Mm; ++j) acc[j] += q * wk[j];
      ad += q * bk[i];
    }
    for (int j = 0; j < Mm; ++j) W[static_cast<size_t>(d) * Mm + j] = static_cast<float>(acc[j]);
    b[d] = static_cast<float>(ad);
  }
  // P rows: B = Wout Wv, b = Wout bv + bout
  for (int d = 0; d < D; ++d) {
    std::fill(acc.begin(), acc.end(), 0.0);
    double bd = bo[d];
    for (int i = 0; i < Mm; ++i) {
      const double o = Wo[static_cast<size_t>(d) * Mm + i];
      const float* wv = &Wv[static_cast<size_t>(i) * Mm];
      for (int j = 0; j < Mm; ++j) acc[j] += o * wv[j];
      bd += o * bv[i];
    }
    for (int j = 0; j < Mm; ++j) W[static_cast<size_t>(D + d) * Mm + j] = static_cast<float>(acc[j]);
    b[D + d] = static_cast<float>(bd);
  }
  // the c row: Wk^T bq, and its bias bq . bk; three zero rows pad the entry to 16 bytes
  double c0 = 0.0;
  for (int i = 0; i < Mm; ++i) c0 += static_cast<double>(bq[i]) * bk[i];
  for (int j = 0; j < Mm; ++j) {
    double c = 0.0;
    for (int i = 0; i < Mm; ++i) c += static_cast<double>(bq[i]) * Wk[static_cast<size_t>(i) * Mm + j];
    W[static_cast<size_t>(2 * D) * Mm + j] = static_cast<float>(c);
  }
  b[2 * D] = static_cast<float>(c0);

  std::vector<uint16_t> wb(W.size());
  for (size_t i = 0; i < W.size(); ++i) {
    const float r = round_bf16(W[i]);
    uint32_t u;
    std::memcpy(&u, &r, 4);
    wb[i] = static_cast<uint16_t>(u >> 16);
  }
  uint16_t* dw = m.alloc<uint16_t>(wb.size());
  check(lx_rt_copy(dw, wb.data(), static_cast<int64_t>(wb.size() * 2), LX_COPY_H2D, nullptr), "host to device copy");
  m.bank_b = m.alloc<float>(rows);
  check(lx_rt_copy(m.bank_b, b.data(), static_cast<int64_t>(rows) * 4, LX_COPY_H2D, nullptr), "host to device copy");
  check(lx_rt_stream_sync(nullptr), "host to device copy");
  m.bank_w = {LX_W_BF16, dw, nullptr, 1.0f};
}

}  // namespace

MetisModel::MetisModel(const MetisConfig& c, const WeightLookup& lw, int max_slots, int kv_capacity,
                       int max_tokens, offload::ExpertOffload* experts, bool speculative)
    : cfg_(c), slot_len_(max_slots, 0), folded_(max_slots, 0), pending_(max_slots, 0), impl_(new Impl) {
  // A constructor that throws (a device allocation that fails, say) never reaches the destructor, so the device
  // memory this build took would stay allocated for the life of the process: free it on the way out.
  struct FreeOnThrow {
    Impl*& impl;
    bool armed = true;
    ~FreeOnThrow() {
      if (armed) {
        delete impl;
        impl = nullptr;
      }
    }
  } free_on_throw{impl_};
  Impl& m = *impl_;
  m.offload = experts;
  if (const char* trace = std::getenv("LX_ROUTE_TRACE")) {
    m.route_trace = std::fopen(trace, "wb");
    if (!m.route_trace) throw std::runtime_error(std::string("cannot open ") + trace);
  }
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

  // Decode reads int8 copies of every large projection except the Mamba-2 output projection: it is
  // the one part of Metis that int8 visibly hurts (alone it takes the tool-prompt KL from 0.014 to
  // 0.026), as it was for the GGUF, FP8 and MLX releases. LX_W8_KEEP overrides the list.
  m.w8_keep = std::getenv("LX_W8_KEEP") ? std::getenv("LX_W8_KEEP") : "out_proj";
  if (std::getenv("LX_NO_W8")) m.w8_keep = "layers,depth_memory,embedding";
  m.prefill_keep = std::getenv("LX_PREFILL_KEEP") ? std::getenv("LX_PREFILL_KEEP")
                                                  : "out_proj,embedding,depth_memory,latent_down,latent_up,shared_expert";
  m.grouped_experts = std::getenv("LX_OLD_EXPERTS") == nullptr;
  if (const char* p = std::getenv("LX_PREFILL")) {
    const std::string v = p;
    struct Mode {
      const char* name;
      lx_wtype w;
      lx_math math;
    };
    const Mode modes[] = {{"bf16", LX_W_BF16, LX_MATH_BF16},   {"w8a16", LX_W_INT8, LX_MATH_BF16},
                          {"w8a16f", LX_W_FP8, LX_MATH_BF16},  {"w4a16", LX_W_NVFP4, LX_MATH_BF16},
                          {"int8", LX_W_INT8, LX_MATH_INT8},   {"fp8", LX_W_FP8, LX_MATH_FP8},
                          {"w4a8", LX_W_NVFP4, LX_MATH_FP8},   {"fp4", LX_W_NVFP4, LX_MATH_FP4}};
    bool found = false;
    for (const Mode& mode : modes)
      if (v == mode.name) {
        m.prefill_w = mode.w;
        m.prefill_math = mode.math;
        found = true;
      }
    if (!found) throw std::runtime_error("LX_PREFILL: unknown mode " + v);
  }
  // LX_W_BF16X (bf16x.cu) is lossless, so either switch keeps the logits of the path it replaces to
  // the bit. LX_BF16X: every decode projection is read compressed instead of int8 or bf16 (BF16
  // logits). LX_KEEP_BF16X: int8 stays, and what stays bf16 (LX_W8_KEEP) is read compressed.
  // LX_DECODE_BF16 (for measuring the format against the same kernels): plain bf16 experts through
  // bf16x.cu's expert kernels. Its GEMV runs lx_gemv's own order, so lx_gemv is already that baseline.
  m.bf16x = std::getenv("LX_BF16X") != nullptr;
  m.keep_bf16x = std::getenv("LX_KEEP_BF16X") != nullptr;
  m.decode_bf16 = std::getenv("LX_DECODE_BF16") != nullptr;
  m.embedding = need(lw, "embedding.weight", LX_BF16, {V, D});
  m.head = make_linear(m, lw, "embedding.weight", "", V, D, true);
  m.stream_emb = need(lw, "stream_embeddings", LX_BF16, {S, D});
  m.final_norm = need(lw, "final_norm.weight", LX_BF16, {D});
  const auto pass_embeddings = host_values(lw("pass_embeddings"));
  need(lw, "pass_embeddings", LX_BF16, {c.max_passes, c.mhc_pass_embedding_dim});

  const std::string dm = "depth_memory.";
  m.m_state = make_linear(m, lw, dm + "state_write.weight", dm + "state_write.bias", Mm, D, true);
  m.m_meta = make_linear(m, lw, dm + "metadata_write.weight", dm + "metadata_write.bias", Mm, R + 4, false);
  m.m_pass_emb = need(lw, dm + "pass_embeddings", LX_BF16, {c.max_passes, Mm});
  m.m_anchor_emb =
      need(lw, dm + "anchor_embeddings", LX_BF16, {static_cast<int64_t>(c.attention_layer_indices.size() + 1), Mm});
  m.m_query = make_linear(m, lw, dm + "query.weight", dm + "query.bias", Mm, D, true);
  m.m_key = make_linear(m, lw, dm + "key.weight", dm + "key.bias", Mm, Mm, false);
  m.m_value = make_linear(m, lw, dm + "value.weight", dm + "value.bias", Mm, Mm, false);
  m.m_out = make_linear(m, lw, dm + "output.weight", dm + "output.bias", D, Mm, true);
  m.m_route = make_linear(m, lw, dm + "route_projection.weight", dm + "route_projection.bias", R, 3 * D + R, true);
  m.m_gate = need(lw, dm + "stream_gate", LX_BF16, {S, D});
  m.m_gate_b = need(lw, dm + "stream_gate_bias", LX_BF16, {S});
  m.c_hidden_w = static_cast<const float*>(need(lw, "continuation.hidden.weight", LX_F32, {R, 3 * D + R}));
  m.c_hidden_b = static_cast<const float*>(need(lw, "continuation.hidden.bias", LX_F32, {R}));
  m.c_out_w = static_cast<const float*>(need(lw, "continuation.output.weight", LX_F32, {1, R}));
  m.c_out_b = static_cast<const float*>(need(lw, "continuation.output.bias", LX_F32, {1}));
  // The decode program (LX_PROGRAM): not for what it does not cover yet (decode_program_covers), and
  // only on a backend that has it.
  const char* program = std::getenv("LX_PROGRAM");  // opt-in: any value but 0
  m.use_program = program && std::strcmp(program, "0") != 0 && !speculative && !experts && !m.bf16x && !m.keep_bf16x;
  if (m.use_program) {
    lx_program* probe = lx_program_create(1, nullptr, nullptr, nullptr, nullptr);
    m.use_program = probe != nullptr;
    if (probe) lx_program_destroy(probe);
  }
  if (m.use_program) {
    m.route_in = route_without_difference(m, m.m_route, D, R);
    build_bank_projection(m, lw, dm);
  }

  for (int i = 0; i < m.L_layers; ++i) {
    Impl::Layer layer{};
    const std::string p = "layers." + std::to_string(i) + ".";
    layer.attention = c.is_attention(i);
    if (layer.attention) {
      const std::string a = p + "mixer.";
      layer.qkv = make_linear(m, lw, a + "qkv.weight", "", m.qkv_width, D, true);
      layer.out = make_linear(m, lw, a + "out.weight", "", D, m.AH * m.Dh, true);
      layer.lora_down = make_linear(m, lw, a + "pass_lora_down.weight", "", m.rank, D, false);
      layer.lora_up = make_linear(m, lw, a + "pass_lora_up.weight", "", m.qkv_width, m.rank, false);
      auto gates = host_values(lw(a + "pass_gates"));
      need(lw, a + "pass_gates", LX_BF16, {c.max_passes});
      for (int pass = 0; pass < m.P_passes; ++pass) layer.pass_gate.push_back(round_bf16(std::tanh(gates[pass])));
    } else {
      const std::string a = p + "mixer.impl.mixer.";
      layer.in_proj = make_linear(m, lw, a + "in_proj.weight", "", m.proj, D, true);
      layer.conv_w = need(lw, a + "conv1d.weight", LX_BF16, {m.conv_dim, 1, m.W});
      layer.conv_b = need(lw, a + "conv1d.bias", LX_BF16, {m.conv_dim});
      layer.norm_w = need(lw, a + "norm.weight", LX_BF16, {m.I});
      layer.out_proj = make_linear(m, lw, a + "out_proj.weight", "", D, m.I, true);
      layer.A_log = static_cast<const float*>(need(lw, a + "A_log", LX_F32, {m.H}));
      layer.D = static_cast<const float*>(need(lw, a + "D", LX_F32, {m.H}));
      layer.dt_bias = static_cast<const float*>(need(lw, a + "dt_bias", LX_F32, {m.H}));
    }
    const std::string e = p + "moe.";
    layer.latent_down = make_linear(m, lw, e + "latent_down.weight", "", Lat, D, true);
    layer.latent_up = make_linear(m, lw, e + "latent_up.weight", "", D, Lat, true);
    layer.expert_emb = need(lw, e + "expert_embeddings", LX_BF16, {E, R});
    if (!m.offload) {
      layer.gate_up = need(lw, e + "local_experts.gate_up.weight", LX_BF16, {E, 2 * m.Iexp, Lat});
      layer.down = need(lw, e + "local_experts.down.weight", LX_BF16, {E, Lat, m.Iexp});
    }
    layer.shared_gate_up = make_linear(m, lw, e + "shared_expert.gate_up.weight", "", 2 * m.Ishared, Lat, true);
    layer.shared_down = make_linear(m, lw, e + "shared_expert.down.weight", "", Lat, m.Ishared, true);
    if (!m.offload && (m.bf16x || (m.keep_bf16x && kept_bf16(m, e + "local_experts.gate_up.weight")))) {
      layer.gate_up_x16 = encode_bf16x(m, layer.gate_up, static_cast<int64_t>(E) * 2 * m.Iexp, Lat);
      layer.down_x16 = encode_bf16x(m, layer.down, static_cast<int64_t>(E) * Lat, m.Iexp);
    } else if (!m.offload && !kept_bf16(m, e + "local_experts.gate_up.weight")) {
      const int64_t gu_rows = static_cast<int64_t>(E) * 2 * m.Iexp, dn_rows = static_cast<int64_t>(E) * Lat;
      layer.gate_up_q = m.alloc<int8_t>(gu_rows * Lat);
      layer.gate_up_scale = m.alloc<float>(gu_rows);
      layer.down_q = m.alloc<int8_t>(dn_rows * m.Iexp);
      layer.down_scale = m.alloc<float>(dn_rows);
      LX_CHECK(lx_quantize_w8(layer.gate_up, layer.gate_up_q, layer.gate_up_scale, gu_rows, Lat, nullptr));
      LX_CHECK(lx_quantize_w8(layer.down, layer.down_q, layer.down_scale, dn_rows, m.Iexp, nullptr));
    }
    if (listed(m.prefill_keep, e + "local_experts.gate_up.weight")) {
      layer.gate_up_pw = lx_weight{LX_W_BF16, layer.gate_up, nullptr, 1.0f};
      layer.down_pw = lx_weight{LX_W_BF16, layer.down, nullptr, 1.0f};
    } else {
      layer.gate_up_pw = prefill_weight(m, layer.gate_up, layer.gate_up_q, layer.gate_up_scale,
                                        static_cast<int64_t>(E) * 2 * m.Iexp, Lat);
      layer.down_pw = prefill_weight(m, layer.down, layer.down_q, layer.down_scale, static_cast<int64_t>(E) * Lat, m.Iexp);
    }
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
      const bool hist = speculative && !m.layers[i].attention;
      m.conv_hist.push_back(hist ? m.alloc<float>(static_cast<size_t>(max_slots) * LX_SPEC_MAX_PENDING * m.conv_dim)
                                 : nullptr);
      m.ssm_hist.push_back(
          hist ? m.alloc<float>(static_cast<size_t>(max_slots) * lx_spec_ssm_hist_floats(m.H, m.Pd, m.N)) : nullptr);
    }
  }
  m.speculative = speculative;
  m.tap_col.assign(static_cast<size_t>(m.P_passes) * (m.L_layers + 1), -1);

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
  if (m.use_program) {
    const size_t P = std::min<size_t>(T, 8);  // decode steps: one token per sequence, at most 8
    m.pbank = m.alloc<float>(P * m.Emax * (2 * D + 4));
    m.lora_f = m.alloc<float>(P * m.rank);
    m.head_ss = m.alloc<float>(P * m.H);
    m.next_src = m.alloc<float>(P * D);
    m.next_parts_cap = 4 * D;
    m.next_parts = m.alloc<float>(P * m.next_parts_cap);
    m.rms = m.alloc<float>(2 * P);
    m.wstate = m.alloc<float>(P * D);
  }
  m.cast_capacity = TS * static_cast<size_t>(std::max({3 * D + R, m.I, m.proj, D}));
  m.cast_bf = m.alloc<uint16_t>(m.cast_capacity);
  m.moe_workspace = m.alloc<char>(static_cast<size_t>(std::max(lx_moe_workspace_bytes(Impl::kGemvRows, m.Iexp, m.K),
                                                                 lx_moe_bf16x_workspace_bytes(Impl::kGemvRows, m.Iexp, m.K))));
  {
    const int kmax = std::max({3 * D + R, m.I, m.proj, D, Lat, m.Ishared, m.AH * m.Dh});
    const int rows = static_cast<int>(T);
    const int64_t ws =
        std::max(lx_gemm_workspace_bytes(rows, kmax, m.prefill_math), lx_gemm_workspace_bytes(rows, kmax, LX_MATH_BF16));
    m.gemm_ws = m.alloc<char>(static_cast<size_t>(ws));
    m.moe_grouped_ws =
        m.alloc<char>(static_cast<size_t>(lx_moe_grouped_workspace_bytes(rows, Lat, m.Iexp, E, m.K, m.prefill_math)));
  }
  m.meta_count = T + (max_slots + 1) + 5 * static_cast<size_t>(max_slots);
  m.d_meta = m.alloc<int32_t>(m.meta_count);
  m.d_tokens = m.d_meta;
  m.d_cu = m.d_tokens + T;
  m.d_slots = m.d_cu + max_slots + 1;
  m.d_has_state = m.d_slots + max_slots;
  m.d_start_pos = m.d_has_state + max_slots;
  m.d_fold = m.d_start_pos + max_slots;
  m.d_keep = m.d_fold + max_slots;
  void* pinned = nullptr;
  check(lx_rt_host_alloc(&pinned, static_cast<int64_t>(m.meta_count * 4)), "pinned host allocation");
  m.h_meta = static_cast<int32_t*>(pinned);
  check(lx_rt_event_create(&m.meta_copied), "event create");
  check(lx_rt_stream_create(&m.route_side), "stream create");
  check(lx_rt_event_create(&m.route_fork), "event create");
  check(lx_rt_event_create(&m.route_join), "event create");
  if (m.offload && std::getenv("LX_OFFLOAD_PREFETCH")) m.prefetch = std::atoi(std::getenv("LX_OFFLOAD_PREFETCH"));
  if (m.route_trace || m.prefetch > 0) {
    m.pred_mixed = m.alloc<float>(TS * D);
    m.pred_h = m.alloc<uint16_t>(T * D);
    m.pred_router_in = m.alloc<float>(T * (Lat + R));
    m.pred_logits = m.alloc<float>(T * E);
    m.pred_ids = m.alloc<int32_t>(T * Impl::kPredicted);
    m.pred_w = m.alloc<float>(T * Impl::kPredicted);
    m.pred_entropy = m.alloc<float>(T);
    m.pred_confidence = m.alloc<float>(T);
  }
  if (m.prefetch > 0) {
    check(lx_rt_stream_create(&m.side), "stream create");
    for (int k = 0; k < m.P_passes * m.L_layers; ++k) {
      void *a = nullptr, *b = nullptr;
      check(lx_rt_event_create(&a), "event create");
      check(lx_rt_event_create(&b), "event create");
      m.fork_events.push_back(a);
      m.join_events.push_back(b);
    }
  }
  if (m.stage_bytes) {
    for (int b = 0; b < 2; ++b) {
      m.stage[b] = m.alloc<uint16_t>(m.stage_bytes / 2);
      check(lx_rt_event_create(&m.stage_copied[b]), "event create");
      check(lx_rt_event_create(&m.stage_free[b]), "event create");
    }
    check(lx_rt_stream_create(&m.stage_stream), "stream create");
  }
  if (m.stage_bytes) m.use_program = false;  // projections in host memory (LX_OFFLOAD_DENSE_HOST)
  m.use_graphs = std::getenv("LX_NO_GRAPHS") == nullptr && !m.route_trace;
  free_on_throw.armed = false;
}

MetisModel::~MetisModel() { delete impl_; }

void MetisModel::reset_slot(int slot) {
  slot_len_.at(slot) = 0;
  folded_.at(slot) = 0;
  pending_.at(slot) = 0;
}

int MetisModel::max_verify_tokens() const {
  return std::min({LX_SPEC_MAX_PENDING, Impl::kGemvRows, 63 / impl_->KVH});
}

void MetisModel::set_feature_taps(const std::vector<FeatureTap>& taps) {
  Impl& m = *impl_;
  auto same = [](const FeatureTap& a, const FeatureTap& b) { return a.pass == b.pass && a.layer == b.layer; };
  if (taps.size() == m.taps.size() && std::equal(taps.begin(), taps.end(), m.taps.begin(), same)) return;
  std::fill(m.tap_col.begin(), m.tap_col.end(), -1);
  int col = 0;
  for (const FeatureTap& t : taps) {
    if (t.pass < 0 || t.pass >= m.P_passes || t.layer < -1 || t.layer >= m.L_layers)
      throw std::runtime_error("feature tap out of range");
    int& slot = m.tap_col[static_cast<size_t>(t.pass) * (m.L_layers + 1) + t.layer + 1];
    if (slot >= 0) throw std::runtime_error("feature tap given twice");
    slot = col;
    col += m.D;
  }
  m.taps = taps;
  if (col != m.tap_dim) {
    if (m.tap_out) {
      lx_rt_free(m.tap_out);
      lx_rt_free(m.tap_mean);
      m.allocations.erase(std::remove(m.allocations.begin(), m.allocations.end(), static_cast<void*>(m.tap_out)),
                          m.allocations.end());
      m.allocations.erase(std::remove(m.allocations.begin(), m.allocations.end(), static_cast<void*>(m.tap_mean)),
                          m.allocations.end());
      m.tap_out = m.tap_mean = nullptr;
    }
    m.tap_dim = col;
    if (col > 0) {
      m.tap_out = m.alloc<float>(static_cast<size_t>(m.max_tokens) * col);
      m.tap_mean = m.alloc<float>(static_cast<size_t>(m.max_tokens) * m.D);
    }
  }
  // captured steps have the old taps baked in
  for (auto& [key, exec] : m.graphs) lx_rt_graph_destroy(exec);
  m.graphs.clear();
  m.seen.clear();
}

int MetisModel::feature_dim() const { return impl_->tap_dim; }
const float* MetisModel::features() const { return impl_->tap_out; }

MetisModel::HeadWeights MetisModel::head_weights() const {
  return {impl_->head.w, impl_->head.q, impl_->head.scale};
}

namespace {

// The snapshot's pieces for one slot, in layout order: `piece(device_ptr, bytes, is_kv)`.
template <typename Fn>
void each_piece(const Impl& m, int slot, int length, Fn&& piece) {
  const size_t kv_row = static_cast<size_t>(m.KVH) * m.Dh;  // elements per position
  for (int pass = 0; pass < m.P_passes; ++pass) {
    for (int i = 0; i < m.L_layers; ++i) {
      const int idx = pass * m.L_layers + i;
      if (m.layers[i].attention) {
        const size_t base = static_cast<size_t>(slot) * m.kv_capacity * kv_row;  // elements
        const size_t bytes = sizeof(uint16_t) * static_cast<size_t>(length) * kv_row;
        piece(static_cast<char*>(m.k_cache[idx]) + sizeof(uint16_t) * base, bytes, true);
        piece(static_cast<char*>(m.v_cache[idx]) + sizeof(uint16_t) * base, bytes, true);
      } else {
        const size_t conv = static_cast<size_t>(m.W - 1) * m.conv_dim;
        const size_t ssm = static_cast<size_t>(m.H) * m.Pd * m.N;
        piece(m.conv_state[idx] + static_cast<size_t>(slot) * conv, sizeof(float) * conv, false);
        piece(m.ssm_state[idx] + static_cast<size_t>(slot) * ssm, sizeof(float) * ssm, false);
      }
    }
  }
}

}  // namespace

size_t MetisModel::slot_state_bytes(int length) const {
  const Impl& m = *impl_;
  size_t total = 0;
  each_piece(m, 0, length, [&total](const void*, size_t bytes, bool) { total += bytes; });
  return total;
}

void MetisModel::save_slot(int slot, void* dst, int length, void* stream) {
  Impl& m = *impl_;
  if (slot < 0 || slot >= m.max_slots || length < 0 || length > m.kv_capacity)
    throw std::runtime_error("save_slot: slot or length out of range");
  // A speculative slot's Mamba state stops at its folded positions: fold the pending ones first.
  if (pending_.at(slot) != 0) throw std::runtime_error("save_slot: the slot has pending positions (fold_pending)");
  char* out = static_cast<char*>(dst);
  size_t off = 0;
  each_piece(m, slot, length, [&](const void* src, size_t bytes, bool) {
    check(lx_rt_copy(out + off, src, static_cast<int64_t>(bytes), LX_COPY_DEFAULT, stream), "slot state copy");
    off += bytes;
  });
}

void MetisModel::restore_slot(int slot, const void* src, int length, void* stream) {
  Impl& m = *impl_;
  if (slot < 0 || slot >= m.max_slots || length < 0 || length > m.kv_capacity)
    throw std::runtime_error("restore_slot: slot or length out of range");
  const char* in = static_cast<const char*>(src);
  size_t off = 0;
  each_piece(m, slot, length, [&](const void* dst, size_t bytes, bool) {
    check(lx_rt_copy(const_cast<void*>(dst), in + off, static_cast<int64_t>(bytes), LX_COPY_DEFAULT, stream),
          "slot state copy");
    off += bytes;
  });
  slot_len_.at(slot) = length;
  folded_.at(slot) = length;
  pending_.at(slot) = 0;
}

namespace {

// y = alpha * (x @ W^T + bias) + beta * y for M rows of x (f32 or bf16). Decode-sized products run
// as GEMVs on the int8 copy when there is one, reading x as it is; larger ones go to the dense GEMM on
// the bf16 weight, through a bf16 copy of x when x is f32. `rows_per_token` > 1 marks a product
// with that many rows per token (the depth memory's per-stream rows): in a decode-sized step those
// stay GEMVs, 8 rows at a time, so that a token's rows get the same bits whatever the step's size.
void linear(Impl& m, const Linear& l, const void* x, lx_dtype x_dtype, int ld_x, void* y, lx_dtype y_dtype, int ld_y,
            int M, void* s, float alpha = 1.0f, float beta = 0.0f, int rows_per_token = 1) {
  const bool decode_step = M / rows_per_token <= Impl::kGemvRows;
  if (M <= Impl::kGemvRows || decode_step) {
    const size_t xs = x_dtype == LX_F32 ? 4 : 2, ys = y_dtype == LX_F32 ? 4 : 2;
    for (int r0 = 0; r0 < M; r0 += Impl::kGemvRows) {
      const int rows = std::min(Impl::kGemvRows, M - r0);
      const void* xr = static_cast<const char*>(x) + xs * r0 * ld_x;
      void* yr = static_cast<char*>(y) + ys * r0 * ld_y;
      if (l.x16)
        LX_CHECK(lx_gemv_bf16x(xr, x_dtype, ld_x, l.x16, l.bias, LX_BF16, yr, y_dtype, ld_y, rows, l.N, l.K, alpha, beta, s));
      else
        LX_CHECK(lx_gemv(xr, x_dtype, ld_x, l.q ? static_cast<const void*>(l.q) : l.w, l.q ? LX_W_INT8 : LX_W_BF16,
                         l.scale, l.bias, LX_BF16, yr, y_dtype, ld_y, rows, l.N, l.K, alpha, beta, s));
    }
    return;
  }
  if (l.pw.type != LX_W_BF16 || m.prefill_math != LX_MATH_BF16) {
    // a weight kept bf16 multiplies under BF16 math whatever the mode
    const lx_math math = l.pw.type == LX_W_BF16 ? LX_MATH_BF16 : m.prefill_math;
    LX_CHECK(lx_gemm_w(x, x_dtype, ld_x, &l.pw, math, l.bias, LX_BF16, y, y_dtype, ld_y, M, l.N, l.K, alpha, beta,
                       m.gemm_ws, s));
    return;
  }
  if (x_dtype == LX_F32) {
    if (static_cast<size_t>(M) * l.K > m.cast_capacity) throw std::runtime_error("cast scratch too small");
    LX_CHECK(lx_copy2d(x, LX_F32, ld_x, m.cast_bf, LX_BF16, l.K, M, l.K, s));
    x = m.cast_bf;
    ld_x = l.K;
  }
  if (l.host) {  // copy the weight in on the copy stream once the product that last used the buffer is done
    const int b = m.stage_next;
    m.stage_next ^= 1;
    check(lx_rt_stream_wait(m.stage_stream, m.stage_free[b]), "stream wait");
    check(lx_rt_copy(m.stage[b], l.w, static_cast<int64_t>(static_cast<size_t>(l.N) * l.K * 2), LX_COPY_H2D,
                     m.stage_stream),
          "weight staging copy");
    check(lx_rt_event_record(m.stage_copied[b], m.stage_stream), "event record");
    check(lx_rt_stream_wait(s, m.stage_copied[b]), "stream wait");
    LX_CHECK(lx_gemm_bf16(x, ld_x, m.stage[b], l.bias, y, y_dtype, ld_y, M, l.N, l.K, alpha, beta, s));
    check(lx_rt_event_record(m.stage_free[b], s), "event record");
    return;
  }
  LX_CHECK(lx_gemm_bf16(x, ld_x, l.w, l.bias, y, y_dtype, ld_y, M, l.N, l.K, alpha, beta, s));
}

// The f32 products (routers, the depth head): batch invariant in decode-sized steps (kernels.h,
// lx_spec_gemv_f32), the dense GEMM otherwise.
void gemm_f32(const float* x, int ld_x, const float* w, const float* bias, float* y, int ld_y, int M, int N, int K,
              void* s) {
  if (M <= Impl::kGemvRows)
    LX_CHECK(lx_spec_gemv_f32(x, ld_x, w, bias, y, ld_y, M, N, K, s));
  else
    LX_CHECK(lx_gemm_f32(x, ld_x, w, bias, y, ld_y, M, N, K, s));
}

void to_bf16(const float* src, int ld_src, void* dst, int ld_dst, int rows, int cols, void* s) {
  LX_CHECK(lx_copy2d(src, LX_F32, ld_src, dst, LX_BF16, ld_dst, rows, cols, s));
}

void copy_f32(const float* src, int ld_src, float* dst, int ld_dst, int rows, int cols, void* s) {
  LX_CHECK(lx_copy2d(src, LX_F32, ld_src, dst, LX_F32, ld_dst, rows, cols, s));
}

// Depth-memory read: attend over the token's bank and gate the result into the streams. Leaves
// the mean retrieved vector in mem_summary (zeros while the bank is empty).
void memory_retrieve(Impl& m, int T, int n_entries, void* s) {
  const int S = m.S, D = m.D, Mm = m.M;
  if (n_entries == 0) {
    lx_rt_memset(m.mem_summary, 0, static_cast<int64_t>(sizeof(float) * T * D), s);
    return;
  }
  linear(m, m.m_query, m.streams, LX_F32, D, m.q, LX_F32, Mm, T * S, s, 1.0f, 0.0f, S);
  LX_CHECK(lx_memory_attend(m.q, m.bank_k, m.bank_v, n_entries, m.Emax, m.retrieved_bf, T, S, Mm, s));
  linear(m, m.m_out, m.retrieved_bf, LX_BF16, Mm, m.projected, LX_F32, D, T * S, s, 1.0f, 0.0f, S);
  LX_CHECK(lx_memory_fuse(m.streams, m.projected, m.m_gate, m.m_gate_b, m.mem_summary, T, S, D, s));
}

// [a | b | c | d] into one row per token, as bf16 (for a GEMM) or f32.
void concat4(Impl& m, const float* a, const float* b, const float* c, const float* d, void* out, lx_dtype dtype,
             int T, void* s) {
  const int D = m.D, R = m.R, ld = 3 * D + R;
  const size_t es = dtype == LX_F32 ? 4 : 2;
  char* o = static_cast<char*>(out);
  LX_CHECK(lx_copy2d(a, LX_F32, D, o, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(b, LX_F32, D, o + es * D, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(c, LX_F32, D, o + es * 2 * D, dtype, ld, T, D, s));
  LX_CHECK(lx_copy2d(d, LX_F32, R, o + es * 3 * D, dtype, ld, T, R, s));
}

// Routing features: tanh(route_projection([state, retrieved, difference, history])).
void routing_features(Impl& m, int T, void* s) {
  concat4(m, m.state, m.mem_summary, m.difference, m.history, m.feat_in_bf, LX_BF16, T, s);
  linear(m, m.m_route, m.feat_in_bf, LX_BF16, 3 * m.D + m.R, m.features, LX_F32, m.R, T, s);
  LX_CHECK(lx_activate_f32(m.features, static_cast<int64_t>(T) * m.R, 0, s));
}

// A new memory entry from the streams and the last MoE's routing; stores its key and value.
void memory_write(Impl& m, const MetisConfig& c, int T, int pass, int anchor, int& n_entries, void* s) {
  const int D = m.D, R = m.R, Mm = m.M;
  LX_CHECK(lx_stream_mean(m.streams, m.state, T, m.S, D, s));
  linear(m, m.m_state, m.state, LX_F32, D, m.entry, LX_F32, Mm, T, s);
  LX_CHECK(lx_copy2d(m.summary, LX_F32, R, m.meta_bf, LX_BF16, R + 4, T, R, s));
  const float mean_k = static_cast<float>(c.num_experts_per_tok) / static_cast<float>(c.max_routed_k);
  const float inv_log_e = c.n_routed_experts > 1 ? 1.0f / std::log(static_cast<float>(c.n_routed_experts)) : 1.0f;
  LX_CHECK(glue::meta_scalars(m.entropy, m.confidence, m.continuation, mean_k, inv_log_e, m.meta_bf, R + 4, R, T, s));
  linear(m, m.m_meta, m.meta_bf, LX_BF16, R + 4, m.entry, LX_F32, Mm, T, s, 1.0f, 1.0f);
  LX_CHECK(lx_add_row_bf16(m.entry, static_cast<const uint16_t*>(m.m_pass_emb) + pass * Mm, T, Mm, s));
  LX_CHECK(lx_add_row_bf16(m.entry, static_cast<const uint16_t*>(m.m_anchor_emb) + anchor * Mm, T, Mm, s));
  const int ld_bank = m.Emax * Mm;
  linear(m, m.m_key, m.entry, LX_F32, Mm, m.bank_k + n_entries * Mm, LX_F32, ld_bank, T, s);
  linear(m, m.m_value, m.entry, LX_F32, Mm, m.bank_v + n_entries * Mm, LX_F32, ld_bank, T, s);
  ++n_entries;
}

// A feature tap after block `layer` of `pass`: the stream mean, into its columns of tap_out.
void write_tap(Impl& m, int pass, int layer, int T, void* s) {
  const int col = m.tap_col[static_cast<size_t>(pass) * (m.L_layers + 1) + layer + 1];
  if (col < 0) return;
  LX_CHECK(lx_stream_mean(m.streams, m.tap_mean, T, m.S, m.D, s));
  copy_f32(m.tap_mean, m.D, m.tap_out + col, m.tap_dim, T, m.D, s);
}

// ---- The decode program ------------------------------------------------------------------------

lx_fx fx_plain(const void* x, lx_dtype dtype, int ld) {
  lx_fx f{};
  f.kind = LX_FX_PLAIN;
  f.x = x;
  f.dtype = dtype;
  f.ld = ld;
  return f;
}

lx_fx fx_concat(std::initializer_list<std::pair<const float*, int>> segs, bool round) {
  lx_fx f{};
  f.kind = LX_FX_CONCAT;
  int i = 0;
  for (const auto& [ptr, width] : segs) {
    f.seg[i] = ptr;
    f.width[i] = width;
    ++i;
  }
  f.round_bf16 = round ? 1 : 0;
  return f;
}

lx_fy fy_store(void* out, lx_dtype dtype, int ld) {
  lx_fy f{};
  f.kind = LX_FY_STORE;
  f.out = out;
  f.dtype = dtype;
  f.ld = ld;
  f.alpha = 1.0f;
  return f;
}

// The hyper-connection write of a sublayer's update into the streams, with the mix of the read the
// sublayer started from (connection params: matrix [S, S], read [S], write [S]).
lx_fy fy_write(Impl& m, const float* params) {
  lx_fy f{};
  f.kind = LX_FY_WRITE;
  f.streams = m.streams;
  f.matrix = params;
  f.write = params + m.S * m.S + m.S;
  f.S = m.S;
  return f;
}

// The ops of the program being built, by name (for LX_PROGRAM_TRACE_OUT).
std::vector<std::pair<int, std::string>>* g_op_names = nullptr;

int add(int op, const char* what) {
  if (op < 0) throw std::runtime_error(std::string("decode program: ") + what + " was refused");
  if (g_op_names) g_op_names->emplace_back(op, what);
  return op;
}

// A memory entry for where the streams are now (state[m] their mean), and its bank row.
void program_memory_write(Impl& m, const MetisConfig& c, lx_program* p, const float* state, int pass, int anchor,
                          int& n_entries) {
  const int D = m.D, R = m.R, Mm = m.M, stride = 2 * D + 4;
  lx_entry_args e{};
  e.state = state;
  e.w_state = wview(m.m_state);
  e.b_state = m.m_state.bias;
  e.summary = m.summary;
  e.entropy = m.entropy;
  e.confidence = m.confidence;
  e.continuation = m.continuation;
  e.mean_k = static_cast<float>(c.num_experts_per_tok) / static_cast<float>(c.max_routed_k);
  e.inv_log_e = c.n_routed_experts > 1 ? 1.0f / std::log(static_cast<float>(c.n_routed_experts)) : 1.0f;
  e.w_meta = m.m_meta.w;
  e.b_meta = m.m_meta.bias;
  e.pass_emb = static_cast<const uint16_t*>(m.m_pass_emb) + pass * Mm;
  e.anchor_emb = static_cast<const uint16_t*>(m.m_anchor_emb) + anchor * Mm;
  e.entry = m.entry;
  e.D = D;
  e.R = R;
  e.mem_dim = Mm;
  add(lx_program_memory_entry(p, &e), "memory entry");
  const lx_fx x = fx_plain(m.entry, LX_F32, Mm);
  lx_fy y = fy_store(m.pbank + static_cast<size_t>(n_entries) * stride, LX_F32, m.Emax * stride);
  y.bias = m.bank_b;
  y.bias_dtype = LX_F32;
  add(lx_program_gemv(p, &x, &m.bank_w, &y, stride, Mm), "bank projection");
  ++n_entries;
}

// The whole decode step for T tokens (one per sequence) as one program, in the order run() takes;
// see run() for the step itself. What differs: the depth-memory read works on composite bank rows
// (no projections per read), the hyper-connection mixes happen in the writes, the routing inside a
// pass drops the difference (zero there), and nothing after the last pass's final read is built
// but the head.
lx_program* build_program(Impl& m, const MetisConfig& c, int T, float* logits) {
  const int S = m.S, D = m.D, R = m.R, Lat = m.Lat, E = m.E, Mm = m.M, params = S * S + 2 * S;
  const int stride = 2 * D + 4;
  lx_program* p = lx_program_create(T, m.d_tokens, m.d_slots, m.d_has_state, m.d_start_pos);
  if (!p) throw std::runtime_error("lx_program_create failed");
  try {
    add(lx_program_embed(p, m.embedding, m.stream_emb, m.streams, m.previous, m.history, m.continuation, S, D, R),
        "embed");
    int n_entries = 0;
    for (int pass = 0; pass < m.P_passes; ++pass) {
      int anchor = 0;
      for (int i = 0; i < m.L_layers; ++i) {
        Impl::Layer& layer = m.layers[i];
        const int slot_index = pass * m.L_layers + i;
        const float* mc = layer.mixer_conn.params + pass * params;
        const float* ec = layer.moe_conn.params + pass * params;

        // depth-memory read, the stream mean, the mixer's read (and an attention layer's LoRA down)
        lx_memread_args r{};
        r.streams = m.streams;
        r.bank = m.pbank;
        r.n_entries = n_entries;
        r.entry_stride = stride;
        r.token_stride = m.Emax * stride;
        r.mem_dim = Mm;
        r.gate = m.m_gate;
        r.gate_bias = m.m_gate_b;
        r.state = m.state;
        r.summary = m.mem_summary;
        r.read = mc + S * S;
        r.norm_w = layer.mixer_conn.norm;
        r.eps = c.rms_norm_eps;
        r.h = m.h_bf;
        if (layer.attention) {
          r.lora_down = layer.lora_down.w;
          r.rank = m.rank;
          r.lora = m.lora_f;
        }
        r.S = S;
        r.D = D;
        add(lx_program_memory_read(p, &r), "memory read");

        // routing features: tanh(route_projection([state, retrieved, history]))
        {
          const lx_fx x = fx_concat({{m.state, D}, {m.mem_summary, D}, {m.history, R}}, true);
          const lx_weight w = wview(m.route_in);
          lx_fy y = fy_store(m.features, LX_F32, R);
          y.bias = m.m_route.bias;
          y.bias_dtype = LX_BF16;
          y.act = 1;
          add(lx_program_gemv(p, &x, &w, &y, R, 2 * D + R), "routing features");
        }

        // mixer
        int write_op;
        lx_fy wr = fy_write(m, mc);
        wr.next_read = ec + S * S;
        wr.next_src = m.next_src;
        wr.next_parts = m.next_parts;
        if (layer.attention) {
          const lx_fx x = fx_plain(m.h_bf, LX_BF16, D);
          const lx_weight w = wview(layer.qkv);
          lx_fy y = fy_store(m.qkv_buf, LX_F32, m.qkv_width);
          y.lora_up = layer.lora_up.w;
          y.lora = m.lora_f;
          y.rank = m.rank;
          y.lora_gate = layer.pass_gate[pass];
          add(lx_program_gemv(p, &x, &w, &y, m.qkv_width, D), "qkv");
          lx_attention_args a{};
          a.qkv = m.qkv_buf;
          a.ld_qkv = m.qkv_width;
          a.k_cache = m.k_cache[slot_index];
          a.v_cache = m.v_cache[slot_index];
          a.capacity = m.kv_capacity;
          a.out = m.att_bf;
          a.H = m.AH;
          a.KVH = m.KVH;
          a.Dh = m.Dh;
          a.scale = 1.0f / std::sqrt(static_cast<float>(m.Dh));
          add(lx_program_attention(p, &a), "attention");
          const lx_fx xo = fx_plain(m.att_bf, LX_BF16, m.AH * m.Dh);
          const lx_weight wo = wview(layer.out);
          write_op = add(lx_program_gemv(p, &xo, &wo, &wr, D, m.AH * m.Dh), "attention out");
        } else {
          const lx_fx x = fx_plain(m.h_bf, LX_BF16, D);
          const lx_weight w = wview(layer.in_proj);
          const lx_fy y = fy_store(m.zxbcdt, LX_F32, m.proj);
          add(lx_program_gemv(p, &x, &w, &y, m.proj, D), "in_proj");
          lx_ssm_args a{};
          a.zxbcdt = m.zxbcdt;
          a.ld = m.proj;
          a.conv_w = layer.conv_w;
          a.conv_b = layer.conv_b;
          a.conv_state = m.conv_state[slot_index];
          a.ssm_state = m.ssm_state[slot_index];
          a.A_log = layer.A_log;
          a.D = layer.D;
          a.dt_bias = layer.dt_bias;
          a.v = m.ssm_y;
          a.head_ss = m.head_ss;
          a.I = m.I;
          a.H = m.H;
          a.P = m.Pd;
          a.G = m.G;
          a.N = m.N;
          a.W = m.W;
          add(lx_program_ssm(p, &a), "Mamba step");
          lx_fx xo{};
          xo.kind = LX_FX_GATED;
          xo.x = m.ssm_y;
          xo.dtype = LX_F32;
          xo.ld = m.I;
          xo.parts = m.head_ss;
          xo.n_parts = m.H;
          xo.norm_w = layer.norm_w;
          xo.eps = c.mamba_norm_eps;
          xo.group = m.I / m.G;
          const lx_weight wo = wview(layer.out_proj);
          write_op = add(lx_program_gemv(p, &xo, &wo, &wr, D, m.I), "out_proj");
        }
        const int n_parts = lx_program_tasks(p, write_op);
        if (n_parts > m.next_parts_cap) throw std::runtime_error("decode program: next_parts too small");

        // latent mixture of experts, read from the mixer's write
        {
          lx_fx x{};
          x.kind = LX_FX_RMS;
          x.x = m.next_src;
          x.dtype = LX_F32;
          x.ld = D;
          x.parts = m.next_parts;
          x.n_parts = n_parts;
          x.norm_w = layer.moe_conn.norm;
          x.eps = c.rms_norm_eps;
          const lx_weight w = wview(layer.latent_down);
          lx_fy y = fy_store(m.latent, LX_F32, Lat);
          y.out_bf = m.latent_bf;
          y.ld_bf = Lat;
          add(lx_program_gemv(p, &x, &w, &y, Lat, D), "latent_down");
        }
        {
          const lx_fx x = fx_concat({{m.latent, Lat}, {m.features, R}}, false);
          const lx_weight w = {LX_W_F32, layer.router_w, nullptr, 1.0f};
          lx_fy y = fy_store(m.router_logits, LX_F32, E);
          y.bias = layer.router_b;
          y.bias_dtype = LX_F32;
          add(lx_program_gemv(p, &x, &w, &y, E, Lat + R), "router");
        }
        {
          lx_select_args a{};
          a.logits = m.router_logits;
          a.selection_bias = layer.selection_bias;
          a.ids = m.route_ids;
          a.weights = m.route_w;
          a.entropy = m.entropy;
          a.confidence = m.confidence;
          a.expert_emb = layer.expert_emb;
          a.summary = m.summary;
          a.history = m.history;
          a.E = E;
          a.top_k = m.K;
          a.R = R;
          add(lx_program_select(p, &a), "routing");
        }
        {
          lx_experts_args a{};
          a.latent = m.latent_bf;
          a.ids = m.route_ids;
          a.weights = m.route_w;
          if (layer.gate_up_q) {
            a.gate_up = {LX_W_INT8, layer.gate_up_q, layer.gate_up_scale, 1.0f};
            a.down = {LX_W_INT8, layer.down_q, layer.down_scale, 1.0f};
          } else {
            a.gate_up = {LX_W_BF16, layer.gate_up, nullptr, 1.0f};
            a.down = {LX_W_BF16, layer.down, nullptr, 1.0f};
          }
          a.shared_gate_up = wview(layer.shared_gate_up);
          a.shared_down = wview(layer.shared_down);
          a.out = m.routed;
          a.L = Lat;
          a.I = m.Iexp;
          a.Is = m.Ishared;
          a.top_k = m.K;
          add(lx_program_experts(p, &a), "experts");
        }
        {
          const lx_fx x = fx_plain(m.routed, LX_F32, Lat);
          const lx_weight w = wview(layer.latent_up);
          lx_fy y = fy_write(m, ec);
          if (layer.attention) y.state = m.wstate;
          add(lx_program_gemv(p, &x, &w, &y, D, Lat), "latent_up");
        }
        if (layer.attention) program_memory_write(m, c, p, m.wstate, pass, anchor++, n_entries);
      }

      // the end of a pass: one more read; then (but after the last pass) the depth head and an entry
      const bool last = pass == m.P_passes - 1;
      lx_memread_args r{};
      r.streams = m.streams;
      r.bank = m.pbank;
      r.n_entries = n_entries;
      r.entry_stride = stride;
      r.token_stride = m.Emax * stride;
      r.mem_dim = Mm;
      r.gate = m.m_gate;
      r.gate_bias = m.m_gate_b;
      r.state = m.state;
      r.summary = m.mem_summary;
      r.rms = m.rms;
      if (!last) {
        r.previous = m.previous;
        r.difference = m.difference;
      }
      r.S = S;
      r.D = D;
      add(lx_program_memory_read(p, &r), "memory read");
      if (last) break;
      {
        const lx_fx x = fx_concat({{m.state, D}, {m.mem_summary, D}, {m.difference, D}, {m.history, R}}, true);
        const lx_weight w = wview(m.m_route);
        lx_fy y = fy_store(m.features, LX_F32, R);
        y.bias = m.m_route.bias;
        y.bias_dtype = LX_BF16;
        y.act = 1;
        add(lx_program_gemv(p, &x, &w, &y, R, 3 * D + R), "routing features");
      }
      {
        const lx_fx x = fx_concat({{m.state, D}, {m.mem_summary, D}, {m.difference, D}, {m.features, R}}, false);
        const lx_weight w = {LX_W_F32, m.c_hidden_w, nullptr, 1.0f};
        lx_fy y = fy_store(m.cont_hidden, LX_F32, R);
        y.bias = m.c_hidden_b;
        y.bias_dtype = LX_F32;
        y.act = 2;  // silu
        add(lx_program_gemv(p, &x, &w, &y, R, 3 * D + R), "depth head");
      }
      add(lx_program_continuation(p, m.cont_hidden, m.c_out_w, m.c_out_b, m.rms, c.continuation_logit_limit,
                                  m.continuation, R, D),
          "continuation");
      program_memory_write(m, c, p, m.state, pass, anchor, n_entries);
    }

    // the final norm of the stream mean (its sum of squares from the last read) and the head
    lx_fx x{};
    x.kind = LX_FX_RMS;
    x.x = m.state;
    x.dtype = LX_F32;
    x.ld = D;
    x.parts = m.rms;
    x.n_parts = 1;
    x.norm_w = m.final_norm;
    x.eps = c.rms_norm_eps;
    const lx_weight w = wview(m.head);
    const lx_fy y = fy_store(logits, LX_F32, c.vocab_size);
    add(lx_program_gemv(p, &x, &w, &y, c.vocab_size, D), "head");
    const int e = lx_program_build(p);
    if (e) throw std::runtime_error("lx_program_build failed with CUDA error " + std::to_string(e));
  } catch (...) {
    lx_program_destroy(p);
    throw;
  }
  return p;
}



// Whether the decode program runs this step: a decode step (one token per sequence, at most 8) of a
// model whose program was built (LX_PROGRAM, and none of what it does not cover yet: speculative
// models, so a verify step and the steps around it share one path, offloaded experts, projections
// in host memory, BF16X copies), with no route trace or feature taps to write.
bool decode_program_covers(const Impl& m, int T, int n_seq, bool spec) {
  if (!m.use_program || spec || T != n_seq || T > 8 || m.route_trace) return false;
  for (int col : m.tap_col)
    if (col >= 0) return false;
  return true;
}

// Runs a decode step as the shape's program, building it on the shape's first step. One token per
// sequence: every row is a last row, so all_rows changes nothing.
void run_program(Impl& m, const MetisConfig& c, int T, float* logits, void* s) {
  auto it = m.programs.find({T, logits});
  if (it == m.programs.end()) {
    std::vector<std::pair<int, std::string>> names;
    g_op_names = &names;
    try {
      it = m.programs.emplace(std::make_pair(T, logits), build_program(m, c, T, logits)).first;
    } catch (...) {
      g_op_names = nullptr;
      throw;
    }
    g_op_names = nullptr;
    m.program_names[it->second] = std::move(names);
  }
  LX_CHECK(lx_program_run(it->second, s));
  // LX_PROGRAM_TRACE_OUT=DIR (with LX_PROGRAM_TRACE): one step's task timeline and op names, the
  // 32nd step's or LX_PROGRAM_TRACE_STEP's
  const char* dir = std::getenv("LX_PROGRAM_TRACE_OUT");
  const char* at = std::getenv("LX_PROGRAM_TRACE_STEP");
  if (dir && ++m.traced_steps == (at ? std::atoi(at) : 32)) {
    check(lx_rt_stream_sync(s), "stream sync");
    std::vector<int64_t> rec(static_cast<size_t>(1) << 24);
    const int n = lx_program_trace(it->second, rec.data(), static_cast<int>(rec.size() / 4));
    std::ofstream(std::string(dir) + "/trace.bin", std::ios::binary)
        .write(reinterpret_cast<const char*>(rec.data()), static_cast<std::streamsize>(n) * 32);
    std::ofstream names(std::string(dir) + "/ops.txt");
    for (const auto& [op, what] : m.program_names[it->second]) names << op << " " << what << "\n";
  }
}

}  // namespace

void MetisModel::forward(const StepBatch& b, float* logits, bool all_rows, void* s) {
  Impl& m = *impl_;
  const int T = static_cast<int>(b.tokens.size());
  const int n_seq = static_cast<int>(b.slots.size());
  if (T == 0) return;
  if (T > m.max_tokens) throw std::runtime_error("step has more tokens than max_tokens");
  if (n_seq > m.max_slots || static_cast<int>(b.cu_seqlens.size()) != n_seq + 1)
    throw std::runtime_error("bad step batch");
  fold_pending(b, s);
  stage(b, false, s);
  launch(T, n_seq, all_rows, logits, false, s);
  for (int i = 0; i < n_seq; ++i) {
    slot_len_[b.slots[i]] += b.cu_seqlens[i + 1] - b.cu_seqlens[i];
    folded_[b.slots[i]] = slot_len_[b.slots[i]];
  }
}

void MetisModel::forward_spec(const StepBatch& b, float* logits, void* s) {
  Impl& m = *impl_;
  if (!m.speculative) throw std::runtime_error("forward_spec needs a model built with speculative = true");
  const int T = static_cast<int>(b.tokens.size());
  const int n_seq = static_cast<int>(b.slots.size());
  if (T == 0) return;
  if (T > m.max_tokens) throw std::runtime_error("step has more tokens than max_tokens");
  if (n_seq > m.max_slots || static_cast<int>(b.cu_seqlens.size()) != n_seq + 1)
    throw std::runtime_error("bad step batch");
  for (int i = 0; i < n_seq; ++i)
    if (b.cu_seqlens[i + 1] - b.cu_seqlens[i] > LX_SPEC_MAX_PENDING)
      throw std::runtime_error("a verify step takes at most LX_SPEC_MAX_PENDING tokens per sequence");
  stage(b, true, s);
  launch(T, n_seq, true, logits, true, s);
  for (int i = 0; i < n_seq; ++i) {
    const int slot = b.slots[i], n = b.cu_seqlens[i + 1] - b.cu_seqlens[i];
    // as stage() decided: fold only what makes room for the new tokens
    const int fold = std::max(0, pending_[slot] + n - LX_SPEC_MAX_PENDING);
    folded_[slot] += fold;
    pending_[slot] += n - fold;
    slot_len_[slot] += n;
  }
}

void MetisModel::truncate(int slot, int length) {
  if (length < folded_.at(slot) || length > slot_len_[slot])
    throw std::runtime_error("truncate past the slot's pending positions");
  pending_[slot] -= slot_len_[slot] - length;
  slot_len_[slot] = length;
}

// Stages a step's metadata for the device (waiting until the previous step's copy has left the
// staging buffer). A verify step folds a slot's oldest pending positions only when the new tokens
// would not fit after them, so each position is folded once and the base state is written on few
// steps; its has_state says whether the base state holds anything.
void MetisModel::stage(const StepBatch& b, bool spec, void* s) {
  Impl& m = *impl_;
  const int T = static_cast<int>(b.tokens.size());
  const int n_seq = static_cast<int>(b.slots.size());
  lx_rt_event_sync(m.meta_copied);
  int32_t* tokens = m.h_meta;
  int32_t* cu = tokens + (m.d_cu - m.d_tokens);
  int32_t* slots = tokens + (m.d_slots - m.d_tokens);
  int32_t* has_state = tokens + (m.d_has_state - m.d_tokens);
  int32_t* start_pos = tokens + (m.d_start_pos - m.d_tokens);
  int32_t* fold = tokens + (m.d_fold - m.d_tokens);
  int32_t* keep = tokens + (m.d_keep - m.d_tokens);
  std::memcpy(tokens, b.tokens.data(), 4 * T);
  std::memcpy(cu, b.cu_seqlens.data(), 4 * (n_seq + 1));
  std::memcpy(slots, b.slots.data(), 4 * n_seq);
  for (int i = 0; i < n_seq; ++i) {
    const int slot = b.slots[i], n = b.cu_seqlens[i + 1] - b.cu_seqlens[i];
    start_pos[i] = slot_len_[slot];
    if (start_pos[i] + n > m.kv_capacity) throw std::runtime_error("sequence longer than the KV capacity");
    if (spec) {
      has_state[i] = folded_[slot] > 0;
      fold[i] = std::max(0, pending_[slot] + n - LX_SPEC_MAX_PENDING);
      keep[i] = pending_[slot] - fold[i];
    } else {
      has_state[i] = slot_len_[slot] > 0;
      fold[i] = keep[i] = 0;
    }
  }
  lx_rt_copy(m.d_meta, m.h_meta, static_cast<int64_t>(m.meta_count * 4), LX_COPY_H2D, s);
  lx_rt_event_record(m.meta_copied, s);
}

// Runs the staged step: as the decode program when it covers the step, else replaying a captured
// graph for decode and verify shapes.
void MetisModel::launch(int T, int n_seq, bool all_rows, float* logits, bool spec, void* s) {
  Impl& m = *impl_;
  if (decode_program_covers(m, T, n_seq, spec)) {
    run_program(m, cfg_, T, logits, s);
    return;
  }
  // Offloaded steps larger than a decode stage their experts from the host, which is not capturable.
  const bool capturable = spec ? T <= Impl::kGemvRows : T == n_seq && !(m.offload && T > Impl::kGemvRows);
  const Impl::GraphKey key{T, n_seq, all_rows, logits, spec};
  if (m.use_graphs && capturable) {
    auto it = m.graphs.find(key);
    if (it == m.graphs.end() && m.seen.count(key)) {
      // A backend without graphs declines the capture (LX_ERR_NOT_SUPPORTED), and every step
      // runs eagerly from then on.
      const int capture = lx_rt_capture_begin(s);
      if (capture == LX_ERR_NOT_SUPPORTED) {
        m.use_graphs = false;
      } else {
        check(capture, "graph capture start");
        run(T, n_seq, all_rows, logits, spec, s);
        void* exec = nullptr;
        check(lx_rt_capture_end(s, &exec), "graph capture end");
        it = m.graphs.emplace(key, exec).first;
      }
    }
    if (it != m.graphs.end()) {
      check(lx_rt_graph_launch(it->second, s), "graph launch");
    } else {
      run(T, n_seq, all_rows, logits, spec, s);
      m.seen.insert(key);
    }
  } else {
    run(T, n_seq, all_rows, logits, spec, s);
  }
}

// Folds every pending position of the step's slots into their base state, so that a normal step
// (which has no history) can continue them: the Mamba layers' spec kernels with no new tokens.
void MetisModel::fold_pending(const StepBatch& b, void* s) {
  Impl& m = *impl_;
  std::vector<int32_t> slots, has, fold;
  for (int slot : b.slots)
    if (pending_[slot] > 0) {
      slots.push_back(slot);
      has.push_back(folded_[slot] > 0);
      fold.push_back(pending_[slot]);
    }
  if (slots.empty()) return;
  const int n = static_cast<int>(slots.size());
  // [slots | has | fold | keep | cu], uploaded after the previous step's metadata copy is done
  lx_rt_event_sync(m.meta_copied);
  int32_t* h = m.h_meta;
  int32_t* d = m.d_meta;
  std::memcpy(h, slots.data(), 4 * n);
  std::memcpy(h + n, has.data(), 4 * n);
  std::memcpy(h + 2 * n, fold.data(), 4 * n);
  std::memset(h + 3 * n, 0, 4 * (2 * n + 1));
  lx_rt_copy(d, h, static_cast<int64_t>(4 * (5 * n + 1)), LX_COPY_H2D, s);
  lx_rt_event_record(m.meta_copied, s);
  for (size_t li = 0; li < m.conv_hist.size(); ++li) {
    if (!m.conv_hist[li]) continue;
    const Impl::Layer& layer = m.layers[li % m.L_layers];
    LX_CHECK(lx_spec_conv1d(m.xbc, m.proj, layer.conv_w, layer.conv_b, m.conv_state[li], m.conv_hist[li], d, d + n,
                            d + 2 * n, d + 3 * n, d + 4 * n, n, m.xbc, m.conv_dim, m.W, s));
    LX_CHECK(lx_spec_ssm_scan(m.xbc, m.conv_dim, m.xbc, m.proj, m.xbc, m.xbc, m.conv_dim, layer.A_log, layer.D,
                              layer.dt_bias, m.ssm_state[li], m.ssm_hist[li], d, d + n, d + 2 * n, d + 3 * n, d + 4 * n,
                              n, m.ssm_y, m.H, m.Pd, m.G, m.N, s));
  }
  for (int slot : slots) {
    folded_[slot] = slot_len_[slot];
    pending_[slot] = 0;
  }
}

// The step's device work, from the token ids in d_meta to the logits. Everything it decides on the
// host depends only on (T, n_seq, all_rows), which is what makes it capturable.
void MetisModel::run(int T, int n_seq, bool all_rows, float* logits, bool spec, void* s) {
  Impl& m = *impl_;
  const MetisConfig& c = cfg_;
  const int S = m.S, D = m.D, R = m.R, Lat = m.Lat, E = m.E;

  LX_CHECK(lx_embed_streams(m.embedding, m.d_tokens, m.stream_emb, m.streams, T, S, D, s));
  LX_CHECK(lx_stream_mean(m.streams, m.previous, T, S, D, s));
  lx_rt_memset(m.history, 0, static_cast<int64_t>(sizeof(float) * T * R), s);
  lx_rt_memset(m.continuation, 0, static_cast<int64_t>(sizeof(float) * T), s);
  int n_entries = 0;

  for (int pass = 0; pass < m.P_passes; ++pass) {
    LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
    LX_CHECK(glue::sub(m.state, m.previous, m.difference, static_cast<int64_t>(T) * D, s));
    int anchor = 0;
    for (int i = 0; i < m.L_layers; ++i) {
      Impl::Layer& layer = m.layers[i];
      const int slot_index = pass * m.L_layers + i;
      const int params = S * S + 2 * S;

      // depth-memory read, then the routing features beside the mixer. They read the streams as the read
      // left them and write nothing the mixer touches, so on a decode-sized step they run on route_side and
      // join before the MoE reads them. A larger step keeps its order: its GEMMs share one workspace.
      memory_retrieve(m, T, n_entries, s);
      const bool beside = T <= Impl::kGemvRows;
      void* rs = beside ? m.route_side : s;
      if (beside) {
        check(lx_rt_event_record(m.route_fork, s), "event record");
        check(lx_rt_stream_wait(rs, m.route_fork), "stream wait");
      }
      LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, rs));
      routing_features(m, T, rs);
      if (beside) check(lx_rt_event_record(m.route_join, rs), "event record");

      // mixer
      const float* mc = layer.mixer_conn.params + pass * params;
      LX_CHECK(lx_mhc_read(m.streams, mc, mc + S * S, layer.mixer_conn.norm, c.rms_norm_eps, m.mixed, m.h_bf, T, S, D, s));
      if (layer.attention) {
        linear(m, layer.qkv, m.h_bf, LX_BF16, D, m.qkv_buf, LX_F32, m.qkv_width, T, s);
        linear(m, layer.lora_down, m.h_bf, LX_BF16, D, m.lora_bf, LX_BF16, m.rank, T, s);
        linear(m, layer.lora_up, m.lora_bf, LX_BF16, m.rank, m.qkv_buf, LX_F32, m.qkv_width, T, s,
             layer.pass_gate[pass], 1.0f);
        LX_CHECK(lx_attention_varlen(m.qkv_buf, m.qkv_width, m.k_cache[slot_index], m.v_cache[slot_index], m.d_slots,
                                     m.d_start_pos, m.d_cu, n_seq, m.kv_capacity, m.att_bf, m.AH, m.KVH, m.Dh,
                                     1.0f / std::sqrt(static_cast<float>(m.Dh)), s));
        linear(m, layer.out, m.att_bf, LX_BF16, m.AH * m.Dh, m.update, LX_F32, D, T, s);
      } else {
        const int gn = m.G * m.N;
        linear(m, layer.in_proj, m.h_bf, LX_BF16, D, m.zxbcdt, LX_F32, m.proj, T, s);
        if (spec) {
          // the step's tokens become pending positions (kernels.h, "Speculative decoding")
          LX_CHECK(lx_spec_conv1d(m.zxbcdt + m.I, m.proj, layer.conv_w, layer.conv_b, m.conv_state[slot_index],
                                  m.conv_hist[slot_index], m.d_slots, m.d_has_state, m.d_fold, m.d_keep, m.d_cu, n_seq,
                                  m.xbc, m.conv_dim, m.W, s));
          LX_CHECK(lx_spec_ssm_scan(m.xbc, m.conv_dim, m.zxbcdt + m.I + m.conv_dim, m.proj, m.xbc + m.I,
                                    m.xbc + m.I + gn, m.conv_dim, layer.A_log, layer.D, layer.dt_bias,
                                    m.ssm_state[slot_index], m.ssm_hist[slot_index], m.d_slots, m.d_has_state,
                                    m.d_fold, m.d_keep, m.d_cu, n_seq, m.ssm_y, m.H, m.Pd, m.G, m.N, s));
        } else {
          LX_CHECK(lx_conv1d_varlen(m.zxbcdt + m.I, m.proj, layer.conv_w, layer.conv_b, m.conv_state[slot_index],
                                    m.d_slots, m.d_has_state, m.d_cu, n_seq, m.xbc, m.conv_dim, m.W, s));
          LX_CHECK(lx_ssm_scan_varlen(m.xbc, m.conv_dim, m.zxbcdt + m.I + m.conv_dim, m.proj, m.xbc + m.I,
                                      m.xbc + m.I + gn, m.conv_dim, layer.A_log, layer.D, layer.dt_bias,
                                      m.ssm_state[slot_index], m.d_slots, m.d_has_state, m.d_cu, n_seq, m.ssm_y, m.H,
                                      m.Pd, m.G, m.N, s));
        }
        LX_CHECK(lx_gated_rmsnorm(m.ssm_y, m.zxbcdt, m.proj, layer.norm_w, m.normed_bf, T, m.I, m.G,
                                  c.mamba_norm_eps, s));
        linear(m, layer.out_proj, m.normed_bf, LX_BF16, m.I, m.update, LX_F32, D, T, s);
      }
      LX_CHECK(lx_mhc_write(m.mixed, mc + S * S + S, m.update, m.streams, T, S, D, s));

      // latent mixture of experts
      const float* ec = layer.moe_conn.params + pass * params;
      LX_CHECK(lx_mhc_read(m.streams, ec, ec + S * S, layer.moe_conn.norm, c.rms_norm_eps, m.mixed, m.h_bf, T, S, D, s));
      // the latent lands in the router's input: [latent | features], one row of Lat + R per token
      linear(m, layer.latent_down, m.h_bf, LX_BF16, D, m.router_in, LX_F32, Lat + R, T, s);
      to_bf16(m.router_in, Lat + R, m.latent_bf, Lat, T, Lat, s);
      if (beside) check(lx_rt_stream_wait(s, m.route_join), "stream wait");
      copy_f32(m.features, R, m.router_in + Lat, Lat + R, T, R, s);
      gemm_f32(m.router_in, Lat + R, layer.router_w, layer.router_b, m.router_logits, E, T, E, Lat + R, s);
      LX_CHECK(lx_moe_route(m.router_logits, layer.selection_bias, m.route_ids, m.route_w, m.entropy, m.confidence, T,
                            E, m.K, s));
      if (m.route_trace) {  // pass, layer, T, then T * K expert ids (int32)
        std::vector<int32_t> ids(static_cast<size_t>(T) * m.K);
        check(lx_rt_copy(ids.data(), m.route_ids, static_cast<int64_t>(ids.size() * 4), LX_COPY_D2H, s),
              "route trace copy");
        check(lx_rt_stream_sync(s), "stream sync");
        const int32_t head[3] = {pass, i, T};
        std::fwrite(head, 4, 3, m.route_trace);
        std::fwrite(ids.data(), 4, ids.size(), m.route_trace);
      }
      const bool prefetching = m.prefetch > 0 && T <= Impl::kGemvRows;
      if (m.offload) {
        if (prefetching && i > 0)  // the branch that prefetched this layer's experts must be done
          check(lx_rt_stream_wait(s, m.join_events[slot_index - 1]), "stream wait");
        if (T <= Impl::kGemvRows)
          m.offload->decode(i, m.route_ids, m.route_w, m.latent_bf, m.routed, T, s);
        else
          m.offload->prefill(i, m.route_ids, m.route_w, m.latent_bf, m.routed, T, s);
      } else if (T <= Impl::kGemvRows && layer.gate_up_x16)
        LX_CHECK(lx_moe_experts_bf16x(m.latent_bf, m.route_ids, m.route_w, layer.gate_up_x16, layer.down_x16, m.routed,
                                      m.moe_workspace, T, Lat, m.Iexp, m.K, s));
      else if (T <= Impl::kGemvRows && m.decode_bf16 && !layer.gate_up_q)
        LX_CHECK(lx_moe_experts_decode_bf16(m.latent_bf, m.route_ids, m.route_w, layer.gate_up, layer.down, m.routed,
                                            m.moe_workspace, T, Lat, m.Iexp, m.K, s));
      else if (T <= Impl::kGemvRows && layer.gate_up_q)
        LX_CHECK(lx_moe_experts_w8(m.latent_bf, m.route_ids, m.route_w, layer.gate_up_q, layer.gate_up_scale,
                                   layer.down_q, layer.down_scale, m.routed, m.moe_workspace, T, Lat, m.Iexp, m.K, s));
      else if (m.grouped_experts && T > Impl::kGemvRows)  // decode-sized steps keep moe.cu's decode kernel
        LX_CHECK(lx_moe_experts_grouped(m.latent_bf, m.route_ids, m.route_w, &layer.gate_up_pw, &layer.down_pw,
                                        layer.gate_up_pw.type == LX_W_BF16 ? LX_MATH_BF16 : m.prefill_math, m.routed,
                                        m.moe_grouped_ws, T, Lat, m.Iexp, E, m.K, s));
      else
        LX_CHECK(lx_moe_experts(m.latent_bf, m.route_ids, m.route_w, layer.gate_up, layer.down, m.routed, T, Lat,
                                m.Iexp, m.K, s));
      linear(m, layer.shared_gate_up, m.latent_bf, LX_BF16, Lat, m.shared_gu, LX_F32, 2 * m.Ishared, T, s);
      LX_CHECK(lx_silu_mul(m.shared_gu, 2 * m.Ishared, m.shared_bf, LX_BF16, T, m.Ishared, s));
      linear(m, layer.shared_down, m.shared_bf, LX_BF16, m.Ishared, m.routed, LX_F32, Lat, T, s, 1.0f, 1.0f);
      linear(m, layer.latent_up, m.routed, LX_F32, Lat, m.update, LX_F32, D, T, s);
      LX_CHECK(lx_moe_summary(m.route_ids, m.route_w, layer.expert_emb, m.summary, T, R, m.K, s));
      LX_CHECK(lx_mhc_write(m.mixed, ec + S * S + S, m.update, m.streams, T, S, D, s));

      if (prefetching && i + 1 < m.L_layers) {
        // On this stream, the inputs the branch needs, so nothing it reads changes under it: the next
        // layer's MoE read of the streams, and this layer's routing features. Then the branch: the
        // next layer's latent_down and router, the top n experts, and their copies into VRAM.
        const Impl::Layer& nx = m.layers[i + 1];
        const float* nc = nx.moe_conn.params + pass * params;
        LX_CHECK(lx_mhc_read(m.streams, nc, nc + S * S, nx.moe_conn.norm, c.rms_norm_eps, m.pred_mixed, m.pred_h, T, S,
                             D, s));
        copy_f32(m.features, R, m.pred_router_in + Lat, Lat + R, T, R, s);
        check(lx_rt_event_record(m.fork_events[slot_index], s), "event record");
        void* b = m.side;
        check(lx_rt_stream_wait(b, m.fork_events[slot_index]), "stream wait");
        linear(m, nx.latent_down, m.pred_h, LX_BF16, D, m.pred_router_in, LX_F32, Lat + R, T, b);
        LX_CHECK(lx_gemm_f32(m.pred_router_in, Lat + R, nx.router_w, nx.router_b, m.pred_logits, E, T, E, Lat + R, b));
        LX_CHECK(lx_moe_route(m.pred_logits, nx.selection_bias, m.pred_ids, m.pred_w, m.pred_entropy, m.pred_confidence,
                              T, E, Impl::kPredicted, b));
        // one token: its top n; several: the first n of the flattened [T, 8] (each token's best first
        // would need a reorder; a decode batch of one is what this is for)
        m.offload->prefetch(i + 1, m.pred_ids, std::min(m.prefetch, T * Impl::kPredicted), b);
        check(lx_rt_event_record(m.join_events[slot_index], b), "event record");
      }

      if (m.route_trace && i + 1 < m.L_layers) {  // record: pass | 0x10000, layer, T, then T * 8 ids
        const Impl::Layer& nx = m.layers[i + 1];
        const float* nc = nx.moe_conn.params + pass * params;
        LX_CHECK(lx_mhc_read(m.streams, nc, nc + S * S, nx.moe_conn.norm, c.rms_norm_eps, m.pred_mixed, m.pred_h, T, S,
                             D, s));
        linear(m, nx.latent_down, m.pred_h, LX_BF16, D, m.pred_router_in, LX_F32, Lat + R, T, s);
        copy_f32(m.features, R, m.pred_router_in + Lat, Lat + R, T, R, s);
        LX_CHECK(lx_gemm_f32(m.pred_router_in, Lat + R, nx.router_w, nx.router_b, m.pred_logits, E, T, E, Lat + R, s));
        LX_CHECK(lx_moe_route(m.pred_logits, nx.selection_bias, m.pred_ids, m.pred_w, m.pred_entropy, m.pred_confidence,
                              T, E, Impl::kPredicted, s));
        std::vector<int32_t> ids(static_cast<size_t>(T) * Impl::kPredicted);
        check(lx_rt_copy(ids.data(), m.pred_ids, static_cast<int64_t>(ids.size() * 4), LX_COPY_D2H, s),
              "route trace copy");
        check(lx_rt_stream_sync(s), "stream sync");
        const int32_t head[3] = {pass | 0x10000, i + 1, T};
        std::fwrite(head, 4, 3, m.route_trace);
        std::fwrite(ids.data(), 4, ids.size(), m.route_trace);
      }

      LX_CHECK(glue::axpby(m.history, m.summary, 0.5f, 0.5f, static_cast<int64_t>(T) * R, s));
      if (layer.attention) memory_write(m, c, T, pass, anchor++, n_entries, s);
      write_tap(m, pass, i, T, s);
    }

    // the end of a pass: one more read, the depth head, and an entry for where the pass left the token
    memory_retrieve(m, T, n_entries, s);
    LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
    if (const int col = m.tap_col[static_cast<size_t>(pass) * (m.L_layers + 1)]; col >= 0)
      copy_f32(m.state, D, m.tap_out + col, m.tap_dim, T, D, s);
    LX_CHECK(glue::sub(m.state, m.previous, m.difference, static_cast<int64_t>(T) * D, s));
    routing_features(m, T, s);
    concat4(m, m.state, m.mem_summary, m.difference, m.features, m.cont_in, LX_F32, T, s);
    gemm_f32(m.cont_in, 3 * D + R, m.c_hidden_w, m.c_hidden_b, m.cont_hidden, R, T, R, 3 * D + R, s);
    LX_CHECK(lx_activate_f32(m.cont_hidden, static_cast<int64_t>(T) * R, 1, s));
    gemm_f32(m.cont_hidden, R, m.c_out_w, m.c_out_b, m.cont_logit, 1, T, 1, R, s);
    LX_CHECK(lx_continuation_finish(m.cont_logit, m.state, m.difference, c.continuation_logit_limit, m.continuation, T,
                                    D, s));
    memory_write(m, c, T, pass, anchor, n_entries, s);
    copy_f32(m.state, D, m.previous, D, T, D, s);
  }

  // mean over streams, the final norm and the tied LM head
  LX_CHECK(lx_stream_mean(m.streams, m.state, T, S, D, s));
  if (all_rows) {
    LX_CHECK(lx_rmsnorm(m.state, LX_F32, m.final_norm, m.out_bf, LX_BF16, T, D, c.rms_norm_eps, s));
    linear(m, m.head, m.out_bf, LX_BF16, D, logits, LX_F32, c.vocab_size, T, s);
  } else {
    LX_CHECK(glue::gather_last(m.state, m.d_cu, m.last_rows, n_seq, D, s));
    LX_CHECK(lx_rmsnorm(m.last_rows, LX_F32, m.final_norm, m.out_bf, LX_BF16, n_seq, D, c.rms_norm_eps, s));
    linear(m, m.head, m.out_bf, LX_BF16, D, logits, LX_F32, c.vocab_size, n_seq, s);
  }
}

}  // namespace lx
