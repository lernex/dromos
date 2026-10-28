// The decode step's memory traffic, counted over the default decode path (metis.cpp, run()). One traversal serves both
// providers (traffic.h): it walks the layers, the passes and the depth memory's retrievals and writes in the order run()
// makes them, and asks for each projection's descriptor by name.
#include "model/traffic.h"

#include <cmath>
#include <stdexcept>
#include <utility>

#include "core/checkpoint.h"

namespace lx {
namespace {

// Whether the BF16X encoding of rows of K values is smaller than the bf16 rows (metis.cpp and plan.cpp agree on this).
bool bf16x_smaller(int64_t K) {
  const int64_t row = lx_bf16x_row_bytes(static_cast<int>(K));
  return row > 0 && row < 2 * K;
}

}  // namespace

MatrixBytes matrix_bytes(lx_wtype format, bool copy, int64_t N, int64_t K) {
  if (copy) format = LX_W_BF16X;
  MatrixBytes b;
  const double n = static_cast<double>(N), k = static_cast<double>(K);
  switch (format) {
    case LX_W_BF16:
      b.data = 2.0 * n * k;
      break;
    case LX_W_BF16X: {
      // The encoding's fixed rows (lx_bf16x_row_bytes); the rare overflow records of a row are not counted.
      const int64_t row = lx_bf16x_row_bytes(static_cast<int>(K));
      if (row <= 0) throw std::runtime_error("decode traffic: BF16X rows are not available on this backend");
      b.data = n * static_cast<double>(row);
      break;
    }
    case LX_W_INT8:
    case LX_W_FP8:
      b.data = n * k;
      b.scales = 4.0 * n;  // one f32 per output row
      break;
    case LX_W_NVFP4:
      b.data = n * k / 2.0;   // two e2m1 per byte
      b.scales = n * k / 16.0;  // one e4m3 block scale per 16 values
      break;
    default:
      b.data = 4.0 * n * k;  // f32: the routers and the depth head
      break;
  }
  return b;
}

double DecodeTraffic::expected_experts(int streams) const {
  if (experts <= 0) return 0;
  const double E = experts, k = top_k;
  return E * (1.0 - std::pow(1.0 - k / E, streams));
}

double DecodeTraffic::item_bytes(const TrafficItem& it, int streams) const {
  const double per = it.data + it.scales;
  switch (it.scope) {
    case Scope::Shared:
      return it.count * per;
    case Scope::PerStream:
      return streams * it.count * (per + it.per_position * context);
    case Scope::Expert:
      return it.count * expected_experts(streams) * per;
  }
  return 0;
}

double DecodeTraffic::step_bytes(int streams) const {
  double total = 0;
  for (const TrafficItem& it : items) total += item_bytes(it, streams);
  return total;
}

double DecodeTraffic::macs_per_token() const {
  double macs = 0;
  for (const TrafficItem& it : items) macs += it.count * it.macs * (it.scope == Scope::Expert ? top_k : 1);
  return macs;
}

DecodeTraffic count_decode_traffic(const MetisConfig& c, const MatrixLookup& lookup, int context) {
  DecodeTraffic t;
  t.context = context;
  t.top_k = c.num_experts_per_tok;
  t.experts = c.n_routed_experts;
  std::vector<TrafficItem>& items = t.items;

  // Every layer runs in every pass, so a layer's items count the passes; the depth memory's count its own events.
  const double P = c.num_passes, L = c.num_hidden_layers;
  const int64_t D = c.hidden_size, SR = c.n_streams, Mm = c.memory_dim, R = c.route_feature_dim;
  const int64_t E = c.n_routed_experts, k = c.num_experts_per_tok, Lat = c.latent_dim;
  const int64_t Iexp = c.expert_intermediate_size, Ish = c.shared_expert_intermediate_size;
  const int64_t I = c.mamba_intermediate(), H = c.mamba_heads(), Pd = c.mamba_head_dim;
  const int64_t G = c.mamba_n_groups, N = c.mamba_d_state, W = c.mamba_d_conv;
  const int64_t conv_dim = I + 2 * G * N, proj = 2 * I + 2 * G * N + H;
  const int64_t AH = c.num_attention_heads, KVH = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t qkv = (AH + 2 * KVH) * Dh, rank = c.attention_pass_lora_rank, V = c.vocab_size;
  const int64_t hyper = SR * SR + 2 * SR;  // a hyper-connection's f32 parameters for one pass: matrix, read, write
  const int64_t route_in = 3 * D + R;      // the routing features' input: state, retrieved, difference, history

  auto push = [&](std::string name, std::string group, int layer, Scope scope, double count, double data, double scales,
                  double per_position, double macs) {
    TrafficItem it;
    it.name = std::move(name);
    it.group = std::move(group);
    it.layer = layer;
    it.scope = scope;
    it.count = count;
    it.data = data;
    it.scales = scales;
    it.per_position = per_position;
    it.macs = macs;
    items.push_back(std::move(it));
  };
  // A projection every token multiplies `rows` rows of (the depth memory's per-residual-stream rows, else one).
  auto dense = [&](const std::string& key, const std::string& name, const std::string& group, int layer, double count,
                   int64_t Ne, int64_t Nn, int64_t Kk, double rows) {
    const MatrixDesc d = lookup(key, Ne, Nn, Kk);
    const MatrixBytes b = matrix_bytes(d.format, d.copy, Nn, Kk);
    push(name, group, layer, Scope::Shared, count, b.data, b.scales, 0, rows * double(Nn) * double(Kk));
  };
  // One routed expert's matrix. A NVFP4 expert also reads its global, one f32, with its rows.
  auto expert = [&](const std::string& key, const std::string& name, int layer, double count, int64_t Nn, int64_t Kk) {
    const MatrixDesc d = lookup(key, E, Nn, Kk);
    MatrixBytes b = matrix_bytes(d.format, d.copy, Nn, Kk);
    if (d.format == LX_W_NVFP4) b.scales += 4.0;
    push(name, "moe", layer, Scope::Expert, count, b.data, b.scales, 0, double(Nn) * double(Kk));
  };

  // ---- each layer, once per pass ----
  for (int i = 0; i < c.num_hidden_layers; ++i) {
    const std::string p = "layers." + std::to_string(i) + ".";
    if (c.is_attention(i)) {
      const std::string a = p + "mixer.";
      dense(a + "qkv", "attention qkv", "mixer", i, P, 1, qkv, D, 1);
      dense(a + "out", "attention out", "mixer", i, P, 1, D, AH * Dh, 1);
      dense(a + "pass_lora_down", "attention pass LoRA down", "mixer", i, P, 1, rank, D, 1);
      dense(a + "pass_lora_up", "attention pass LoRA up", "mixer", i, P, 1, qkv, rank, 1);
      // K and V of every position up to the context are read (bf16), and the new position is written.
      push("attention cache read", "state and cache", i, Scope::PerStream, P, 0, 0, 2.0 * KVH * Dh * 2, 0);
      push("attention cache write", "state and cache", i, Scope::PerStream, P, 2.0 * KVH * Dh * 2, 0, 0, 0);
    } else {
      const std::string a = p + "mixer.impl.mixer.";
      dense(a + "in_proj", "Mamba in_proj", "mixer", i, P, 1, proj, D, 1);
      dense(a + "out_proj", "Mamba out_proj", "mixer", i, P, 1, D, I, 1);
      push("Mamba conv weights", "mixer", i, Scope::Shared, P, 2.0 * conv_dim * W + 2.0 * conv_dim, 0, 0, 0);
      push("Mamba gated norm and scan constants", "mixer", i, Scope::Shared, P, 2.0 * I + 3.0 * H * 4, 0, 0, 0);
      // The SSM state (f32 H x P x N) and the conv state (f32 W - 1 x conv_dim), each read and written once.
      push("Mamba state", "state and cache", i, Scope::PerStream, P, 2.0 * 4.0 * (H * Pd * N + (W - 1) * conv_dim), 0,
           0, 0);
    }

    // The two hyper-connections: their norms (bf16 [D]) and their f32 mixes for the pass.
    push("hyper-connection norms", "connections", i, Scope::Shared, P, 2.0 * 2.0 * D, 0, 0, 0);
    push("hyper-connection mixes", "connections", i, Scope::Shared, P, 2.0 * 4.0 * hyper, 0, 0, 0);

    const std::string e = p + "moe.";
    dense(e + "latent_down", "latent_down", "moe", i, P, 1, Lat, D, 1);
    dense(e + "latent_up", "latent_up", "moe", i, P, 1, D, Lat, 1);
    dense(e + "shared_expert.gate_up", "shared expert gate_up", "moe", i, P, 1, 2 * Ish, Lat, 1);
    dense(e + "shared_expert.down", "shared expert down", "moe", i, P, 1, Lat, Ish, 1);
    // The router is f32 in every format; its bias and the selection bias are f32 [E] each.
    push("router (f32)", "moe", i, Scope::Shared, P, 4.0 * E * (Lat + R), 0, 0, double(E) * double(Lat + R));
    push("router and selection bias", "moe", i, Scope::Shared, P, 2.0 * 4.0 * E, 0, 0, 0);
    expert(e + "local_experts.gate_up", "routed experts gate_up", i, P, 2 * Iexp, Lat);
    expert(e + "local_experts.down", "routed experts down", i, P, Lat, Iexp);
    // The routing summary reads the expert embedding of each of the token's top_k experts (bf16 [R]).
    push("expert embeddings", "moe", i, Scope::PerStream, P, 2.0 * k * R, 0, 0, 0);
  }

  // ---- the depth memory, from the events of the step ----
  // A retrieval runs at the start of every layer and at each pass's end. It reads the query and output projections and
  // the stream gate only once the bank holds an entry (memory_retrieve returns early on an empty bank), and reads the
  // bank's entries, one key and one value each. An entry is written by every attention layer and by each pass's end.
  double retrievals = 0, active = 0, entry_reads = 0, writes = 0;
  int64_t entries = 0;
  for (int pass = 0; pass < c.num_passes; ++pass) {
    for (int i = 0; i < c.num_hidden_layers; ++i) {
      ++retrievals;
      if (entries > 0) {
        ++active;
        entry_reads += static_cast<double>(entries);
      }
      if (c.is_attention(i)) {
        ++writes;
        ++entries;
      }
    }
    ++retrievals;
    if (entries > 0) {
      ++active;
      entry_reads += static_cast<double>(entries);
    }
    ++writes;
    ++entries;
  }

  const std::string dm = "depth_memory.";
  // The routing features run once per layer and once per pass's end: tanh of route_projection over [state, retrieved,
  // difference, history] (3D + R wide).
  dense(dm + "route_projection", "depth memory routing features", "depth memory", -1, (L + 1) * P, 1, R, route_in, 1);
  // The read's query and output run on every residual stream's row of the token (SR rows per token, one weight read).
  dense(dm + "query", "depth memory read query", "depth memory", -1, active, 1, Mm, D, double(SR));
  dense(dm + "output", "depth memory read output", "depth memory", -1, active, 1, D, Mm, double(SR));
  push("depth memory read gate", "depth memory", -1, Scope::Shared, active, 2.0 * SR * D + 2.0 * SR, 0, 0, 0);
  // Per entry read: a key and a value, f32 [Mm] each. Per entry written: the same, plus the pass and anchor rows (bf16).
  push("depth memory bank read", "depth memory", -1, Scope::PerStream, entry_reads, 2.0 * Mm * 4, 0, 0, 0);
  push("depth memory bank write", "depth memory", -1, Scope::PerStream, writes, 2.0 * Mm * 4, 0, 0, 0);
  push("depth memory pass and anchor rows", "depth memory", -1, Scope::Shared, writes, 2.0 * 2.0 * Mm, 0, 0, 0);
  dense(dm + "state_write", "depth memory entry state", "depth memory", -1, writes, 1, Mm, D, 1);
  dense(dm + "metadata_write", "depth memory entry metadata", "depth memory", -1, writes, 1, Mm, R + 4, 1);
  dense(dm + "key", "depth memory entry key", "depth memory", -1, writes, 1, Mm, Mm, 1);
  dense(dm + "value", "depth memory entry value", "depth memory", -1, writes, 1, Mm, Mm, 1);
  // The depth head, once per pass (run() computes its continuation at every pass's end): f32 hidden layer [R, 3D + R]
  // and its bias, then the output row and its bias.
  push("depth head (f32)", "depth memory", -1, Scope::Shared, P, 4.0 * (double(R) * route_in + 2.0 * R + 1), 0, 0,
       double(R) * double(route_in) + double(R));

  // ---- the embedding, the final norm and the tied LM head ----
  push("embedding row", "head", -1, Scope::PerStream, 1, 2.0 * D, 0, 0, 0);
  push("residual stream embeddings", "head", -1, Scope::PerStream, 1, 2.0 * SR * D, 0, 0, 0);
  push("final norm", "head", -1, Scope::Shared, 1, 2.0 * D, 0, 0, 0);
  dense("embedding", "LM head", "head", -1, 1, 1, V, D, 1);
  return t;
}

DecodeTraffic config_decode_traffic(const MetisConfig& config, lx_wtype target, bool bf16x_decode, int context,
                                    const std::set<std::string>* dropped_copies) {
  const bool copies = bf16x_decode && target != LX_W_BF16X;  // the lossless target is its own encoding
  return count_decode_traffic(
      config,
      [&](const std::string& name, int64_t E, int64_t N, int64_t K) {
        MatrixDesc d;
        d.format = projection_format(target, name);
        d.E = E;
        d.N = N;
        d.K = K;
        const bool expert = name.find("local_experts.") != std::string::npos;
        // The copy unit, as the planner names it: an expert pair is one unit, the tied head is "lm_head".
        const std::string unit = expert ? name.substr(0, name.rfind('.')) : name == "embedding" ? "lm_head" : name;
        const bool left_out = dropped_copies && dropped_copies->count(unit);
        d.copy = copies && !left_out && d.format == LX_W_BF16 && (expert || bf16x_smaller(K));
        return d;
      },
      context);
}

}  // namespace lx
