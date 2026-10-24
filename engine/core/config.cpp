#include "core/config.h"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include <nlohmann/json.hpp>

namespace lx {
namespace {

using json = nlohmann::json;

// A field the engine cannot run without. The message names the file and the field.
template <typename T>
T required(const json& j, const char* key, const std::string& where) {
  const auto it = j.find(key);
  if (it == j.end()) throw std::runtime_error(where + ": missing field \"" + key + "\"");
  try {
    return it->get<T>();
  } catch (const json::exception&) {
    throw std::runtime_error(where + ": field \"" + key + "\" has the wrong type (" + it->type_name() + ")");
  }
}

// A field that may be absent or null; absent gives `fallback`.
template <typename T>
T optional_field(const json& j, const char* key, T fallback, const std::string& where) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return fallback;
  try {
    return it->get<T>();
  } catch (const json::exception&) {
    throw std::runtime_error(where + ": field \"" + key + "\" has the wrong type (" + it->type_name() + ")");
  }
}

// A special-token id that the checkpoint may leave null (bos_token_id is null in Praxis).
std::optional<int> optional_token(const json& j, const char* key, const std::string& where) {
  const auto it = j.find(key);
  if (it == j.end() || it->is_null()) return std::nullopt;
  return optional_field<int>(j, key, 0, where);
}

void require(bool ok, const std::string& where, const std::string& message) {
  if (!ok) throw std::runtime_error(where + ": " + message);
}

std::vector<int> int_or_list(const json& j, const char* key, const std::string& where) {
  const auto it = j.find(key);
  if (it == j.end()) throw std::runtime_error(where + ": missing field \"" + key + "\"");
  try {
    if (it->is_array()) return it->get<std::vector<int>>();
    return {it->get<int>()};
  } catch (const json::exception&) {
    throw std::runtime_error(where + ": field \"" + key + "\" must be an integer or a list of integers");
  }
}

}  // namespace

bool MetisConfig::is_attention(int layer) const noexcept {
  return std::find(attention_layer_indices.begin(), attention_layer_indices.end(), layer) !=
         attention_layer_indices.end();
}

MetisConfig parse_config(std::string_view json_text, const std::string& where) {
  json j;
  try {
    j = json::parse(json_text);
  } catch (const json::exception& e) {
    throw std::runtime_error(where + ": not valid JSON (" + e.what() + ")");
  }
  require(j.is_object(), where, "the top level must be a JSON object");

  MetisConfig c;
  c.vocab_size = required<int>(j, "vocab_size", where);
  c.hidden_size = required<int>(j, "hidden_size", where);
  c.num_hidden_layers = required<int>(j, "num_hidden_layers", where);
  c.attention_layer_indices = required<std::vector<int>>(j, "attention_layer_indices", where);
  c.num_attention_heads = required<int>(j, "num_attention_heads", where);
  c.num_key_value_heads = required<int>(j, "num_key_value_heads", where);
  c.head_dim = required<int>(j, "head_dim", where);
  c.attention_pass_lora_rank = required<int>(j, "attention_pass_lora_rank", where);

  c.mamba_d_state = required<int>(j, "mamba_d_state", where);
  c.mamba_d_conv = required<int>(j, "mamba_d_conv", where);
  c.mamba_expand = required<int>(j, "mamba_expand", where);
  c.mamba_head_dim = required<int>(j, "mamba_head_dim", where);
  c.mamba_n_groups = required<int>(j, "mamba_n_groups", where);
  c.mamba_chunk_size = required<int>(j, "mamba_chunk_size", where);
  c.mamba_norm_eps = required<float>(j, "mamba_norm_eps", where);

  c.n_streams = required<int>(j, "n_streams", where);
  c.mhc_pass_embedding_dim = required<int>(j, "mhc_pass_embedding_dim", where);
  c.mhc_sinkhorn_iterations = required<int>(j, "mhc_sinkhorn_iterations", where);

  c.latent_dim = required<int>(j, "latent_dim", where);
  c.n_routed_experts = required<int>(j, "n_routed_experts", where);
  c.num_experts_per_tok = required<int>(j, "num_experts_per_tok", where);
  c.min_routed_k = required<int>(j, "min_routed_k", where);
  c.max_routed_k = required<int>(j, "max_routed_k", where);
  c.expert_intermediate_size = required<int>(j, "expert_intermediate_size", where);
  c.shared_expert_intermediate_size = required<int>(j, "shared_expert_intermediate_size", where);
  c.route_feature_dim = required<int>(j, "route_feature_dim", where);

  c.memory_dim = required<int>(j, "memory_dim", where);
  c.num_passes = required<int>(j, "num_passes", where);
  c.max_passes = required<int>(j, "max_passes", where);
  c.continuation_logit_limit = required<float>(j, "continuation_logit_limit", where);

  c.rms_norm_eps = required<float>(j, "rms_norm_eps", where);
  c.tie_word_embeddings = required<bool>(j, "tie_word_embeddings", where);
  c.max_position_embeddings = optional_field<int>(j, "max_position_embeddings", 0, where);
  c.bos_token_id = optional_token(j, "bos_token_id", where);
  c.pad_token_id = optional_token(j, "pad_token_id", where);
  c.eos_token_ids = int_or_list(j, "eos_token_id", where);
  c.family = optional_field<std::string>(j, "family", "", where);
  c.model_type = optional_field<std::string>(j, "model_type", "", where);
  c.torch_dtype = optional_field<std::string>(j, "torch_dtype", "", where);

  // Checks that a wrong config would otherwise turn into a bad shape deep inside a kernel.
  for (const int v : {c.vocab_size, c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
                      c.num_key_value_heads, c.head_dim, c.mamba_d_state, c.mamba_d_conv, c.mamba_expand,
                      c.mamba_head_dim, c.mamba_n_groups, c.n_streams, c.latent_dim, c.n_routed_experts,
                      c.num_experts_per_tok, c.expert_intermediate_size, c.memory_dim, c.num_passes})
    require(v > 0, where, "every size must be positive");
  require(c.num_attention_heads % c.num_key_value_heads == 0, where,
          "num_attention_heads must be a multiple of num_key_value_heads");
  require(c.mamba_intermediate() % c.mamba_head_dim == 0, where,
          "hidden_size * mamba_expand must be a multiple of mamba_head_dim");
  require(c.mamba_heads() % c.mamba_n_groups == 0, where,
          "the Mamba heads must divide evenly into mamba_n_groups");
  require(c.num_experts_per_tok <= c.n_routed_experts, where,
          "num_experts_per_tok cannot exceed n_routed_experts");
  require(c.min_routed_k >= 1 && c.min_routed_k <= c.max_routed_k, where,
          "need 1 <= min_routed_k <= max_routed_k");
  require(c.max_passes >= c.num_passes && c.num_passes >= 1, where, "need 1 <= num_passes <= max_passes");
  require(!c.eos_token_ids.empty(), where, "eos_token_id must name at least one token");
  for (const int layer : c.attention_layer_indices)
    require(layer >= 0 && layer < c.num_hidden_layers, where,
            "attention_layer_indices entry " + std::to_string(layer) + " is outside the layer range");
  return c;
}

MetisConfig load_config(const std::string& model_dir) {
  const std::filesystem::path path = std::filesystem::path(model_dir) / "config.json";
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot open " + path.string());
  std::ostringstream text;
  text << in.rdbuf();
  return parse_config(text.str(), path.string());
}

}  // namespace lx
