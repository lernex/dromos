// The Metis config.json, typed. Read once at load; everything after it works from these fields.
#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace lx {

struct MetisConfig {
  // Trunk
  int vocab_size = 0;
  int hidden_size = 0;
  int num_hidden_layers = 0;
  std::vector<int> attention_layer_indices;  // the layers that are GQA attention; the rest are Mamba-2

  // Attention (no positional encoding; a LoRA per pass on the QKV projection)
  int num_attention_heads = 0;
  int num_key_value_heads = 0;
  int head_dim = 0;
  int attention_pass_lora_rank = 0;

  // Mamba-2
  int mamba_d_state = 0;
  int mamba_d_conv = 0;
  int mamba_expand = 0;     // intermediate = hidden_size * mamba_expand
  int mamba_head_dim = 0;
  int mamba_n_groups = 0;
  int mamba_chunk_size = 0;
  float mamba_norm_eps = 0;

  // Hyper-connections: S residual streams mixed by Sinkhorn-balanced matrices
  int n_streams = 0;
  int mhc_pass_embedding_dim = 0;
  int mhc_sinkhorn_iterations = 0;

  // Latent mixture of experts
  int latent_dim = 0;
  int n_routed_experts = 0;
  int num_experts_per_tok = 0;
  int min_routed_k = 0;
  int max_routed_k = 0;
  int expert_intermediate_size = 0;
  int shared_expert_intermediate_size = 0;
  int route_feature_dim = 0;

  // Depth memory and the continuation head
  int memory_dim = 0;
  int num_passes = 0;
  int max_passes = 0;
  float continuation_logit_limit = 0;

  // Norms, embeddings and generation
  float rms_norm_eps = 0;
  bool tie_word_embeddings = false;
  int max_position_embeddings = 0;  // informational: the model has no positional encoding
  std::optional<int> bos_token_id;
  std::optional<int> pad_token_id;
  std::vector<int> eos_token_ids;  // config.json allows one id or a list
  std::string family;
  std::string model_type;
  std::string torch_dtype;

  // Derived
  int mamba_intermediate() const noexcept { return hidden_size * mamba_expand; }
  int mamba_heads() const noexcept { return mamba_intermediate() / mamba_head_dim; }
  bool is_attention(int layer) const noexcept;
};

// Reads MODEL_DIR/config.json. Throws std::runtime_error naming the file and the field on any
// missing, mistyped or inconsistent value.
MetisConfig load_config(const std::string& model_dir);

// The same from JSON text; `where` names the source in error messages.
MetisConfig parse_config(std::string_view json_text, const std::string& where);

}  // namespace lx
