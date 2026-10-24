// lernex inspect MODEL_DIR
//
// Loads the config and the checkpoint, uploads every tensor to the GPU, reads each one back and
// compares it with the file, then reports what landed. This is the loader's end-to-end check.

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"

namespace {

using Clock = std::chrono::steady_clock;

double seconds_since(Clock::time_point start) {
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + ": " + lx_rt_error_string(status));
}

std::string join_ints(const std::vector<int>& values) {
  std::string out;
  for (size_t i = 0; i < values.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(values[i]);
  }
  return out;
}

std::string shape_text(const std::vector<int64_t>& shape) {
  std::string out = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    if (i) out += ", ";
    out += std::to_string(shape[i]);
  }
  return out + "]";
}

void print_config(const std::string& dir, const lx::MetisConfig& c) {
  std::printf("config   %s\n", dir.c_str());
  std::printf("         %s / %s: %d layers (attention at %s), hidden %d, vocab %d\n",
              c.family.c_str(), c.model_type.c_str(), c.num_hidden_layers,
              join_ints(c.attention_layer_indices).c_str(), c.hidden_size, c.vocab_size);
  std::printf("         attention: %d heads, %d kv heads, head_dim %d, pass LoRA rank %d\n",
              c.num_attention_heads, c.num_key_value_heads, c.head_dim, c.attention_pass_lora_rank);
  std::printf("         mamba: d_state %d, d_conv %d, expand %d -> %d channels, %d heads of %d, %d groups\n",
              c.mamba_d_state, c.mamba_d_conv, c.mamba_expand, c.mamba_intermediate(), c.mamba_heads(),
              c.mamba_head_dim, c.mamba_n_groups);
  std::printf("         experts: %d routed (top %d, k %d..%d), latent %d, expert ffn %d, shared ffn %d\n",
              c.n_routed_experts, c.num_experts_per_tok, c.min_routed_k, c.max_routed_k, c.latent_dim,
              c.expert_intermediate_size, c.shared_expert_intermediate_size);
  std::printf("         streams %d, depth memory %d, passes %d (max %d), eos [%s], tie embeddings %s\n",
              c.n_streams, c.memory_dim, c.num_passes, c.max_passes, join_ints(c.eos_token_ids).c_str(),
              c.tie_word_embeddings ? "yes" : "no");
}

}  // namespace

int cmd_inspect(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: lernex inspect MODEL_DIR\n");
    return 2;
  }
  try {
    const std::string dir = argv[1];
    print_config(dir, lx::load_config(dir));

    lx_device_props props{};
    check(lx_rt_device_props(0, &props), "reading the device properties");
    int64_t free_before = 0, total_device = 0;
    check(lx_rt_mem_info(&free_before, &total_device), "reading the device memory");
    std::printf("gpu      %s, %.2f GB free of %.2f GB\n", props.name, free_before / 1e9, total_device / 1e9);

    const auto t_open = Clock::now();
    const lx::SafetensorsSet set = lx::SafetensorsSet::open(dir);
    std::printf("tensors  %zu in %zu file(s), %.2f GB of data, mapped in %.3f s\n", set.tensors().size(),
                set.file_count(), set.total_bytes() / 1e9, seconds_since(t_open));

    const auto t_upload = Clock::now();
    const lx::Weights weights = lx::Weights::upload(set);
    const double upload_s = seconds_since(t_upload);
    int64_t free_after = 0;
    check(lx_rt_mem_info(&free_after, &total_device), "reading the device memory");
    std::printf("uploaded %.2f GB (%.2f GB with 256-byte alignment) in %.2f s, %.2f GB/s\n",
                set.total_bytes() / 1e9, weights.device_bytes() / 1e9, upload_s,
                set.total_bytes() / 1e9 / upload_s);
    std::printf("device   %.2f GB in use by the weights\n", (free_before - free_after) / 1e9);

    // Every tensor read back from the device must match the file, byte for byte.
    const auto t_verify = Clock::now();
    size_t largest = 0;
    for (const auto& entry : set.tensors()) largest = std::max(largest, entry.second.bytes);
    std::vector<std::byte> host(largest);
    for (const auto& [name, info] : set.tensors()) {
      const lx::Tensor* t = weights.find(name);
      if (!t) throw std::runtime_error("the upload is missing \"" + name + "\"");
      if (info.bytes == 0) continue;
      check(lx_rt_copy(host.data(), t->data, static_cast<int64_t>(info.bytes), LX_COPY_D2H, nullptr),
            "reading back \"" + name + "\"");
      check(lx_rt_stream_sync(nullptr), "reading back \"" + name + "\"");
      if (std::memcmp(host.data(), info.data, info.bytes) != 0)
        throw std::runtime_error("the device copy of \"" + name + "\" differs from the checkpoint");
    }
    std::printf("verified %zu tensors against the file in %.2f s\n", set.tensors().size(),
                seconds_since(t_verify));

    std::printf("first tensors:\n");
    size_t shown = 0;
    for (const auto& [name, t] : weights.tensors()) {
      if (shown++ == 8) break;
      std::printf("  %-48s %-9s %s\n", name.c_str(), std::string(lx::dtype_name(t.dtype)).c_str(),
                  shape_text(t.shape).c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
