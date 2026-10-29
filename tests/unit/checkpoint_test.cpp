// The checkpoint loader (core/checkpoint.h) on the real folders: what each load holds on the device, compared
// byte for byte with the file's own tensors. These need a GPU and the folders, so each case is skipped unless
// the folder it reads is named:
//   LX_CKPT_BF16_DIR    a BF16 folder (the Praxis hf checkpoint)
//   LX_CKPT_FP8_DIR     the published FP8 folder (Praxis fp8)
//   LX_CKPT_NVFP4_DIR   the GPTQ NVFP4 folder (Praxis nvfp4)
// Run them under the GPU lock on the server:
//   scripts/remote.sh ckpt x 'LX_CKPT_BF16_DIR=... LX_CKPT_FP8_DIR=... LX_CKPT_NVFP4_DIR=... build/dromos_unit -tc="checkpoint*"'
#include <algorithm>
#include <functional>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"
#include "doctest.h"

using namespace lx;

namespace {

const char* folder(const char* variable) {
  const char* dir = std::getenv(variable);
  if (dir && *dir) return dir;
  std::printf("  skipped: %s is not set\n", variable);
  return nullptr;
}

void check_status(int status, const char* what) {
  if (status != 0) throw std::runtime_error(std::string(what) + ": " + lx_rt_error_string(status));
}

// Whether `bytes` of device memory equal the file's bytes, copied back in chunks.
bool device_equals(const void* device, const std::byte* host, size_t bytes) {
  constexpr size_t kChunk = 64ull << 20;
  std::vector<std::byte> back(std::min(bytes, kChunk));
  for (size_t off = 0; off < bytes; off += kChunk) {
    const size_t n = std::min(kChunk, bytes - off);
    check_status(lx_rt_copy(back.data(), static_cast<const std::byte*>(device) + off, static_cast<int64_t>(n),
                            LX_COPY_D2H, nullptr),
                 "reading the device");
    check_status(lx_rt_stream_sync(nullptr), "reading the device");
    if (std::memcmp(back.data(), host + off, n) != 0) return false;
  }
  return true;
}

const TensorInfo& tensor(const SafetensorsSet& set, const std::string& name) {
  const TensorInfo* t = set.find(name);
  if (!t) throw std::runtime_error("the folder has no " + name);
  return *t;
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool is_expert(const std::string& name) { return name.find("local_experts.") != std::string::npos; }

// The format a stored tensor's dtype names.
lx_wtype stored_format_of(DType dtype) {
  switch (dtype) {
    case DType::F8_E4M3: return LX_W_FP8;
    case DType::U8: return LX_W_NVFP4;
    default: return LX_W_BF16;
  }
}

// The file's name for a loaded projection: the tied head is the embedding table.
std::string file_name(const std::string& projection) { return projection == "lm_head" ? "embedding" : projection; }

// Every loaded projection is the file's own bytes in the format the file stores it: the data, the scales,
// and the per-expert globals of an NVFP4 stack.
void expect_as_stored(const ModelWeights& m, const SafetensorsSet& set) {
  for (const std::string& name : m.projection_names()) {
    const lx_weight* w = m.projection(name);
    REQUIRE(w);
    const TensorInfo& data = tensor(set, file_name(name) + ".weight");
    CHECK(w->type == stored_format_of(data.dtype));
    CHECK(device_equals(w->data, data.data, data.bytes));
    if (w->scale) {
      const TensorInfo& scale = tensor(set, file_name(name) + ".weight_scale");
      CHECK(device_equals(w->scale, scale.data, scale.bytes));
    }
    if (w->expert_global) {
      const TensorInfo& global = tensor(set, file_name(name) + ".weight_scale_2");
      CHECK(device_equals(w->expert_global, global.data, global.bytes));
    }
  }
}

// The format each projection is held in, against a target's recipe, for the projections of every name.
void expect_formats(const ModelWeights& m, const std::function<lx_wtype(const std::string&)>& want) {
  for (const std::string& name : m.projection_names()) {
    const lx_weight* w = m.projection(name);
    REQUIRE(w);
    CHECK(w->type == want(name));
  }
}

void report(const char* what, const ModelWeights& m) {
  std::printf("  %-44s %8.2f GiB resident, %8.2f GiB peak during the load\n", what,
              static_cast<double>(m.device_bytes()) / (1ull << 30),
              static_cast<double>(m.peak_device_bytes()) / (1ull << 30));
}

}  // namespace

TEST_CASE("checkpoint: a BF16 folder loads as BF16 with every projection the file's bytes") {
  const char* dir = folder("LX_CKPT_BF16_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  CHECK(ModelWeights::stored_format(dir) == LX_W_BF16);
  LoadOptions options;
  options.target = LX_W_BF16;
  const ModelWeights m = ModelWeights::load(dir, config, options);
  CHECK(m.format() == LX_W_BF16);
  expect_formats(m, [](const std::string&) -> lx_wtype { return LX_W_BF16; });
  expect_as_stored(m, set);
  report("BF16 target from a BF16 folder", m);
}

TEST_CASE("checkpoint: an FP8 folder loads as FP8, the file's own keep set") {
  const char* dir = folder("LX_CKPT_FP8_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  CHECK(ModelWeights::stored_format(dir) == LX_W_FP8);
  LoadOptions options;
  options.target = LX_W_FP8;
  const ModelWeights m = ModelWeights::load(dir, config, options);
  CHECK(m.format() == LX_W_FP8);
  // The published FP8 folders quantize the routed experts, the Mamba in_proj, the attention qkv and out;
  // everything else (the head, out_proj, the shared expert, the latent projections, the depth memory) is bf16.
  expect_formats(m, [](const std::string& name) -> lx_wtype {
    const bool fp8 = is_expert(name) || ends_with(name, "mixer.impl.mixer.in_proj") ||
                     ends_with(name, "mixer.qkv") || ends_with(name, "mixer.out");
    return fp8 ? LX_W_FP8 : LX_W_BF16;
  });
  expect_as_stored(m, set);
  report("FP8 target from the FP8 folder", m);
}

TEST_CASE("checkpoint: the GPTQ NVFP4 folder loads as NVFP4, its tensors unconverted") {
  const char* dir = folder("LX_CKPT_NVFP4_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  CHECK(ModelWeights::stored_format(dir) == LX_W_NVFP4);
  LoadOptions options;
  options.target = LX_W_NVFP4;
  const ModelWeights m = ModelWeights::load(dir, config, options);
  CHECK(m.format() == LX_W_NVFP4);
  // The GPTQ folder: experts NVFP4, the depth memory's key and value and the dense projections FP8, the
  // Mamba out_proj, the MoE latent_up, the pass LoRA and the metadata write bf16.
  expect_formats(m, [](const std::string& name) -> lx_wtype {
    if (is_expert(name)) return LX_W_NVFP4;
    if (ends_with(name, "mixer.impl.mixer.out_proj") || ends_with(name, "moe.latent_up") ||
        ends_with(name, "pass_lora_down") || ends_with(name, "pass_lora_up") ||
        ends_with(name, "depth_memory.metadata_write"))
      return LX_W_BF16;
    return LX_W_FP8;
  });
  expect_as_stored(m, set);
  // The per-expert globals, read back through projection_expert, are the file's weight_scale_2.
  const TensorInfo& global = tensor(set, "layers.0.moe.local_experts.gate_up.weight_scale_2");
  const auto* values = reinterpret_cast<const float*>(global.data);
  for (int64_t e : {int64_t{0}, int64_t{127}}) {
    const lx_weight w = m.projection_expert("layers.0.moe.local_experts.gate_up", e);
    CHECK(w.type == LX_W_NVFP4);
    CHECK(w.global == values[e]);
  }
  report("NVFP4 target from the GPTQ folder", m);
}

TEST_CASE("checkpoint: NVFP4 from a BF16 or FP8 folder is refused, naming the GPTQ checkpoint") {
  const char* bf16 = folder("LX_CKPT_BF16_DIR");
  const char* fp8 = folder("LX_CKPT_FP8_DIR");
  LoadOptions options;
  options.target = LX_W_NVFP4;
  const struct {
    const char* dir;
    const char* stored;
  } cases[] = {{bf16, "a BF16"}, {fp8, "an FP8"}};
  for (const auto& c : cases) {
    if (!c.dir) continue;
    const MetisConfig config = load_config(c.dir);
    std::string what;
    bool threw = false;
    try {
      ModelWeights::load(c.dir, config, options);
    } catch (const std::runtime_error& e) {
      threw = true;
      what = e.what();
    }
    CHECK(threw);
    const std::string expected = std::string("NVFP4 from ") + c.stored +
                                 " checkpoint is refused: round-to-nearest conversion fails its KL. Use the GPTQ "
                                 "NVFP4 checkpoint at /home/aiadmin/kronos-scratch/logos/engines/praxis/nvfp4 "
                                 "instead.";
    CHECK(config.family == "praxis");
    CHECK(what == expected);
    if (what != expected) std::printf("  got: %s\n", what.c_str());
  }
}

TEST_CASE("checkpoint: INT8 keeps only the engine's wide projections, and every other one is int8") {
  const char* dir = folder("LX_CKPT_BF16_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  LoadOptions options;
  options.target = LX_W_INT8;
  const ModelWeights m = ModelWeights::load(dir, config, options);
  CHECK(m.format() == LX_W_INT8);
  // The engine's int8 decode policy: every projection int8 but the Mamba out_proj, and the five it makes
  // with quantize = false (the pass LoRA, the depth memory's key, value and metadata write).
  expect_formats(m, [](const std::string& name) -> lx_wtype {
    const bool wide = ends_with(name, "mixer.impl.mixer.out_proj") || ends_with(name, "pass_lora_down") ||
                      ends_with(name, "pass_lora_up") || ends_with(name, "depth_memory.key") ||
                      ends_with(name, "depth_memory.value") || ends_with(name, "depth_memory.metadata_write");
    return wide ? LX_W_BF16 : LX_W_INT8;
  });
  report("INT8 target from the BF16 folder", m);
}

TEST_CASE("checkpoint: FP8 from a BF16 folder quantizes the published FP8 set and nothing else") {
  const char* dir = folder("LX_CKPT_BF16_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  LoadOptions options;
  options.target = LX_W_FP8;
  const ModelWeights m = ModelWeights::load(dir, config, options);
  CHECK(m.format() == LX_W_FP8);
  expect_formats(m, [](const std::string& name) -> lx_wtype {
    const bool fp8 = is_expert(name) || ends_with(name, "mixer.impl.mixer.in_proj") ||
                     ends_with(name, "mixer.qkv") || ends_with(name, "mixer.out");
    return fp8 ? LX_W_FP8 : LX_W_BF16;
  });
  report("FP8 target from the BF16 folder", m);
}

TEST_CASE("checkpoint: skip_experts leaves the expert stacks off the device, their sizes still readable") {
  const char* dir = folder("LX_CKPT_NVFP4_DIR");
  if (!dir) return;
  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  LoadOptions options;
  options.target = LX_W_NVFP4;

  // The expert stacks' bytes on the device: their data, block scales and per-expert globals, from the file.
  size_t expert_device = 0;
  for (const auto& [name, t] : set.tensors()) {
    if (!is_expert(name)) continue;
    if (ends_with(name, ".weight") || ends_with(name, ".weight_scale") || ends_with(name, ".weight_scale_2"))
      expert_device += t.bytes;
  }

  std::vector<std::string> experts;
  std::vector<ModelWeights::Shape> shapes;
  std::vector<std::pair<size_t, size_t>> sizes;
  size_t full_bytes = 0;
  {
    const ModelWeights full = ModelWeights::load(dir, config, options);
    full_bytes = full.device_bytes();
    report("NVFP4 target, experts loaded", full);
    for (const std::string& name : full.projection_names()) {
      if (!is_expert(name)) continue;
      experts.push_back(name);
      shapes.push_back(full.projection_shape(name));
      sizes.push_back(full.expert_bytes(name));
    }
  }
  REQUIRE(!experts.empty());

  options.skip_experts = true;
  const ModelWeights skipped = ModelWeights::load(dir, config, options);
  report("NVFP4 target, experts skipped", skipped);
  CHECK(skipped.device_bytes() == full_bytes - expert_device);
  std::printf("  expert stacks off the device: %.2f GiB (%zu bytes)\n", static_cast<double>(expert_device) / (1ull << 30),
              expert_device);

  for (size_t i = 0; i < experts.size(); ++i) {
    const std::string& name = experts[i];
    CHECK(skipped.projection(name) == nullptr);
    const ModelWeights::Shape s = skipped.projection_shape(name);
    CHECK(s.E == shapes[i].E);
    CHECK(s.N == shapes[i].N);
    CHECK(s.K == shapes[i].K);
    CHECK(skipped.expert_bytes(name) == sizes[i]);
    CHECK_THROWS_AS(skipped.projection_expert(name, 0), std::runtime_error);
  }
  for (const std::string& name : skipped.projection_names()) CHECK_FALSE(is_expert(name));
  // The dense projections are still the file's bytes.
  expect_as_stored(skipped, set);
}
