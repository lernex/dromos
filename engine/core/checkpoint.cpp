// The Metis checkpoint loader (checkpoint.h).
//
// A folder is read as the file says: a tensor's dtype names its format (BF16; F8_E4M3 with a per-row
// F32 weight_scale for FP8; U8 packed nibbles with an e4m3 weight_scale per 16 values and an F32
// weight_scale_2 per expert for NVFP4). Projections are the weights the engine multiplies by; every
// other BF16 or F32 tensor is a plain tensor, uploaded as it is.
//
// The target (LoadOptions::target) decides the format each projection is held in:
//   - the target equal to the folder's own format keeps every projection exactly as the file stores it;
//   - otherwise the recipe for the target decides, per projection (recipe() below), and a projection
//     whose file format differs is converted: its bf16 values (copied from the file, or dequantized
//     from its quantized bytes) are rounded to the target with lx_quantize_weight, one expert at a time,
//     so the peak device memory is the final footprint plus one expert's scratch.
//
// Recipes. The FP8 recipe is the published FP8 folder's own (the projections it quantizes: the Mamba
// in_proj, the attention qkv and out, and the routed experts). The NVFP4 recipe is the published NVFP4
// folder's, with two choices that follow the design: the MoE latent_up stays BF16, and the tied LM head
// is FP8. The INT8 recipe is the engine's int8 policy (every large projection but the Mamba out_proj).
// Whatever stays wide, stays wide in every target: the Mamba out_proj, the pass LoRA, and the depth
// memory's key, value and metadata write (small, and their rounding is the most visible).
// The lossless target (BF16X) is no rounding at all: every projection the other targets quantize is held
// in the lossless encoding of bf16x.cu, which reproduces the checkpoint's bf16 bits exactly. The wide ones
// stay bf16 there too, as in every target.
#include "core/checkpoint.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "core/safetensors.h"
#include "cuda/kernels.h"

namespace lx {
namespace {

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + ": " + lx_rt_error_string(status));
}

void check_lx(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed with error code " + std::to_string(status));
}

size_t align256(size_t n) { return (n + 255) / 256 * 256; }

// Whether `name` ends with the dotted component `suffix` ("mixer.qkv" matches "layers.4.mixer.qkv").
bool ends_with_part(std::string_view name, std::string_view suffix) {
  if (name.size() < suffix.size() || name.substr(name.size() - suffix.size()) != suffix) return false;
  return name.size() == suffix.size() || name[name.size() - suffix.size() - 1] == '.';
}

bool contains(std::string_view s, std::string_view part) { return s.find(part) != std::string_view::npos; }

// Bytes of one row of an [N, K] matrix in a format, and of its scales (kernels.h, "Weight formats").
size_t row_bytes(lx_wtype t, int64_t K) {
  switch (t) {
    case LX_W_BF16: return 2 * static_cast<size_t>(K);
    case LX_W_NVFP4: return static_cast<size_t>(K / 2);
    case LX_W_BF16X: return static_cast<size_t>(lx_bf16x_row_bytes(static_cast<int>(K)));
    default: return static_cast<size_t>(K);
  }
}

// The projection a name refers to: the ".weight" suffix is optional.
std::string projection_key(const std::string& name) {
  if (name.size() > 7 && name.compare(name.size() - 7, 7, ".weight") == 0) return name.substr(0, name.size() - 7);
  return name;
}
size_t scale_row_bytes(lx_wtype t, int64_t K) {
  switch (t) {
    case LX_W_INT8:
    case LX_W_FP8: return 4;
    case LX_W_NVFP4: return static_cast<size_t>(K / 16);
    default: return 0;
  }
}

// The format a projection's data has, from its dtype in the file.
lx_wtype format_of(DType dtype) {
  switch (dtype) {
    case DType::BF16: return LX_W_BF16;
    case DType::F8_E4M3: return LX_W_FP8;
    case DType::U8: return LX_W_NVFP4;
    case DType::I8: return LX_W_INT8;
    default: throw std::runtime_error("a projection cannot be stored as " + std::string(dtype_name(dtype)));
  }
}

// The weights the engine multiplies by, named by the tensor name without ".weight". The tied head is
// "embedding" (its gather table is the plain "embedding.weight").
constexpr std::string_view kProjectionParts[] = {
    "mixer.impl.mixer.in_proj", "mixer.impl.mixer.out_proj", "mixer.qkv",
    "mixer.out",                "mixer.pass_lora_down",      "mixer.pass_lora_up",
    "moe.latent_down",          "moe.latent_up",             "moe.shared_expert.gate_up",
    "moe.shared_expert.down",   "moe.local_experts.gate_up", "moe.local_experts.down",
    "depth_memory.query",       "depth_memory.state_write",  "depth_memory.output",
    "depth_memory.route_projection", "depth_memory.key",     "depth_memory.value",
    "depth_memory.metadata_write",
};

bool is_projection_key(std::string_view key) {
  if (key == "embedding") return true;
  for (const std::string_view part : kProjectionParts)
    if (ends_with_part(key, part)) return true;
  return false;
}

bool is_scale_name(std::string_view name) {
  return ends_with_part(name, "weight_scale") || ends_with_part(name, "weight_scale_2") ||
         ends_with_part(name, "input_scale");
}

// The format a projection is held in for a target (see the recipes at the top of this file).
lx_wtype recipe(lx_wtype target, std::string_view key) {
  const bool expert = contains(key, "local_experts.");
  if (key == "embedding") {
    // The tied head: FP8 for NVFP4 (as the published folder), its own format otherwise. The FP8 folder
    // keeps it BF16, so FP8 is the one target that does too.
    switch (target) {
      case LX_W_NVFP4: return LX_W_FP8;
      case LX_W_FP8: return LX_W_BF16;
      default: return target;
    }
  }
  if (ends_with_part(key, "mixer.impl.mixer.out_proj") || ends_with_part(key, "mixer.pass_lora_down") ||
      ends_with_part(key, "mixer.pass_lora_up") || ends_with_part(key, "depth_memory.key") ||
      ends_with_part(key, "depth_memory.value") || ends_with_part(key, "depth_memory.metadata_write"))
    return LX_W_BF16;
  switch (target) {
    case LX_W_BF16:
      return LX_W_BF16;
    case LX_W_INT8:
      return LX_W_INT8;
    case LX_W_FP8:
      if (expert || ends_with_part(key, "mixer.impl.mixer.in_proj") || ends_with_part(key, "mixer.qkv") ||
          ends_with_part(key, "mixer.out"))
        return LX_W_FP8;
      return LX_W_BF16;
    case LX_W_NVFP4:
      if (expert) return LX_W_NVFP4;
      if (ends_with_part(key, "moe.latent_up")) return LX_W_BF16;
      return LX_W_FP8;
    case LX_W_BF16X:
      return LX_W_BF16X;
    default:
      return LX_W_BF16;
  }
}

}  // namespace

// One projection of the folder: its tensors, the format the file holds it in, and the format it loads in.
struct Projection {
  std::string key;                     // "layers.4.mixer.qkv", "moe" stacks included; "embedding" for the head
  const TensorInfo* data = nullptr;    // .weight
  const TensorInfo* scale = nullptr;   // .weight_scale (quantized folders)
  const TensorInfo* global = nullptr;  // .weight_scale_2 (NVFP4: [E] per expert, or a scalar)
  lx_wtype src = LX_W_BF16;
  lx_wtype dst = LX_W_BF16;
  int64_t E = 1, N = 0, K = 0;  // a stack of E [N, K] matrices (E = 1 for a plain matrix)
  bool stack = false;
};

// Builds a ModelWeights from a folder. Friend of ModelWeights, so the loader can fill its members.
struct CheckpointBuilder {
  const SafetensorsSet& set;
  const MetisConfig& config;
  lx_wtype target;
  lx_wtype stored;
  ModelWeights& out;

  size_t live = 0, peak = 0;
  DeviceBuffer scratch_bf16, scratch_data, scratch_scale;

  CheckpointBuilder(const SafetensorsSet& s, const MetisConfig& c, lx_wtype t, lx_wtype st, ModelWeights& o)
      : set(s), config(c), target(t), stored(st), out(o) {}

  void grow(size_t bytes) {
    live += bytes;
    peak = std::max(peak, live);
  }

  DeviceBuffer device(size_t bytes) {
    grow(bytes);
    return DeviceBuffer(bytes);
  }

  // Collects the projections of the folder, with their shapes checked against the config.
  std::vector<Projection> projections() const {
    std::vector<Projection> list;
    for (const auto& [name, t] : set.tensors()) {
      std::string key;
      if (name == "embedding.weight") {
        key = "embedding";
      } else if (ends_with_part(name, "weight") && name.size() > 7) {
        key = name.substr(0, name.size() - 7);
      } else {
        continue;
      }
      if (contains(name, "ngram") || name == "lm_head.weight" || !is_projection_key(key)) continue;
      Projection p;
      p.key = key;
      p.data = &t;
      p.scale = set.find(key + ".weight_scale");
      p.global = set.find(key + ".weight_scale_2");
      p.src = format_of(t.dtype);
      if (t.shape.size() == 3) {
        p.stack = true;
        p.E = t.shape[0];
        p.N = t.shape[1];
      } else if (t.shape.size() == 2) {
        p.N = t.shape[0];
      } else {
        throw std::runtime_error("projection " + name + " has rank " + std::to_string(t.shape.size()));
      }
      p.K = p.src == LX_W_NVFP4 ? 2 * t.shape.back() : t.shape.back();
      if (p.src != LX_W_BF16 && !p.scale)
        throw std::runtime_error("projection " + name + " is quantized but has no weight_scale");
      if (p.src == LX_W_NVFP4 && !p.global) throw std::runtime_error("projection " + name + " has no weight_scale_2");
      p.dst = target == stored ? p.src : recipe(target, key);
      list.push_back(std::move(p));
    }
    return list;
  }

  void check_shapes(const std::vector<Projection>& list) const {
    const MetisConfig& c = config;
    for (const Projection& p : list) {
      if (p.key == "embedding" && (p.N != c.vocab_size || p.K != c.hidden_size))
        throw std::runtime_error("embedding is [" + std::to_string(p.N) + ", " + std::to_string(p.K) +
                                 "], the config says [" + std::to_string(c.vocab_size) + ", " +
                                 std::to_string(c.hidden_size) + "]");
      if (ends_with_part(p.key, "local_experts.gate_up") &&
          (p.E != c.n_routed_experts || p.N != 2 * c.expert_intermediate_size || p.K != c.latent_dim))
        throw std::runtime_error("expert gate_up in " + p.key + " does not match the config");
      if (ends_with_part(p.key, "local_experts.down") &&
          (p.E != c.n_routed_experts || p.N != c.latent_dim || p.K != c.expert_intermediate_size))
        throw std::runtime_error("expert down in " + p.key + " does not match the config");
    }
  }

  // The float values of a global scale (NVFP4): one per expert, or one scalar.
  std::vector<float> globals(const Projection& p) const {
    const size_t n = static_cast<size_t>(p.stack ? p.E : 1);
    std::vector<float> g(n, 1.0f);
    if (p.src == LX_W_NVFP4) {
      if (p.global->bytes < n * sizeof(float)) throw std::runtime_error("short weight_scale_2 for " + p.key);
      std::memcpy(g.data(), p.global->data, n * sizeof(float));
    }
    return g;
  }

  // Copies a host range into a device buffer.
  static void upload(void* dst, const void* src, size_t bytes, const std::string& what) {
    if (bytes) {
      check(lx_rt_copy(dst, src, static_cast<int64_t>(bytes), LX_COPY_H2D, nullptr), "uploading " + what);
      check(lx_rt_stream_sync(nullptr), "uploading " + what);
    }
  }

  // Expert e of a projection (or the whole matrix for a plain one), as bf16 at `dst` on the device.
  void to_bf16(const Projection& p, int64_t e, void* dst, float global) {
    const int64_t N = p.N, K = p.K;
    if (p.src == LX_W_BF16) {
      const auto* src = static_cast<const std::byte*>(p.data->data) + e * N * K * 2;
      upload(dst, src, static_cast<size_t>(N * K * 2), p.key);
      return;
    }
    const size_t data_bytes = static_cast<size_t>(N) * row_bytes(p.src, K);
    const size_t scale_bytes = static_cast<size_t>(N) * scale_row_bytes(p.src, K);
    upload(scratch_data.data(), static_cast<const std::byte*>(p.data->data) + e * static_cast<int64_t>(data_bytes),
           data_bytes, p.key);
    upload(scratch_scale.data(), static_cast<const std::byte*>(p.scale->data) + e * static_cast<int64_t>(scale_bytes),
           scale_bytes, p.key + ".weight_scale");
    lx_weight w{};
    w.type = p.src;
    w.data = scratch_data.data();
    w.scale = scratch_scale.data();
    w.global = global;
    check_lx(lx_dequantize_weight(&w, dst, N, static_cast<int>(K), nullptr), "dequantizing " + p.key);
  }

  // The scratch every conversion needs: one expert's bf16 values, and one expert's quantized bytes when
  // the file holds them quantized. Sized once, for the largest expert (or matrix) of the folder.
  void size_scratch(const std::vector<Projection>& list) {
    size_t bf16 = 0, data = 0, scale = 0;
    // The lossless target encodes a stack whole, so its scratch holds the largest stack; the others go an
    // expert at a time.
    for (const Projection& p : list) {
      if (p.key != "embedding")
        bf16 = std::max<size_t>(bf16, static_cast<size_t>(target == LX_W_BF16X ? p.E : 1) * p.N * p.K * 2);
      if (p.src == LX_W_BF16) continue;
      data = std::max(data, static_cast<size_t>(p.N) * row_bytes(p.src, p.K));
      scale = std::max(scale, static_cast<size_t>(p.N) * scale_row_bytes(p.src, p.K));
    }
    scratch_bf16 = device(bf16);
    scratch_data = device(data);
    scratch_scale = device(scale);
  }

  // Holds a projection in the format it is stored in: the file's bytes, copied.
  lx_weight as_is(const Projection& p) {
    const std::vector<float> g = globals(p);
    DeviceBuffer data = device(p.data->bytes);
    upload(data.data(), p.data->data, p.data->bytes, p.key);
    lx_weight w{};
    w.type = p.src;
    w.data = data.data();
    out.buffers_.push_back(std::move(data));
    if (p.scale) {
      DeviceBuffer s = device(p.scale->bytes);
      upload(s.data(), p.scale->data, p.scale->bytes, p.key + ".weight_scale");
      w.scale = s.data();
      out.buffers_.push_back(std::move(s));
    }
    w.global = g[0];
    if (p.src == LX_W_NVFP4 && p.stack) w.expert_global = upload_globals(g);
    return w;
  }

  // The per-expert globals on the device (NVFP4 stacks); a copy is kept with the buffers.
  const float* upload_globals(const std::vector<float>& g) {
    DeviceBuffer d = device(g.size() * sizeof(float));
    upload(d.data(), g.data(), g.size() * sizeof(float), "expert globals");
    const float* ptr = static_cast<const float*>(d.data());
    out.buffers_.push_back(std::move(d));
    return ptr;
  }

  // Converts a projection to its target format, one expert at a time. `bf16_source` (a device matrix)
  // replaces the file's bf16 values when they are already on the device, as the tied head's are.
  // The lossless target (LX_W_BF16X). A stack is one encoding of E * N rows: its rows come first and the
  // overflow records of every row follow them in one area (bf16x.cu), so the whole stack's bf16 values go
  // to the scratch first, and one lx_bf16x_encode writes it. The scratch holds the largest stack.
  lx_weight convert_lossless(const Projection& p, const void* bf16_source) {
    const int64_t rows = p.E * p.N;
    void* bf16 = const_cast<void*>(bf16_source);
    if (!bf16) {
      const std::vector<float> g = globals(p);
      for (int64_t e = 0; e < p.E; ++e) {
        auto* at = static_cast<std::byte*>(scratch_bf16.data()) + static_cast<size_t>(e * p.N * p.K) * 2;
        to_bf16(p, e, at, g[static_cast<size_t>(e)]);
      }
      bf16 = scratch_bf16.data();
    }
    const int64_t need = lx_bf16x_bytes(bf16, rows, static_cast<int>(p.K), nullptr);
    if (need < 0) throw std::runtime_error("lossless size of " + p.key + " failed");
    DeviceBuffer data = device(static_cast<size_t>(need));
    check_lx(lx_bf16x_encode(bf16, rows, static_cast<int>(p.K), data.data(), need, nullptr), "encoding " + p.key);
    lx_weight w{};
    w.type = LX_W_BF16X;
    w.data = data.data();
    w.global = 1.0f;
    out.buffers_.push_back(std::move(data));
    return w;
  }

  lx_weight convert(const Projection& p, const void* bf16_source) {
    if (p.dst == LX_W_BF16X) return convert_lossless(p, bf16_source);
    const size_t per_row = row_bytes(p.dst, p.K), per_scale = scale_row_bytes(p.dst, p.K);
    DeviceBuffer data = device(static_cast<size_t>(p.E * p.N) * per_row);
    DeviceBuffer scale = per_scale ? device(static_cast<size_t>(p.E * p.N) * per_scale) : DeviceBuffer();
    const std::vector<float> src_globals = globals(p);
    std::vector<float> dst_globals(static_cast<size_t>(p.E), 1.0f);
    auto* d = static_cast<std::byte*>(data.data());
    auto* s = static_cast<std::byte*>(scale.data());
    for (int64_t e = 0; e < p.E; ++e) {
      const void* bf16 = bf16_source;
      if (!bf16) {
        to_bf16(p, e, scratch_bf16.data(), src_globals[static_cast<size_t>(e)]);
        bf16 = scratch_bf16.data();
      }
      if (p.dst == LX_W_BF16) {
        check(lx_rt_copy(d + e * p.N * per_row, bf16, static_cast<int64_t>(p.N * p.K * 2), LX_COPY_D2D, nullptr),
              "copying " + p.key);
        check(lx_rt_stream_sync(nullptr), "copying " + p.key);
        continue;
      }
      lx_weight q{};
      q.type = p.dst;
      q.data = d + e * p.N * static_cast<int64_t>(per_row);
      q.scale = per_scale ? s + e * p.N * static_cast<int64_t>(per_scale) : nullptr;
      q.global = 1.0f;
      check_lx(lx_quantize_weight(bf16, &q, p.N, static_cast<int>(p.K), nullptr), "quantizing " + p.key);
      dst_globals[static_cast<size_t>(e)] = q.global;
    }
    lx_weight w{};
    w.type = p.dst;
    w.data = data.data();
    w.scale = scale.data();
    w.global = dst_globals[0];
    if (p.dst == LX_W_NVFP4 && p.stack) w.expert_global = upload_globals(dst_globals);
    out.buffers_.push_back(std::move(data));
    if (scale.size()) out.buffers_.push_back(std::move(scale));
    return w;
  }

  // The plain tensors (BF16 and F32, not projections), uploaded in one buffer, plus the gather table.
  void load_plain() {
    std::vector<const TensorInfo*> plain;
    for (const auto& [name, t] : set.tensors()) {
      if (is_scale_name(name) || contains(name, "ngram") || name == "lm_head.weight") continue;
      if (t.dtype != DType::BF16 && t.dtype != DType::F32) continue;
      // Projections are held by their own buffers, except the BF16 gather table.
      if (name != "embedding.weight" && ends_with_part(name, "weight") && name.size() > 7 &&
          is_projection_key(name.substr(0, name.size() - 7)))
        continue;
      plain.push_back(&t);
    }
    size_t total = 0;
    for (const TensorInfo* t : plain) total += align256(t->bytes);
    DeviceBuffer buffer = device(total);
    auto* base = static_cast<std::byte*>(buffer.data());
    size_t offset = 0;
    for (const TensorInfo* t : plain) {
      upload(base + offset, t->data, t->bytes, t->name);
      WeightView v;
      v.data = base + offset;
      v.dtype = t->dtype == DType::BF16 ? LX_BF16 : LX_F32;
      v.shape = t->shape;
      out.plain_.emplace(t->name, std::move(v));
      offset += align256(t->bytes);
    }
    out.buffers_.push_back(std::move(buffer));
  }

  void run() {
    std::vector<Projection> list = projections();
    check_shapes(list);
    // Name order, so the device layout does not depend on hash order. Taken before any pointer into it.
    std::sort(list.begin(), list.end(), [](const Projection& a, const Projection& b) { return a.key < b.key; });
    const Projection* head = nullptr;
    for (const Projection& p : list)
      if (p.key == "embedding") head = &p;
    if (!head) throw std::runtime_error("the folder has no embedding.weight");

    load_plain();
    size_scratch(list);
    // The gather table is BF16 on the device whatever the folder holds; the tied head comes from it.
    const void* gather = nullptr;
    if (head->src == LX_W_BF16) {
      gather = out.plain_.at("embedding.weight").data;
    } else {
      // A quantized folder: the table is the dequantized embedding, held in its own buffer.
      DeviceBuffer table = device(static_cast<size_t>(head->N * head->K * 2));
      to_bf16(*head, 0, table.data(), globals(*head)[0]);
      gather = table.data();
      out.plain_.emplace("embedding.weight", WeightView{table.data(), LX_BF16, {head->N, head->K}});
      out.buffers_.push_back(std::move(table));
    }

    for (const Projection& p : list) {
      if (p.key == "embedding") continue;
      out.projections_.emplace(p.key, p.dst == p.src ? as_is(p) : convert(p, nullptr));
      out.shapes_.emplace(p.key, ModelWeights::Shape{p.E, p.N, p.K, p.stack});
    }

    // The tied head: its format from the recipe (or the folder's own, when the target is the folder's).
    lx_weight h{};
    if (head->dst == LX_W_BF16) {
      h.type = LX_W_BF16;
      h.data = gather;
      h.global = 1.0f;
    } else if (head->dst == head->src) {
      h = as_is(*head);
    } else {
      // Converted from the gather table, which is already on the device as bf16.
      h = convert(*head, gather);
    }
    out.projections_.emplace("lm_head", h);
    out.shapes_.emplace("lm_head", ModelWeights::Shape{1, head->N, head->K, false});
    out.peak_bytes_ = peak;
  }
};

ModelWeights ModelWeights::load(const std::filesystem::path& dir, const MetisConfig& config,
                                const LoadOptions& options) {
  const SafetensorsSet set = SafetensorsSet::open(dir);
  const lx_wtype stored = stored_format(dir);
  ModelWeights out;
  out.format_ = options.target;
  CheckpointBuilder builder(set, config, options.target, stored, out);
  builder.run();
  return out;
}

lx_wtype ModelWeights::stored_format(const std::filesystem::path& dir) {
  const SafetensorsSet set = SafetensorsSet::open(dir);
  bool fp8 = false;
  for (const auto& [name, t] : set.tensors()) {
    if (t.dtype == DType::U8 && ends_with_part(name, "weight")) return LX_W_NVFP4;
    if (t.dtype == DType::F8_E4M3) fp8 = true;
  }
  return fp8 ? LX_W_FP8 : LX_W_BF16;
}

const lx_weight* ModelWeights::projection(const std::string& name) const {
  const auto it = projections_.find(projection_key(name));
  return it == projections_.end() ? nullptr : &it->second;
}

std::vector<std::string> ModelWeights::projection_names() const {
  std::vector<std::string> names;
  names.reserve(projections_.size());
  for (const auto& entry : projections_) names.push_back(entry.first);
  return names;
}

ModelWeights::Shape ModelWeights::projection_shape(const std::string& name) const {
  const auto it = shapes_.find(projection_key(name));
  if (it == shapes_.end()) throw std::runtime_error("no projection " + name);
  return it->second;
}

lx_weight ModelWeights::projection_expert(const std::string& name, int64_t e) const {
  const lx_weight* w = projection(name);
  if (!w) throw std::runtime_error("no projection " + name);
  const Shape s = projection_shape(name);
  if (e < 0 || e >= s.E) throw std::runtime_error("expert " + std::to_string(e) + " is out of range for " + name);
  lx_weight v = *w;
  v.data = static_cast<const std::byte*>(w->data) + static_cast<size_t>(e * s.N) * row_bytes(w->type, s.K);
  if (w->scale)
    v.scale = static_cast<const std::byte*>(w->scale) + static_cast<size_t>(e * s.N) * scale_row_bytes(w->type, s.K);
  v.expert_global = nullptr;
  if (w->expert_global) {
    float g = 1.0f;
    const std::string what = "reading the expert global of " + name;
    check(lx_rt_copy(&g, w->expert_global + e, static_cast<int64_t>(sizeof(float)), LX_COPY_D2H, nullptr), what);
    check(lx_rt_stream_sync(nullptr), what);
    v.global = g;
  }
  return v;
}

std::pair<size_t, size_t> ModelWeights::expert_bytes(const std::string& name) const {
  const lx_weight* w = projection(name);
  if (!w) throw std::runtime_error("no projection " + name);
  const Shape s = projection_shape(name);
  const size_t data = static_cast<size_t>(s.N) * row_bytes(w->type, s.K);
  const size_t scale = w->scale ? static_cast<size_t>(s.N) * scale_row_bytes(w->type, s.K) : 0;
  return {data, scale};
}

WeightView ModelWeights::find(const std::string& name) const {
  const auto it = plain_.find(name);
  return it == plain_.end() ? WeightView{} : it->second;
}

size_t ModelWeights::device_bytes() const noexcept {
  size_t total = 0;
  for (const DeviceBuffer& b : buffers_) total += b.size();
  return total;
}

}  // namespace lx
