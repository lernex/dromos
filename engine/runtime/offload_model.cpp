#include "runtime/offload_model.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>

namespace lx::offload {
namespace {

using json = nlohmann::json;

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + " failed with error " + std::to_string(status));
}

struct FileOffset {
  std::filesystem::path file;
  int64_t offset = -1;
};

// Absolute byte offsets of every tensor in the checkpoint's files (for the NVMe tier's O_DIRECT reads).
std::map<std::string, FileOffset> file_offsets(const std::filesystem::path& dir) {
  std::vector<std::filesystem::path> files;
  const auto index = dir / "model.safetensors.index.json";
  if (std::filesystem::exists(index)) {
    std::ifstream in(index);
    json j = json::parse(in);
    std::set<std::string> names;
    for (const auto& [tensor, shard] : j.at("weight_map").items()) names.insert(shard.get<std::string>());
    for (const auto& n : names) files.push_back(dir / n);
  } else {
    files.push_back(dir / "model.safetensors");
  }
  std::map<std::string, FileOffset> out;
  for (const auto& f : files) {
    std::ifstream in(f, std::ios::binary);
    uint64_t len = 0;
    in.read(reinterpret_cast<char*>(&len), 8);
    std::string text(len, '\0');
    in.read(text.data(), static_cast<std::streamsize>(len));
    const json h = json::parse(text);
    for (const auto& [name, entry] : h.items()) {
      if (name == "__metadata__") continue;
      out[name] = {std::filesystem::canonical(f), static_cast<int64_t>(8 + len + entry.at("data_offsets")[0].get<uint64_t>())};
    }
  }
  return out;
}

const char* format_name(lx_wtype t) {
  switch (t) {
    case LX_W_BF16: return "BF16";
    case LX_W_INT8: return "INT8";
    case LX_W_FP8: return "FP8";
    case LX_W_NVFP4: return "NVFP4";
    case LX_W_BF16X: return "BF16X";
  }
  return "?";
}

// The format an expert stack has in the file, from its dtype.
lx_wtype stored_format(DType d) {
  switch (d) {
    case DType::BF16: return LX_W_BF16;
    case DType::F8_E4M3: return LX_W_FP8;
    case DType::I8: return LX_W_INT8;
    case DType::U8: return LX_W_NVFP4;
    default: throw std::runtime_error("offload: experts stored as " + std::string(dtype_name(d)));
  }
}

void upload(void* dst, const void* src, size_t bytes) {
  check(lx_rt_copy(dst, src, static_cast<int64_t>(bytes), LX_COPY_H2D, nullptr), "upload");
}

// One expert stack of the checkpoint ([rows, cols]: E experts' rows; per-row f32 scales for INT8 / FP8) in
// another format, on the host: through the GPU, dequantized to bf16 exactly when it is stored quantized, then
// rounded to `to` with the engine's quantizer (lx_quantize_weight, as ModelWeights converts).
struct HostStack {
  std::vector<char> data;
  std::vector<float> scale;  // [rows] for INT8 / FP8
};
HostStack convert_stack(const TensorInfo& w, const TensorInfo* scale, lx_wtype from, lx_wtype to, int64_t rows,
                        int cols) {
  const size_t n = static_cast<size_t>(rows) * cols;
  DeviceBuffer src(w.bytes), src_scale, bf16, dst, dst_scale;
  upload(src.data(), w.data, w.bytes);
  const void* values = src.data();
  if (from != LX_W_BF16) {
    if (!scale || scale->bytes != static_cast<size_t>(rows) * 4)
      throw std::runtime_error("offload: " + w.name + " needs one f32 scale per row");
    src_scale = DeviceBuffer(scale->bytes);
    upload(src_scale.data(), scale->data, scale->bytes);
    bf16 = DeviceBuffer(n * 2);
    const lx_weight q{from, src.data(), src_scale.data(), 1.0f, nullptr};
    check(lx_dequantize_weight(&q, bf16.data(), rows, cols, nullptr), "dequantizing " + w.name);
    values = bf16.data();
  }
  HostStack out;
  if (to == LX_W_BF16) {
    out.data.resize(n * 2);
    check(lx_rt_copy(out.data.data(), values, static_cast<int64_t>(n * 2), LX_COPY_D2H, nullptr), "copy");
  } else {
    dst = DeviceBuffer(n);
    dst_scale = DeviceBuffer(static_cast<size_t>(rows) * 4);
    lx_weight q{to, dst.data(), dst_scale.data(), 1.0f, nullptr};
    check(lx_quantize_weight(values, &q, rows, cols, nullptr), "quantizing " + w.name);
    out.data.resize(n);
    out.scale.resize(static_cast<size_t>(rows));
    check(lx_rt_copy(out.data.data(), dst.data(), static_cast<int64_t>(n), LX_COPY_D2H, nullptr), "copy");
    check(lx_rt_copy(out.scale.data(), dst_scale.data(), rows * 4, LX_COPY_D2H, nullptr), "copy");
  }
  check(lx_rt_stream_sync(nullptr), "sync");
  return out;
}

}  // namespace

std::unique_ptr<OffloadedModel> OffloadedModel::load(const std::filesystem::path& dir, const MetisConfig& c,
                                                     const OffloadLoadOptions& options) {
  std::unique_ptr<OffloadedModel> m(new OffloadedModel(SafetensorsSet::open(dir)));
  m->options_ = options;
  m->budget_ = options.vram_budget;
  check(lx_rt_stream_sync(nullptr), "sync");  // the CUDA context exists from here
  if (const int64_t ctx = process_vram_bytes(); ctx >= 0)
    m->notes_.push_back("this process before loading (the CUDA context): " + std::to_string(ctx >> 20) + " MiB");
  if (!options.routing_stats.empty()) m->stats_ = RoutingStats::load(options.routing_stats);
  if (!m->stats_.empty() && (m->stats_.n_layers != c.num_hidden_layers || m->stats_.n_experts != c.n_routed_experts))
    throw std::runtime_error("routing stats do not match the model");

  // Dense weights: ModelWeights in the plan's format, the routed experts left out (they live in the offload).
  LoadOptions dense;
  dense.target = options.weights < 0 ? ModelWeights::stored_format(dir) : static_cast<lx_wtype>(options.weights);
  dense.skip_experts = true;
  m->dense_ = ModelWeights::load(dir, c, dense);
  m->dense_vram_ = static_cast<int64_t>(m->dense_.device_bytes());
  m->notes_.push_back(std::string("dense weights: ") + format_name(dense.target));

  // Experts, in the plan's format: the offload kernels read BF16, INT8 and FP8 (an expert with its per-row scales).
  const std::string first = "layers.0.moe.local_experts.gate_up.weight";
  const TensorInfo* g0 = m->set_.find(first);
  if (!g0) throw std::runtime_error("no routed experts in the checkpoint");
  const lx_wtype stored = stored_format(g0->dtype);
  const lx_wtype target = options.experts < 0 ? stored : static_cast<lx_wtype>(options.experts);
  if (target != LX_W_BF16 && target != LX_W_INT8 && target != LX_W_FP8)
    throw std::runtime_error(std::string("offload: the experts cannot run as ") + format_name(target) +
                             " (the offload kernels read BF16, INT8 and FP8)");
  if (target != stored && stored == LX_W_NVFP4)
    throw std::runtime_error("offload: NVFP4 experts cannot be converted (their per-block scales do not survive)");
  ExpertShape shape;
  shape.n_layers = c.num_hidden_layers;
  shape.n_experts = c.n_routed_experts;
  shape.top_k = c.num_experts_per_tok;
  shape.L = c.latent_dim;
  shape.I = c.expert_intermediate_size;
  shape.type = target;
  OffloadOptions policy = options.policy;
  policy.ram_bytes = options.ram_budget;
  m->offload_ = std::make_unique<ExpertOffload>(shape, policy);
  ExpertOffload& off = *m->offload_;
  const int64_t all = static_cast<int64_t>(shape.n_layers) * shape.n_experts * shape.expert_bytes();
  if (target != stored && options.ram_budget >= 0 && options.ram_budget < all)
    throw std::runtime_error("offload: converted experts must all fit in RAM (the NVMe tier reads the stored format)");
  off.plan_host(m->stats_);
  const auto offsets = file_offsets(dir);
  const int E = shape.n_experts, I = shape.I, L = shape.L;
  for (int l = 0; l < shape.n_layers; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".moe.local_experts.";
    const TensorInfo* gu = m->set_.find(p + "gate_up.weight");
    const TensorInfo* dn = m->set_.find(p + "down.weight");
    if (!gu || !dn) throw std::runtime_error("missing experts of layer " + std::to_string(l));
    if (gu->shape != std::vector<int64_t>{E, 2 * I, L} || dn->shape != std::vector<int64_t>{E, L, I})
      throw std::runtime_error("experts of layer " + std::to_string(l) + " have an unexpected shape");
    const bool scaled = stored == LX_W_INT8 || stored == LX_W_FP8;
    const TensorInfo* gus = scaled ? m->set_.find(p + "gate_up.weight_scale") : nullptr;
    const TensorInfo* dns = scaled ? m->set_.find(p + "down.weight_scale") : nullptr;
    if (scaled && (!gus || !dns))
      throw std::runtime_error("quantized experts without scales in layer " + std::to_string(l));
    if (target == stored) {
      for (int e = 0; e < E; ++e) {
        ExpertSource s;
        s.gate_up = reinterpret_cast<const char*>(gu->data) + e * shape.gate_up_bytes();
        s.down = reinterpret_cast<const char*>(dn->data) + e * shape.down_bytes();
        DiskLocation d;
        const FileOffset& fg = offsets.at(gu->name);
        d.file = fg.file;
        d.gate_up = fg.offset + e * shape.gate_up_bytes();
        d.down = offsets.at(dn->name).offset + e * shape.down_bytes();
        if (offsets.at(dn->name).file != fg.file) d.file.clear();  // stacks in different shards: no NVMe tier
        if (gus) {
          s.gate_up_scale = reinterpret_cast<const char*>(gus->data) + e * shape.gate_up_scale_bytes();
          s.down_scale = reinterpret_cast<const char*>(dns->data) + e * shape.down_scale_bytes();
          d.gate_up_scale = offsets.at(gus->name).offset + e * shape.gate_up_scale_bytes();
          d.down_scale = offsets.at(dns->name).offset + e * shape.down_scale_bytes();
          if (offsets.at(gus->name).file != fg.file || offsets.at(dns->name).file != fg.file) d.file.clear();
        }
        off.add_expert(l, e, s, d);
      }
    } else {
      // Another format: converted on the GPU one stack at a time, kept on the host (no NVMe tier: the drive
      // holds the stored format)
      const int64_t gu_rows = static_cast<int64_t>(E) * 2 * I, dn_rows = static_cast<int64_t>(E) * L;
      const HostStack hg = convert_stack(*gu, gus, stored, target, gu_rows, L);
      const HostStack hd = convert_stack(*dn, dns, stored, target, dn_rows, I);
      for (int e = 0; e < E; ++e) {
        ExpertSource s;
        s.gate_up = hg.data.data() + e * shape.gate_up_bytes();
        s.down = hd.data.data() + e * shape.down_bytes();
        if (!hg.scale.empty()) {
          s.gate_up_scale = hg.scale.data() + static_cast<int64_t>(e) * 2 * I;
          s.down_scale = hd.scale.data() + static_cast<int64_t>(e) * L;
        }
        off.add_expert(l, e, s, DiskLocation{});
      }
    }
  }
  return m;
}

OffloadedModel::~OffloadedModel() { offload_.reset(); }

void OffloadedModel::finalize() {
  int64_t free_b = 0, total_b = 0;
  check(lx_rt_stream_sync(nullptr), "sync");
  check(lx_rt_mem_info(&free_b, &total_b), "lx_rt_mem_info");
  const int64_t budget = budget_ > 0 ? std::min(budget_, total_b) : total_b;
  // What this process holds now (the context, the dense weights, the model's scratch and caches),
  // and a margin for what is allocated lazily (cuBLASLt workspaces, graph memory). Without NVML the
  // whole card's use stands in for it, which charges other processes to this budget too.
  const int64_t mine = process_vram_bytes();
  const int64_t used = mine >= 0 ? mine : total_b - free_b;
  const int64_t margin = static_cast<int64_t>(std::getenv("LX_OFFLOAD_MARGIN_MB") ? std::atof(std::getenv("LX_OFFLOAD_MARGIN_MB")) : 384) << 20;
  int64_t cache = std::max<int64_t>(0, std::min(budget - used, free_b) - margin);
  const ExpertShape& s = offload_->shape();
  const int64_t all = static_cast<int64_t>(s.n_layers) * s.n_experts * s.expert_bytes();
  cache = std::min(cache, all);
  notes_.push_back("VRAM budget " + std::to_string(budget >> 20) + " MiB, in use before the expert cache " +
                   std::to_string(used >> 20) + " MiB (" + (mine >= 0 ? "this process" : "the whole card") +
                   "), expert cache " + std::to_string(cache >> 20) + " MiB");
  offload_->finalize(stats_, cache);
}

std::string OffloadedModel::describe() const {
  std::string s;
  char buf[256];
  std::snprintf(buf, sizeof(buf), "dense weights: %.2f GB in VRAM\n", dense_vram_ / 1e9);
  s += buf;
  for (const auto& n : notes_) s += n + "\n";
  s += offload_->describe();
  return s;
}

}  // namespace lx::offload
