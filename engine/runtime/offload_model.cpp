#include "runtime/offload_model.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <nlohmann/json.hpp>
#include <set>
#include <stdexcept>
#include <thread>

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

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() && s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

// layers.N.moe.local_experts.{gate_up,down}.weight[_scale]
bool expert_tensor(const std::string& name, int& layer, bool& gate_up, bool& scale) {
  if (name.rfind("layers.", 0) != 0) return false;
  const size_t dot = name.find('.', 7);
  const std::string rest = name.substr(dot + 1);
  for (const char* part : {"gate_up", "down"}) {
    for (const char* suffix : {".weight", ".weight_scale"}) {
      if (rest == std::string("moe.local_experts.") + part + suffix) {
        layer = std::stoi(name.substr(7, dot - 7));
        gate_up = std::strcmp(part, "gate_up") == 0;
        scale = std::strcmp(suffix, ".weight_scale") == 0;
        return true;
      }
    }
  }
  return false;
}

// The projections whose decode reads an int8 copy (metis.cpp builds one for every projection made with
// quantize = true unless LX_W8_KEEP names it; the default keeps out_proj): their bf16 weights only
// feed prefill GEMMs and the int8 quantization at load, so they can live in host memory.
bool prefill_only(const std::string& name) {
  static const char* suffixes[] = {"mixer.impl.mixer.in_proj.weight", "mixer.qkv.weight", "mixer.out.weight",
                                   "moe.latent_down.weight", "moe.latent_up.weight",
                                   "moe.shared_expert.gate_up.weight", "moe.shared_expert.down.weight",
                                   "depth_memory.state_write.weight", "depth_memory.query.weight",
                                   "depth_memory.output.weight", "depth_memory.route_projection.weight",
                                   "embedding.weight"};
  bool listed = false;
  for (const char* s : suffixes) listed |= ends_with(name, s);
  if (!listed) return false;
  const char* keep_env = std::getenv("LX_W8_KEEP");
  const std::string keep = keep_env ? keep_env : "out_proj";
  for (size_t pos = 0; pos < keep.size();) {
    const size_t end = keep.find(',', pos);
    const std::string frag = keep.substr(pos, end == std::string::npos ? std::string::npos : end - pos);
    if (!frag.empty() && name.find(frag) != std::string::npos) return false;
    pos = end == std::string::npos ? keep.size() : end + 1;
  }
  return true;
}

float e4m3(uint8_t b) {
  const int s = b >> 7, e = (b >> 3) & 15, m = b & 7;
  float v = e == 0 ? std::ldexp(static_cast<float>(m), -9) : std::ldexp(1.0f + m / 8.0f, e - 7);
  return s ? -v : v;
}

uint16_t to_bf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7FFFu + ((u >> 16) & 1u);
  return static_cast<uint16_t>(u >> 16);
}

// An FP8 projection [N, K] with per-row scales, dequantized to bf16 (the dense model runs in bf16 and
// its int8 decode copies; the experts keep FP8).
void dequant_fp8(const uint8_t* q, const float* scale, uint16_t* out, int64_t N, int64_t K) {
  float lut[256];
  for (int i = 0; i < 256; ++i) lut[i] = e4m3(static_cast<uint8_t>(i));
  const int threads = std::max(1u, std::min(16u, std::thread::hardware_concurrency()));
  std::vector<std::thread> pool;
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      for (int64_t n = t; n < N; n += threads)
        for (int64_t k = 0; k < K; ++k) out[n * K + k] = to_bf16(lut[q[n * K + k]] * scale[n]);
    });
  for (auto& th : pool) th.join();
}

int lx_dtype_of(DType d) { return d == DType::F32 ? LX_F32 : d == DType::BF16 ? LX_BF16 : -1; }

}  // namespace

std::unique_ptr<OffloadedModel> OffloadedModel::load(const std::filesystem::path& dir, const MetisConfig& c,
                                                     const OffloadLoadOptions& options) {
  std::unique_ptr<OffloadedModel> m(new OffloadedModel(SafetensorsSet::open(dir)));
  m->options_ = options;
  m->budget_ = options.vram_budget;
  check(lx_rt_stream_sync(nullptr), "sync");  // the CUDA context exists from here
  if (const int64_t ctx = process_vram_bytes(); ctx >= 0)
    m->notes_.push_back("this process before loading (the CUDA context): " + std::to_string(ctx >> 20) + " MiB");
  const auto& tensors = m->set_.tensors();
  if (!options.routing_stats.empty()) m->stats_ = RoutingStats::load(options.routing_stats);
  if (!m->stats_.empty() && (m->stats_.n_layers != c.num_hidden_layers || m->stats_.n_experts != c.n_routed_experts))
    throw std::runtime_error("routing stats do not match the model");

  // Dense tensors: where each goes, and its bytes there.
  struct Item {
    const TensorInfo* info;
    const TensorInfo* scale = nullptr;  // FP8 projections: dequantized to bf16
    bool host = false;
    size_t offset = 0, bytes = 0;
  };
  std::vector<Item> items;
  size_t vram_total = 0, host_total = 0;
  for (const auto& [name, t] : tensors) {
    int layer;
    bool gu, sc;
    if (expert_tensor(name, layer, gu, sc)) continue;
    if (ends_with(name, "_scale")) continue;  // consumed with its weight
    if (name == "lm_head.weight" && c.tie_word_embeddings) continue;
    Item it{&t};
    if (t.dtype == DType::F8_E4M3) {
      it.scale = m->set_.find(name + "_scale");
      if (!it.scale) throw std::runtime_error(name + " is FP8 without a scale");
      it.bytes = static_cast<size_t>(t.numel()) * 2;
    } else {
      it.bytes = t.bytes;
    }
    it.host = options.dense_prefill_on_host && options.w8_decode && prefill_only(name);
    size_t& total = it.host ? host_total : vram_total;
    it.offset = total;
    total += (it.bytes + 255) / 256 * 256;
    items.push_back(it);
  }
  void* vram = nullptr;
  check(lx_rt_alloc(&vram, static_cast<int64_t>(std::max<size_t>(vram_total, 256))), "device allocation for weights");
  m->device_blocks_.push_back(vram);
  void *host = nullptr, *host_dev = nullptr;
  if (host_total) {
    check(lx_rt_host_alloc_mapped(&host, &host_dev, static_cast<int64_t>(host_total)), "pinned allocation for weights");
    m->host_blocks_.push_back(host);
  }
  m->dense_vram_ = static_cast<int64_t>(vram_total);
  m->dense_host_ = static_cast<int64_t>(host_total);
  std::vector<uint16_t> deq;
  for (const Item& it : items) {
    const TensorInfo& t = *it.info;
    const void* src = t.data;
    if (it.scale) {
      const int64_t N = t.shape[0], K = t.numel() / N;
      deq.resize(static_cast<size_t>(t.numel()));
      dequant_fp8(reinterpret_cast<const uint8_t*>(t.data), reinterpret_cast<const float*>(it.scale->data), deq.data(),
                  N, K);
      src = deq.data();
    }
    WeightView v;
    v.dtype = it.scale ? LX_BF16 : lx_dtype_of(t.dtype);
    v.shape = t.shape;
    if (it.host) {
      std::memcpy(static_cast<char*>(host) + it.offset, src, it.bytes);
      v.data = static_cast<char*>(host_dev) + it.offset;
    } else {
      check(lx_rt_copy(static_cast<char*>(vram) + it.offset, src, static_cast<int64_t>(it.bytes), LX_COPY_H2D, nullptr),
            "upload");
      check(lx_rt_stream_sync(nullptr), "sync");
      v.data = static_cast<char*>(vram) + it.offset;
    }
    m->views_[t.name] = v;
  }

  // Experts.
  const std::string first = "layers.0.moe.local_experts.gate_up.weight";
  const TensorInfo* g0 = m->set_.find(first);
  if (!g0) throw std::runtime_error("no routed experts in the checkpoint");
  const lx_wtype stored = g0->dtype == DType::BF16 ? LX_W_BF16 : g0->dtype == DType::F8_E4M3 ? LX_W_FP8 : LX_W_NVFP4;
  if (stored == LX_W_NVFP4) throw std::runtime_error("offload: experts stored as " + std::string(dtype_name(g0->dtype)));
  const lx_wtype target = options.experts < 0 ? stored : static_cast<lx_wtype>(options.experts);
  if (target != stored && !(stored == LX_W_BF16 && target == LX_W_INT8))
    throw std::runtime_error("offload: experts convert only from BF16 to INT8");
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
    if (target == stored) {
      const TensorInfo* gus = stored == LX_W_FP8 ? m->set_.find(p + "gate_up.weight_scale") : nullptr;
      const TensorInfo* dns = stored == LX_W_FP8 ? m->set_.find(p + "down.weight_scale") : nullptr;
      if (stored == LX_W_FP8 && (!gus || !dns)) throw std::runtime_error("FP8 experts without scales");
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
      // BF16 -> INT8 on the GPU, the engine's own quantizer, one layer at a time
      void *d_gu = nullptr, *d_dn = nullptr, *q_gu = nullptr, *q_dn = nullptr, *s_gu = nullptr, *s_dn = nullptr;
      const int64_t gu_rows = static_cast<int64_t>(E) * 2 * I, dn_rows = static_cast<int64_t>(E) * L;
      check(lx_rt_alloc(&d_gu, static_cast<int64_t>(gu->bytes)), "alloc");
      check(lx_rt_alloc(&d_dn, static_cast<int64_t>(dn->bytes)), "alloc");
      check(lx_rt_alloc(&q_gu, gu_rows * L), "alloc");
      check(lx_rt_alloc(&q_dn, dn_rows * I), "alloc");
      check(lx_rt_alloc(&s_gu, gu_rows * 4), "alloc");
      check(lx_rt_alloc(&s_dn, dn_rows * 4), "alloc");
      check(lx_rt_copy(d_gu, gu->data, static_cast<int64_t>(gu->bytes), LX_COPY_H2D, nullptr), "copy");
      check(lx_rt_copy(d_dn, dn->data, static_cast<int64_t>(dn->bytes), LX_COPY_H2D, nullptr), "copy");
      check(lx_quantize_w8(d_gu, static_cast<int8_t*>(q_gu), static_cast<float*>(s_gu), gu_rows, L, nullptr), "quantize");
      check(lx_quantize_w8(d_dn, static_cast<int8_t*>(q_dn), static_cast<float*>(s_dn), dn_rows, I, nullptr), "quantize");
      std::vector<char> h_gu(gu_rows * L), h_dn(dn_rows * I);
      std::vector<float> h_sgu(gu_rows), h_sdn(dn_rows);
      check(lx_rt_copy(h_gu.data(), q_gu, gu_rows * L, LX_COPY_D2H, nullptr), "copy");
      check(lx_rt_copy(h_dn.data(), q_dn, dn_rows * I, LX_COPY_D2H, nullptr), "copy");
      check(lx_rt_copy(h_sgu.data(), s_gu, gu_rows * 4, LX_COPY_D2H, nullptr), "copy");
      check(lx_rt_copy(h_sdn.data(), s_dn, dn_rows * 4, LX_COPY_D2H, nullptr), "copy");
      check(lx_rt_stream_sync(nullptr), "sync");
      for (void* ptr : {d_gu, d_dn, q_gu, q_dn, s_gu, s_dn}) lx_rt_free(ptr);
      for (int e = 0; e < E; ++e) {
        ExpertSource s;
        s.gate_up = h_gu.data() + e * shape.gate_up_bytes();
        s.down = h_dn.data() + e * shape.down_bytes();
        s.gate_up_scale = h_sgu.data() + static_cast<int64_t>(e) * 2 * I;
        s.down_scale = h_sdn.data() + static_cast<int64_t>(e) * L;
        off.add_expert(l, e, s, DiskLocation{});
      }
    }
  }
  return m;
}

OffloadedModel::~OffloadedModel() {
  offload_.reset();
  for (void* p : device_blocks_) lx_rt_free(p);
  for (void* p : host_blocks_) lx_rt_host_free(p);
}

WeightLookup OffloadedModel::lookup() const {
  return [this](const std::string& name) {
    const auto it = views_.find(name);
    return it == views_.end() ? WeightView{} : it->second;
  };
}

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
  std::snprintf(buf, sizeof(buf), "dense weights: %.2f GB in VRAM, %.2f GB in pinned RAM (read by prefill only)\n",
                dense_vram_ / 1e9, dense_host_ / 1e9);
  s += buf;
  for (const auto& n : notes_) s += n + "\n";
  s += offload_->describe();
  return s;
}

}  // namespace lx::offload
