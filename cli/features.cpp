// lernex features: the target's hidden states over a tokenized corpus, for training draft models.
//
//   lernex features MODEL_DIR DATA_JSONL OUT_DIR --taps P:L,P:L,... [--epochs E] [--queue Q]
//                   [--shard-tokens N] [--max-len L] [--seed S] [--limit N] [--skip-seqs N]
//
// DATA_JSONL has one sequence per line: {"tokens": [ids], "mask": [0|1 per token]} (mask marks the
// tokens a drafter learns to predict: the responses). Each sequence runs through the model as a
// prefill with the feature taps P:L (MetisModel::set_feature_taps; L = -1 is the end of pass P),
// and its rows go to shards of about N tokens in OUT_DIR, in a shuffled order per epoch:
//
//   shard-EEE-NNNNNN.bin   "LXFS" | u32 version 1 | u32 sequences | u32 feature_dim | u64 tokens
//                          | u32 length[sequences] | i32 tokens[tokens] | u8 mask[tokens]
//                          | bf16 features[tokens][feature_dim]
//
// A shard appears whole (written as .tmp, then renamed). When Q shards are waiting, the command
// waits for the consumer (the trainer deletes each shard it has read), so a corpus far larger than
// the disk streams through a bounded window. DONE marks the end. --skip-seqs N resumes a stream:
// the first N sequences it would have written (over all epochs, same seed) are skipped. Run it with
// LX_NO_W8=1: prefill reads the bf16 weights, and the int8 decode copies would only take memory.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "core/config.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace {

namespace fs = std::filesystem;

struct Sequence {
  std::vector<int32_t> tokens;
  std::vector<uint8_t> mask;
};

std::vector<lx::FeatureTap> parse_taps(const std::string& s) {
  std::vector<lx::FeatureTap> taps;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');) {
    const size_t colon = item.find(':');
    if (colon == std::string::npos) throw std::runtime_error("a tap is PASS:LAYER, got " + item);
    taps.push_back({std::stoi(item.substr(0, colon)), std::stoi(item.substr(colon + 1))});
  }
  return taps;
}

size_t waiting_shards(const fs::path& dir) {
  size_t n = 0;
  for (const auto& e : fs::directory_iterator(dir))
    if (e.path().extension() == ".bin") ++n;
  return n;
}

// Device memory for `count` values of T.
template <typename T>
T* device_alloc(size_t count) {
  void* p = nullptr;
  const int64_t bytes = static_cast<int64_t>(sizeof(T) * count);
  const int status = lx_rt_alloc(&p, bytes);
  if (status != 0)
    throw std::runtime_error("device allocation of " + std::to_string(bytes) + " bytes: " + lx_rt_error_string(status));
  return static_cast<T*>(p);
}

}  // namespace

int cmd_features(int argc, char** argv) {
  if (argc < 4) {
    std::fprintf(stderr, "usage: lernex features MODEL_DIR DATA_JSONL OUT_DIR --taps P:L,... [options]\n");
    return 2;
  }
  try {
    const std::string dir = argv[1], data_path = argv[2];
    const fs::path out = argv[3];
    std::string taps_arg;
    int epochs = 1, queue = 4, max_len = 2048, limit = 0;
    int64_t shard_tokens = 32768, skip = 0;
    uint64_t seed = 0;
    for (int i = 4; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (a == "--taps") taps_arg = val();
      else if (a == "--epochs") epochs = std::stoi(val());
      else if (a == "--queue") queue = std::stoi(val());
      else if (a == "--shard-tokens") shard_tokens = std::stoll(val());
      else if (a == "--max-len") max_len = std::stoi(val());
      else if (a == "--seed") seed = std::stoull(val());
      else if (a == "--limit") limit = std::stoi(val());
      else if (a == "--skip-seqs") skip = std::stoll(val());
      else throw std::runtime_error("unknown option " + a);
    }
    const std::vector<lx::FeatureTap> taps = parse_taps(taps_arg);
    if (taps.empty()) throw std::runtime_error("--taps is required");

    std::vector<Sequence> data;
    std::ifstream in(data_path);
    for (std::string line; std::getline(in, line);) {
      if (line.empty()) continue;
      const auto j = nlohmann::json::parse(line);
      Sequence s;
      s.tokens = j.at("tokens").get<std::vector<int32_t>>();
      s.mask = j.contains("mask") ? j["mask"].get<std::vector<uint8_t>>() : std::vector<uint8_t>(s.tokens.size(), 1);
      if (s.tokens.size() > static_cast<size_t>(max_len)) {
        s.tokens.resize(max_len);
        s.mask.resize(max_len);
      }
      if (s.tokens.size() >= 2) data.push_back(std::move(s));
      if (limit > 0 && static_cast<int>(data.size()) >= limit) break;
    }
    if (data.empty()) throw std::runtime_error("no sequences in " + data_path);
    fs::create_directories(out);
    fs::remove(out / "DONE");

    const lx::MetisConfig config = lx::load_config(dir);
    const lx::SafetensorsSet set = lx::SafetensorsSet::open(dir);
    const lx::Weights weights = lx::Weights::upload(set);
    auto lookup = [&weights](const std::string& name) {
      lx::WeightView view;
      const lx::Tensor* t = weights.find(name);
      if (!t) return view;
      view.data = t->data;
      view.dtype = t->dtype == lx::DType::F32 ? LX_F32 : t->dtype == lx::DType::BF16 ? LX_BF16 : -1;
      view.shape = t->shape;
      return view;
    };
    lx::MetisModel model(config, lookup, 1, max_len, max_len);
    model.set_feature_taps(taps);
    const int F = model.feature_dim(), V = config.vocab_size;
    float* d_logits = device_alloc<float>(static_cast<size_t>(V));
    void* d_bf = device_alloc<uint16_t>(static_cast<size_t>(max_len) * F);
    void* stream = nullptr;
    lx_rt_stream_create(&stream);
    std::printf("%zu sequences, %d taps (%d features per token), %d epochs\n", data.size(), static_cast<int>(taps.size()),
                F, epochs);

    std::mt19937_64 rng(seed);
    int64_t done_tokens = 0;
    const auto t0 = std::chrono::steady_clock::now();
    for (int epoch = 0; epoch < epochs; ++epoch) {
      std::vector<size_t> order(data.size());
      for (size_t i = 0; i < order.size(); ++i) order[i] = i;
      std::shuffle(order.begin(), order.end(), rng);
      int shard = 0;
      std::vector<uint32_t> lengths;
      std::vector<int32_t> tok;
      std::vector<uint8_t> mask;
      std::vector<uint16_t> feat;
      auto flush = [&]() {
        if (lengths.empty()) return;
        while (static_cast<int>(waiting_shards(out)) >= queue) std::this_thread::sleep_for(std::chrono::milliseconds(200));
        char name[64];
        std::snprintf(name, sizeof(name), "shard-%03d-%06d", epoch, shard++);
        const fs::path tmp = out / (std::string(name) + ".tmp");
        std::ofstream f(tmp, std::ios::binary);
        const uint32_t version = 1, n = static_cast<uint32_t>(lengths.size()), fd = static_cast<uint32_t>(F);
        const uint64_t total = tok.size();
        f.write("LXFS", 4);
        f.write(reinterpret_cast<const char*>(&version), 4);
        f.write(reinterpret_cast<const char*>(&n), 4);
        f.write(reinterpret_cast<const char*>(&fd), 4);
        f.write(reinterpret_cast<const char*>(&total), 8);
        f.write(reinterpret_cast<const char*>(lengths.data()), 4 * lengths.size());
        f.write(reinterpret_cast<const char*>(tok.data()), 4 * tok.size());
        f.write(reinterpret_cast<const char*>(mask.data()), mask.size());
        f.write(reinterpret_cast<const char*>(feat.data()), 2 * feat.size());
        f.close();
        fs::rename(tmp, out / (std::string(name) + ".bin"));
        lengths.clear();
        tok.clear();
        mask.clear();
        feat.clear();
      };
      for (size_t idx : order) {
        if (skip > 0) {
          --skip;
          continue;
        }
        const Sequence& s = data[idx];
        const int T = static_cast<int>(s.tokens.size());
        model.reset_slot(0);
        model.forward({s.tokens, {0, T}, {0}}, d_logits, false, stream);
        if (lx_copy2d(model.features(), LX_F32, F, d_bf, LX_BF16, F, T, F, stream) != 0)
          throw std::runtime_error("lx_copy2d failed");
        const size_t at = feat.size();
        feat.resize(at + static_cast<size_t>(T) * F);
        lx_rt_copy(feat.data() + at, d_bf, static_cast<int64_t>(sizeof(uint16_t) * static_cast<size_t>(T) * F),
                   LX_COPY_D2H, stream);
        lx_rt_stream_sync(stream);
        lengths.push_back(static_cast<uint32_t>(T));
        tok.insert(tok.end(), s.tokens.begin(), s.tokens.end());
        mask.insert(mask.end(), s.mask.begin(), s.mask.end());
        done_tokens += T;
        if (static_cast<int64_t>(tok.size()) >= shard_tokens) flush();
      }
      flush();
      const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      std::printf("epoch %d done: %lld tokens, %.0f tok/s\n", epoch, static_cast<long long>(done_tokens), done_tokens / secs);
      std::fflush(stdout);
    }
    if (lx_rt_last_error() != 0) throw std::runtime_error("device error during the run");
    std::ofstream(out / "DONE") << "ok\n";
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
