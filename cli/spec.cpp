// dromos spec: speculative decoding against plain decoding, per prompt and per drafter.
//
//   dromos spec MODEL_DIR PROMPTS_JSON [options]
//     --tokens N        tokens to generate per prompt (256)
//     --k N             most drafts per step (6; clamped to the model's verify limit)
//     --drafters LIST   comma list of plain, ngram, ngram-fixed, self, model, block, block-fixed
//                       (plain,ngram). ngram and block pick each draft's length from the verify cost
//                       (block from its confidence head); the -fixed ones always draft the most
//     --draft-head DIR  the lernex.block/1 drafter for `block` (tools/draft/train_block.py)
//     --bf16-draft      the block drafter reads bf16 weights (its own, and the target's bf16 embedding
//                       as its head) instead of int8 copies of its own and the head as the target holds it
//     --cost LIST       verify cost of T = 1, 2, .. tokens in plain steps (default: the 5070 Ti, Praxis)
//     --draft DIR       the draft model for `model` (a Metis model with the same tokenizer)
//     --temp T --top-p P --top-k K --min-p P --seed S    sampling (temp 0, the default, is greedy)
//     --ngram MAX,MIN   prompt-lookup match lengths (4,1)
//     --min-conf C      a model drafter stops drafting below this probability of its proposal (0)
//     --json FILE       every number, per prompt and drafter
//     --repeat R        time each run R times and report the median (1)
//   Under greedy decoding every speculative run must emit exactly the plain run's tokens; the
//   command reports the first divergence if one does not.
//
//   dromos spec-cost MODEL_DIR [CONTEXT]
//     Milliseconds per plain step and per verify step of each size, after a CONTEXT-token prompt.
//
//   dromos block-check MODEL_DIR DRAFT_HEAD_DIR PROMPTS_JSON OUT_DIR
//     For each prompt: the target prefills all but its last token, the block drafter proposes a
//     block anchored on that token (greedy, bf16 weights), and OUT_DIR gets what
//     tools/draft/check_block.py needs to recompute it in PyTorch: the feature rows of the window,
//     the anchor, and the drafter's logits and confidences.
//
//   dromos spec-check MODEL_DIR PROMPTS_JSON [N]
//     Feeds each prompt's second half one token at a time, then again as verify steps of random
//     sizes with random rollbacks (and an occasional normal step, which folds), and requires every
//     verify row's logits to be bit-identical to the one-token step at the same position.
//
//   spec, spec-cost and spec-check take --weights bf16|int8|fp8|nvfp4|bf16x and --no-bf16x: the target model loads
//   in the folder's format unless --weights names another (cli/common.h, load_weights).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/common.h"
#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"
#include "runtime/block_drafter.h"
#include "runtime/engine.h"
#include "runtime/offload_model.h"
#include "runtime/speculative.h"

namespace {

struct Loaded {
  lx::MetisConfig config;
  std::unique_ptr<lx::ModelWeights> weights;
  std::unique_ptr<lx::offload::OffloadedModel> offloaded;  // offloaded models: the dense weights and the experts
  std::unique_ptr<lx::MetisModel> model;
};

// The model in the folder's format, or the one the flags name, with the plan's maths for it on this GPU. With `offload`,
// a model that does not fit the GPU runs as the engine runs it: its routed experts in the offload runtime, under the
// plan's budgets, and the verify and one-token steps read the same expert copies.
Loaded load(const std::string& dir, int slots, int max_tokens, const lx::cli::FormatFlags& format = {},
            bool offload = false) {
  Loaded l;
  l.config = lx::load_config(dir);
  if (offload) {
    const lx::runtime::Plan plan =
        lx::cli::plan_for(lx::cli::find_model(dir, false), slots, 8192, format.weights, 0.92, format.bf16x,
                                          format.fast_math);
    if (plan.offload.enabled) {
      l.offloaded = lx::offload::OffloadedModel::load(dir, l.config, lx::runtime::offload_options(plan));
      l.model = std::make_unique<lx::MetisModel>(l.config, l.offloaded->weights(), plan.formats(), slots, 8192,
                                                 max_tokens, &l.offloaded->experts(), true);
      l.offloaded->finalize();
      std::fprintf(stderr, "%s", l.offloaded->describe().c_str());
      return l;
    }
  }
  lx::cli::LoadedWeights w = lx::cli::load_weights(dir, l.config, format);
  std::fprintf(stderr, "%s\n", w.description.c_str());
  l.weights = std::move(w.weights);
  l.model = std::make_unique<lx::MetisModel>(l.config, *l.weights, w.formats, slots, 8192, max_tokens, nullptr, true);
  return l;
}

// argv without the format flags (which go to `format`), for the commands that read their arguments by position.
std::vector<char*> without_format_flags(int argc, char** argv, lx::cli::FormatFlags& format) {
  std::vector<char*> out;
  for (int i = 0; i < argc; ++i)
    if (i == 0 || !lx::cli::parse_format_flag(argc, argv, i, format)) out.push_back(argv[i]);
  return out;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty()) out.push_back(item);
  return out;
}

int run_check(int argc, char** argv, const lx::cli::FormatFlags& format) {
  const std::string dir = argv[1];
  nlohmann::json prompts;
  std::ifstream(argv[2]) >> prompts;
  const int K = argc >= 4 ? std::stoi(argv[3]) : 96;
  Loaded l = load(dir, 1, 512, format, true);
  lx::MetisModel& model = *l.model;
  const int V = model.vocab_size(), maxv = model.max_verify_tokens();
  float* d_logits = nullptr;
  const int64_t logit_bytes = static_cast<int64_t>(sizeof(float) * 512 * V);
  if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
    throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
  void* s = nullptr;
  if (lx_rt_stream_create(&s) != 0) throw std::runtime_error("creating a device stream");
  std::mt19937 rng(1234);
  bool all_ok = true;
  for (auto& [name, ids_json] : prompts.items()) {
    const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
    const int N = std::min<int>(static_cast<int>(ids.size()) / 2, 512);
    const int n = std::min<int>(K, static_cast<int>(ids.size()) - N);
    std::vector<int32_t> prompt(ids.begin(), ids.begin() + N);
    auto prefill = [&]() {
      model.reset_slot(0);
      for (int off = 0; off < N; off += 512) {
        const int c = std::min(512, N - off);
        model.forward({{prompt.begin() + off, prompt.begin() + off + c}, {0, c}, {0}}, d_logits, false, s);
      }
    };
    // one token at a time
    prefill();
    std::vector<float> plain(static_cast<size_t>(n) * V);
    for (int t = 0; t < n; ++t) {
      model.forward({{ids[N + t]}, {0, 1}, {0}}, d_logits, false, s);
      lx_rt_copy(plain.data() + static_cast<size_t>(t) * V, d_logits, static_cast<int64_t>(sizeof(float)) * V,
                 LX_COPY_D2H, s);
    }
    lx_rt_stream_sync(s);
    // verify steps with rollbacks
    prefill();
    std::vector<float> rows(static_cast<size_t>(maxv) * V);
    int pos = 0, compared = 0, identical = 0, steps = 0, folds = 0;
    double worst = 0;
    while (pos < n) {
      const bool normal = rng() % 7 == 0;  // a normal step folds the pending positions first
      const int g = normal ? 1 : std::min<int>(1 + rng() % maxv, n - pos);
      const int before = model.slot_length(0);
      lx::StepBatch b{{ids.begin() + N + pos, ids.begin() + N + pos + g}, {0, g}, {0}};
      if (normal) {
        model.forward(b, d_logits, true, s);
        ++folds;
      } else {
        model.forward_spec(b, d_logits, s);
      }
      lx_rt_copy(rows.data(), d_logits, static_cast<int64_t>(sizeof(float)) * g * V, LX_COPY_D2H, s);
      lx_rt_stream_sync(s);
      for (int r = 0; r < g; ++r) {
        const float* a = rows.data() + static_cast<size_t>(r) * V;
        const float* e = plain.data() + static_cast<size_t>(pos + r) * V;
        ++compared;
        if (std::memcmp(a, e, sizeof(float) * V) == 0) {
          ++identical;
        } else {
          for (int v = 0; v < V; ++v) worst = std::max(worst, static_cast<double>(std::abs(a[v] - e[v])));
        }
      }
      const int keep = normal ? 1 : 1 + rng() % g;
      if (!normal) model.truncate(0, before + keep);
      pos += keep;
      ++steps;
    }
    if (const int err = lx_rt_last_error())
      throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the check");
    const bool ok = identical == compared;
    all_ok &= ok;
    std::printf("%-22s prefill %4d, %3d positions in %3d steps (%d normal): %d of %d rows bit-identical%s",
                name.c_str(), N, n, steps, folds, identical, compared, ok ? "\n" : "");
    if (!ok) std::printf(", max |diff| %.3g\n", worst);
  }
  lx_rt_free(d_logits);
  return all_ok ? 0 : 1;
}

// Milliseconds per step: plain one-token steps, and verify steps of T = 1 .. max tokens, each
// followed by the host sync and the rollback a real step does (one token kept, so folds happen at
// their real rate).
int run_cost(int argc, char** argv, const lx::cli::FormatFlags& format) {
  const std::string dir = argv[1];
  const int context = argc >= 3 ? std::stoi(argv[2]) : 1024;
  Loaded l = load(dir, 1, 512, format);
  lx::MetisModel& model = *l.model;
  const int V = model.vocab_size(), maxv = model.max_verify_tokens();
  float* d_logits = nullptr;
  const int64_t logit_bytes = static_cast<int64_t>(sizeof(float) * 512 * V);
  if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
    throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
  void* s = nullptr;
  if (lx_rt_stream_create(&s) != 0) throw std::runtime_error("creating a device stream");
  std::mt19937 rng(7);
  std::vector<int32_t> ids(context + 4096);
  for (auto& t : ids) t = 100 + static_cast<int32_t>(rng() % 30000);
  auto prefill = [&]() {
    model.reset_slot(0);
    for (int off = 0; off < context; off += 512) {
      const int c = std::min(512, context - off);
      model.forward({{ids.begin() + off, ids.begin() + off + c}, {0, c}, {0}}, d_logits, false, s);
    }
  };
  const int iters = 40;
  auto time_steps = [&](int T, bool spec) {
    prefill();
    int pos = context;
    double ms = 0;
    for (int it = 0; it < iters + 5; ++it) {
      lx_rt_stream_sync(s);
      const auto t0 = std::chrono::steady_clock::now();
      lx::StepBatch b{{ids.begin() + pos, ids.begin() + pos + T}, {0, T}, {0}};
      const int before = model.slot_length(0);
      if (spec) {
        model.forward_spec(b, d_logits, s);
      } else {
        model.forward(b, d_logits, false, s);
      }
      lx_rt_stream_sync(s);
      if (spec) model.truncate(0, before + 1);
      if (it >= 5) ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
      pos += 1;
    }
    return ms / iters;
  };
  // Rounds over every shape, medians across rounds: the server is shared, so any one round can be
  // slowed by someone else's work.
  const int rounds = argc >= 4 ? std::stoi(argv[3]) : 7;
  std::vector<std::vector<double>> samples(maxv + 1);
  for (int r = 0; r < rounds; ++r) {
    samples[0].push_back(time_steps(1, false));
    for (int T = 1; T <= maxv; ++T) samples[T].push_back(time_steps(T, true));
  }
  auto median = [](std::vector<double> v) {
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
  };
  const double plain = median(samples[0]);
  std::printf("context %d, median of %d rounds: plain one-token step %.3f ms\n", context, rounds, plain);
  for (int T = 1; T <= maxv; ++T) {
    const double t = median(samples[T]);
    std::printf("  verify step T=%d  %.3f ms  (x%.3f of a plain step)\n", T, t, t / plain);
  }
  lx_rt_free(d_logits);
  return 0;
}

struct Run {
  std::string drafter;
  std::vector<int32_t> tokens;
  lx::runtime::SpecStats stats;
  double propose_ms = 0;  // block drafter: milliseconds per proposal
  int proposals = 0;      // block drafter: steps that ran it
};

}  // namespace

int cmd_spec_cost(int argc, char** argv) {
  try {
    lx::cli::FormatFlags format;
    std::vector<char*> args = without_format_flags(argc, argv, format);
    const int n = static_cast<int>(args.size());
    if (n < 2 || n > 4) {
      std::fprintf(stderr, "usage: dromos spec-cost MODEL_DIR [CONTEXT [ROUNDS]] %s\n", lx::cli::kFormatUsage);
      return 2;
    }
    return run_cost(n, args.data(), format);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

int cmd_spec_check(int argc, char** argv) {
  try {
    lx::cli::FormatFlags format;
    std::vector<char*> args = without_format_flags(argc, argv, format);
    const int n = static_cast<int>(args.size());
    if (n != 3 && n != 4) {
      std::fprintf(stderr, "usage: dromos spec-check MODEL_DIR PROMPTS_JSON [N] %s\n", lx::cli::kFormatUsage);
      return 2;
    }
    return run_check(n, args.data(), format);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

int cmd_block_check(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: dromos block-check MODEL_DIR DRAFT_HEAD_DIR PROMPTS_JSON OUT_DIR\n");
    return 2;
  }
  try {
    nlohmann::json prompts;
    std::ifstream(argv[3]) >> prompts;
    const std::filesystem::path out = argv[4];
    std::filesystem::create_directories(out);
    Loaded l = load(argv[1], 1, 512);
    lx::MetisModel& model = *l.model;
    const lx_spec_filter greedy{0.0f, 1.0f, 0, 0.0f};
    lx::runtime::BlockDrafter drafter(argv[2], model, greedy, 0, lx::runtime::DraftCost::rtx5070ti_praxis(), true);
    drafter.set_adaptive(false);
    model.set_feature_taps(drafter.feature_taps());
    const int F = model.feature_dim(), V = model.vocab_size(), rows = drafter.block_size() - 1;
    nlohmann::json head;
    std::ifstream(std::filesystem::path(argv[2]) / "config.json") >> head;
    const int W = head.at("window");
    float* d_logits = nullptr;
    const int64_t logit_bytes = static_cast<int64_t>(sizeof(float) * V);
    if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
      throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
    void* s = nullptr;
    if (lx_rt_stream_create(&s) != 0) throw std::runtime_error("creating a device stream");
    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
      const int P = static_cast<int>(ids.size()) - 1;  // the anchor's position
      model.reset_slot(0);
      drafter.begin(ids, s);
      // keep the window's feature rows as the drafter saw them
      std::vector<float> window;
      const int first = std::max(0, P - W);
      for (int off = 0; off < P; off += 512) {
        const int n = std::min(512, P - off);
        model.forward({{ids.begin() + off, ids.begin() + off + n}, {0, n}, {0}}, d_logits, false, s);
        drafter.observe_features(model.features(), n, F, s);
        for (int r = std::max(first, off); r < off + n; ++r) {
          const size_t at = window.size();
          window.resize(at + F);
          lx_rt_copy(window.data() + at, model.features() + static_cast<size_t>(r - off) * F,
                     static_cast<int64_t>(sizeof(float)) * F, LX_COPY_D2H, s);
        }
        lx_rt_stream_sync(s);
      }
      std::vector<int32_t> context(ids.begin(), ids.end());
      lx::runtime::DraftRequest req;
      req.context = &context;
      req.max_drafts = rows;
      req.stream = s;
      drafter.propose(req);
      std::vector<float> logits(static_cast<size_t>(rows) * V);
      if (lx_rt_copy(logits.data(), drafter.last_logits(), static_cast<int64_t>(sizeof(float) * logits.size()),
                     LX_COPY_D2H, nullptr) != 0)
        throw std::runtime_error("reading the block drafter's logits");
      if (lx_rt_stream_sync(nullptr) != 0) throw std::runtime_error("reading the block drafter's logits");
      std::ofstream(out / (name + ".features.f32"), std::ios::binary)
          .write(reinterpret_cast<const char*>(window.data()), sizeof(float) * window.size());
      std::ofstream(out / (name + ".logits.f32"), std::ios::binary)
          .write(reinterpret_cast<const char*>(logits.data()), sizeof(float) * logits.size());
      nlohmann::json meta{{"anchor", ids.back()},     {"position", P},
                          {"first", first},           {"feature_dim", F},
                          {"rows", rows},             {"tokens", drafter.last_tokens()},
                          {"confidence", drafter.last_confidence()}};
      std::ofstream(out / (name + ".json")) << meta.dump() << "\n";
      std::printf("%-22s anchor at %5d, window %3d rows, drafts", name.c_str(), P, P - first);
      for (int32_t t : drafter.last_tokens()) std::printf(" %d", t);
      std::printf("\n");
    }
    lx_rt_free(d_logits);
    if (const int err = lx_rt_last_error())
      throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the check");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

int cmd_spec(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr, "usage: dromos spec MODEL_DIR PROMPTS_JSON [options] (see cli/spec.cpp)\n");
    return 2;
  }
  try {
    const std::string dir = argv[1];
    nlohmann::json prompts;
    std::ifstream(argv[2]) >> prompts;
    lx::runtime::SpecConfig cfg;
    cfg.max_new_tokens = 256;
    std::vector<std::string> drafters{"plain", "ngram"};
    std::string draft_dir, json_out, head_dir;
    bool bf16_head = false;
    int ngram_max = 4, ngram_min = 1;
    float min_conf = 0.0f;
    int repeat = 1;
    std::string cost_list;
    lx::cli::FormatFlags format;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (lx::cli::parse_format_flag(argc, argv, i, format)) continue;
      if (a == "--tokens") cfg.max_new_tokens = std::stoi(val());
      else if (a == "--k") cfg.max_drafts = std::stoi(val());
      else if (a == "--drafters") drafters = split(val(), ',');
      else if (a == "--draft") draft_dir = val();
      else if (a == "--draft-head") head_dir = val();
      else if (a == "--bf16-draft") bf16_head = true;
      else if (a == "--temp") cfg.filter.temperature = std::stof(val());
      else if (a == "--top-p") cfg.filter.top_p = std::stof(val());
      else if (a == "--top-k") cfg.filter.top_k = std::stoi(val());
      else if (a == "--min-p") cfg.filter.min_p = std::stof(val());
      else if (a == "--seed") cfg.seed = std::stoull(val());
      else if (a == "--min-conf") min_conf = std::stof(val());
      else if (a == "--json") json_out = val();
      else if (a == "--repeat") repeat = std::stoi(val());
      else if (a == "--cost") cost_list = val();
      else if (a == "--ngram") {
        const auto parts = split(val(), ',');
        ngram_max = std::stoi(parts.at(0));
        ngram_min = parts.size() > 1 ? std::stoi(parts[1]) : 1;
      } else throw std::runtime_error("unknown option " + a);
    }
    const bool greedy = !(cfg.filter.temperature > 0.0f);
    const bool self = std::find(drafters.begin(), drafters.end(), "self") != drafters.end();
    Loaded target = load(dir, self ? 2 : 1, cfg.prefill_chunk, format);
    cfg.stop_ids = std::vector<int32_t>(target.config.eos_token_ids.begin(), target.config.eos_token_ids.end());
    Loaded draft;
    if (std::find(drafters.begin(), drafters.end(), "model") != drafters.end()) {
      if (draft_dir.empty()) throw std::runtime_error("drafter `model` needs --draft DIR");
      draft = load(draft_dir, 1, cfg.prefill_chunk);
      if (draft.config.vocab_size != target.config.vocab_size) throw std::runtime_error("vocabularies differ");
    }
    void* s = nullptr;
    if (lx_rt_stream_create(&s) != 0) throw std::runtime_error("creating a device stream");
    lx::runtime::SpecDecoder decoder(*target.model);
    lx::runtime::DraftCost cost = lx::runtime::DraftCost::rtx5070ti_praxis();
    if (!cost_list.empty()) {
      cost.cost = {0.0};
      for (const auto& c : split(cost_list, ',')) cost.cost.push_back(std::stod(c));
    }
    lx::runtime::NgramDrafter ngram(ngram_max, ngram_min, cost), ngram_fixed(ngram_max, ngram_min, cost);
    ngram_fixed.set_adaptive(false);
    std::unique_ptr<lx::runtime::ModelDrafter> self_drafter, model_drafter;
    if (self) self_drafter = std::make_unique<lx::runtime::ModelDrafter>(*target.model, 1, cfg.filter, cfg.seed, min_conf);
    if (draft.model) model_drafter = std::make_unique<lx::runtime::ModelDrafter>(*draft.model, 0, cfg.filter, cfg.seed, min_conf);
    std::unique_ptr<lx::runtime::BlockDrafter> block, block_fixed;
    for (const std::string& d : drafters)
      if (d == "block" || d == "block-fixed") {
        if (head_dir.empty()) throw std::runtime_error("drafter `" + d + "` needs --draft-head DIR");
        auto& b = d == "block" ? block : block_fixed;
        b = std::make_unique<lx::runtime::BlockDrafter>(head_dir, *target.model, cfg.filter, cfg.seed, cost, bf16_head);
        b->set_adaptive(d == "block");
      }
    std::printf("model %s, verify steps up to %d tokens, k %d, %s\n", dir.c_str(), target.model->max_verify_tokens(),
                std::min(cfg.max_drafts, target.model->max_verify_tokens() - 1),
                greedy ? "greedy" : "sampled");
    nlohmann::json report = nlohmann::json::array();
    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> prompt = ids_json.get<std::vector<int32_t>>();
      std::vector<Run> runs;
      for (const std::string& d : drafters) {
        lx::runtime::Drafter* drafter = nullptr;
        if (d == "ngram") drafter = &ngram;
        else if (d == "ngram-fixed") drafter = &ngram_fixed;
        else if (d == "self") drafter = self_drafter.get();
        else if (d == "model") drafter = model_drafter.get();
        else if (d == "block") drafter = block.get();
        else if (d == "block-fixed") drafter = block_fixed.get();
        else if (d != "plain") throw std::runtime_error("unknown drafter " + d);
        Run r{d, {}, {}, 0};
        // a short warm-up of the same shapes, so graph capture and GEMM plans stay out of the timing
        lx::runtime::SpecConfig warm = cfg;
        warm.max_new_tokens = 24;
        lx::runtime::SpecStats ws;
        const auto drafted = ngram.drafted_by_n, accepted = ngram.accepted_by_n;
        decoder.generate(drafter, prompt, warm, ws, s);
        ngram.drafted_by_n = drafted;
        ngram.accepted_by_n = accepted;
        // the median decode time of `repeat` runs (the tokens are the same every run: greedy, or the
        // same seed)
        std::vector<double> times;
        for (int rep = 0; rep < repeat; ++rep) {
          const auto d0 = ngram.drafted_by_n, a0 = ngram.accepted_by_n;
          r.tokens = decoder.generate(drafter, prompt, cfg, r.stats, s);
          times.push_back(r.stats.decode_s);
          if (rep + 1 < repeat) ngram.drafted_by_n = d0, ngram.accepted_by_n = a0;
        }
        std::sort(times.begin(), times.end());
        r.stats.decode_s = times[times.size() / 2];
        if (auto* b = dynamic_cast<lx::runtime::BlockDrafter*>(drafter); b && b->proposals())
          r.propose_ms = b->propose_ms() / b->proposals(), r.proposals = b->proposals();
        runs.push_back(std::move(r));
      }
      const Run* plain = nullptr;
      for (const Run& r : runs)
        if (r.drafter == "plain") plain = &r;
      for (const Run& r : runs) {
        const auto& st = r.stats;
        const double tps = (st.emitted - 1) / st.decode_s;
        std::string same = "";
        int diverge = -1;
        if (plain && &r != plain && greedy) {
          const size_t n = std::min(r.tokens.size(), plain->tokens.size());
          for (size_t i = 0; i < n && diverge < 0; ++i)
            if (r.tokens[i] != plain->tokens[i]) diverge = static_cast<int>(i);
          if (diverge < 0 && r.tokens.size() != plain->tokens.size()) diverge = static_cast<int>(n);
          same = diverge < 0 ? "  identical" : "  DIVERGES at " + std::to_string(diverge);
        }
        const double plain_tps = plain ? (plain->stats.emitted - 1) / plain->stats.decode_s : 0;
        std::printf("%-22s %-6s prompt %5zu  out %4d  %7.1f tok/s  x%.2f  steps %4d  accepted/step %.2f  accept %.0f%%%s",
                    name.c_str(), r.drafter.c_str(), prompt.size(), st.emitted, tps, plain_tps > 0 ? tps / plain_tps : 0,
                    st.steps, st.steps ? static_cast<double>(st.accepted) / st.steps : 0,
                    st.drafted ? 100.0 * st.accepted / st.drafted : 0, same.c_str());
        if (r.propose_ms > 0) std::printf("  propose %.3f ms in %d steps", r.propose_ms, r.proposals);
        std::printf("\n");
        report.push_back({{"prompt", name},
                          {"drafter", r.drafter},
                          {"prompt_tokens", prompt.size()},
                          {"emitted", st.emitted},
                          {"steps", st.steps},
                          {"drafted", st.drafted},
                          {"accepted", st.accepted},
                          {"accept_hist", st.accept_hist},
                          {"prefill_s", st.prefill_s},
                          {"decode_s", st.decode_s},
                          {"tok_s", tps},
                          {"speedup", plain_tps > 0 ? tps / plain_tps : 0},
                          {"greedy", greedy},
                          {"propose_ms", r.propose_ms},
                          {"proposals", r.proposals},
                          {"diverges_at", diverge}});
      }
    }
    std::printf("ngram acceptance by match length:");
    for (size_t n = 1; n < ngram.drafted_by_n.size(); ++n)
      if (ngram.drafted_by_n[n])
        std::printf("  n=%zu %lld/%lld (%.0f%%)", n, static_cast<long long>(ngram.accepted_by_n[n]),
                    static_cast<long long>(ngram.drafted_by_n[n]),
                    100.0 * ngram.accepted_by_n[n] / ngram.drafted_by_n[n]);
    std::printf("\n");
    if (!json_out.empty()) std::ofstream(json_out) << report.dump(1) << "\n";
    if (const int err = lx_rt_last_error())
      throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the run");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

// ---- dromos pass1: the free drafters of the pipelined two-pass schedule ----------------------------------------
//
//   dromos pass1 MODEL_DIR PROMPTS_JSON [options]
//     Decodes each prompt one token at a time, as plain decoding does, and at every step records what three free
//     drafters would have proposed for the next token and whether it held:
//       early   the argmax of the first pass's readout (metis.h, set_early_logits); under sampling, a draw from it
//       ngram   prompt lookup: the most recent earlier occurrence of the longest suffix of 3, 2 or 1 tokens, and the
//               token after it (runtime/speculative.cpp's NgramDrafter, with match lengths 3..1)
//       comb    ngram when it has a match, else early
//     Greedy: a draft holds when it equals the target's argmax. Sampled, with p the target's filtered distribution and
//     q the drafter's (the same filter): the expected acceptance is sum_x min(p, q) (p(g) for a point-mass ngram draft),
//     and `held` is a simulated accept (draw the draft from q, accept with probability min(1, p / q)). Runs are maximal
//     stretches of consecutive held steps.
//     Options (defaults: greedy, 256 tokens):
//       --tokens N                    tokens per prompt, stopping at an end token
//       --temp T --top-p P --top-k K --min-p P --rep R --seed S     sampling (temp 0 is greedy)
//       --json FILE                   every step's numbers and the run-length histograms
//       --check                       a one-pass model with the same weights runs beside, on the same tokens; its
//                                     logits must equal this model's first-pass readout bit for bit. Uses two models on
//                                     the card, so run it with --no-bf16x
//       --time                        milliseconds per step, medians: plain one-token steps at M = 1 and at M = 2 (two
//                                     streams), and one-pass sweeps at M = 1 and M = 2 (the pipelined sweep reads the
//                                     weights once for its two rows). Then the speedup each measured acceptance
//                                     projects: plain / (sweep at M = 2 + (1 - a) x one-pass step at M = 1).
//                                     Run it alone, in the production format (no --check, no --no-bf16x)
namespace {

struct Sampling {
  double temp = 0.0, top_p = 1.0, min_p = 0.0, rep = 1.0;
  int top_k = 0;
  uint64_t seed = 1;
};

// The sampler's distribution over the vocabulary for one row (kernels.h, lx_sample): repetition penalty on the tokens
// seen so far, then temperature, top-k, top-p over the top-k distribution, and min-p. Temperature 0 is a point mass on
// the argmax (ties to the lower index).
struct Dist {
  std::vector<double> p;
  int argmax = 0;
};

Dist make_dist(const float* logits, int V, const std::vector<char>& seen, const Sampling& sm) {
  std::vector<double> x(V);
  for (int i = 0; i < V; ++i) {
    const double v = logits[i];
    x[i] = seen[i] && sm.rep != 1.0 ? (v > 0 ? v / sm.rep : v * sm.rep) : v;
  }
  Dist d;
  d.p.assign(V, 0.0);
  d.argmax = static_cast<int>(std::max_element(x.begin(), x.end()) - x.begin());
  if (sm.temp <= 0) {
    d.p[d.argmax] = 1.0;
    return d;
  }
  const int k = sm.top_k > 0 && sm.top_k < V ? sm.top_k : V;
  std::vector<int> idx(V);
  std::iota(idx.begin(), idx.end(), 0);
  std::partial_sort(idx.begin(), idx.begin() + k, idx.end(), [&](int a, int b) {
    return x[a] > x[b] || (x[a] == x[b] && a < b);
  });
  const double s0 = x[idx[0]] / sm.temp;
  std::vector<double> w(k);
  double Z = 0.0;
  for (int j = 0; j < k; ++j) {
    w[j] = std::exp(x[idx[j]] / sm.temp - s0);
    Z += w[j];
  }
  // top-p: the smallest prefix of the top-k distribution holding top_p of its mass
  int keep = 0;
  double cum = 0.0;
  for (int j = 0; j < k && cum < sm.top_p; ++j) {
    cum += w[j] / Z;
    ++keep;
  }
  std::vector<int> kept;
  double mass = 0.0;
  for (int j = 0; j < keep; ++j) {
    if (sm.min_p > 0 && x[idx[j]] / sm.temp < s0 + std::log(sm.min_p)) continue;
    kept.push_back(j);
    mass += w[j];
  }
  for (int j : kept) d.p[idx[j]] = w[j] / mass;
  return d;
}

// The token at which a distribution's cumulative mass reaches u (u in [0, 1)).
int draw(const std::vector<double>& p, double u) {
  double c = 0.0;
  int last = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    if (p[i] <= 0) continue;
    c += p[i];
    last = static_cast<int>(i);
    if (u < c) return last;
  }
  return last;
}

double overlap(const Dist& p, const Dist& q) {
  double s = 0.0;
  for (size_t i = 0; i < p.p.size(); ++i) s += std::min(p.p[i], q.p[i]);
  return s;
}

// Prompt lookup (NgramDrafter): the longest suffix of 3, 2 or 1 tokens that occurred before, ending before the last
// token, and the token after its most recent occurrence. -1 when there is none.
int ngram_guess(const std::vector<int32_t>& ctx, int max_n, int min_n) {
  const int L = static_cast<int>(ctx.size());
  for (int n = max_n; n >= min_n; --n) {
    if (L - 1 < n) continue;
    for (int e = L - 2; e >= n - 1; --e) {
      bool same = true;
      for (int j = 0; j < n && same; ++j) same = ctx[e - j] == ctx[L - 1 - j];
      if (same) return ctx[e + 1];
    }
  }
  return -1;
}

// Lengths of the maximal runs of held steps, as a histogram.
void add_runs(const std::vector<char>& held, std::map<int, int>& out) {
  int run = 0;
  for (char h : held) {
    if (h) {
      ++run;
    } else if (run) {
      ++out[run];
      run = 0;
    }
  }
  if (run) ++out[run];
}

struct Trace {
  std::vector<int32_t> tokens;  // the emitted tokens (the output after the prompt)
  std::vector<char> match, agree, held_early, held_ngram, held_comb;
  std::vector<double> a_early, a_ngram, a_comb;  // expected acceptance per step
  bool ended = false;                            // stopped at an end token
  std::vector<int> rank_early;                   // rank of the emitted token under the first-pass logits (0 = top)
  std::vector<double> lp_p, lp_q;                // log-probability of the emitted token, raw softmax, both readouts
  std::vector<int> top_early;                    // the first-pass argmax, for the histogram
  int identical = 0, compared = 0;               // --check: one-pass logits equal to the first-pass readout
  double worst = 0.0;
};

// One prompt: prefill all but its last token, then decode the last one and on, with the first pass's readout on every
// step. `one`, when set, runs beside on the same tokens and its logits are compared with the readout.
Trace decode_prompt(lx::MetisModel& model, lx::MetisModel* one, const std::vector<int32_t>& prompt, int tokens,
                    const Sampling& sm, const std::vector<int>& end_ids, float* d_logits, float* d_early, float* d_one,
                    void* s) {
  const int V = model.vocab_size();
  const int L = static_cast<int>(prompt.size());
  model.set_early_logits(nullptr);
  model.reset_slot(0);
  if (one) one->reset_slot(0);
  for (int off = 0; off + 1 < L; off += 512) {
    const int c = std::min(512, L - 1 - off);
    model.forward({{prompt.begin() + off, prompt.begin() + off + c}, {0, c}, {0}}, d_logits, false, s);
    if (one) one->forward({{prompt.begin() + off, prompt.begin() + off + c}, {0, c}, {0}}, d_one, false, s);
  }
  model.set_early_logits(d_early);
  std::vector<char> seen(V, 0);
  for (int t : prompt) seen[t] = 1;
  std::vector<int32_t> ctx(prompt);
  std::mt19937_64 rng(sm.seed);
  std::uniform_real_distribution<double> unif(0.0, 1.0);
  const bool greedy = sm.temp <= 0;
  std::vector<float> lp(V), lq(V), lo(V);
  Trace tr;
  int32_t input = prompt.back();
  for (int t = 0; t < tokens; ++t) {
    model.forward({{input}, {0, 1}, {0}}, d_logits, false, s);
    lx_rt_copy(lp.data(), d_logits, static_cast<int64_t>(sizeof(float)) * V, LX_COPY_D2H, s);
    lx_rt_copy(lq.data(), d_early, static_cast<int64_t>(sizeof(float)) * V, LX_COPY_D2H, s);
    if (one) {
      one->forward({{input}, {0, 1}, {0}}, d_one, false, s);
      lx_rt_copy(lo.data(), d_one, static_cast<int64_t>(sizeof(float)) * V, LX_COPY_D2H, s);
    }
    lx_rt_stream_sync(s);
    if (one) {
      ++tr.compared;
      if (std::memcmp(lq.data(), lo.data(), sizeof(float) * V) == 0) ++tr.identical;
      for (int v = 0; v < V; ++v) tr.worst = std::max(tr.worst, static_cast<double>(std::abs(lq[v] - lo[v])));
    }
    const Dist P = make_dist(lp.data(), V, seen, sm), Q = make_dist(lq.data(), V, seen, sm);
    const int emitted = greedy ? P.argmax : draw(P.p, unif(rng));
    {
      // diagnostics on the raw logits: where the emitted token ranks under the first pass, and its log-probability
      auto logsoft = [&](const float* x, int i) {
        double mx = x[0];
        for (int v = 1; v < V; ++v) mx = std::max(mx, static_cast<double>(x[v]));
        double z = 0.0;
        for (int v = 0; v < V; ++v) z += std::exp(static_cast<double>(x[v]) - mx);
        return static_cast<double>(x[i]) - mx - std::log(z);
      };
      int rank = 0;
      for (int v = 0; v < V; ++v) rank += lq[v] > lq[emitted];
      tr.rank_early.push_back(rank);
      tr.lp_p.push_back(logsoft(lp.data(), emitted));
      tr.lp_q.push_back(logsoft(lq.data(), emitted));
      tr.top_early.push_back(Q.argmax);
    }

    // early: the draft from the first pass, its argmax (greedy) or a draw from q
    const int de = greedy ? Q.argmax : draw(Q.p, unif(rng));
    const double a_e = greedy ? static_cast<double>(de == P.argmax) : overlap(P, Q);
    const bool h_e = greedy ? de == emitted : unif(rng) < std::min(1.0, P.p[de] / Q.p[de]);

    // ngram: a point mass, which holds under sampling with probability p(g)
    const int g = ngram_guess(ctx, 3, 1);
    const bool match = g >= 0;
    const double a_n = !match ? 0.0 : greedy ? static_cast<double>(g == P.argmax) : P.p[g];
    const bool h_n = match && (greedy ? g == emitted : unif(rng) < P.p[g]);

    tr.match.push_back(match);
    tr.agree.push_back(Q.argmax == P.argmax);
    tr.a_early.push_back(a_e);
    tr.held_early.push_back(h_e);
    tr.a_ngram.push_back(a_n);
    tr.held_ngram.push_back(h_n);
    tr.a_comb.push_back(match ? a_n : a_e);
    tr.held_comb.push_back(match ? h_n : h_e);
    tr.tokens.push_back(emitted);
    ctx.push_back(emitted);
    seen[emitted] = 1;
    input = emitted;
    if (std::find(end_ids.begin(), end_ids.end(), emitted) != end_ids.end()) {
      tr.ended = true;
      break;
    }
  }
  model.set_early_logits(nullptr);
  return tr;
}

// Median milliseconds of `iters` steps (after `warm`) of `n_seq` one-token rows, one row per slot 0 .. n_seq - 1, each
// synced as a decode loop is.
double median_step_ms(lx::MetisModel& model, int n_seq, float* logits, void* s, int warm, int iters) {
  std::vector<int32_t> cu(n_seq + 1), slots(n_seq), toks(n_seq, 1000);
  for (int i = 0; i <= n_seq; ++i) cu[i] = i;
  for (int i = 0; i < n_seq; ++i) slots[i] = i;
  std::vector<double> ms;
  for (int i = 0; i < warm + iters; ++i) {
    const auto t0 = std::chrono::steady_clock::now();
    model.forward({toks, cu, slots}, logits, false, s);
    lx_rt_stream_sync(s);
    const auto t1 = std::chrono::steady_clock::now();
    if (i >= warm) ms.push_back(std::chrono::duration<double, std::milli>(t1 - t0).count());
  }
  std::sort(ms.begin(), ms.end());
  return ms[ms.size() / 2];
}

void prefill_slot(lx::MetisModel& model, int slot, const std::vector<int32_t>& ids, float* logits, void* s) {
  model.reset_slot(slot);
  for (size_t off = 0; off < ids.size(); off += 512) {
    const int c = static_cast<int>(std::min<size_t>(512, ids.size() - off));
    model.forward({{ids.begin() + off, ids.begin() + off + c}, {0, c}, {slot}}, logits, false, s);
  }
}

// The projected single-stream speedup of the pipelined schedule at acceptance a (plain = 1 step).
double projected(double plain, double sweep2, double pass1, double a) { return plain / (sweep2 + (1.0 - a) * pass1); }

}  // namespace

int cmd_pass1(int argc, char** argv) {
  if (argc < 3) {
    std::fprintf(stderr,
                 "usage: dromos pass1 MODEL_DIR PROMPTS_JSON [--tokens N] [--temp T --top-p P --top-k K --min-p P --rep R "
                 "--seed S] [--json FILE] [--check] [--time] [--weights F] [--no-bf16x]\n");
    return 2;
  }
  try {
    const std::string dir = argv[1];
    nlohmann::json prompts;
    std::ifstream(argv[2]) >> prompts;
    int tokens = 256;
    std::string json_out;
    bool check = false, timing = false;
    Sampling sm;
    lx::cli::FormatFlags format;
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
      if (lx::cli::parse_format_flag(argc, argv, i, format)) continue;
      if (a == "--tokens") tokens = std::stoi(val());
      else if (a == "--temp") sm.temp = std::stod(val());
      else if (a == "--top-p") sm.top_p = std::stod(val());
      else if (a == "--top-k") sm.top_k = std::stoi(val());
      else if (a == "--min-p") sm.min_p = std::stod(val());
      else if (a == "--rep") sm.rep = std::stod(val());
      else if (a == "--seed") sm.seed = std::stoull(val());
      else if (a == "--json") json_out = val();
      else if (a == "--check") check = true;
      else if (a == "--time") timing = true;
      else throw std::runtime_error("unknown option " + a);
    }
    if (check && timing) throw std::runtime_error("--check and --time need separate runs (two models do not time cleanly)");

    const lx::MetisConfig config = lx::load_config(dir);
    lx::cli::LoadedWeights w = lx::cli::load_weights(dir, config, format);
    std::fprintf(stderr, "%s\n", w.description.c_str());
    void* s = nullptr;
    if (lx_rt_stream_create(&s) != 0) throw std::runtime_error("creating a device stream");
    const int V = config.vocab_size;
    const std::vector<int> end_ids = config.eos_token_ids;
    auto alloc = [&](int64_t floats) {
      void* p = nullptr;
      if (lx_rt_alloc(&p, static_cast<int64_t>(sizeof(float)) * floats) != 0)
        throw std::runtime_error("device allocation failed");
      return static_cast<float*>(p);
    };
    float* d_logits = alloc(2 * V);
    float* d_early = alloc(2 * V);
    float* d_one = alloc(2 * V);

    // two slots: slot 0 decodes the prompts; slots 0 and 1 are the two streams of --time
    auto two_pass = std::make_unique<lx::MetisModel>(config, *w.weights, w.formats, 2, 8192, 512, nullptr, true);
    std::unique_ptr<lx::MetisModel> one;
    if (check) {
      lx::MetisConfig c1 = config;
      c1.num_passes = 1;  // the same weights, read for the first pass alone
      one = std::make_unique<lx::MetisModel>(c1, *w.weights, w.formats, 2, 8192, 512, nullptr, true);
    }
    if (sm.temp <= 0) std::printf("%s, greedy", dir.c_str());
    else
      std::printf("%s, sampled: temp %.2f top-p %.2f top-k %d min-p %.2f rep %.2f", dir.c_str(), sm.temp, sm.top_p,
                  sm.top_k, sm.min_p, sm.rep);
    std::printf(", %d tokens at most per prompt\n", tokens);
    std::printf("%-22s %5s %6s | %-19s | %-19s | %-19s\n", "prompt", "steps", "match", "early exp / held",
                "ngram exp / held", "comb exp / held");

    nlohmann::json report = nlohmann::json::object();
    report["model"] = dir;
    report["greedy"] = sm.temp <= 0;
    report["temperature"] = sm.temp;
    report["top_p"] = sm.top_p;
    report["top_k"] = sm.top_k;
    report["min_p"] = sm.min_p;
    report["repetition"] = sm.rep;
    report["seed"] = sm.seed;
    report["tokens"] = tokens;
    nlohmann::json per_prompt = nlohmann::json::array();

    std::map<std::string, std::map<int, int>> runs_all;
    double all_e = 0, all_n = 0, all_c = 0, held_e = 0, held_n = 0, held_c = 0;
    long long steps_all = 0;
    int identical = 0, compared = 0;
    double worst = 0.0;
    std::vector<int32_t> first_prompt;
    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
      if (first_prompt.empty()) first_prompt = ids;
      Trace tr = decode_prompt(*two_pass, one.get(), ids, tokens, sm, end_ids, d_logits, d_early, d_one, s);
      const double n = static_cast<double>(tr.tokens.size());
      auto mean = [&](const std::vector<double>& v) {
        double acc = 0;
        for (double x : v) acc += x;
        return v.empty() ? 0.0 : acc / v.size();
      };
      auto rate = [&](const std::vector<char>& v) {
        double acc = 0;
        for (char x : v) acc += x;
        return v.empty() ? 0.0 : acc / v.size();
      };
      auto runs = [&](const std::vector<char>& v) {
        std::map<int, int> h;
        add_runs(v, h);
        return h;
      };
      auto longest = [&](const std::vector<char>& v) {
        const auto h = runs(v);
        return h.empty() ? 0 : h.rbegin()->first;
      };
      const double match_rate = rate(tr.match);
      std::printf("%-22s %5.0f %5.0f%% | %5.1f%% / %5.1f%%    | %5.1f%% / %5.1f%%    | %5.1f%% / %5.1f%%\n",
                  name.c_str(), n, 100 * match_rate, 100 * mean(tr.a_early), 100 * rate(tr.held_early),
                  100 * mean(tr.a_ngram), 100 * rate(tr.held_ngram), 100 * mean(tr.a_comb), 100 * rate(tr.held_comb));
      {
        int top5 = 0, top1 = 0;
        for (int r : tr.rank_early) { top5 += r < 5; top1 += r == 0; }
        std::vector<int> r = tr.rank_early;
        std::sort(r.begin(), r.end());
        double lpp = 0, lpq = 0;
        for (size_t i = 0; i < tr.lp_p.size(); ++i) lpp += tr.lp_p[i], lpq += tr.lp_q[i];
        const double m = std::max<double>(1.0, static_cast<double>(tr.rank_early.size()));
        std::printf("%-22s diag: emitted is pass-1 top-1 %.1f%%, top-5 %.1f%%, median rank %d; mean logp(emitted) pass-2 %.3f, pass-1 %.3f\n",
                    "", 100 * top1 / m, 100 * top5 / m, r.empty() ? -1 : r[r.size() / 2], lpp / m, lpq / m);
      }
      std::printf("%-22s longest held run: early %d, ngram %d, comb %d; %s\n", "", longest(tr.held_early),
                  longest(tr.held_ngram), longest(tr.held_comb),
                  tr.ended ? "ended at an end token" : "stopped at the token limit");
      all_e += mean(tr.a_early) * n;
      all_n += mean(tr.a_ngram) * n;
      all_c += mean(tr.a_comb) * n;
      held_e += rate(tr.held_early) * n;
      held_n += rate(tr.held_ngram) * n;
      held_c += rate(tr.held_comb) * n;
      steps_all += static_cast<long long>(n);
      for (auto& [len, cnt] : runs(tr.held_early)) runs_all["early"][len] += cnt;
      for (auto& [len, cnt] : runs(tr.held_ngram)) runs_all["ngram"][len] += cnt;
      for (auto& [len, cnt] : runs(tr.held_comb)) runs_all["comb"][len] += cnt;
      identical += tr.identical;
      compared += tr.compared;
      worst = std::max(worst, tr.worst);
      per_prompt.push_back(
          {{"prompt", name},
           {"steps", tr.tokens.size()},
           {"ended", tr.ended},
           {"match", match_rate},
           {"agree", rate(tr.agree)},
           {"expected", {{"early", mean(tr.a_early)}, {"ngram", mean(tr.a_ngram)}, {"comb", mean(tr.a_comb)}}},
           {"held", {{"early", rate(tr.held_early)}, {"ngram", rate(tr.held_ngram)}, {"comb", rate(tr.held_comb)}}},
           {"runs", {{"early", runs(tr.held_early)}, {"ngram", runs(tr.held_ngram)}, {"comb", runs(tr.held_comb)}}},
           {"steps_detail",
            {{"a_early", tr.a_early}, {"held_early", tr.held_early}, {"a_ngram", tr.a_ngram},
             {"held_ngram", tr.held_ngram}, {"a_comb", tr.a_comb}, {"held_comb", tr.held_comb},
             {"match", tr.match}, {"tokens", tr.tokens}}}});
    }
    const double N = static_cast<double>(std::max<long long>(steps_all, 1));
    std::printf("%-22s %5lld %5s | %5.1f%% / %5.1f%%    | %5.1f%% / %5.1f%%    | %5.1f%% / %5.1f%%\n", "all prompts",
                steps_all, "", 100 * all_e / N, 100 * held_e / N, 100 * all_n / N, 100 * held_n / N, 100 * all_c / N,
                100 * held_c / N);
    for (const char* d : {"early", "ngram", "comb"}) {
      std::printf("held runs, %-5s (length:count):", d);
      int shown = 0;
      for (auto& [len, cnt] : runs_all[d]) {
        if (shown++ == 16) break;
        std::printf(" %d:%d", len, cnt);
      }
      std::printf("\n");
    }
    report["prompts"] = per_prompt;
    report["expected_all"] = {{"early", all_e / N}, {"ngram", all_n / N}, {"comb", all_c / N}};
    report["held_all"] = {{"early", held_e / N}, {"ngram", held_n / N}, {"comb", held_c / N}};
    if (check) {
      std::printf("one-pass readout vs first-pass readout, teacher forced: %d of %d steps bit-identical, max |diff| %.3g\n",
                  identical, compared, worst);
      report["check"] = {{"identical", identical}, {"compared", compared}, {"max_abs_diff", worst}};
    }

    if (timing) {
      if (first_prompt.empty()) throw std::runtime_error("--time needs a prompt for its context");
      const int ctx_len = static_cast<int>(std::min<size_t>(first_prompt.size(), 1024));
      const std::vector<int32_t> context(first_prompt.begin(), first_prompt.begin() + ctx_len);
      const int warm = 30, iters = 200;
      prefill_slot(*two_pass, 0, context, d_logits, s);
      prefill_slot(*two_pass, 1, context, d_logits, s);
      const double plain1 = median_step_ms(*two_pass, 1, d_logits, s, warm, iters);
      const double plain2 = median_step_ms(*two_pass, 2, d_logits, s, warm, iters);
      if (!one) {
        two_pass.reset();  // frees the two-pass model's decode copies before the one-pass model takes its own
        lx::MetisConfig c1 = config;
        c1.num_passes = 1;
        one = std::make_unique<lx::MetisModel>(c1, *w.weights, w.formats, 2, 8192, 512, nullptr, true);
      }
      prefill_slot(*one, 0, context, d_logits, s);
      prefill_slot(*one, 1, context, d_logits, s);
      const double sweep1 = median_step_ms(*one, 1, d_logits, s, warm, iters);
      const double sweep2 = median_step_ms(*one, 2, d_logits, s, warm, iters);
      std::printf("\ntiming at a %d-token context, medians of %d steps (ms per step):\n", ctx_len, iters);
      std::printf("  plain two-pass step, M = 1           %8.3f   (1.000)\n", plain1);
      std::printf("  plain two-pass step, M = 2 streams   %8.3f   (%.3f plain steps)\n", plain2, plain2 / plain1);
      std::printf("  one-pass step, M = 1                 %8.3f   (%.3f plain steps)  pass-1-only sweep\n", sweep1,
                  sweep1 / plain1);
      std::printf("  one-pass sweep, M = 2                %8.3f   (%.3f plain steps)  pipelined sweep\n", sweep2,
                  sweep2 / plain1);
      std::printf("  pipelined sweep / M = 2 two-pass step  %.3f\n", sweep2 / plain2);
      std::printf("projected single-stream speedup, plain / (sweep + (1 - a) x pass-1-only), a = mean expected acceptance:\n");
      const double ae = all_e / N, an = all_n / N, ac = all_c / N;
      std::printf("  early %.3f -> x%.3f,  ngram %.3f -> x%.3f,  comb %.3f -> x%.3f\n", ae,
                  projected(plain1, sweep2, sweep1, ae), an, projected(plain1, sweep2, sweep1, an), ac,
                  projected(plain1, sweep2, sweep1, ac));
      std::printf("  ceiling at a = 1: x%.3f\n", projected(plain1, sweep2, sweep1, 1.0));
      report["timing_ms"] = {{"plain_m1", plain1}, {"plain_m2", plain2}, {"one_pass_m1", sweep1}, {"one_pass_m2", sweep2},
                             {"context", ctx_len}};
    }
    if (!json_out.empty()) std::ofstream(json_out) << report.dump(1) << "\n";
    if (const int err = lx_rt_last_error())
      throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during pass1");
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}

// dromos spec-exactness: the next token of one prompt, drawn DRAWS times by the verify path and DRAWS times by the plain
// sampler, at the think setting (temperature 0.8, top-p 0.95, top-k 20). Speculation runs without penalties
// (engine.cpp spec_eligible), so the distributions compared are the unpenalised ones. The verify path draws against a
// forced draft: a point mass on one token, the way the n-gram drafter proposes (SpecDecoder::verify, q null). The
// reference is make_dist, the host reading of sampler.cu's distribution. Each set uses its own seed, and speculative
// positions are spaced 4 apart, so no two draws share a Philox key.
namespace {

// Upper tail of a chi-square at x with df degrees of freedom (Wilson-Hilferty).
double chi_square_tail(double x, int df) {
  const double k = static_cast<double>(df);
  const double z = (std::cbrt(x / k) - (1.0 - 2.0 / (9.0 * k))) / std::sqrt(2.0 / (9.0 * k));
  return 0.5 * std::erfc(z / std::sqrt(2.0));
}

// Chi-square of counts against a reference distribution. Tokens expected at least 5 times each form a bin, the rest
// one bin. A draw outside the reference's support fails outright.
double chi_square_p(const std::vector<long>& counts, const std::vector<double>& p, long draws) {
  double x = 0.0, rest_expected = 0.0, rest_observed = 0.0;
  int bins = 0;
  for (size_t i = 0; i < p.size(); ++i) {
    if (p[i] <= 0.0) {
      if (counts[i] > 0) return 0.0;
      continue;
    }
    const double e = p[i] * static_cast<double>(draws);
    if (e >= 5.0) {
      x += (counts[i] - e) * (counts[i] - e) / e;
      ++bins;
    } else {
      rest_expected += e;
      rest_observed += static_cast<double>(counts[i]);
    }
  }
  if (rest_expected > 0.0) {
    x += (rest_observed - rest_expected) * (rest_observed - rest_expected) / rest_expected;
    ++bins;
  }
  if (bins < 2) return 1.0;
  return chi_square_tail(x, bins - 1);
}

double tv_to_reference(const std::vector<long>& counts, const std::vector<double>& p, long draws) {
  double s = 0.0;
  for (size_t i = 0; i < p.size(); ++i) s += std::abs(static_cast<double>(counts[i]) / static_cast<double>(draws) - p[i]);
  return 0.5 * s;
}

double tv_between(const std::vector<long>& a, const std::vector<long>& b, long na, long nb) {
  double s = 0.0;
  for (size_t i = 0; i < a.size(); ++i)
    s += std::abs(static_cast<double>(a[i]) / static_cast<double>(na) - static_cast<double>(b[i]) / static_cast<double>(nb));
  return 0.5 * s;
}

void check_rc(int rc, const char* what) {
  if (rc != 0) throw std::runtime_error(std::string(what) + ": " + lx_rt_error_string(rc));
}

int run_exactness(const std::string& dir, const std::string& prompts_path, const std::string& name, int draws,
                  const lx::cli::FormatFlags& format) {
  nlohmann::json prompts;
  std::ifstream(prompts_path) >> prompts;
  if (!prompts.contains(name)) throw std::runtime_error("no prompt named " + name + " in " + prompts_path);
  const std::vector<int32_t> ids = prompts[name].get<std::vector<int32_t>>();
  if (ids.size() < 2) throw std::runtime_error("the prompt needs at least two tokens");
  const std::vector<int32_t> prefix(ids.begin(), ids.end() - 1);
  const int32_t last = ids.back();
  const int N = static_cast<int>(prefix.size());
  const Sampling sm{0.8, 0.95, 0.0, 1.0, 20, 0};  // temp, top_p, min_p, rep, top_k, seed (unused: the seeds below)

  Loaded l = load(dir, 1, 512, format, false);
  lx::MetisModel& model = *l.model;
  const int V = model.vocab_size();
  void* s = nullptr;
  check_rc(lx_rt_stream_create(&s), "creating a device stream");
  float* d_logits = nullptr;
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_logits), static_cast<int64_t>(sizeof(float)) * 512 * V),
           "device allocation for logits");
  auto prefill = [&]() {
    model.reset_slot(0);
    for (int off = 0; off < N; off += 512) {
      const int c = std::min(512, N - off);
      model.forward({{prefix.begin() + off, prefix.begin() + off + c}, {0, c}, {0}}, d_logits, false, s);
    }
  };

  // The reference: the next token's distribution after the prefix and the pending token.
  prefill();
  model.forward({{last}, {0, 1}, {0}}, d_logits, false, s);
  std::vector<float> base(static_cast<size_t>(V));
  check_rc(lx_rt_copy(base.data(), d_logits, static_cast<int64_t>(sizeof(float)) * V, LX_COPY_D2H, s), "copy of logits");
  check_rc(lx_rt_stream_sync(s), "stream sync");
  const Dist ref = make_dist(base.data(), V, std::vector<char>(static_cast<size_t>(V), 0), sm);

  // The plain sampler (lx_sample) on the same logits, as the engine calls it.
  float* d_base = nullptr;
  int32_t *d_seen = nullptr, *d_counts = nullptr, *d_tok = nullptr;
  float* d_work = nullptr;
  lx_sampling* d_params = nullptr;
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_base), static_cast<int64_t>(sizeof(float)) * V), "alloc");
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_seen), static_cast<int64_t>(sizeof(int32_t)) * V), "alloc");
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_counts), static_cast<int64_t>(sizeof(int32_t)) * V), "alloc");
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_work), static_cast<int64_t>(sizeof(float)) * V), "alloc");
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_tok), sizeof(int32_t)), "alloc");
  check_rc(lx_rt_alloc(reinterpret_cast<void**>(&d_params), sizeof(lx_sampling)), "alloc");
  check_rc(lx_rt_copy(d_base, base.data(), static_cast<int64_t>(sizeof(float)) * V, LX_COPY_H2D, s), "copy");
  check_rc(lx_sampler_reset(d_seen, d_counts, 0, V, s), "sampler reset");
  auto plain_draws = [&](uint64_t seed) {
    std::vector<long> counts(static_cast<size_t>(V), 0);
    for (int i = 0; i < draws; ++i) {
      lx_sampling sp{};
      sp.seed = seed;
      sp.position = static_cast<uint64_t>(i);
      sp.slot = 0;
      sp.row = 0;
      sp.temperature = static_cast<float>(sm.temp);
      sp.top_p = static_cast<float>(sm.top_p);
      sp.top_k = sm.top_k;
      sp.min_p = 0.0f;
      sp.repetition_penalty = 1.0f;
      check_rc(lx_rt_copy(d_params, &sp, static_cast<int64_t>(sizeof(sp)), LX_COPY_H2D, s), "copy of params");
      check_rc(lx_sample(d_base, V, d_params, 1, d_seen, d_counts, d_work, d_tok, nullptr, 0, nullptr, nullptr, s),
               "lx_sample");
      int32_t tok = 0;
      check_rc(lx_rt_copy(&tok, d_tok, sizeof(int32_t), LX_COPY_D2H, s), "copy of a draw");
      check_rc(lx_rt_stream_sync(s), "stream sync");
      counts[static_cast<size_t>(tok)] += 1;
    }
    return counts;
  };

  // The verify path: the state after the prefix is restored before every draw, and the step is [pending, draft].
  lx::runtime::SpecDecoder decoder(model);
  lx::runtime::SpecConfig config;
  config.filter = lx_spec_filter{static_cast<float>(sm.temp), static_cast<float>(sm.top_p), sm.top_k,
                                 static_cast<float>(sm.min_p)};
  config.seed = 3;
  auto spec_draws = [&](int draft, long& accepted_total) {
    std::vector<long> counts(static_cast<size_t>(V), 0);
    prefill();
    const int before = model.slot_length(0);
    for (int i = 0; i < draws; ++i) {
      model.truncate(0, before);
      int accepted = 0;
      int32_t next = 0;
      decoder.verify(0, config, {last, static_cast<int32_t>(draft)}, nullptr, static_cast<uint64_t>(4 * i), accepted,
                     next, s);
      const int32_t first = accepted >= 1 ? static_cast<int32_t>(draft) : next;
      counts[static_cast<size_t>(first)] += 1;
      accepted_total += accepted;
    }
    return counts;
  };

  // Two forced drafts: the reference's second most likely token (often accepted) and its least likely kept token.
  std::vector<int> kept;
  for (int t = 0; t < V; ++t)
    if (ref.p[static_cast<size_t>(t)] > 0.0) kept.push_back(t);
  std::sort(kept.begin(), kept.end(), [&](int a, int b) {
    return ref.p[static_cast<size_t>(a)] > ref.p[static_cast<size_t>(b)] ||
           (ref.p[static_cast<size_t>(a)] == ref.p[static_cast<size_t>(b)] && a < b);
  });
  const int high = kept.size() > 1 ? kept[1] : kept[0];
  const int low = kept.back();

  const std::vector<long> plain_a = plain_draws(1);
  const std::vector<long> plain_b = plain_draws(2);
  long acc_high = 0, acc_low = 0;
  const std::vector<long> spec_high = spec_draws(high, acc_high);
  const std::vector<long> spec_low = spec_draws(low, acc_low);
  if (const int err = lx_rt_last_error()) throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the check");

  std::printf("%s: %d prefix tokens, the next token drawn %d times at temperature %.1f, top-p %.2f, top-k %d\n",
              name.c_str(), N, draws, sm.temp, sm.top_p, sm.top_k);
  std::printf("reference: %zu tokens kept, most likely %d (p %.3f)\n", kept.size(), kept[0],
              ref.p[static_cast<size_t>(kept[0])]);
  bool ok = true;
  auto report = [&](const char* label, const std::vector<long>& c) {
    const double p = chi_square_p(c, ref.p, draws);
    ok = ok && p > 1e-3;
    std::printf("  %-38s chi-square p %.3f   total variation to reference %.4f\n", label, p, tv_to_reference(c, ref.p, draws));
    return p;
  };
  report("plain sampler (lx_sample), seed 1", plain_a);
  report("plain sampler (lx_sample), seed 2", plain_b);
  report("verify, forced draft: rank 2 token", spec_high);
  report("verify, forced draft: least likely kept", spec_low);
  const double floor = tv_between(plain_a, plain_b, draws, draws);
  std::printf("  noise: total variation between the two plain sets %.4f; verify vs plain (rank 2) %.4f, (least likely) %.4f\n",
              floor, tv_between(spec_high, plain_a, draws, draws), tv_between(spec_low, plain_a, draws, draws));
  std::printf("  draft accepted in %.1f%% (rank 2) and %.1f%% (least likely) of draws\n",
              100.0 * static_cast<double>(acc_high) / draws, 100.0 * static_cast<double>(acc_low) / draws);
  std::printf("%s\n", ok ? "exact: every sample set follows the reference (chi-square p > 0.001)"
                         : "NOT exact: a sample set departs from the reference");

  lx_rt_free(d_base);
  lx_rt_free(d_seen);
  lx_rt_free(d_counts);
  lx_rt_free(d_work);
  lx_rt_free(d_tok);
  lx_rt_free(d_params);
  lx_rt_free(d_logits);
  return ok ? 0 : 1;
}

}  // namespace

int cmd_spec_exactness(int argc, char** argv) {
  try {
    lx::cli::FormatFlags format;
    std::vector<char*> args = without_format_flags(argc, argv, format);
    const int n = static_cast<int>(args.size());
    if (n != 4 && n != 5) {
      std::fprintf(stderr, "usage: dromos spec-exactness MODEL_DIR PROMPTS_JSON PROMPT [DRAWS] %s\n", lx::cli::kFormatUsage);
      return 2;
    }
    const int draws = n == 5 ? std::stoi(args[4]) : 2000;
    return run_exactness(args[1], args[2], args[3], draws, format);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}
