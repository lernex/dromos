// lernex spec: speculative decoding against plain decoding, per prompt and per drafter.
//
//   lernex spec MODEL_DIR PROMPTS_JSON [options]
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
//   lernex spec-cost MODEL_DIR [CONTEXT]
//     Milliseconds per plain step and per verify step of each size, after a CONTEXT-token prompt.
//
//   lernex block-check MODEL_DIR DRAFT_HEAD_DIR PROMPTS_JSON OUT_DIR
//     For each prompt: the target prefills all but its last token, the block drafter proposes a
//     block anchored on that token (greedy, bf16 weights), and OUT_DIR gets what
//     tools/draft/check_block.py needs to recompute it in PyTorch: the feature rows of the window,
//     the anchor, and the drafter's logits and confidences.
//
//   lernex spec-check MODEL_DIR PROMPTS_JSON [N]
//     Feeds each prompt's second half one token at a time, then again as verify steps of random
//     sizes with random rollbacks (and an occasional normal step, which folds), and requires every
//     verify row's logits to be bit-identical to the one-token step at the same position.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <nlohmann/json.hpp>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"
#include "runtime/block_drafter.h"
#include "runtime/speculative.h"

namespace {

struct Loaded {
  lx::MetisConfig config;
  std::unique_ptr<lx::ModelWeights> weights;
  std::unique_ptr<lx::MetisModel> model;
};

Loaded load(const std::string& dir, int slots, int max_tokens) {
  Loaded l;
  l.config = lx::load_config(dir);
  l.weights = std::make_unique<lx::ModelWeights>(lx::ModelWeights::load(dir, l.config, lx::LoadOptions{LX_W_BF16}));
  l.model = std::make_unique<lx::MetisModel>(l.config, *l.weights, lx::ModelFormats{}, slots, 8192, max_tokens,
                                             nullptr, true);
  return l;
}

std::vector<std::string> split(const std::string& s, char sep) {
  std::vector<std::string> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, sep);)
    if (!item.empty()) out.push_back(item);
  return out;
}

int run_check(int argc, char** argv) {
  const std::string dir = argv[1];
  nlohmann::json prompts;
  std::ifstream(argv[2]) >> prompts;
  const int K = argc >= 4 ? std::stoi(argv[3]) : 96;
  Loaded l = load(dir, 1, 512);
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
int run_cost(int argc, char** argv) {
  const std::string dir = argv[1];
  const int context = argc >= 3 ? std::stoi(argv[2]) : 1024;
  Loaded l = load(dir, 1, 512);
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
  if (argc < 2 || argc > 4) {
    std::fprintf(stderr, "usage: lernex spec-cost MODEL_DIR [CONTEXT [ROUNDS]]\n");
    return 2;
  }
  try {
    return run_cost(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

int cmd_spec_check(int argc, char** argv) {
  if (argc != 3 && argc != 4) {
    std::fprintf(stderr, "usage: lernex spec-check MODEL_DIR PROMPTS_JSON [N]\n");
    return 2;
  }
  try {
    return run_check(argc, argv);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
}

int cmd_block_check(int argc, char** argv) {
  if (argc != 5) {
    std::fprintf(stderr, "usage: lernex block-check MODEL_DIR DRAFT_HEAD_DIR PROMPTS_JSON OUT_DIR\n");
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
    std::fprintf(stderr, "usage: lernex spec MODEL_DIR PROMPTS_JSON [options] (see cli/spec.cpp)\n");
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
    for (int i = 3; i < argc; ++i) {
      const std::string a = argv[i];
      auto val = [&]() -> std::string {
        if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
        return argv[++i];
      };
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
    Loaded target = load(dir, self ? 2 : 1, cfg.prefill_chunk);
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
