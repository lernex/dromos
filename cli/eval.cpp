// dromos eval: run reference prompts through the model and write every position's logits, for
// tools/compare.py. With a split point N the first N tokens go in as one prefill and the rest as
// single-token decode steps, which checks that the caches carry a sequence exactly.
//
//   dromos eval MODEL_DIR PROMPTS_JSON OUT_DIR [N] [--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x]
//
// The weights load in the format the folder stores (the published FP8 and GPTQ NVFP4 folders as they are), or
// the one --weights names, with the maths the plan gives that format on this GPU; stderr says which.
//
// LX_EVAL_STREAMS=K runs K streams at once instead: stream s takes prompt s % P (P prompts), is prefilled with its
// first N tokens, and then all streams step together one token per step. So the decode steps carry K rows, the batch
// the decode GEMMs see when K streams decode, and the logits are those of K concurrent sequences. The stream that
// starts prompt p writes NAME.f32; the replicas write NAME.rS.f32 (they must match their first stream to the bit).
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <nlohmann/json.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "cli/common.h"
#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"
#include "platform/platform.h"

namespace {

using Clock = std::chrono::steady_clock;

struct Stream {
  std::string name;
  std::vector<int32_t> ids;
  int slot = 0;
  int next = 0;  // the position whose token the stream feeds next
  std::vector<float> logits;  // [T, V]
  std::filesystem::path file;
};

// The concurrent run of LX_EVAL_STREAMS: prefill each stream on its own slot, then decode in lockstep.
void run_streams(lx::MetisModel& model, int V, const std::vector<std::pair<std::string, std::vector<int32_t>>>& prompts,
                 int split, int K, const std::filesystem::path& out_dir, void* stream, float* d_logits) {
  std::vector<Stream> streams;
  for (int s = 0; s < K; ++s) {
    const auto& [name, ids] = prompts[static_cast<size_t>(s) % prompts.size()];
    Stream st;
    st.name = name;
    st.ids = ids;
    st.slot = s;
    const int T = static_cast<int>(ids.size());
    st.next = split > 0 && split < T ? split : T;
    st.logits.assign(static_cast<size_t>(T) * V, 0.f);
    st.file = s < static_cast<int>(prompts.size()) ? out_dir / (name + ".f32") : out_dir / (name + ".r" + std::to_string(s) + ".f32");
    streams.push_back(std::move(st));
  }
  const auto t0 = Clock::now();
  for (Stream& st : streams) {
    model.reset_slot(st.slot);
    const int first = st.next;
    lx::StepBatch step{{st.ids.begin(), st.ids.begin() + first}, {0, first}, {st.slot}};
    model.forward(step, d_logits, true, stream);
    lx_rt_copy(st.logits.data(), d_logits, static_cast<int64_t>(sizeof(float) * first * V), LX_COPY_D2H, stream);
    lx_rt_stream_sync(stream);
  }
  const double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
  const auto t1 = Clock::now();
  int64_t decoded = 0;
  for (;;) {
    std::vector<Stream*> active;
    for (Stream& st : streams)
      if (st.next < static_cast<int>(st.ids.size())) active.push_back(&st);
    if (active.empty()) break;
    lx::StepBatch step;
    step.cu_seqlens.push_back(0);
    for (size_t j = 0; j < active.size(); ++j) {
      step.tokens.push_back(active[j]->ids[active[j]->next]);
      step.cu_seqlens.push_back(static_cast<int32_t>(j + 1));
      step.slots.push_back(active[j]->slot);
    }
    model.forward(step, d_logits, true, stream);
    for (size_t j = 0; j < active.size(); ++j)
      lx_rt_copy(active[j]->logits.data() + static_cast<size_t>(active[j]->next) * V, d_logits + j * V,
                 static_cast<int64_t>(sizeof(float) * V), LX_COPY_D2H, stream);
    lx_rt_stream_sync(stream);
    for (Stream* st : active) ++st->next;
    decoded += static_cast<int64_t>(active.size());
  }
  const double decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
  if (const int err = lx_rt_last_error())
    throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the run");
  for (const Stream& st : streams)
    std::ofstream(st.file, std::ios::binary)
        .write(reinterpret_cast<const char*>(st.logits.data()), st.logits.size() * sizeof(float));
  std::printf("%d streams: prefill %.1f ms, %lld decode tokens in %.1f ms (%.1f tok/s over the lockstep steps)\n", K,
              prefill_s * 1e3, static_cast<long long>(decoded), decode_s * 1e3, decoded / decode_s);
}

}  // namespace

int cmd_eval(int argc, char** argv) {
  try {
    lx::cli::FormatFlags format;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i)
      if (!lx::cli::parse_format_flag(argc, argv, i, format)) args.push_back(argv[i]);
    if (args.size() != 3 && args.size() != 4) {
      std::fprintf(stderr, "usage: dromos eval MODEL_DIR PROMPTS_JSON OUT_DIR [N] %s\n", lx::cli::kFormatUsage);
      return 2;
    }
    const std::string dir = args[0];
    const int split = args.size() == 4 ? std::stoi(args[3]) : 0;
    const std::filesystem::path out_dir = args[2];
    std::filesystem::create_directories(out_dir);
    nlohmann::json prompts;
    std::ifstream(args[1]) >> prompts;
    // Eval runs batch-invariant, so K concurrent streams (LX_EVAL_STREAMS) give each prompt the bits one stream does.
    lx::platform::set_env("LX_BATCH_INVARIANT", "1");
    const char* streams_env = std::getenv("LX_EVAL_STREAMS");
    const int streams = streams_env != nullptr ? std::max(0, std::atoi(streams_env)) : 0;

    const lx::MetisConfig config = lx::load_config(dir);
    const lx::cli::LoadedWeights loaded = lx::cli::load_weights(dir, config, format);
    std::fprintf(stderr, "%s\n", loaded.description.c_str());
    size_t longest = 1;
    for (auto& [name, ids] : prompts.items()) longest = std::max(longest, ids.size());
    // Concurrent streams need a slot each and a step of up to one token per stream; their caches need only the prompts.
    const int slots = std::max(1, streams);
    const int step_tokens = std::max(static_cast<int>(longest), streams);
    const int kv_capacity = streams > 0 ? static_cast<int>(longest) + 64 : 4096;
    lx::MetisModel model(config, *loaded.weights, loaded.formats, slots, kv_capacity, step_tokens);

    const int V = config.vocab_size;
    float* d_logits = nullptr;
    const auto logit_bytes = static_cast<int64_t>(sizeof(float) * step_tokens * V);
    if (lx_rt_alloc(reinterpret_cast<void**>(&d_logits), logit_bytes) != 0)
      throw std::runtime_error("device allocation of " + std::to_string(logit_bytes) + " bytes");
    void* stream = nullptr;
    if (lx_rt_stream_create(&stream) != 0) throw std::runtime_error("creating a device stream");

    if (streams > 0) {
      std::vector<std::pair<std::string, std::vector<int32_t>>> list;
      for (auto& [name, ids_json] : prompts.items()) list.push_back({name, ids_json.get<std::vector<int32_t>>()});
      if (list.empty()) throw std::runtime_error("no prompts");
      run_streams(model, V, list, split, streams, out_dir, stream, d_logits);
      lx_rt_free(d_logits);
      return 0;
    }

    for (auto& [name, ids_json] : prompts.items()) {
      const std::vector<int32_t> ids = ids_json.get<std::vector<int32_t>>();
      const int T = static_cast<int>(ids.size());
      const int first = split > 0 && split < T ? split : T;
      std::vector<float> logits(static_cast<size_t>(T) * V);
      model.reset_slot(0);
      const auto t0 = Clock::now();
      // the prefill
      lx::StepBatch step{{ids.begin(), ids.begin() + first}, {0, first}, {0}};
      model.forward(step, d_logits, true, stream);
      lx_rt_copy(logits.data(), d_logits, static_cast<int64_t>(sizeof(float) * first * V), LX_COPY_D2H, stream);
      lx_rt_stream_sync(stream);
      const double prefill_s = std::chrono::duration<double>(Clock::now() - t0).count();
      // then one token at a time
      const auto t1 = Clock::now();
      for (int t = first; t < T; ++t) {
        lx::StepBatch one{{ids[t]}, {0, 1}, {0}};
        model.forward(one, d_logits, true, stream);
        lx_rt_copy(logits.data() + static_cast<size_t>(t) * V, d_logits, static_cast<int64_t>(sizeof(float) * V),
                   LX_COPY_D2H, stream);
      }
      lx_rt_stream_sync(stream);
      const double decode_s = std::chrono::duration<double>(Clock::now() - t1).count();
      if (const int err = lx_rt_last_error())
        throw std::runtime_error(std::string(lx_rt_error_string(err)) + " during the run");
      std::ofstream(out_dir / (name + ".f32"), std::ios::binary)
          .write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
      std::printf("%-20s %4d tokens: prefill %d in %.1f ms", name.c_str(), T, first, prefill_s * 1e3);
      if (first < T) std::printf(", decode %d in %.1f ms (%.1f tok/s)", T - first, decode_s * 1e3, (T - first) / decode_s);
      std::printf("\n");
    }
    lx_rt_free(d_logits);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
