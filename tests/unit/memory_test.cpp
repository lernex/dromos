// The plan's memory against the allocator: load a checkpoint (ModelWeights, in its own format), build the model the
// plan would give it (4 sequences of 4096 positions, 512 tokens per step), and compare the device memory the load
// and the model take with the plan's accounting: the weights (and the BF16X decode copies), the state,
// the cache and the step's scratch for the format's maths. Needs the GPU and the folders: LX_MODEL_DIR (BF16, with
// and without the copies, and the lossless BF16X target), LX_CKPT_FP8_DIR and LX_CKPT_NVFP4_DIR; each is skipped
// without its folder.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

#include "core/checkpoint.h"
#include "core/config.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "cuda/kernels.h"
#include "doctest.h"
#include "model/metis.h"
#include "runtime/plan.h"

namespace {

// Tensors the engine never reads that the loader still uploads as plain tensors: the k_router / k_budget /
// depth_budget family (ModelWeights skips the tied lm_head and the n-gram tables).
bool unused_tensor(const std::string& name) {
  return name.find("ngram") == std::string::npos && name.find("lm_head") == std::string::npos &&
         (name.find("k_router") != std::string::npos || name.find("k_budget") != std::string::npos ||
          name.find("depth_budget") != std::string::npos);
}

}  // namespace

namespace {

// Loads `dir` in `target` (with the BF16X decode copies when asked), builds the model the plan would give it on this
// GPU (4 sequences of 4096 positions, 512 tokens per step), and compares the device memory the load and the model
// take with the plan's accounting.
void check_memory(const char* dir, lx_wtype target, bool bf16x) {
  using namespace lx;
  using namespace lx::runtime;
  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  size_t unused = 0;
  for (const auto& [name, t] : set.tensors())
    if (unused_tensor(name)) unused += t.bytes;

  const lx::ModelFormats formats = model_formats(target, query_device(0), bf16x);
  int64_t free_before = 0, total = 0;
  REQUIRE(lx_rt_device_sync() == 0);
  REQUIRE(lx_rt_mem_info(&free_before, &total) == 0);
  {
    LoadOptions options;
    options.target = target;
    const ModelWeights weights = ModelWeights::load(dir, config, options);
    const int slots = 4, capacity = 4096, tokens = 512;
    MetisModel model(config, weights, formats, slots, capacity, tokens);
    REQUIRE(lx_rt_device_sync() == 0);
    int64_t free_after = 0;
    REQUIRE(lx_rt_mem_info(&free_after, &total) == 0);

    const double measured = static_cast<double>(free_before - free_after);
    const double predicted = static_cast<double>(weight_bytes(config, target, bf16x).total() + unused +
                                                 slots * state_bytes_per_slot(config) +
                                                 slots * capacity * kv_bytes_per_position(config) +
                                                 scratch_bytes(config, tokens, slots, formats));
    std::printf("  %-6s%s measured %.3f GiB, planned %.3f GiB (%+.2f%%)\n", format_name(target).c_str(),
                bf16x ? " + BF16X copies" : "", measured / (1ull << 30), predicted / (1ull << 30),
                100.0 * (predicted - measured) / measured);
    INFO(format_name(target) << (bf16x ? " with BF16X copies" : "") << ": measured " << measured << " predicted "
                             << predicted);
    CHECK(std::fabs(measured - predicted) / predicted < 0.02);
  }
}

}  // namespace

TEST_CASE("the plan's memory matches what the model allocates") {
  if (const char* dir = std::getenv("LX_MODEL_DIR")) {
    check_memory(dir, LX_W_BF16, false);
    if (lx::runtime::bf16x_available()) {
      check_memory(dir, LX_W_BF16, true);
      check_memory(dir, LX_W_BF16X, false);
    }
  }
  // The quantized folders with and without the BF16X copies of the BF16 tensors they keep wide.
  const bool copies = lx::runtime::bf16x_available();
  if (const char* dir = std::getenv("LX_CKPT_FP8_DIR")) {
    check_memory(dir, LX_W_FP8, false);
    if (copies) check_memory(dir, LX_W_FP8, true);
  }
  if (const char* dir = std::getenv("LX_CKPT_NVFP4_DIR")) {
    check_memory(dir, LX_W_NVFP4, false);
    if (copies) check_memory(dir, LX_W_NVFP4, true);
  }
}
