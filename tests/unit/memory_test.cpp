// The plan's memory against the allocator: load the checkpoint named by LX_MODEL_DIR (ModelWeights, BF16), build the
// model the plan would give it (BF16, 4 sequences of 4096 positions, 512 tokens per step), and compare the device
// memory the load and the model take with the plan's accounting. Needs the GPU and the checkpoint, so it
// is skipped without LX_MODEL_DIR.
#include <cmath>
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

TEST_CASE("the plan's memory matches what the model allocates") {
  const char* dir = std::getenv("LX_MODEL_DIR");
  if (!dir) return;
  using namespace lx;
  using namespace lx::runtime;

  const MetisConfig config = load_config(dir);
  const SafetensorsSet set = SafetensorsSet::open(dir);
  size_t unused = 0;
  for (const auto& [name, t] : set.tensors())
    if (unused_tensor(name)) unused += t.bytes;

  // The model as the plan describes it: one copy of each weight, in BF16.
  int64_t free_before = 0, total = 0;
  REQUIRE(lx_rt_mem_info(&free_before, &total) == 0);
  {
    const ModelWeights weights = ModelWeights::load(dir, config, LoadOptions{LX_W_BF16});
    const int slots = 4, capacity = 4096, tokens = 512;
    MetisModel model(config, weights, ModelFormats{}, slots, capacity, tokens);
    REQUIRE(lx_rt_device_sync() == 0);
    int64_t free_after = 0;
    REQUIRE(lx_rt_mem_info(&free_after, &total) == 0);

    const double measured = static_cast<double>(free_before - free_after);
    const double predicted = static_cast<double>(weight_bytes(config, recipe_for(LX_W_BF16)).total() + unused +
                                                 slots * state_bytes_per_slot(config) +
                                                 slots * capacity * kv_bytes_per_position(config) +
                                                 scratch_bytes(config, tokens, slots));
    INFO("measured " << measured << " predicted " << predicted);
    CHECK(std::fabs(measured - predicted) / predicted < 0.02);
  }
}
