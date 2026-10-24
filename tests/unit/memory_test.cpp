// The plan's memory against the allocator: load the checkpoint named by LX_MODEL_DIR, build the model the
// plan would give it (BF16, 4 sequences of 4096 positions, 512 tokens per step), and compare the device
// memory the load and the model take with the plan's accounting. Needs the GPU and the checkpoint, so it
// is skipped without LX_MODEL_DIR.
#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <string>

#include "core/config.h"
#include "core/device.h"
#include "core/safetensors.h"
#include "doctest.h"
#include "model/metis.h"
#include "runtime/plan.h"

namespace {

// Tensors the engine never reads: the tied lm_head and the k_router / k_budget / depth_budget family.
bool unused_tensor(const std::string& name) {
  return name.find("lm_head") != std::string::npos || name.find("k_router") != std::string::npos ||
         name.find("k_budget") != std::string::npos || name.find("depth_budget") != std::string::npos;
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

  // The model as the plan describes it: one copy of each weight, in BF16. The current metis.cpp also keeps
  // int8 decode copies of its projections unless LX_NO_W8 turns them off; that is measured separately.
  setenv("LX_NO_W8", "1", 1);
  size_t free_before = 0, total = 0;
  REQUIRE(cudaMemGetInfo(&free_before, &total) == cudaSuccess);
  {
    const lx::Weights weights = lx::Weights::upload(set);
    const int slots = 4, capacity = 4096, tokens = 512;
    const WeightLookup lookup = [&weights](const std::string& name) {
      WeightView view;
      const Tensor* t = weights.find(name);
      if (!t) return view;
      view.data = t->data;
      view.dtype = t->dtype == DType::F32 ? LX_F32 : LX_BF16;
      view.shape = t->shape;
      return view;
    };
    MetisModel model(config, lookup, slots, capacity, tokens);
    REQUIRE(cudaDeviceSynchronize() == cudaSuccess);
    size_t free_after = 0;
    REQUIRE(cudaMemGetInfo(&free_after, &total) == cudaSuccess);

    const double measured = static_cast<double>(free_before - free_after);
    const double predicted = static_cast<double>(weight_bytes(config, recipe_for(LX_W_BF16)).total() + unused +
                                                 slots * state_bytes_per_slot(config) +
                                                 slots * capacity * kv_bytes_per_position(config) +
                                                 scratch_bytes(config, tokens, slots));
    INFO("measured " << measured << " predicted " << predicted);
    CHECK(std::fabs(measured - predicted) / predicted < 0.02);
  }
  unsetenv("LX_NO_W8");
}
