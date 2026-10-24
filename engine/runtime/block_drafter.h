// A one-pass block drafter (DFlash / DSpark style), format lernex.block/1: the model and its training
// live in tools/draft (block_drafter.py, train_block.py); this runs it in the engine.
//
// Each proposal is one forward pass over a block of B rows: the pending token (the anchor) and
// B - 1 mask rows, attending to each other and to the target's fused hidden features over the
// last `window` committed positions, which the drafter keeps as per-layer keys and values in a ring
// (built from observe_features, so it costs nothing at proposal time). Row j's logits come through
// the target's own LM head (MetisModel::head_weights, shared, not copied). A confidence head gives,
// per row, the chance its draft holds once the drafts before it held, and DraftCost turns those
// into how many of the B - 1 drafts to verify.
//
// Proposals are point masses under greedy decoding (each row's argmax) and draws from each row's
// filtered distribution under sampling (Draft::q is then those rows). The rows don't depend on each
// other's drawn tokens, which speculative sampling allows: any proposal distribution chosen before
// the target's draws keeps the output exact.
//
// The confidence head is trusted as far as it keeps being right: per sequence and per draft
// position, its predictions are scaled by how many of them held lately (decayed counts, starting
// from trust). And a proposal has to pay for itself: the drafter keeps a running average of what its
// proposals gained (accepted tokens, minus the verify step's extra cost and the proposal's own, in
// tokens at the sequence's rate). While that is negative it sits out steps, 1, then 3, 7 and at most
// 15 in a row, proposing between them to find out whether the text has turned predictable, so text
// it cannot predict costs a proposal every 16 steps instead of one every step.
#pragma once

#include <cuda_runtime.h>

#include <memory>
#include <string>
#include <vector>

#include "cuda/kernels.h"
#include "model/metis.h"
#include "runtime/speculative.h"

namespace lx::runtime {

class BlockDrafter final : public Drafter {
 public:
  // dir: config.json and model.safetensors. The target's head serves as the drafter's embedding and
  // LM head. The drafter's products read int8 copies of its weights and the target's int8 head
  // (per-row scales, as the target's decode does), or with bf16 the bf16 weights themselves (what
  // the checks against PyTorch use). filter: the target's sampling (greedy when temperature <= 0);
  // sampled proposals are drawn with it.
  BlockDrafter(const std::string& dir, const MetisModel& target, lx_spec_filter filter, uint64_t seed,
               DraftCost cost = DraftCost::rtx5070ti_praxis(), bool bf16 = false);
  ~BlockDrafter() override;
  BlockDrafter(const BlockDrafter&) = delete;
  BlockDrafter& operator=(const BlockDrafter&) = delete;

  const char* name() const override { return "block"; }
  std::vector<FeatureTap> feature_taps() const override;
  void begin(const std::vector<int32_t>& prompt, cudaStream_t stream) override;
  void observe_features(const float* rows, int n, int feature_dim, cudaStream_t stream) override;
  Draft propose(const DraftRequest& request) override;
  void observe(int accepted, int32_t next, cudaStream_t stream) override;

  // Off: always propose every draft the request allows, whatever the confidence head says.
  void set_adaptive(bool on) { adaptive_ = on; }
  int block_size() const;

  // The last proposal, for checks: logits [B - 1, V] (device), confidences [B - 1][greedy, sampled]
  // and proposal tokens of every row (host).
  const float* last_logits() const;
  const std::vector<float>& last_confidence() const { return conf_; }
  const std::vector<int32_t>& last_tokens() const { return tokens_; }
  double propose_ms() const { return propose_ms_; }  // summed over the proposals since begin
  int proposals() const { return proposals_; }        // steps that ran the drafter (not sat out)

 private:
  struct Impl;
  std::unique_ptr<Impl> m_;
  DraftCost cost_;
  lx_spec_filter filter_;
  uint64_t seed_;
  bool adaptive_ = true;
  int last_k_ = 0;
  double lambda_ = 1.0;
  std::vector<float> conf_;
  std::vector<int32_t> tokens_;
  double propose_ms_ = 0;
  int proposals_ = 0;
  // per draft position: decayed sums of the head's hold predictions that were judged and of the
  // drafts that held; the raw predictions of the last proposal
  std::vector<double> predicted_, held_, last_p_;
  double surplus_ = 1.0;  // running average of a proposal's gain over a plain step, in tokens
  bool proposed_ = false;  // the last step ran the drafter
  int skip_ = 0, backoff_ = 0;
};

}  // namespace lx::runtime
