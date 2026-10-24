// The continuous-batching policy: which running sequences feed how many tokens in the next step.
// Pure host code, so the policy is tested without a GPU.
#pragma once

#include <vector>

namespace lx::runtime {

// What the scheduler needs to know about one running sequence.
struct SchedSeq {
  int prompt_left = 0;        // prompt tokens not yet fed to the model
  bool decode_ready = false;  // a sampled token is waiting to be fed (the sequence is decoding)
};

struct SchedEntry {
  int seq = 0;     // index into the running list
  int tokens = 0;  // tokens fed this step: 1 for a decoding sequence, a prompt chunk otherwise
};

struct SchedPlan {
  std::vector<SchedEntry> entries;  // in the order their tokens go into the step
  int total_tokens = 0;
};

// Decoding sequences come first, one token each, so a long prompt never holds up a reply in
// progress. They start from the cursor, so when the budget cannot cover them all, each takes its
// turn; the cursor advances by one each step. The rest of the budget goes to prompt chunks (at
// most prefill_chunk tokens each), in the order the sequences were admitted.
SchedPlan plan_step(const std::vector<SchedSeq>& seqs, int budget, int prefill_chunk, unsigned& cursor);

}  // namespace lx::runtime
