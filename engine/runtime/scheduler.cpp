#include "runtime/scheduler.h"

#include <algorithm>

namespace lx::runtime {

SchedPlan plan_step(const std::vector<SchedSeq>& seqs, int budget, int prefill_chunk, unsigned& cursor, int grid) {
  SchedPlan plan;
  const int n = static_cast<int>(seqs.size());
  if (n == 0 || budget <= 0) {
    cursor += 1;
    return plan;
  }
  const int start = static_cast<int>(cursor % static_cast<unsigned>(n));
  cursor += 1;
  int left = budget;

  for (int k = 0; k < n && left > 0; ++k) {
    const int i = (start + k) % n;
    if (!seqs[i].decode_ready) continue;
    plan.entries.push_back({i, 1});
    --left;
  }

  // Prompts go in admission order, so the first request to arrive is the first to get its first
  // token; the budget a finished prompt leaves over goes to the next one in line.
  const int chunk = std::max(1, prefill_chunk);
  for (int i = 0; i < n && left > 0; ++i) {
    if (seqs[i].decode_ready || seqs[i].prompt_left <= 0) continue;
    int take = std::min(chunk, seqs[i].prompt_left);
    if (grid > 0) {
      take = std::min(take, grid - seqs[i].prompt_pos % grid);  // stop at the next grid line
      if (take > left) continue;                                 // the whole piece must fit
    } else {
      take = std::min(take, left);
    }
    if (take <= 0) continue;
    plan.entries.push_back({i, take});
    left -= take;
  }

  for (const SchedEntry& e : plan.entries) plan.total_tokens += e.tokens;
  return plan;
}

}  // namespace lx::runtime
