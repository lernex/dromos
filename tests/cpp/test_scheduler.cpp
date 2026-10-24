#include <doctest/doctest.h>

#include "runtime/scheduler.h"

using lx::runtime::SchedPlan;
using lx::runtime::SchedSeq;
using lx::runtime::plan_step;

namespace {

SchedSeq decoding() { return SchedSeq{0, true}; }
SchedSeq prefilling(int left) { return SchedSeq{left, false}; }

int tokens_for(const SchedPlan& p, int seq) {
  int n = 0;
  for (const auto& e : p.entries)
    if (e.seq == seq) n += e.tokens;
  return n;
}

}  // namespace

TEST_CASE("every decoding sequence feeds one token when the budget allows") {
  std::vector<SchedSeq> seqs = {decoding(), prefilling(100), decoding()};
  unsigned cursor = 0;
  SchedPlan p = plan_step(seqs, 64, 512, cursor);
  CHECK(tokens_for(p, 0) == 1);
  CHECK(tokens_for(p, 2) == 1);
  CHECK(tokens_for(p, 1) == 64 - 2);
  CHECK(p.total_tokens == 64);
}

TEST_CASE("prompt chunks are capped by the chunk size and by what is left of the prompt") {
  std::vector<SchedSeq> seqs = {prefilling(5000), prefilling(3)};
  unsigned cursor = 0;
  SchedPlan p = plan_step(seqs, 4096, 512, cursor);
  CHECK(tokens_for(p, 0) == 512);
  CHECK(tokens_for(p, 1) == 3);
}

TEST_CASE("decoding sequences are served first, so a long prompt cannot stall a reply") {
  std::vector<SchedSeq> seqs = {prefilling(4000), decoding()};
  unsigned cursor = 0;
  SchedPlan p = plan_step(seqs, 8, 512, cursor);
  REQUIRE(!p.entries.empty());
  CHECK(p.entries.front().seq == 1);
  CHECK(p.entries.front().tokens == 1);
}

TEST_CASE("a tight budget rotates across decoding sequences, so none starves") {
  std::vector<SchedSeq> seqs = {decoding(), decoding(), decoding(), decoding()};
  std::vector<int> served(4, 0);
  unsigned cursor = 0;
  for (int step = 0; step < 400; ++step) {
    SchedPlan p = plan_step(seqs, 2, 512, cursor);
    for (const auto& e : p.entries) served[e.seq] += e.tokens;
  }
  for (int s : served) CHECK(s == 200);
}

TEST_CASE("prompts take the budget in admission order, so the first in line answers first") {
  std::vector<SchedSeq> seqs = {prefilling(100), prefilling(100)};
  unsigned cursor = 0;
  SchedPlan p = plan_step(seqs, 512, 512, cursor);
  REQUIRE(p.entries.size() == 2);
  CHECK(p.entries[0].seq == 0);
  CHECK(p.entries[0].tokens == 100);
  CHECK(p.entries[1].seq == 1);
}

TEST_CASE("prompts all finish under a small budget, none waiting for the others") {
  std::vector<SchedSeq> seqs = {prefilling(40), prefilling(40), prefilling(40)};
  unsigned cursor = 0;
  std::vector<int> fed(3, 0);
  int steps = 0;
  while (true) {
    bool any = false;
    for (const auto& s : seqs) any = any || s.prompt_left > 0;
    if (!any) break;
    SchedPlan p = plan_step(seqs, 16, 8, cursor);
    REQUIRE(p.total_tokens > 0);
    for (const auto& e : p.entries) {
      fed[e.seq] += e.tokens;
      seqs[e.seq].prompt_left -= e.tokens;
    }
    REQUIRE(++steps < 100);
  }
  for (int f : fed) CHECK(f == 40);
  CHECK(steps <= 15);
}

TEST_CASE("an empty batch or a zero budget plans nothing") {
  unsigned cursor = 0;
  CHECK(plan_step({}, 64, 512, cursor).entries.empty());
  CHECK(plan_step({decoding()}, 0, 512, cursor).entries.empty());
}
