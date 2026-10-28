// The decode traffic's expert union: the measured union where a run measured the stream count, the uniform-routing
// expectation everywhere else, and the modelled expectation always (what the model counts from the config).
#include "model/traffic.h"

#include "doctest.h"

using namespace lx;

namespace {

// One routed expert item in layer 3 of a four-layer model: two passes, 1000 bytes per expert, Logos's 192 experts, top 4.
DecodeTraffic one_expert_item() {
  DecodeTraffic t;
  t.context = 0;
  t.top_k = 4;
  t.experts = 192;
  TrafficItem e;
  e.name = "routed experts gate_up";
  e.group = "moe";
  e.layer = 3;
  e.scope = Scope::Expert;
  e.count = 2;
  e.data = 1000;
  t.items.push_back(e);
  return t;
}

}  // namespace

TEST_CASE("the expert union is the uniform expectation until a run measures the stream count") {
  const DecodeTraffic t = one_expert_item();
  const TrafficItem& e = t.items.front();
  CHECK(t.expected_experts(1) == doctest::Approx(4.0));
  CHECK(t.step_bytes(32) == doctest::Approx(2 * t.expected_experts(32) * 1000));
  CHECK(t.item_bytes(e, 32) == doctest::Approx(2 * t.expected_experts(32) * 1000));
}

TEST_CASE("a measured union replaces the expectation at its stream count, layer by layer") {
  DecodeTraffic t = one_expert_item();
  const TrafficItem& e = t.items.front();
  const double uniform32 = t.expected_experts(32);
  t.measured[32] = std::vector<double>{0, 0, 0, 40};  // layer 3 touched 40 experts over its passes and steps
  CHECK(t.experts_read(e, 32) == doctest::Approx(40.0));
  CHECK(t.step_bytes(32) == doctest::Approx(2 * 40 * 1000));
  CHECK(t.item_bytes(e, 32) == doctest::Approx(2 * 40 * 1000));
  // the modelled count is the uniform expectation, whatever was measured
  CHECK(t.step_bytes(32, true) == doctest::Approx(2 * uniform32 * 1000));
  CHECK(t.experts_read(e, 32, true) == doctest::Approx(uniform32));
  // another stream count has no measurement and reads the expectation
  CHECK(t.step_bytes(8) == doctest::Approx(2 * t.expected_experts(8) * 1000));
}
