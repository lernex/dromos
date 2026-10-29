#include <doctest/doctest.h>

#include <string>

#include "nlohmann/json.hpp"
#include "server/request_args.h"

using lx::server::ApiError;
using lx::server::Sampling;
using lx::server::read_sampling_fields;
using nlohmann::json;

namespace {

// A body's sampling fields, read against the model defaults d. The refusal's parameter and message are empty when
// the body is accepted.
struct Parsed {
  lx::runtime::SamplingParams sp;
  std::string refused_param;
  std::string message;
};

Parsed parse(const json& body, Sampling d = {}) {
  Parsed out;
  try {
    read_sampling_fields(body, d, out.sp);
  } catch (const ApiError& e) {
    out.refused_param = e.param;
    out.message = e.what();
  }
  return out;
}

}  // namespace

TEST_CASE("penalties default to none, and the model's repetition default applies") {
  Sampling d;
  d.repetition_penalty = 1.05;
  const Parsed p = parse(json::object(), d);
  CHECK(p.refused_param.empty());
  CHECK(p.sp.presence_penalty == 0.0f);
  CHECK(p.sp.frequency_penalty == 0.0f);
  CHECK(p.sp.repetition_penalty == doctest::Approx(1.05f));
}

TEST_CASE("presence and frequency penalties are read as given, signs included") {
  const Parsed p = parse(json{{"presence_penalty", 2.0}, {"frequency_penalty", -2.0}});
  CHECK(p.refused_param.empty());
  CHECK(p.sp.presence_penalty == 2.0f);
  CHECK(p.sp.frequency_penalty == -2.0f);
}

TEST_CASE("penalties at the bounds are accepted, and past them refused naming the field") {
  CHECK(parse(json{{"presence_penalty", -2.0}, {"frequency_penalty", 2.0}}).refused_param.empty());

  const Parsed high = parse(json{{"presence_penalty", 2.5}});
  CHECK(high.refused_param == "presence_penalty");
  CHECK(high.message == "presence_penalty must be between -2 and 2");

  const Parsed low = parse(json{{"frequency_penalty", -2.5}});
  CHECK(low.refused_param == "frequency_penalty");
  CHECK(low.message == "frequency_penalty must be between -2 and 2");
}

TEST_CASE("a null penalty is the default, and a non-number is refused naming it") {
  const Parsed nul = parse(json{{"presence_penalty", nullptr}, {"frequency_penalty", nullptr}});
  CHECK(nul.refused_param.empty());
  CHECK(nul.sp.presence_penalty == 0.0f);
  CHECK(nul.sp.frequency_penalty == 0.0f);

  const Parsed text = parse(json{{"frequency_penalty", "high"}});
  CHECK(text.refused_param == "frequency_penalty");
  CHECK(text.message == "frequency_penalty must be a number");
}

TEST_CASE("the other sampling fields keep their bounds") {
  CHECK(parse(json{{"repetition_penalty", 0.0}}).refused_param == "repetition_penalty");
  CHECK(parse(json{{"repetition_penalty", 2.5}}).refused_param == "repetition_penalty");
  CHECK(parse(json{{"temperature", 2.5}}).refused_param == "temperature");
  CHECK(parse(json{{"top_p", 0.0}}).refused_param == "top_p");
  CHECK(parse(json{{"top_k", -1}}).refused_param == "top_k");
  CHECK(parse(json{{"min_p", 1.5}}).refused_param == "min_p");
  CHECK(parse(json{{"top_k", 40}, {"top_p", 0.95}, {"temperature", 0.8}}).refused_param.empty());
}
