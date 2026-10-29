// The request fields that are checked the same way on every endpoint: the bounds of each number, and the sampling
// fields of a request body with the model's defaults where the body says nothing. A field the client got wrong throws
// ApiError naming it, so the server answers 400 with that parameter. Header-only, so the parse has a unit test.
#pragma once

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <utility>

#include "nlohmann/json.hpp"
#include "runtime/engine.h"

namespace lx::server {

// A request the client got wrong, with the HTTP status and the OpenAI error type to say so.
struct ApiError : std::runtime_error {
  int status;
  std::string type;
  std::string param;
  ApiError(int status_code, std::string type_name, const std::string& message, std::string param_name = "")
      : std::runtime_error(message), status(status_code), type(std::move(type_name)), param(std::move(param_name)) {}
};

[[noreturn]] inline void invalid(const std::string& message, const std::string& param = "") {
  throw ApiError(400, "invalid_request_error", message, param);
}

// A number as people write it: 2 for 2.0, 0.95 for 0.95.
inline std::string shown(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

inline double number_in(const nlohmann::json& body, const char* key, double lo, double hi, double fallback) {
  if (!body.contains(key) || body[key].is_null()) return fallback;
  if (!body[key].is_number()) invalid(std::string(key) + " must be a number", key);
  const double v = body[key].get<double>();
  if (!(v >= lo && v <= hi)) invalid(std::string(key) + " must be between " + shown(lo) + " and " + shown(hi), key);
  return v;
}

inline int integer_in(const nlohmann::json& body, const char* key, int lo, int hi, int fallback) {
  if (!body.contains(key) || body[key].is_null()) return fallback;
  if (!body[key].is_number_integer()) invalid(std::string(key) + " must be an integer", key);
  const int64_t v = body[key].get<int64_t>();
  if (v < lo || v > hi) invalid(std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi), key);
  return static_cast<int>(v);
}

// The model's sampling defaults (generation_config.json): the top level is the thinking sampling; "direct" overrides it.
struct Sampling {
  double temperature = 1.0, top_p = 1.0, min_p = 0.0, repetition_penalty = 1.0;
  int top_k = 0;
};

// The sampling fields of a request body into sp; sp's seed and logprobs are left to the caller.
inline void read_sampling_fields(const nlohmann::json& body, const Sampling& d, runtime::SamplingParams& sp) {
  sp.temperature = static_cast<float>(number_in(body, "temperature", 0.0, 2.0, d.temperature));
  sp.top_p = static_cast<float>(number_in(body, "top_p", 1e-6, 1.0, d.top_p));
  sp.top_k = integer_in(body, "top_k", 0, 1 << 20, d.top_k);
  sp.min_p = static_cast<float>(number_in(body, "min_p", 0.0, 1.0, d.min_p));
  sp.repetition_penalty = static_cast<float>(number_in(body, "repetition_penalty", 1e-3, 2.0, d.repetition_penalty));
  sp.presence_penalty = static_cast<float>(number_in(body, "presence_penalty", -2.0, 2.0, 0.0));
  sp.frequency_penalty = static_cast<float>(number_in(body, "frequency_penalty", -2.0, 2.0, 0.0));
}

}  // namespace lx::server
