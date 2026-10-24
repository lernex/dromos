#include "server/server.h"

#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <deque>
#include <fstream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <vector>

#include "httplib.h"
#include "nlohmann/json.hpp"
#include "text/tokenizer.h"

namespace lx::server {
namespace {

using json = nlohmann::json;
using runtime::GenerateRequest;
using runtime::RequestResult;
using runtime::TokenDelta;

// A request the client got wrong, with the HTTP status and the OpenAI error type to say so.
struct ApiError : std::runtime_error {
  int status;
  std::string type;
  std::string param;
  ApiError(int status_code, std::string type_name, const std::string& message, std::string param_name = "")
      : std::runtime_error(message), status(status_code), type(std::move(type_name)), param(std::move(param_name)) {}
};

[[noreturn]] void invalid(const std::string& message, const std::string& param = "") {
  throw ApiError(400, "invalid_request_error", message, param);
}

std::string random_hex(size_t digits) {
  static std::mutex mu;
  static std::mt19937_64 rng{std::random_device{}()};
  std::lock_guard<std::mutex> lock(mu);
  static const char* hex = "0123456789abcdef";
  std::string out;
  while (out.size() < digits) {
    uint64_t v = rng();
    for (int i = 0; i < 16 && out.size() < digits; ++i, v >>= 4) out.push_back(hex[v & 15]);
  }
  return out;
}

uint64_t random_u64() {
  static std::mutex mu;
  static std::mt19937_64 rng{std::random_device{}()};
  std::lock_guard<std::mutex> lock(mu);
  return rng();
}

int64_t unix_time() {
  return std::chrono::duration_cast<std::chrono::seconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

json bytes_array(const std::string& s) {
  json out = json::array();
  for (unsigned char c : s) out.push_back(static_cast<int>(c));
  return out;
}

// The defaults a model ships with (generation_config.json), used where a request says nothing.
struct Defaults {
  double temperature = 1.0, top_p = 1.0;
  int top_k = 0;
};

Defaults read_defaults(const std::string& dir) {
  Defaults d;
  std::ifstream in(dir + "/generation_config.json");
  if (!in) return d;
  try {
    const json j = json::parse(in);
    d.temperature = j.value("temperature", d.temperature);
    d.top_p = j.value("top_p", d.top_p);
    d.top_k = j.value("top_k", d.top_k);
  } catch (const std::exception&) {
  }
  return d;
}

// What a request asks for, once it has been checked.
struct Job {
  GenerateRequest request;
  bool chat = false;
  bool stream = false;
  bool include_usage = false;
  bool logprobs = false;
  int top_logprobs = 0;
  std::string model;
};

// A number as people write it: 2 for 2.0, 0.95 for 0.95.
std::string shown(double v) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%g", v);
  return buf;
}

double number_in(const json& body, const char* key, double lo, double hi, double fallback) {
  if (!body.contains(key) || body[key].is_null()) return fallback;
  if (!body[key].is_number()) invalid(std::string(key) + " must be a number", key);
  const double v = body[key].get<double>();
  if (!(v >= lo && v <= hi)) invalid(std::string(key) + " must be between " + shown(lo) + " and " + shown(hi), key);
  return v;
}

int integer_in(const json& body, const char* key, int lo, int hi, int fallback) {
  if (!body.contains(key) || body[key].is_null()) return fallback;
  if (!body[key].is_number_integer()) invalid(std::string(key) + " must be an integer", key);
  const int64_t v = body[key].get<int64_t>();
  if (v < lo || v > hi) invalid(std::string(key) + " must be between " + std::to_string(lo) + " and " + std::to_string(hi), key);
  return static_cast<int>(v);
}

// Sampling, limits and stop conditions shared by both endpoints.
void read_common(const json& body, Job& job, const Defaults& d) {
  if (!body.is_object()) invalid("the request body must be a JSON object");
  runtime::SamplingParams& sp = job.request.sampling;
  sp.temperature = static_cast<float>(number_in(body, "temperature", 0.0, 2.0, d.temperature));
  sp.top_p = static_cast<float>(number_in(body, "top_p", 1e-6, 1.0, d.top_p));
  sp.top_k = integer_in(body, "top_k", 0, 1 << 20, d.top_k);
  sp.min_p = static_cast<float>(number_in(body, "min_p", 0.0, 1.0, 0.0));
  sp.repetition_penalty = static_cast<float>(number_in(body, "repetition_penalty", 1e-3, 2.0, 1.0));
  sp.presence_penalty = static_cast<float>(number_in(body, "presence_penalty", -2.0, 2.0, 0.0));
  sp.frequency_penalty = static_cast<float>(number_in(body, "frequency_penalty", -2.0, 2.0, 0.0));
  if (body.contains("seed") && !body["seed"].is_null()) {
    if (!body["seed"].is_number_integer()) invalid("seed must be an integer", "seed");
    sp.seed = body["seed"].get<uint64_t>();
  } else {
    sp.seed = random_u64();
  }
  if (body.contains("n") && !body["n"].is_null() && integer_in(body, "n", 1, 128, 1) != 1)
    invalid("only n = 1 is supported", "n");
  if (body.contains("best_of") && !body["best_of"].is_null() && integer_in(body, "best_of", 1, 1, 1) != 1)
    invalid("only best_of = 1 is supported", "best_of");
  if (body.contains("response_format") && !body["response_format"].is_null()) {
    const std::string type = body["response_format"].value("type", "text");
    if (type != "text") invalid("response_format " + type + " is not supported yet", "response_format");
  }
  if (body.contains("stop") && !body["stop"].is_null()) {
    const json& stop = body["stop"];
    if (stop.is_string()) {
      job.request.stop_strings.push_back(stop.get<std::string>());
    } else if (stop.is_array()) {
      if (stop.size() > 4) invalid("at most 4 stop sequences", "stop");
      for (const auto& s : stop) {
        if (!s.is_string()) invalid("stop must be strings", "stop");
        job.request.stop_strings.push_back(s.get<std::string>());
      }
    } else {
      invalid("stop must be a string or an array of strings", "stop");
    }
  }
  for (const std::string& s : job.request.stop_strings)
    if (s.empty()) invalid("a stop sequence is empty", "stop");
}

// The token limit: max_tokens (or max_completion_tokens for chat); the engine trims it to the context.
void read_limit(const json& body, Job& job, const char* key) {
  if (body.contains(key) && !body[key].is_null())
    job.request.max_tokens = integer_in(body, key, 1, 1 << 24, 4096);
  else
    job.request.max_tokens = 4096;
}

json usage_json(int prompt, int completion) {
  return {{"prompt_tokens", prompt}, {"completion_tokens", completion}, {"total_tokens", prompt + completion}};
}

std::string text_of(const json& content, const char* what) {
  if (content.is_null()) return "";
  if (content.is_string()) return content.get<std::string>();
  if (!content.is_array()) invalid(std::string(what) + " must be a string or a list of text parts", "messages");
  std::string out;
  for (const auto& part : content) {
    const std::string type = part.value("type", "");
    if (type != "text") invalid("only text content is supported, not " + type, "messages");
    out += part.value("text", "");
  }
  return out;
}

std::vector<lx::text::Message> read_messages(const json& messages) {
  if (!messages.is_array() || messages.empty()) invalid("messages must be a non-empty list", "messages");
  std::vector<lx::text::Message> out;
  for (const auto& m : messages) {
    if (!m.is_object()) invalid("each message must be an object", "messages");
    lx::text::Message msg;
    msg.role = m.value("role", "");
    if (msg.role == "developer") msg.role = "system";
    if (msg.role != "system" && msg.role != "user" && msg.role != "assistant" && msg.role != "tool")
      invalid("unknown message role '" + msg.role + "'", "messages");
    msg.content = text_of(m.contains("content") ? m["content"] : json(nullptr), "content");
    msg.name = m.value("name", "");
    if (msg.role == "tool") {
      msg.tool_call_id = m.value("tool_call_id", "");
      if (msg.tool_call_id.empty()) invalid("a tool message needs tool_call_id", "messages");
    }
    if (msg.role == "assistant") {
      msg.reasoning = m.value("reasoning_content", "");
      if (m.contains("tool_calls") && m["tool_calls"].is_array()) {
        for (const auto& c : m["tool_calls"]) {
          lx::text::ToolCall call;
          call.id = c.value("id", "");
          const json fn = c.value("function", json::object());
          call.name = fn.value("name", "");
          if (call.name.empty()) invalid("a tool call needs a function name", "messages");
          call.arguments_json = fn.contains("arguments") && fn["arguments"].is_string() ? fn["arguments"].get<std::string>() : "{}";
          msg.tool_calls.push_back(call);
        }
      }
    }
    out.push_back(std::move(msg));
  }
  return out;
}

// The tools the template should show: none, all of them, or the one a named tool_choice picks.
std::string read_tools(const json& body, bool& any) {
  any = false;
  if (!body.contains("tools") || body["tools"].is_null()) return "";
  if (!body["tools"].is_array()) invalid("tools must be a list", "tools");
  json tools = body["tools"];
  std::string choice_name;
  bool none = false;
  if (body.contains("tool_choice") && !body["tool_choice"].is_null()) {
    const json& tc = body["tool_choice"];
    if (tc.is_string()) {
      const std::string s = tc.get<std::string>();
      if (s == "none") none = true;
      else if (s != "auto" && s != "required") invalid("tool_choice must be none, auto or required", "tool_choice");
    } else if (tc.is_object()) {
      choice_name = tc.value("function", json::object()).value("name", "");
      if (choice_name.empty()) invalid("tool_choice names no function", "tool_choice");
    } else {
      invalid("tool_choice must be a string or an object", "tool_choice");
    }
  }
  if (none || tools.empty()) return "";
  if (!choice_name.empty()) {
    json only = json::array();
    for (const auto& t : tools)
      if (t.value("function", json::object()).value("name", "") == choice_name) only.push_back(t);
    if (only.empty()) invalid("tool_choice names " + choice_name + ", which is not in tools", "tool_choice");
    tools = only;
  }
  any = true;
  return tools.dump();
}

const char* finish_name(runtime::FinishReason r, bool tools) {
  if (tools) return "tool_calls";
  return r == runtime::FinishReason::Length ? "length" : "stop";
}

// The logprobs block of one token, in chat form.
json chat_logprob_entry(const TokenDelta& d, const lx::text::Tokenizer* tok) {
  json entry = {{"token", d.piece}, {"logprob", d.logprob}, {"bytes", bytes_array(d.piece)}};
  json top = json::array();
  for (const auto& [id, lp] : d.top) {
    const std::string piece = tok ? tok->decode({id}, false) : std::to_string(id);
    top.push_back({{"token", piece}, {"logprob", lp}, {"bytes", bytes_array(piece)}});
  }
  entry["top_logprobs"] = top;
  return entry;
}

}  // namespace

// ---- streaming -------------------------------------------------------------------------------

// Tokens arrive on the engine thread; the connection's thread takes them here.
struct Pending {
  std::mutex mu;
  std::condition_variable cv;
  std::deque<TokenDelta> deltas;
  bool finished = false;
  RequestResult result;
};

namespace {

// Turns one request's deltas into the JSON the endpoint speaks.
class Wire {
 public:
  Wire(const Job& job, const lx::text::Tokenizer* tokenizer, std::string model)
      : chat_(job.chat),
        logprobs_(job.logprobs),
        include_usage_(job.include_usage),
        tok_(tokenizer),
        model_(std::move(model)),
        created_(unix_time()) {
    id_ = (chat_ ? "chatcmpl-" : "cmpl-") + random_hex(24);
  }

  const std::string& id() const { return id_; }

  // The JSON chunks for a batch of deltas, as one string of SSE events.
  std::string events(const std::vector<TokenDelta>& batch) {
    std::string out;
    for (const TokenDelta& d : batch) {
      if (!role_sent_) {
        role_sent_ = true;
        out += sse(chat_ ? chunk_with(json::object({{"role", "assistant"}, {"content", ""}}), nullptr, nullptr)
                             : chunk_text("", nullptr, nullptr));
      }
      if (!d.reasoning.empty()) out += sse(chunk_with(json{{"reasoning_content", d.reasoning}}, nullptr, nullptr));
      if (!d.content.empty() || (logprobs_ && d.token >= 0)) {
        out += sse(chat_ ? chunk_with(json{{"content", d.content}}, nullptr, token_logprobs(d))
                             : chunk_text(d.content, nullptr, token_logprobs(d)));
      }
      for (const auto& call : d.tool_calls) {
        const std::string call_id = call.id.empty() ? "call_" + random_hex(24) : call.id;
        json tc = {{"index", tool_index_++},
                   {"id", call_id},
                   {"type", "function"},
                   {"function", {{"name", call.name}, {"arguments", call.arguments_json}}}};
        out += sse(chunk_with(json{{"tool_calls", json::array({tc})}}, nullptr, nullptr));
        tools_ = true;
      }
    }
    return out;
  }

  // The last chunk, with the reason, the usage when asked for, and [DONE].
  std::string finish(const RequestResult& r) {
    std::string out;
    if (r.reason == runtime::FinishReason::Error) {
      out += sse(json{{"error", {{"message", r.error}, {"type", "server_error"}, {"param", nullptr}, {"code", nullptr}}}});
      out += "data: [DONE]\n\n";
      return out;
    }
    if (!role_sent_) out += sse(chat_ ? chunk_with(json::object({{"role", "assistant"}, {"content", ""}}), nullptr, nullptr) : chunk_text("", nullptr, nullptr));
    const char* reason = finish_name(r.reason, tools_);
    out += sse(chat_ ? chunk_with(json::object(), reason, nullptr, true) : chunk_text("", reason, nullptr));
    if (include_usage_) {
      json usage_chunk = base_chunk();
      usage_chunk["choices"] = json::array();
      usage_chunk["usage"] = usage_json(r.prompt_tokens, r.completion_tokens);
      out += sse(usage_chunk);
    }
    out += "data: [DONE]\n\n";
    return out;
  }

  // The whole reply, not streamed.
  json whole(const std::vector<TokenDelta>& deltas, const RequestResult& r) {
    std::string content, reasoning;
    std::vector<lx::text::ToolCall> calls;
    json logprob_items = json::array();
    json top_items = json::array();
    for (const TokenDelta& d : deltas) {
      content += d.content;
      reasoning += d.reasoning;
      for (const auto& c : d.tool_calls) calls.push_back(c);
      if (logprobs_ && d.token >= 0) {
        if (chat_) logprob_items.push_back(chat_logprob_entry(d, tok_));
        else {
          logprob_items.push_back(d.piece);
          top_items.push_back(legacy_top(d));
        }
      }
    }
    const bool tools = !calls.empty();
    json choice = json::object();
    choice["index"] = 0;
    if (chat_) {
      json message = {{"role", "assistant"}, {"content", content.empty() && tools ? json(nullptr) : json(content)}};
      if (!reasoning.empty()) message["reasoning_content"] = reasoning;
      if (tools) {
        json list = json::array();
        int index = 0;
        for (const auto& c : calls) {
          list.push_back({{"index", index++},
                          {"id", c.id.empty() ? "call_" + random_hex(24) : c.id},
                          {"type", "function"},
                          {"function", {{"name", c.name}, {"arguments", c.arguments_json}}}});
        }
        message["tool_calls"] = list;
      }
      choice["message"] = message;
      choice["logprobs"] = logprobs_ ? json{{"content", logprob_items}} : json(nullptr);
    } else {
      choice["text"] = content;
      choice["logprobs"] = logprobs_ ? legacy_logprobs(deltas) : json(nullptr);
    }
    choice["finish_reason"] = finish_name(r.reason, tools);
    json out = base_chunk();
    out["object"] = chat_ ? "chat.completion" : "text_completion";
    out["choices"] = json::array({choice});
    out["usage"] = usage_json(r.prompt_tokens, r.completion_tokens);
    return out;
  }

 private:
  json base_chunk() const {
    return {{"id", id_}, {"object", chat_ ? "chat.completion.chunk" : "text_completion"}, {"created", created_}, {"model", model_}};
  }

  json token_logprobs(const TokenDelta& d) const {
    if (!logprobs_ || d.token < 0) return json(nullptr);
    if (chat_) return json{{"content", json::array({chat_logprob_entry(d, tok_)})}};
    return json{{"tokens", json::array({d.piece})}, {"token_logprobs", json::array({d.logprob})},
                {"top_logprobs", json::array({legacy_top(d)})}, {"text_offset", json::array({0})}};
  }

  json legacy_top(const TokenDelta& d) const {
    json top = json::object();
    for (const auto& [id, lp] : d.top) top[tok_ ? tok_->decode({id}, false) : std::to_string(id)] = lp;
    return top;
  }

  json legacy_logprobs(const std::vector<TokenDelta>& deltas) const {
    json tokens = json::array(), lps = json::array(), tops = json::array(), offsets = json::array();
    for (const TokenDelta& d : deltas) {
      if (d.token < 0) continue;
      tokens.push_back(d.piece);
      lps.push_back(d.logprob);
      tops.push_back(legacy_top(d));
      offsets.push_back(0);
    }
    return {{"tokens", tokens}, {"token_logprobs", lps}, {"top_logprobs", tops}, {"text_offset", offsets}};
  }

  json chunk_with(json delta, const char* finish, json logprobs, bool is_final = false) const {
    json c = base_chunk();
    json choice = {{"index", 0}, {"delta", delta}, {"finish_reason", finish ? json(finish) : json(nullptr)}};
    if (logprobs.is_object()) choice["logprobs"] = logprobs;
    c["choices"] = json::array({choice});
    (void)is_final;
    return c;
  }

  json chunk_text(const std::string& text, const char* finish, json logprobs) const {
    json c = base_chunk();
    json choice = {{"index", 0}, {"text", text}, {"finish_reason", finish ? json(finish) : json(nullptr)}};
    if (logprobs.is_object()) choice["logprobs"] = logprobs;
    c["choices"] = json::array({choice});
    return c;
  }

  static std::string sse(const json& j) { return "data: " + j.dump() + "\n\n"; }

  // Copies of what the wire format needs: the request itself may be gone by the time a stream ends.
  bool chat_;
  bool logprobs_;
  bool include_usage_;
  const lx::text::Tokenizer* tok_;
  std::string model_;
  int64_t created_;
  std::string id_;
  bool role_sent_ = false;
  bool tools_ = false;
  int tool_index_ = 0;
};

}  // namespace

// ---- the server ------------------------------------------------------------------------------

Server::Server(runtime::Engine& engine, ServerOptions options)
    : engine_(engine), options_(std::move(options)), http_(std::make_unique<httplib::Server>()) {
  http_->new_task_queue = [n = options_.threads] { return new httplib::ThreadPool(n); };
  http_->set_payload_max_length(64u << 20);
  http_->set_read_timeout(std::chrono::seconds(60));
  http_->set_write_timeout(std::chrono::seconds(60));
}

Server::~Server() { stop(); }

void Server::stop() {
  if (http_) http_->stop();
}

bool Server::run() {
  const lx::text::Tokenizer* tokenizer = engine_.tokenizer();
  const std::string model = options_.model_name;
  const Defaults defaults = read_defaults(options_.model_dir);

  auto error_to = [](httplib::Response& res, int status, const std::string& type, const std::string& message,
                     const std::string& param) {
    res.status = status;
    res.set_content(json{{"error", {{"message", message}, {"type", type}, {"param", param.empty() ? json(nullptr) : json(param)}, {"code", nullptr}}}}.dump(),
                    "application/json");
  };

  // Shared by both completion endpoints: submits and answers, streaming or not.
  auto serve = [this, tokenizer, error_to](Job job, httplib::Response& res) {
    auto state = std::make_shared<Pending>();
    runtime::RequestCallbacks cb;
    cb.on_token = [state](const TokenDelta& d) {
      std::lock_guard<std::mutex> lock(state->mu);
      state->deltas.push_back(d);
      state->cv.notify_one();
    };
    cb.on_finish = [state](const RequestResult& r) {
      std::lock_guard<std::mutex> lock(state->mu);
      state->result = r;
      state->finished = true;
      state->cv.notify_one();
    };
    const uint64_t id = engine_.submit(std::move(job.request), std::move(cb));

    auto wire = std::make_shared<Wire>(job, tokenizer, job.model);
    if (job.stream) {
      // The connection's thread drains the deltas and writes them out. Losing the client cancels.
      res.set_chunked_content_provider(
          "text/event-stream",
          [this, state, wire, id](size_t, httplib::DataSink& sink) {
            std::vector<TokenDelta> batch;
            RequestResult result;
            bool finished = false;
            {
              std::unique_lock<std::mutex> lock(state->mu);
              state->cv.wait(lock, [&] { return !state->deltas.empty() || state->finished; });
              batch.assign(state->deltas.begin(), state->deltas.end());
              state->deltas.clear();
              finished = state->finished;
              result = state->result;
            }
            if (finished) {
              const std::string tail = wire->events(batch) + wire->finish(result);
              sink.write(tail.data(), tail.size());
              sink.done();
              return true;
            }
            const std::string text = wire->events(batch);
            if (!text.empty() && !sink.write(text.data(), text.size())) {
              engine_.cancel(id);
              return false;
            }
            return true;
          },
          [this, id](bool success) {
            if (!success) engine_.cancel(id);
          });
      return;
    }

    std::vector<TokenDelta> deltas;
    RequestResult result;
    {
      std::unique_lock<std::mutex> lock(state->mu);
      state->cv.wait(lock, [&] { return state->finished; });
      deltas.assign(state->deltas.begin(), state->deltas.end());
      result = state->result;
    }
    if (result.reason == runtime::FinishReason::Error) {
      error_to(res, 500, "server_error", result.error, "");
      return;
    }
    res.status = 200;
    res.set_content(wire->whole(deltas, result).dump(), "application/json");
  };

  http_->Get("/health", [this](const httplib::Request&, httplib::Response& res) {
    const runtime::EngineStats st = engine_.stats();
    json health = {{"status", "ok"}, {"running", st.running}, {"waiting", st.waiting}, {"slots", st.slots}};
    res.set_content(health.dump(), "application/json");
  });

  http_->Get("/v1/models", [model](const httplib::Request&, httplib::Response& res) {
    json list = {{"object", "list"},
                 {"data", json::array({{{"id", model}, {"object", "model"}, {"created", unix_time()}, {"owned_by", "lernex"}}})}};
    res.set_content(list.dump(), "application/json");
  });

  auto checked_model = [model](const json& body) {
    if (!body.contains("model") || body["model"].is_null()) return model;
    const std::string requested = body["model"].is_string() ? body["model"].get<std::string>() : "";
    if (requested != model)
      throw ApiError(404, "invalid_request_error", "the model `" + requested + "` does not exist; this server serves `" + model + "`", "model");
    return model;
  };

  http_->Post("/v1/chat/completions", [=, &defaults](const httplib::Request& req, httplib::Response& res) {
    try {
      const json body = json::parse(req.body);
      Job job;
      job.chat = true;
      job.model = checked_model(body);
      read_common(body, job, defaults);
      read_limit(body, job, body.contains("max_completion_tokens") ? "max_completion_tokens" : "max_tokens");
      job.stream = body.value("stream", false);
      job.include_usage = body.value("stream_options", json::object()).value("include_usage", false);
      job.logprobs = body.value("logprobs", false);
      job.top_logprobs = integer_in(body, "top_logprobs", 0, runtime::kMaxTopLogprobs, 0);
      if (job.top_logprobs > 0 && !job.logprobs) invalid("top_logprobs needs logprobs: true", "top_logprobs");
      job.request.sampling.logprobs = job.logprobs;
      job.request.sampling.top_logprobs = job.top_logprobs;

      bool has_tools = false;
      const std::string tools_text = read_tools(body, has_tools);
      bool thinking = true;
      if (body.contains("chat_template_kwargs") && body["chat_template_kwargs"].is_object())
        thinking = body["chat_template_kwargs"].value("enable_thinking", true);
      job.request.thinking = thinking;

      lx::text::ChatOptions options;
      options.thinking = thinking;
      options.tools_json = tools_text;
      if (!body.contains("messages")) invalid("messages is required", "messages");
      const std::string prompt = lx::text::render_chat(read_messages(body["messages"]), options);
      if (!tokenizer) throw ApiError(500, "server_error", "this server has no tokenizer");
      job.request.prompt = tokenizer->encode(prompt, true);
      serve(std::move(job), res);
    } catch (const json::exception& e) {
      error_to(res, 400, "invalid_request_error", std::string("the body is not valid JSON: ") + e.what(), "");
    } catch (const ApiError& e) {
      error_to(res, e.status, e.type, e.what(), e.param);
    } catch (const std::invalid_argument& e) {
      error_to(res, 400, "invalid_request_error", e.what(), "");
    } catch (const std::runtime_error& e) {
      const bool busy = std::string(e.what()).find("queue is full") != std::string::npos;
      error_to(res, busy ? 503 : 500, busy ? "server_busy" : "server_error", e.what(), "");
    }
  });

  http_->Post("/v1/completions", [=, &defaults](const httplib::Request& req, httplib::Response& res) {
    try {
      const json body = json::parse(req.body);
      Job job;
      job.chat = false;
      job.model = checked_model(body);
      read_common(body, job, defaults);
      read_limit(body, job, "max_tokens");
      job.stream = body.value("stream", false);
      job.include_usage = body.value("stream_options", json::object()).value("include_usage", false);
      job.logprobs = body.contains("logprobs") && !body["logprobs"].is_null() && integer_in(body, "logprobs", 0, runtime::kMaxTopLogprobs, 0) > 0;
      job.top_logprobs = job.logprobs ? integer_in(body, "logprobs", 0, runtime::kMaxTopLogprobs, 0) : 0;
      job.request.sampling.logprobs = job.logprobs;
      job.request.sampling.top_logprobs = job.top_logprobs;
      job.request.thinking = false;
      if (!body.contains("prompt") || body["prompt"].is_null()) invalid("prompt is required", "prompt");
      const json& prompt = body["prompt"];
      if (prompt.is_string()) {
        if (!tokenizer) throw ApiError(500, "server_error", "this server has no tokenizer");
        job.request.prompt = tokenizer->encode(prompt.get<std::string>(), true);
      } else if (prompt.is_array() && !prompt.empty() && prompt[0].is_number_integer()) {
        for (const auto& id : prompt) job.request.prompt.push_back(id.get<int32_t>());
      } else if (prompt.is_array() && prompt.size() == 1 && prompt[0].is_string()) {
        if (!tokenizer) throw ApiError(500, "server_error", "this server has no tokenizer");
        job.request.prompt = tokenizer->encode(prompt[0].get<std::string>(), true);
      } else {
        invalid("prompt must be a string or a list of token ids (one prompt per request)", "prompt");
      }
      serve(std::move(job), res);
    } catch (const json::exception& e) {
      error_to(res, 400, "invalid_request_error", std::string("the body is not valid JSON: ") + e.what(), "");
    } catch (const ApiError& e) {
      error_to(res, e.status, e.type, e.what(), e.param);
    } catch (const std::invalid_argument& e) {
      error_to(res, 400, "invalid_request_error", e.what(), "");
    } catch (const std::runtime_error& e) {
      const bool busy = std::string(e.what()).find("queue is full") != std::string::npos;
      error_to(res, busy ? 503 : 500, busy ? "server_busy" : "server_error", e.what(), "");
    }
  });

  // Unknown paths get the same JSON errors as everything else; a handler's own error is kept.
  http_->set_error_handler([error_to](const httplib::Request&, httplib::Response& res) {
    if (!res.body.empty()) return;
    error_to(res, res.status ? res.status : 404, "not_found", "no such endpoint", "");
  });

  if (!http_->bind_to_port(options_.host, options_.port)) {
    std::fprintf(stderr, "lernex: cannot listen on %s:%d (is the port in use?)\n", options_.host.c_str(), options_.port);
    return false;
  }
  bound_port_ = options_.port;
  http_->listen_after_bind();
  return true;
}

}  // namespace lx::server
