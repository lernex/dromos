// render_chat: the model folder's own chat template, rendered by the Jinja interpreter with the
// variables transformers' apply_chat_template passes. Nothing about a template is written in C++, so a
// new Metis generation with a new chat_template.jinja renders without a code change.
#include <fstream>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "text/json.h"
#include "text/jinja.h"
#include "text/jinja_internal.h"
#include "text/tokenizer.h"

namespace lx::text {
namespace {

bool read_file(const std::string& path, std::string& out) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::ostringstream buffer;
  buffer << in.rdbuf();
  out = buffer.str();
  return true;
}

// tokenizer_config.json, parsed; empty when the file is missing.
JValue tokenizer_config(const std::string& model_dir) {
  std::string text;
  JValue config;
  if (read_file(model_dir + "/tokenizer_config.json", text)) json_loads(text, config);
  return config;
}

// The template text: chat_template.jinja when the folder has one, otherwise tokenizer_config.json's
// chat_template (a string, or a list of named templates of which "default" is used).
std::string template_source(const std::string& model_dir) {
  std::string text;
  if (read_file(model_dir + "/chat_template.jinja", text)) return text;
  const JValue config = tokenizer_config(model_dir);
  const JValue* chat = config.get("chat_template");
  if (chat != nullptr && chat->type == JValue::Type::kString) return chat->text;
  if (chat != nullptr && chat->type == JValue::Type::kArray) {
    for (const JValue& entry : chat->items) {
      const JValue* name = entry.get("name");
      const JValue* body = entry.get("template");
      if (name != nullptr && name->text == "default" && body != nullptr) return body->text;
    }
  }
  throw std::runtime_error("no chat template in " + model_dir);
}

// The special tokens transformers puts in the template's variables (special_tokens_map): every one that is set.
std::vector<std::pair<std::string, jinja::Value>> special_tokens(const std::string& model_dir) {
  static const char* kNames[] = {"bos_token", "eos_token", "unk_token", "sep_token", "pad_token", "cls_token",
                                 "mask_token"};
  const JValue config = tokenizer_config(model_dir);
  std::vector<std::pair<std::string, jinja::Value>> out;
  for (const char* name : kNames) {
    const JValue* v = config.get(name);
    if (v == nullptr) continue;
    if (v->type == JValue::Type::kString) {
      out.emplace_back(name, jinja::Value::string(v->text));
    } else if (v->type == JValue::Type::kObject) {
      const JValue* content = v->get("content");
      if (content != nullptr && content->type == JValue::Type::kString) {
        out.emplace_back(name, jinja::Value::string(content->text));
      }
    }
  }
  const JValue* extra = config.get("additional_special_tokens");
  if (extra != nullptr && extra->type == JValue::Type::kArray) {
    std::vector<jinja::Value> items;
    for (const JValue& item : extra->items) {
      if (item.type == JValue::Type::kString) items.push_back(jinja::Value::string(item.text));
    }
    out.emplace_back("additional_special_tokens", jinja::Value::list(std::move(items)));
  }
  return out;
}

// A message as transformers' chat template sees it: the keys transformers' messages carry, with
// reasoning under "reasoning_content" and tool calls as {"type", "function": {"name", "arguments"}}.
jinja::Value message_value(const Message& m) {
  std::vector<std::pair<jinja::Value, jinja::Value>> items;
  items.emplace_back(jinja::Value::string("role"), jinja::Value::string(m.role));
  items.emplace_back(jinja::Value::string("content"), jinja::Value::string(m.content));
  if (!m.reasoning.empty()) {
    items.emplace_back(jinja::Value::string("reasoning_content"), jinja::Value::string(m.reasoning));
  }
  if (!m.tool_calls.empty()) {
    std::vector<jinja::Value> calls;
    for (const ToolCall& call : m.tool_calls) {
      std::vector<std::pair<jinja::Value, jinja::Value>> fn;
      fn.emplace_back(jinja::Value::string("name"), jinja::Value::string(call.name));
      JValue args;
      jinja::Value arguments = jinja::Value::string(call.arguments_json);
      if (json_loads(call.arguments_json, args)) arguments = jinja::value_from_json(args);
      fn.emplace_back(jinja::Value::string("arguments"), arguments);
      std::vector<std::pair<jinja::Value, jinja::Value>> entry;
      if (!call.id.empty()) entry.emplace_back(jinja::Value::string("id"), jinja::Value::string(call.id));
      entry.emplace_back(jinja::Value::string("type"), jinja::Value::string("function"));
      entry.emplace_back(jinja::Value::string("function"), jinja::Value::dict(std::move(fn)));
      calls.push_back(jinja::Value::dict(std::move(entry)));
    }
    items.emplace_back(jinja::Value::string("tool_calls"), jinja::Value::list(std::move(calls)));
  }
  if (!m.tool_call_id.empty()) {
    items.emplace_back(jinja::Value::string("tool_call_id"), jinja::Value::string(m.tool_call_id));
  }
  if (!m.name.empty()) items.emplace_back(jinja::Value::string("name"), jinja::Value::string(m.name));
  return jinja::Value::dict(std::move(items));
}

// Compiled templates by source text: a folder's template is parsed once.
std::shared_ptr<const jinja::Template> compiled(const std::string& source) {
  static std::mutex mutex;
  static std::map<std::string, std::shared_ptr<const jinja::Template>> cache;
  std::lock_guard<std::mutex> lock(mutex);
  auto it = cache.find(source);
  if (it != cache.end()) return it->second;
  if (cache.size() >= 32) cache.clear();
  auto tpl = std::make_shared<const jinja::Template>(source);
  cache.emplace(source, tpl);
  return tpl;
}

}  // namespace

std::string render_chat(const std::vector<Message>& messages, const ChatOptions& options) {
  if (options.model_dir.empty()) throw std::invalid_argument("render_chat needs ChatOptions::model_dir");
  // transformers' apply_chat_template refuses an empty conversation before it renders anything.
  if (messages.empty()) throw std::invalid_argument("Cannot apply chat template to an empty conversation.");
  const auto tpl = compiled(template_source(options.model_dir));

  std::vector<jinja::Value> list;
  list.reserve(messages.size());
  for (const Message& m : messages) list.push_back(message_value(m));

  jinja::Value tools;  // None, as transformers passes it when there are no tools
  if (!options.tools_json.empty()) {
    JValue parsed;
    if (!json_loads(options.tools_json, parsed) || parsed.type != JValue::Type::kArray) {
      throw std::invalid_argument("ChatOptions::tools_json must be a JSON array");
    }
    tools = jinja::value_from_json(parsed);
  } else {
    tools = jinja::Value::none();
  }

  std::vector<std::pair<std::string, jinja::Value>> variables;
  variables.emplace_back("messages", jinja::Value::list(std::move(list)));
  variables.emplace_back("tools", tools);
  variables.emplace_back("documents", jinja::Value::none());
  variables.emplace_back("add_generation_prompt", jinja::Value::boolean(options.add_generation_prompt));
  variables.emplace_back("enable_thinking", jinja::Value::boolean(options.thinking));
  for (auto& special : special_tokens(options.model_dir)) variables.push_back(std::move(special));
  return tpl->render(variables);
}

}  // namespace lx::text
