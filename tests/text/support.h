// Shared helpers for the text tests: the golden directory, JSON lines, and hex.
#pragma once

#include <cstdlib>
#include <fstream>
#include <functional>
#include <string>
#include <string_view>

#include "text/json.h"

namespace lx::text::test {

// The golden directory from LX_TEXT_GOLDEN, or empty when the goldens are not generated.
inline std::string golden_dir() {
  const char* env = std::getenv("LX_TEXT_GOLDEN");
  return env ? std::string(env) : std::string();
}

// The Metis model folder (tokenizer.json, chat_template.jinja) from LX_TEXT_MODEL.
inline std::string model_dir() {
  const char* env = std::getenv("LX_TEXT_MODEL");
  return env ? std::string(env) : std::string();
}

// Calls fn for each non-empty line of a JSON-lines file, parsed. Returns false when the file cannot
// be read.
inline bool for_each_json_line(const std::string& path, const std::function<void(const JValue&)>& fn) {
  std::ifstream in(path, std::ios::binary);
  if (!in) return false;
  std::string line;
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    JValue value;
    if (!json_loads(line, value)) return false;
    fn(value);
  }
  return true;
}

inline std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
  return text;
}

// The string member `key` of an object, or empty.
inline std::string str_of(const JValue& v, std::string_view key) {
  const JValue* m = v.get(key);
  return m != nullptr && m->type == JValue::Type::kString ? m->text : std::string();
}

inline std::string from_hex(std::string_view hex) {
  std::string out;
  for (size_t i = 0; i + 1 < hex.size(); i += 2) {
    out.push_back(static_cast<char>(std::stoi(std::string(hex.substr(i, 2)), nullptr, 16)));
  }
  return out;
}

}  // namespace lx::text::test
