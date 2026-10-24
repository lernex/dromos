// JSON with Python's semantics, because the template and the tool-call parser are specified by
// what Python's json module does. json_loads reads what json.loads reads (NaN and Infinity
// included; a duplicate key keeps its first position and its last value); json_dumps writes what
// json.dumps(value, ensure_ascii=False) writes, default separators included.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

namespace lx::text {

struct JValue {
  enum class Type : uint8_t { kNull, kBool, kInt, kFloat, kString, kArray, kObject };

  Type type = Type::kNull;
  bool boolean = false;
  double number = 0.0;            // kFloat
  std::string text;               // kInt: canonical decimal digits; kString: UTF-8
  std::vector<JValue> items;      // kArray
  std::vector<std::string> keys;  // kObject: unique, in insertion order
  std::vector<JValue> values;     // kObject: values[i] belongs to keys[i]

  bool is_null() const { return type == Type::kNull; }
  bool is_string() const { return type == Type::kString; }
  bool is_object() const { return type == Type::kObject; }
  bool is_array() const { return type == Type::kArray; }

  // The member under key, or nullptr when this is not an object or has no such key.
  const JValue* get(std::string_view key) const;

  static JValue string(std::string s);
  static JValue integer(std::string canonical_digits);
  static JValue flt(double v);
  static JValue boolean_value(bool v);
};

// Parses one JSON value that spans the whole text (surrounding JSON whitespace allowed).
bool json_loads(std::string_view text, JValue& out);

void json_dumps(const JValue& value, std::string& out);
std::string json_dumps(const JValue& value);

// json.dumps of a str with ensure_ascii=False: quotes, backslashes and control characters escaped.
void json_append_string(std::string& out, std::string_view s);

// repr() of a finite double, as json.dumps writes a float.
std::string py_float_repr(double value);

// True when no NaN or infinity sits anywhere in the value (json.dumps with allow_nan=False would
// accept it).
bool json_is_finite(const JValue& value);

}  // namespace lx::text
