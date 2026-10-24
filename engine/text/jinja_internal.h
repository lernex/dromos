// Helpers shared by the Jinja interpreter's translation units: Python value operations, the filters,
// tests, methods and globals, and the JSON bridge. Not part of the public interface.
#pragma once

#include <string>
#include <string_view>

#include "text/jinja.h"

namespace lx::text {
struct JValue;
}

namespace lx::text::jinja {

// Python operators. `op` is one of: + - * / (true division) d (//) % p (**) ~ (string concatenation).
Value py_arith(char op, const Value& a, const Value& b);
bool py_equal(const Value& a, const Value& b);
// a < b. Throws Error for types Python cannot order.
bool py_less(const Value& a, const Value& b);
// item in container.
bool py_contains(const Value& container, const Value& item);
// The slice container[start:stop:step]; a missing bound is nullptr.
Value py_slice(const Value& container, const Value* start, const Value* stop, const Value* step);
// container[index] (Jinja: a missing item is undefined).
Value py_item(const Value& container, const Value& index);
// The value of attribute `name` on a non-object value (dict keys, str/list/dict methods). Undefined
// when there is none.
Value py_attr(const Value& object, const std::string& name);

// JSON: the JSON text Python's json.dumps writes for a value (ensure_ascii, indent, sort_keys and
// separators as given), and the value for a parsed JSON document.
struct JsonOptions {
  bool ensure_ascii = false;
  bool sort_keys = false;
  int indent = -1;  // -1: none
  std::string item_separator = ", ";
  std::string key_separator = ": ";
};
std::string json_dumps_value(const Value& value, const JsonOptions& options);
Value value_from_json(const lx::text::JValue& json);

// Filters, tests, methods and globals (jinja_filters.cpp).
Value apply_filter(const std::string& name, const Value& subject, const CallArgs& args);
bool apply_test(const std::string& name, const Value& subject, const CallArgs& args);
// The method `name` of a str, list or dict receiver as a callable object; `found` is false when the
// receiver has no such method.
Value bound_method(const Value& receiver, const std::string& name, bool& found);
bool is_global(const std::string& name);
Value call_global(const std::string& name, const CallArgs& args);
// A namespace() object holding the keyword arguments as attributes.
Value make_namespace(const CallArgs& args);
// True when `name` is a dict method, which Python's getattr finds before a dict key.
bool is_dict_method(const std::string& name);

}  // namespace lx::text::jinja
