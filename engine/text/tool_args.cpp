#include "text/tool_args.h"

#include <cmath>
#include <cstdio>
#include <vector>

#include "text/py.h"
#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text {
namespace {

constexpr size_t npos = std::string_view::npos;

size_t skip_regex_space(std::string_view s, size_t i) {
  while (i < s.size()) {
    size_t len = 0;
    const uint32_t cp = utf8_decode(s, i, len);
    if (!uni::is_regex_space(cp)) break;
    i += len;
  }
  return i;
}

bool word_at(std::string_view s, size_t i, std::string_view word) {
  return i <= s.size() && s.substr(i, word.size()) == word;
}

// <\s*parameter\s*=  : the index just past the '=', or npos.
size_t open_tag(std::string_view s, size_t i) {
  if (i >= s.size() || s[i] != '<') return npos;
  size_t j = skip_regex_space(s, i + 1);
  if (!word_at(s, j, "parameter")) return npos;
  j = skip_regex_space(s, j + 9);
  if (j >= s.size() || s[j] != '=') return npos;
  return j + 1;
}

// <\s*/\s*parameter\s*>  : the index just past the '>', or npos.
size_t close_tag(std::string_view s, size_t i) {
  if (i >= s.size() || s[i] != '<') return npos;
  size_t j = skip_regex_space(s, i + 1);
  if (j >= s.size() || s[j] != '/') return npos;
  j = skip_regex_space(s, j + 1);
  if (!word_at(s, j, "parameter")) return npos;
  j = skip_regex_space(s, j + 9);
  if (j >= s.size() || s[j] != '>') return npos;
  return j + 1;
}

// The same search the reference's regex does, one match at a time:
//   <\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))   (DOTALL)
struct ParamMatch {
  std::string_view name;   // raw, before strip()
  std::string_view value;  // raw, before the newline trim
  size_t end;              // where the next search starts
};

bool next_param(std::string_view s, size_t from, ParamMatch& m) {
  for (size_t start = from; start < s.size(); ++start) {
    if (s[start] != '<') continue;
    const size_t after_eq = open_tag(s, start);
    if (after_eq == npos) continue;
    const size_t gt = s.find('>', after_eq);
    if (gt == npos) continue;
    const size_t value_begin = gt + 1;
    // The lazy value ends at the first close tag or open tag that follows it.
    for (size_t q = value_begin; q < s.size(); ++q) {
      if (s[q] != '<') continue;
      const size_t close_end = close_tag(s, q);
      if (close_end != npos) {
        m = ParamMatch{s.substr(after_eq, gt - after_eq), s.substr(value_begin, q - value_begin), close_end};
        return true;
      }
      if (open_tag(s, q) != npos) {
        m = ParamMatch{s.substr(after_eq, gt - after_eq), s.substr(value_begin, q - value_begin), q};
        return true;
      }
    }
  }
  return false;
}

// _trim_template_newlines: the template wraps every value in one newline on each side.
std::string_view trim_template_newlines(std::string_view v) {
  if (!v.empty() && v.front() == '\n') v.remove_prefix(1);
  if (!v.empty() && v.back() == '\n') v.remove_suffix(1);
  return v;
}

void set_param(JValue& object, const std::string& key, std::string value) {
  for (size_t k = 0; k < object.keys.size(); ++k) {
    if (object.keys[k] == key) {
      object.values[k] = JValue::string(std::move(value));
      return;
    }
  }
  object.keys.push_back(key);
  object.values.push_back(JValue::string(std::move(value)));
}

// ---- schema typing -----------------------------------------------------------------------------

struct Coerced {
  bool identity;  // the value came back as the very string it was given: nothing changed
  JValue value;
};

std::string normalize_type(std::string_view raw) {
  const std::string lower = py_lower_ascii(py_strip(raw));
  static const struct {
    const char* alias;
    const char* name;
  } kAliases[] = {
      {"str", "string"},     {"text", "string"},     {"varchar", "string"}, {"char", "string"},
      {"enum", "string"},    {"int", "integer"},     {"int32", "integer"},  {"int64", "integer"},
      {"uint", "integer"},   {"uint32", "integer"},  {"uint64", "integer"}, {"long", "integer"},
      {"short", "integer"},  {"unsigned", "integer"}, {"float", "number"},  {"float32", "number"},
      {"float64", "number"}, {"double", "number"},   {"bool", "boolean"},   {"dict", "object"},
      {"arr", "array"},      {"list", "array"},      {"sequence", "array"},
  };
  for (const auto& a : kAliases) {
    if (lower == a.alias) return a.name;
  }
  return lower;
}

// extract_types_from_schema: the JSON types a property schema admits, "string" when it names none.
std::vector<std::string> extract_types(const JValue& schema) {
  std::vector<std::string> types;
  auto add = [&types](const std::string& t) {
    for (const auto& have : types) {
      if (have == t) return;
    }
    types.push_back(t);
  };
  if (!schema.is_object()) return {"string"};
  if (const JValue* t = schema.get("type")) {
    if (t->is_string()) {
      add(t->text);
    } else if (t->is_array()) {
      for (const JValue& item : t->items) {
        if (item.is_string()) add(item.text);
      }
    }
  }
  if (const JValue* e = schema.get("enum"); e != nullptr && e->is_array() && !e->items.empty()) {
    for (const JValue& v : e->items) {
      switch (v.type) {
        case JValue::Type::kNull: add("null"); break;
        case JValue::Type::kBool: add("boolean"); break;
        case JValue::Type::kInt: add("integer"); break;
        case JValue::Type::kFloat: add("number"); break;
        case JValue::Type::kString: add("string"); break;
        case JValue::Type::kArray: add("array"); break;
        case JValue::Type::kObject: add("object"); break;
      }
    }
  }
  for (const char* field : {"anyOf", "oneOf", "allOf"}) {
    const JValue* choices = schema.get(field);
    if (choices == nullptr || !choices->is_array()) continue;
    for (const JValue& choice : choices->items) {
      for (const std::string& t : extract_types(choice)) add(t);
    }
  }
  if (types.empty()) types.push_back("string");
  return types;
}

// coerce_to_schema_type: the first candidate, in the order null, integer, number, boolean,
// object, array, string, that the raw value fits; otherwise whatever json.loads makes of it.
Coerced coerce_to_schema_type(const std::string& value, const std::vector<std::string>& types) {
  std::vector<std::string> normalized;
  for (const auto& t : types) normalized.push_back(normalize_type(t));
  auto has = [&](const char* name) {
    for (const auto& n : normalized) {
      if (n == name) return true;
    }
    return false;
  };
  for (const char* candidate : {"null", "integer", "number", "boolean", "object", "array", "string"}) {
    if (!has(candidate)) continue;
    const std::string_view name = candidate;
    if (name == "null") {
      if (py_lower_ascii(value) == "null") return {false, JValue()};
      continue;
    }
    if (name == "string") return {true, JValue()};
    if (name == "integer") {
      std::string canonical;
      if (py_int(value, canonical)) return {false, JValue::integer(canonical)};
      continue;
    }
    if (name == "number") {
      double d = 0;
      if (py_float(value, d) && std::isfinite(d)) {
        if (d == std::trunc(d)) {
          if (d == 0) return {false, JValue::integer("0")};
          char buf[400];
          std::snprintf(buf, sizeof(buf), "%.0f", d);
          return {false, JValue::integer(buf)};
        }
        return {false, JValue::flt(d)};
      }
      continue;
    }
    if (name == "boolean") {
      const std::string lower = py_lower_ascii(py_strip(value));
      if (lower == "true" || lower == "1") return {false, JValue::boolean_value(true)};
      if (lower == "false" || lower == "0") return {false, JValue::boolean_value(false)};
      continue;
    }
    // object and array: json.loads, and any JSON value it yields is kept (non-finite ones are not).
    JValue parsed;
    if (json_loads(value, parsed) && json_is_finite(parsed)) return {false, std::move(parsed)};
  }
  JValue parsed;
  if (json_loads(value, parsed) && json_is_finite(parsed)) return {false, std::move(parsed)};
  return {true, JValue::string(value)};
}

bool same_scalar(const JValue& a, const JValue& b) {
  if (a.type != b.type) return false;
  switch (a.type) {
    case JValue::Type::kNull: return true;
    case JValue::Type::kBool: return a.boolean == b.boolean;
    case JValue::Type::kInt: return a.text == b.text;
    case JValue::Type::kFloat: return a.number == b.number;
    default: return false;
  }
}

bool coerce_value(JValue& value, const JValue& schema);

// _coerce_dict: every value whose key has an object schema in properties.
bool coerce_dict(JValue& object, const JValue& properties) {
  bool changed = false;
  for (size_t k = 0; k < object.keys.size(); ++k) {
    const JValue* prop = properties.get(object.keys[k]);
    if (prop == nullptr || !prop->is_object()) continue;
    changed = coerce_value(object.values[k], *prop) || changed;
  }
  return changed;
}

// _coerce_value: a string is parsed by the schema; containers are walked; other scalars are
// rendered as JSON text and parsed the same way.
bool coerce_value(JValue& value, const JValue& schema) {
  if (value.is_string()) {
    Coerced c = coerce_to_schema_type(value.text, extract_types(schema));
    if (c.identity) return false;
    value = std::move(c.value);
    return true;
  }
  if (value.is_object()) {
    const JValue* nested = schema.get("properties");
    if (nested != nullptr && nested->is_object()) return coerce_dict(value, *nested);
    return false;
  }
  if (value.is_array()) {
    const JValue* items = schema.get("items");
    if (items == nullptr || !items->is_object()) return false;
    bool changed = false;
    for (JValue& item : value.items) {
      changed = coerce_value(item, *items) || changed;
    }
    return changed;
  }
  Coerced c = coerce_to_schema_type(json_dumps(value), extract_types(schema));
  Coerced fixed{c.identity, std::move(c.value)};
  // A scalar's JSON text is a str, so it comes back as a str unless the schema turns it into
  // another value. Compare with the original, as the reference does.
  if (fixed.identity) {
    value = JValue::string(json_dumps(value));
    return true;
  }
  if (same_scalar(fixed.value, value)) return false;
  value = std::move(fixed.value);
  return true;
}

}  // namespace

const JValue* find_tool_properties(const JValue& tools, std::string_view name) {
  if (!tools.is_array()) return nullptr;
  for (const JValue& tool : tools.items) {
    if (!tool.is_object()) continue;
    const JValue* type = tool.get("type");
    if (type == nullptr || !type->is_string() || type->text != "function") continue;
    const JValue* fn = tool.get("function");
    if (fn == nullptr || !fn->is_object()) continue;
    const JValue* fn_name = fn->get("name");
    if (fn_name == nullptr || !fn_name->is_string() || fn_name->text != name) continue;
    const JValue* params = fn->get("parameters");
    if (params == nullptr || !params->is_object()) return nullptr;
    const JValue* props = params->get("properties");
    return props != nullptr && props->is_object() ? props : nullptr;
  }
  return nullptr;
}

std::string convert_parameters(std::string_view body) {
  JValue params;
  params.type = JValue::Type::kObject;
  size_t from = 0;
  ParamMatch m;
  while (from <= body.size() && next_param(body, from, m)) {
    set_param(params, std::string(py_strip(m.name)), std::string(trim_template_newlines(m.value)));
    from = m.end;
  }
  return json_dumps(params);
}

std::string name_from_body(std::string_view body) {
  for (size_t p = body.find("\"name\""); p != npos; p = body.find("\"name\"", p + 1)) {
    size_t i = skip_regex_space(body, p + 6);
    if (i >= body.size() || body[i] != ':') continue;
    i = skip_regex_space(body, i + 1);
    if (i >= body.size() || body[i] != '"') continue;
    const size_t close = body.find('"', i + 1);
    if (close == npos) continue;
    return std::string(body.substr(i + 1, close - i - 1));
  }
  return {};
}

bool name_and_args_from_json(std::string_view body, std::string& name, std::string& args) {
  name.clear();
  args.clear();
  JValue parsed;
  if (!json_loads(py_strip(body), parsed) || !parsed.is_object()) return false;
  if (const JValue* n = parsed.get("name"); n != nullptr && n->is_string()) name = n->text;
  for (const char* key : {"arguments", "parameters"}) {
    if (const JValue* v = parsed.get(key)) {
      args = v->is_string() ? v->text : json_dumps(*v);
      return true;
    }
  }
  JValue rest;
  rest.type = JValue::Type::kObject;
  for (size_t k = 0; k < parsed.keys.size(); ++k) {
    if (parsed.keys[k] == "name") continue;
    rest.keys.push_back(parsed.keys[k]);
    rest.values.push_back(parsed.values[k]);
  }
  args = json_dumps(rest);
  return true;
}

std::string fix_arg_types(const std::string& args, std::string_view name, const JValue& tools) {
  if (!tools.is_array() || tools.items.empty() || name.empty()) return args;
  JValue parsed;
  if (!json_loads(args, parsed) || !parsed.is_object()) return args;
  const JValue* properties = find_tool_properties(tools, name);
  if (properties == nullptr || properties->keys.empty()) return args;
  if (!coerce_dict(parsed, *properties)) return args;
  return json_dumps(parsed);
}

}  // namespace lx::text
