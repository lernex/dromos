// Python value semantics for the Jinja interpreter: truthiness, str and repr, equality and ordering,
// the operators, indexing and slicing, and json.dumps.
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string>
#include <vector>

#include "text/json.h"
#include "text/jinja_internal.h"
#include "text/utf8.h"

namespace lx::text::jinja {

const Value* CallArgs::get(size_t index, const char* name) const {
  if (index < positional.size()) return &positional[index];
  for (const auto& kv : keyword) {
    if (kv.first == name) return &kv.second;
  }
  return nullptr;
}

Value Object::getattr(const std::string&) const { return Value(); }

bool Object::setattr(const std::string&, const Value&) { return false; }

Value Object::call(const CallArgs&) const { throw Error(std::string(type_name()) + " object is not callable"); }

Value Value::none() {
  Value v;
  v.kind_ = Kind::kNone;
  return v;
}

Value Value::boolean(bool v) {
  Value out;
  out.kind_ = Kind::kBool;
  out.bool_ = v;
  out.int_ = v ? 1 : 0;
  return out;
}

Value Value::integer(int64_t v) {
  Value out;
  out.kind_ = Kind::kInt;
  out.int_ = v;
  return out;
}

Value Value::real(double v) {
  Value out;
  out.kind_ = Kind::kFloat;
  out.float_ = v;
  return out;
}

Value Value::string(std::string s) {
  Value out;
  out.kind_ = Kind::kString;
  out.string_ = std::move(s);
  return out;
}

Value Value::list(std::vector<Value> items) {
  Value out;
  out.kind_ = Kind::kList;
  out.list_ = std::make_shared<std::vector<Value>>(std::move(items));
  return out;
}

Value Value::tuple(std::vector<Value> items) {
  Value out = list(std::move(items));
  out.tuple_ = true;
  return out;
}

Value Value::dict(std::vector<std::pair<Value, Value>> items) {
  Value out;
  out.kind_ = Kind::kDict;
  out.dict_ = std::make_shared<std::vector<std::pair<Value, Value>>>(std::move(items));
  return out;
}

Value Value::object(std::shared_ptr<Object> obj) {
  Value out;
  out.kind_ = Kind::kObject;
  out.object_ = std::move(obj);
  return out;
}

bool Value::truthy() const {
  switch (kind_) {
    case Kind::kUndefined:
    case Kind::kNone:
      return false;
    case Kind::kBool:
    case Kind::kInt:
      return int_ != 0;
    case Kind::kFloat:
      return float_ != 0.0;
    case Kind::kString:
      return !string_.empty();
    case Kind::kList:
      return !list_->empty();
    case Kind::kDict:
      return !dict_->empty();
    case Kind::kObject:
      return true;
  }
  return false;
}

double Value::as_float() const {
  return kind_ == Kind::kFloat ? float_ : static_cast<double>(int_);
}

namespace {

std::string float_text(double d) {
  if (std::isnan(d)) return "nan";
  if (std::isinf(d)) return d > 0 ? "inf" : "-inf";
  return py_float_repr(d);
}

// Python's repr() of a str: single quotes unless the text has a single quote and no double quote.
std::string string_repr(std::string_view s) {
  const bool single = s.find('\'') != std::string_view::npos;
  const bool dbl = s.find('"') != std::string_view::npos;
  const char quote = (single && !dbl) ? '"' : '\'';
  std::string out(1, quote);
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      if (c == '\\') {
        out += "\\\\";
      } else if (c == static_cast<unsigned char>(quote)) {
        out.push_back('\\');
        out.push_back(quote);
      } else if (c == '\n') {
        out += "\\n";
      } else if (c == '\r') {
        out += "\\r";
      } else if (c == '\t') {
        out += "\\t";
      } else if (c < 0x20 || c == 0x7f) {
        char buf[8];
        std::snprintf(buf, sizeof buf, "\\x%02x", c);
        out += buf;
      } else {
        out.push_back(static_cast<char>(c));
      }
      ++i;
      continue;
    }
    size_t len = 1;
    const uint32_t cp = utf8_decode(s, i, len);
    // Python escapes what str.isprintable() rejects: C1 controls, the separators, and the
    // non-ASCII spaces and format characters.
    const bool escape = (cp >= 0x80 && cp <= 0x9f) || cp == 0xa0 || cp == 0xad || cp == 0x1680 ||
                        (cp >= 0x2000 && cp <= 0x200f) || (cp >= 0x2028 && cp <= 0x202f) ||
                        (cp >= 0x205f && cp <= 0x206f) || cp == 0x3000 || cp == 0xfeff;
    if (escape) {
      char buf[12];
      if (cp <= 0xff) {
        std::snprintf(buf, sizeof buf, "\\x%02x", cp);
      } else {
        std::snprintf(buf, sizeof buf, "\\u%04x", cp);
      }
      out += buf;
    } else {
      out.append(s.substr(i, len));
    }
    i += len;
  }
  out.push_back(quote);
  return out;
}

}  // namespace

std::string Value::str() const {
  switch (kind_) {
    case Kind::kUndefined:
      return "";
    case Kind::kNone:
      return "None";
    case Kind::kBool:
      return bool_ ? "True" : "False";
    case Kind::kInt:
      return std::to_string(int_);
    case Kind::kFloat:
      return float_text(float_);
    case Kind::kString:
      return string_;
    case Kind::kList:
    case Kind::kDict:
      return repr();
    case Kind::kObject:
      return std::string("<") + object_->type_name() + " object>";
  }
  return "";
}

std::string Value::repr() const {
  switch (kind_) {
    case Kind::kUndefined:
      return "Undefined";
    case Kind::kString:
      return string_repr(string_);
    case Kind::kList: {
      const auto& items = *list_;
      std::string out = tuple_ ? "(" : "[";
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ", ";
        out += items[i].repr();
      }
      if (tuple_ && items.size() == 1) out += ",";
      return out + (tuple_ ? ")" : "]");
    }
    case Kind::kDict: {
      std::string out = "{";
      const auto& items = *dict_;
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) out += ", ";
        out += items[i].first.repr();
        out += ": ";
        out += items[i].second.repr();
      }
      return out + "}";
    }
    default:
      return str();
  }
}

const Value* Value::find(const Value& key) const {
  if (kind_ != Kind::kDict) return nullptr;
  for (const auto& kv : *dict_) {
    if (py_equal(kv.first, key)) return &kv.second;
  }
  return nullptr;
}

void Value::set_item(const Value& key, Value value) {
  if (kind_ != Kind::kDict) throw Error("item assignment needs a dict");
  for (auto& kv : *dict_) {
    if (py_equal(kv.first, key)) {
      kv.second = std::move(value);
      return;
    }
  }
  dict_->emplace_back(key, std::move(value));
}

bool py_equal(const Value& a, const Value& b) {
  if (a.is_number() && b.is_number()) {
    if (a.is_float() || b.is_float()) return a.as_float() == b.as_float();
    return a.as_int() == b.as_int();
  }
  if (a.kind() != b.kind()) return false;
  switch (a.kind()) {
    case Value::Kind::kUndefined:
    case Value::Kind::kNone:
      return true;
    case Value::Kind::kString:
      return a.as_string() == b.as_string();
    case Value::Kind::kList: {
      if (a.is_tuple() != b.is_tuple()) return false;
      const auto& x = a.as_list();
      const auto& y = b.as_list();
      if (x.size() != y.size()) return false;
      for (size_t i = 0; i < x.size(); ++i) {
        if (!py_equal(x[i], y[i])) return false;
      }
      return true;
    }
    case Value::Kind::kDict: {
      const auto& x = a.as_dict();
      if (x.size() != b.as_dict().size()) return false;
      for (const auto& kv : x) {
        const Value* other = b.find(kv.first);
        if (other == nullptr || !py_equal(kv.second, *other)) return false;
      }
      return true;
    }
    case Value::Kind::kObject:
      return a.as_object() == b.as_object();
    default:
      return false;
  }
}

bool py_less(const Value& a, const Value& b) {
  if (a.is_number() && b.is_number()) {
    if (a.is_float() || b.is_float()) return a.as_float() < b.as_float();
    return a.as_int() < b.as_int();
  }
  if (a.is_string() && b.is_string()) return a.as_string() < b.as_string();
  if (a.is_list() && b.is_list()) {
    const auto& x = a.as_list();
    const auto& y = b.as_list();
    for (size_t i = 0; i < x.size() && i < y.size(); ++i) {
      if (!py_equal(x[i], y[i])) return py_less(x[i], y[i]);
    }
    return x.size() < y.size();
  }
  throw Error("'<' not supported between these operand types");
}

bool py_contains(const Value& container, const Value& item) {
  switch (container.kind()) {
    case Value::Kind::kList:
      for (const auto& v : container.as_list()) {
        if (py_equal(v, item)) return true;
      }
      return false;
    case Value::Kind::kDict:
      return container.find(item) != nullptr;
    case Value::Kind::kString:
      if (!item.is_string()) throw Error("'in <string>' requires string as left operand");
      return container.as_string().find(item.as_string()) != std::string::npos;
    case Value::Kind::kUndefined:
      return false;
    default:
      throw Error("argument of type is not iterable");
  }
}

namespace {

// The code point start offsets of a UTF-8 string, plus the end.
std::vector<size_t> code_point_offsets(const std::string& s) {
  std::vector<size_t> offsets;
  size_t i = 0;
  while (i < s.size()) {
    offsets.push_back(i);
    size_t len = 1;
    utf8_decode(s, i, len);
    i += len;
  }
  offsets.push_back(s.size());
  return offsets;
}

int64_t floor_div(int64_t a, int64_t b) {
  int64_t q = a / b;
  if ((a % b != 0) && ((a < 0) != (b < 0))) --q;
  return q;
}

int64_t py_mod(int64_t a, int64_t b) {
  int64_t r = a % b;
  if (r != 0 && ((r < 0) != (b < 0))) r += b;
  return r;
}

}  // namespace

Value py_arith(char op, const Value& a, const Value& b) {
  if (op == '~') return Value::string(a.str() + b.str());
  const bool both_int = (a.is_int() || a.is_bool()) && (b.is_int() || b.is_bool());
  if (a.is_number() && b.is_number()) {
    switch (op) {
      case '+':
        return both_int ? Value::integer(a.as_int() + b.as_int()) : Value::real(a.as_float() + b.as_float());
      case '-':
        return both_int ? Value::integer(a.as_int() - b.as_int()) : Value::real(a.as_float() - b.as_float());
      case '*':
        return both_int ? Value::integer(a.as_int() * b.as_int()) : Value::real(a.as_float() * b.as_float());
      case '/':
        if (b.as_float() == 0.0) throw Error("division by zero");
        return Value::real(a.as_float() / b.as_float());
      case 'd':
        if (both_int) {
          if (b.as_int() == 0) throw Error("integer division by zero");
          return Value::integer(floor_div(a.as_int(), b.as_int()));
        }
        if (b.as_float() == 0.0) throw Error("float floor division by zero");
        return Value::real(std::floor(a.as_float() / b.as_float()));
      case '%':
        if (both_int) {
          if (b.as_int() == 0) throw Error("integer modulo by zero");
          return Value::integer(py_mod(a.as_int(), b.as_int()));
        }
        if (b.as_float() == 0.0) throw Error("float modulo");
        {
          double r = std::fmod(a.as_float(), b.as_float());
          if (r != 0.0 && ((r < 0) != (b.as_float() < 0))) r += b.as_float();
          return Value::real(r);
        }
      case 'p':
        if (both_int && b.as_int() >= 0) {
          int64_t base = a.as_int(), result = 1;
          for (int64_t e = b.as_int(); e > 0; --e) result *= base;
          return Value::integer(result);
        }
        return Value::real(std::pow(a.as_float(), b.as_float()));
      default:
        break;
    }
  }
  if (op == '+') {
    if (a.is_string() && b.is_string()) return Value::string(a.as_string() + b.as_string());
    if (a.is_list() && b.is_list()) {
      std::vector<Value> items = a.as_list();
      items.insert(items.end(), b.as_list().begin(), b.as_list().end());
      return Value::list(std::move(items));
    }
  }
  if (op == '*') {
    const Value* seq = nullptr;
    const Value* count = nullptr;
    if ((a.is_string() || a.is_list()) && (b.is_int() || b.is_bool())) {
      seq = &a;
      count = &b;
    } else if ((b.is_string() || b.is_list()) && (a.is_int() || a.is_bool())) {
      seq = &b;
      count = &a;
    }
    if (seq != nullptr) {
      const int64_t n = std::max<int64_t>(0, count->as_int());
      if (seq->is_string()) {
        std::string out;
        for (int64_t i = 0; i < n; ++i) out += seq->as_string();
        return Value::string(std::move(out));
      }
      std::vector<Value> items;
      for (int64_t i = 0; i < n; ++i) items.insert(items.end(), seq->as_list().begin(), seq->as_list().end());
      return Value::list(std::move(items));
    }
  }
  throw Error(std::string("unsupported operand types for ") + op);
}

Value py_item(const Value& container, const Value& index) {
  switch (container.kind()) {
    case Value::Kind::kList: {
      if (!(index.is_int() || index.is_bool())) return Value();
      const auto& items = container.as_list();
      int64_t i = index.as_int();
      if (i < 0) i += static_cast<int64_t>(items.size());
      if (i < 0 || i >= static_cast<int64_t>(items.size())) return Value();
      return items[static_cast<size_t>(i)];
    }
    case Value::Kind::kString: {
      if (!(index.is_int() || index.is_bool())) return Value();
      const auto offsets = code_point_offsets(container.as_string());
      const int64_t count = static_cast<int64_t>(offsets.size()) - 1;
      int64_t i = index.as_int();
      if (i < 0) i += count;
      if (i < 0 || i >= count) return Value();
      const size_t at = offsets[static_cast<size_t>(i)];
      const size_t end = offsets[static_cast<size_t>(i) + 1];
      return Value::string(container.as_string().substr(at, end - at));
    }
    case Value::Kind::kDict: {
      if (const Value* found = container.find(index)) return *found;
      if (index.is_string() && is_dict_method(index.as_string())) return py_attr(container, index.as_string());
      return Value();
    }
    case Value::Kind::kObject:
      if (index.is_string()) return container.as_object()->getattr(index.as_string());
      return Value();
    default:
      return Value();
  }
}

Value py_slice(const Value& container, const Value* start, const Value* stop, const Value* step) {
  if (!container.is_list() && !container.is_string()) {
    if (container.is_undefined()) return Value();
    throw Error("object is not sliceable");
  }
  const bool is_str = container.is_string();
  std::vector<size_t> offsets;
  int64_t n;
  if (is_str) {
    offsets = code_point_offsets(container.as_string());
    n = static_cast<int64_t>(offsets.size()) - 1;
  } else {
    n = static_cast<int64_t>(container.as_list().size());
  }
  int64_t s = 1;
  if (step != nullptr && !step->is_undefined() && !step->is_none()) s = step->as_int();
  if (s == 0) throw Error("slice step cannot be zero");
  auto bound = [&](const Value* v, int64_t dflt) -> int64_t {
    if (v == nullptr || v->is_undefined() || v->is_none()) return dflt;
    int64_t x = v->as_int();
    if (x < 0) x += n;
    if (s > 0) {
      if (x < 0) x = 0;
      if (x > n) x = n;
    } else {
      if (x < 0) x = -1;
      if (x >= n) x = n - 1;
    }
    return x;
  };
  const int64_t lo = s > 0 ? bound(start, 0) : bound(start, n - 1);
  const int64_t hi = s > 0 ? bound(stop, n) : bound(stop, -1);
  std::vector<Value> picked;
  std::string text;
  for (int64_t i = lo; s > 0 ? i < hi : i > hi; i += s) {
    if (is_str) {
      const size_t a = offsets[static_cast<size_t>(i)];
      const size_t b = offsets[static_cast<size_t>(i) + 1];
      text.append(container.as_string(), a, b - a);
    } else {
      picked.push_back(container.as_list()[static_cast<size_t>(i)]);
    }
  }
  return is_str ? Value::string(std::move(text)) : Value::list(std::move(picked));
}

Value py_attr(const Value& object, const std::string& name) {
  switch (object.kind()) {
    case Value::Kind::kUndefined:
      throw Error("'" + name + "' is undefined");
    case Value::Kind::kDict:
      if (is_dict_method(name)) {
        bool found = false;
        Value method = bound_method(object, name, found);
        if (found) return method;
      }
      if (const Value* found = object.find(Value::string(name))) return *found;
      return Value();
    case Value::Kind::kString:
    case Value::Kind::kList: {
      bool found = false;
      Value method = bound_method(object, name, found);
      return found ? method : Value();
    }
    case Value::Kind::kObject:
      return object.as_object()->getattr(name);
    default:
      return Value();
  }
}

namespace {

void append_json_string(std::string& out, std::string_view s, bool ensure_ascii) {
  out.push_back('"');
  size_t i = 0;
  while (i < s.size()) {
    const unsigned char c = static_cast<unsigned char>(s[i]);
    if (c < 0x80) {
      switch (c) {
        case '"':
          out += "\\\"";
          break;
        case '\\':
          out += "\\\\";
          break;
        case '\n':
          out += "\\n";
          break;
        case '\r':
          out += "\\r";
          break;
        case '\t':
          out += "\\t";
          break;
        case '\b':
          out += "\\b";
          break;
        case '\f':
          out += "\\f";
          break;
        default:
          if (c < 0x20) {
            char buf[8];
            std::snprintf(buf, sizeof buf, "\\u%04x", c);
            out += buf;
          } else {
            out.push_back(static_cast<char>(c));
          }
      }
      ++i;
      continue;
    }
    size_t len = 1;
    const uint32_t cp = utf8_decode(s, i, len);
    if (ensure_ascii) {
      char buf[16];
      if (cp >= 0x10000) {
        const uint32_t v = cp - 0x10000;
        std::snprintf(buf, sizeof buf, "\\u%04x\\u%04x", 0xd800 + (v >> 10), 0xdc00 + (v & 0x3ff));
      } else {
        std::snprintf(buf, sizeof buf, "\\u%04x", cp);
      }
      out += buf;
    } else {
      out.append(s.substr(i, len));
    }
    i += len;
  }
  out.push_back('"');
}

void newline_indent(std::string& out, const JsonOptions& o, size_t depth) {
  out.push_back('\n');
  out.append(static_cast<size_t>(o.indent) * depth, ' ');
}

void dump(const Value& v, const JsonOptions& o, size_t depth, std::string& out) {
  switch (v.kind()) {
    case Value::Kind::kUndefined:
    case Value::Kind::kNone:
      out += "null";
      return;
    case Value::Kind::kBool:
      out += v.as_bool() ? "true" : "false";
      return;
    case Value::Kind::kInt:
      out += std::to_string(v.as_int());
      return;
    case Value::Kind::kFloat: {
      const double d = v.as_float();
      if (std::isnan(d)) {
        out += "NaN";
      } else if (std::isinf(d)) {
        out += d > 0 ? "Infinity" : "-Infinity";
      } else {
        out += py_float_repr(d);
      }
      return;
    }
    case Value::Kind::kString:
      append_json_string(out, v.as_string(), o.ensure_ascii);
      return;
    case Value::Kind::kList: {
      const auto& items = v.as_list();
      if (items.empty()) {
        out += "[]";
        return;
      }
      out.push_back('[');
      if (o.indent >= 0) newline_indent(out, o, depth + 1);
      for (size_t i = 0; i < items.size(); ++i) {
        if (i) {
          out += o.item_separator;
          if (o.indent >= 0) newline_indent(out, o, depth + 1);
        }
        dump(items[i], o, depth + 1, out);
      }
      if (o.indent >= 0) newline_indent(out, o, depth);
      out.push_back(']');
      return;
    }
    case Value::Kind::kDict: {
      const auto& items = v.as_dict();
      if (items.empty()) {
        out += "{}";
        return;
      }
      std::vector<const std::pair<Value, Value>*> order;
      for (const auto& kv : items) order.push_back(&kv);
      if (o.sort_keys) {
        std::stable_sort(order.begin(), order.end(), [](const auto* a, const auto* b) {
          return a->first.str() < b->first.str();
        });
      }
      out.push_back('{');
      if (o.indent >= 0) newline_indent(out, o, depth + 1);
      for (size_t i = 0; i < order.size(); ++i) {
        if (i) {
          out += o.item_separator;
          if (o.indent >= 0) newline_indent(out, o, depth + 1);
        }
        const Value& key = order[i]->first;
        if (key.is_string()) {
          append_json_string(out, key.as_string(), o.ensure_ascii);
        } else if (key.is_bool()) {
          append_json_string(out, key.as_bool() ? "true" : "false", o.ensure_ascii);
        } else if (key.is_none()) {
          append_json_string(out, "null", o.ensure_ascii);
        } else if (key.is_number()) {
          append_json_string(out, key.str(), o.ensure_ascii);
        } else {
          throw Error("keys must be str, int, float, bool or None");
        }
        out += o.key_separator;
        dump(order[i]->second, o, depth + 1, out);
      }
      if (o.indent >= 0) newline_indent(out, o, depth);
      out.push_back('}');
      return;
    }
    case Value::Kind::kObject:
      throw Error(std::string("Object of type ") + v.as_object()->type_name() + " is not JSON serializable");
  }
}

}  // namespace

std::string json_dumps_value(const Value& value, const JsonOptions& options) {
  std::string out;
  dump(value, options, 0, out);
  return out;
}

Value value_from_json(const JValue& json) {
  switch (json.type) {
    case JValue::Type::kNull:
      return Value::none();
    case JValue::Type::kBool:
      return Value::boolean(json.boolean);
    case JValue::Type::kInt: {
      errno = 0;
      const long long parsed = std::strtoll(json.text.c_str(), nullptr, 10);
      if (errno == ERANGE) return Value::real(std::strtod(json.text.c_str(), nullptr));
      return Value::integer(parsed);
    }
    case JValue::Type::kFloat:
      return Value::real(json.number);
    case JValue::Type::kString:
      return Value::string(json.text);
    case JValue::Type::kArray: {
      std::vector<Value> items;
      items.reserve(json.items.size());
      for (const auto& item : json.items) items.push_back(value_from_json(item));
      return Value::list(std::move(items));
    }
    case JValue::Type::kObject: {
      std::vector<std::pair<Value, Value>> items;
      items.reserve(json.keys.size());
      for (size_t i = 0; i < json.keys.size(); ++i) {
        items.emplace_back(Value::string(json.keys[i]), value_from_json(json.values[i]));
      }
      return Value::dict(std::move(items));
    }
  }
  return Value();
}

}  // namespace lx::text::jinja
