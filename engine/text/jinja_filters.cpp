// The filters, tests, str/list/dict methods and globals of the Jinja interpreter, with Python's
// behaviour for the cases templates use. Case mapping is ASCII only.
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <map>
#include <string>
#include <vector>

#include "text/json.h"
#include "text/jinja_internal.h"
#include "text/py.h"
#include "platform/posix.h"
#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text::jinja {
namespace {

using Pair = std::pair<Value, Value>;

std::vector<std::string> code_points(const std::string& s) {
  std::vector<std::string> out;
  size_t i = 0;
  while (i < s.size()) {
    size_t len = 1;
    utf8_decode(s, i, len);
    out.push_back(s.substr(i, len));
    i += len;
  }
  return out;
}

size_t code_point_count(const std::string& s) {
  size_t n = 0;
  for (size_t i = 0; i < s.size();) {
    size_t len = 1;
    utf8_decode(s, i, len);
    i += len;
    ++n;
  }
  return n;
}

// Python's str.strip / lstrip / rstrip. With `chars`, the code points in it are removed instead of
// whitespace.
std::string strip_text(const std::string& s, bool left, bool right, const std::string* chars) {
  auto removable = [&](uint32_t cp, const std::string& piece) {
    if (chars == nullptr) return uni::is_py_space(cp);
    return chars->find(piece) != std::string::npos;
  };
  size_t begin = 0;
  if (left) {
    while (begin < s.size()) {
      size_t len = 1;
      const uint32_t cp = utf8_decode(s, begin, len);
      if (!removable(cp, s.substr(begin, len))) break;
      begin += len;
    }
  }
  size_t end = s.size();
  if (right) {
    while (end > begin) {
      size_t start = end - 1;
      while (start > begin && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
      size_t len = 1;
      const uint32_t cp = utf8_decode(s, start, len);
      if (!removable(cp, s.substr(start, end - start))) break;
      end = start;
    }
  }
  return s.substr(begin, end - begin);
}

std::string ascii_lower(std::string s) {
  for (char& c : s) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return s;
}

std::string ascii_upper(std::string s) {
  for (char& c : s) {
    if (c >= 'a' && c <= 'z') c = static_cast<char>(c - 'a' + 'A');
  }
  return s;
}

std::string python_replace(const std::string& s, const std::string& old, const std::string& repl, int64_t count) {
  if (old.empty()) {
    std::string out;
    int64_t done = 0;
    for (const std::string& piece : code_points(s)) {
      if (count < 0 || done < count) {
        out += repl;
        ++done;
      }
      out += piece;
    }
    if (count < 0 || done < count) out += repl;
    return out;
  }
  std::string out;
  size_t pos = 0;
  int64_t done = 0;
  while (true) {
    const size_t hit = s.find(old, pos);
    if (hit == std::string::npos || (count >= 0 && done >= count)) break;
    out.append(s, pos, hit - pos);
    out += repl;
    pos = hit + old.size();
    ++done;
  }
  out.append(s, pos, std::string::npos);
  return out;
}

const Value* opt(const CallArgs& args, size_t index, const char* name) {
  const Value* v = args.get(index, name);
  if (v == nullptr || v->is_undefined()) return nullptr;
  return v;
}

int64_t int_arg(const CallArgs& args, size_t index, const char* name, int64_t dflt) {
  const Value* v = opt(args, index, name);
  if (v == nullptr || v->is_none()) return dflt;
  if (!(v->is_int() || v->is_bool())) throw Error(std::string("'") + name + "' must be an integer");
  return v->as_int();
}

bool bool_arg(const CallArgs& args, size_t index, const char* name, bool dflt) {
  const Value* v = opt(args, index, name);
  return v == nullptr ? dflt : v->truthy();
}

Value attribute_of(const Value& item, const std::string& attribute) {
  // Dotted attribute paths, as Jinja's make_attrgetter does.
  Value cur = item;
  size_t start = 0;
  while (start <= attribute.size()) {
    const size_t dot = attribute.find('.', start);
    const std::string part = attribute.substr(start, dot == std::string::npos ? std::string::npos : dot - start);
    cur = py_attr(cur, part);
    if (dot == std::string::npos) break;
    start = dot + 1;
  }
  return cur;
}

// The test-or-truthiness predicate of select/reject: a named test, or the item's truthiness.
bool passes(const CallArgs& args, size_t test_index, const Value& item) {
  const Value* test = args.get(test_index, "test");
  if (test == nullptr) return item.truthy();
  CallArgs none;
  if (test_index + 1 < args.positional.size()) {
    none.positional.assign(args.positional.begin() + static_cast<long>(test_index) + 1, args.positional.end());
  }
  return apply_test(test->str(), item, none);
}

std::vector<Value> items_of(const Value& v) {
  if (v.is_list()) return v.as_list();
  if (v.is_undefined()) return {};
  if (v.is_string()) {
    std::vector<Value> chars;
    for (const std::string& piece : code_points(v.as_string())) chars.push_back(Value::string(piece));
    return chars;
  }
  if (v.is_dict()) {
    std::vector<Value> keys;
    for (const auto& kv : v.as_dict()) keys.push_back(kv.first);
    return keys;
  }
  throw Error("'" + v.str() + "' is not iterable");
}

size_t utf8_len_at(const std::string& s, size_t i) {
  size_t len = 1;
  utf8_decode(s, i, len);
  return len;
}

bool space_here(const std::string& s, size_t i) {
  size_t len = 1;
  return uni::is_py_space(utf8_decode(s, i, len));
}

size_t skip_space_text(const std::string& s, size_t i) {
  while (i < s.size() && space_here(s, i)) i += utf8_len_at(s, i);
  return i;
}

std::string html_escape(const std::string& s) {
  std::string out;
  for (char c : s) {
    switch (c) {
      case '&': out += "&amp;"; break;
      case '<': out += "&lt;"; break;
      case '>': out += "&gt;"; break;
      case '"': out += "&#34;"; break;
      case '\'': out += "&#39;"; break;
      default: out.push_back(c);
    }
  }
  return out;
}

// A dict method bound to its receiver.
class BoundMethod : public Object {
 public:
  BoundMethod(Value receiver, std::string name) : receiver_(std::move(receiver)), name_(std::move(name)) {}
  bool callable() const override { return true; }
  Value call(const CallArgs& args) const override;
  const char* type_name() const override { return "builtin_function_or_method"; }

 private:
  Value receiver_;
  std::string name_;
};

class NamespaceObject : public Object {
 public:
  Value getattr(const std::string& name) const override {
    auto it = attrs_.find(name);
    return it == attrs_.end() ? Value() : it->second;
  }
  bool setattr(const std::string& name, const Value& value) override {
    attrs_[name] = value;
    return true;
  }
  const char* type_name() const override { return "Namespace"; }

 private:
  std::map<std::string, Value> attrs_;
};

}  // namespace

Value call_method(const Value& recv, const std::string& name, const CallArgs& args);

Value BoundMethod::call(const CallArgs& args) const { return call_method(receiver_, name_, args); }

bool is_dict_method(const std::string& name) {
  static const char* names[] = {"items", "keys", "values", "get", "copy", "pop", "update", "setdefault", "popitem", "clear"};
  for (const char* n : names) {
    if (name == n) return true;
  }
  return false;
}

Value bound_method(const Value& receiver, const std::string& name, bool& found) {
  found = false;
  if (receiver.is_string()) {
    static const char* names[] = {"strip", "lstrip", "rstrip", "split", "splitlines", "startswith", "endswith",
                                  "replace", "lower", "upper", "title", "capitalize", "find", "rfind", "index",
                                  "count", "isdigit", "isalpha", "isspace", "isupper", "islower", "join"};
    for (const char* n : names) {
      if (name == n) found = true;
    }
  } else if (receiver.is_list()) {
    found = name == "count" || name == "index";
  } else if (receiver.is_dict()) {
    found = is_dict_method(name);
  }
  if (!found) return Value();
  return Value::object(std::make_shared<BoundMethod>(receiver, name));
}

Value call_method(const Value& recv, const std::string& name, const CallArgs& args) {
  if (recv.is_string()) {
    const std::string& s = recv.as_string();
    if (name == "strip" || name == "lstrip" || name == "rstrip") {
      const Value* chars = opt(args, 0, "chars");
      std::string set;
      if (chars != nullptr && !chars->is_none()) set = chars->as_string();
      const bool has_chars = chars != nullptr && !chars->is_none();
      return Value::string(strip_text(s, name != "rstrip", name != "lstrip", has_chars ? &set : nullptr));
    }
    if (name == "split") {
      const Value* sep = opt(args, 0, "sep");
      const int64_t maxsplit = int_arg(args, 1, "maxsplit", -1);
      std::vector<Value> parts;
      if (sep == nullptr || sep->is_none()) {
        size_t i = 0;
        int64_t made = 0;
        while (true) {
          i = skip_space_text(s, i);
          if (i >= s.size()) break;
          if (maxsplit >= 0 && made >= maxsplit) {
            parts.push_back(Value::string(strip_text(s.substr(i), true, true, nullptr)));
            break;
          }
          size_t j = i;
          while (j < s.size() && !space_here(s, j)) j += utf8_len_at(s, j);
          parts.push_back(Value::string(s.substr(i, j - i)));
          ++made;
          i = j;
        }
        return Value::list(std::move(parts));
      }
      const std::string& delim = sep->as_string();
      if (delim.empty()) throw Error("empty separator");
      size_t pos = 0;
      int64_t made = 0;
      while (true) {
        const size_t hit = (maxsplit >= 0 && made >= maxsplit) ? std::string::npos : s.find(delim, pos);
        if (hit == std::string::npos) {
          parts.push_back(Value::string(s.substr(pos)));
          break;
        }
        parts.push_back(Value::string(s.substr(pos, hit - pos)));
        pos = hit + delim.size();
        ++made;
      }
      return Value::list(std::move(parts));
    }
    if (name == "splitlines") {
      std::vector<Value> lines;
      size_t pos = 0;
      while (pos < s.size()) {
        const size_t nl = s.find('\n', pos);
        if (nl == std::string::npos) {
          lines.push_back(Value::string(s.substr(pos)));
          break;
        }
        lines.push_back(Value::string(s.substr(pos, nl - pos)));
        pos = nl + 1;
      }
      return Value::list(std::move(lines));
    }
    if (name == "startswith" || name == "endswith") {
      const Value* arg = args.get(0, "prefix");
      if (arg == nullptr) throw Error(name + " needs an argument");
      std::vector<Value> options = arg->is_list() ? arg->as_list() : std::vector<Value>{*arg};
      for (const Value& option : options) {
        const std::string& t = option.as_string();
        const bool hit = name == "startswith" ? s.compare(0, t.size(), t) == 0 && s.size() >= t.size()
                                              : s.size() >= t.size() && s.compare(s.size() - t.size(), t.size(), t) == 0;
        if (hit) return Value::boolean(true);
      }
      return Value::boolean(false);
    }
    if (name == "replace") {
      const Value* old = args.get(0, "old");
      const Value* repl = args.get(1, "new");
      if (old == nullptr || repl == nullptr) throw Error("replace needs old and new");
      return Value::string(python_replace(s, old->as_string(), repl->as_string(), int_arg(args, 2, "count", -1)));
    }
    if (name == "lower") return Value::string(ascii_lower(s));
    if (name == "upper") return Value::string(ascii_upper(s));
    if (name == "capitalize") {
      std::string out = ascii_lower(s);
      if (!out.empty()) out[0] = static_cast<char>(ascii_upper(out.substr(0, 1))[0]);
      return Value::string(out);
    }
    if (name == "title") {
      std::string out = ascii_lower(s);
      bool start = true;
      for (char& c : out) {
        const bool letter = (c >= 'a' && c <= 'z');
        if (letter && start) c = static_cast<char>(c - 'a' + 'A');
        start = !letter;
      }
      return Value::string(out);
    }
    if (name == "find" || name == "rfind" || name == "index") {
      const Value* sub = args.get(0, "sub");
      if (sub == nullptr || !sub->is_string()) throw Error(name + " needs a string");
      const size_t hit = name == "rfind" ? s.rfind(sub->as_string()) : s.find(sub->as_string());
      if (hit == std::string::npos) {
        if (name == "index") throw Error("substring not found");
        return Value::integer(-1);
      }
      return Value::integer(static_cast<int64_t>(code_point_count(s.substr(0, hit))));
    }
    if (name == "count") {
      const Value* sub = args.get(0, "sub");
      if (sub == nullptr || !sub->is_string() || sub->as_string().empty()) throw Error("count needs a string");
      int64_t n = 0;
      size_t pos = 0;
      while ((pos = s.find(sub->as_string(), pos)) != std::string::npos) {
        ++n;
        pos += sub->as_string().size();
      }
      return Value::integer(n);
    }
    if (name == "isdigit" || name == "isalpha" || name == "isspace" || name == "isupper" || name == "islower") {
      if (s.empty()) return Value::boolean(false);
      for (char c : s) {
        const bool ok = name == "isdigit"   ? (c >= '0' && c <= '9')
                        : name == "isalpha" ? ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z'))
                        : name == "isspace" ? (c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v')
                        : name == "isupper" ? (c >= 'A' && c <= 'Z')
                                            : (c >= 'a' && c <= 'z');
        if (!ok) return Value::boolean(false);
      }
      return Value::boolean(true);
    }
    if (name == "join") {
      const Value* items = args.get(0, "iterable");
      if (items == nullptr) throw Error("join needs an iterable");
      std::string out;
      bool first = true;
      for (const Value& item : items_of(*items)) {
        if (!first) out += s;
        first = false;
        out += item.str();
      }
      return Value::string(std::move(out));
    }
  }
  if (recv.is_list()) {
    const auto& items = recv.as_list();
    if (name == "count") {
      const Value* x = args.get(0, "x");
      if (x == nullptr) throw Error("count needs an argument");
      int64_t n = 0;
      for (const Value& v : items) {
        if (py_equal(v, *x)) ++n;
      }
      return Value::integer(n);
    }
    if (name == "index") {
      const Value* x = args.get(0, "x");
      if (x == nullptr) throw Error("index needs an argument");
      for (size_t i = 0; i < items.size(); ++i) {
        if (py_equal(items[i], *x)) return Value::integer(static_cast<int64_t>(i));
      }
      throw Error("value is not in list");
    }
  }
  if (recv.is_dict()) {
    const auto& entries = recv.as_dict();
    if (name == "items" || name == "keys" || name == "values") {
      std::vector<Value> out;
      for (const Pair& kv : entries) {
        if (name == "keys") {
          out.push_back(kv.first);
        } else if (name == "values") {
          out.push_back(kv.second);
        } else {
          out.push_back(Value::tuple({kv.first, kv.second}));
        }
      }
      return Value::list(std::move(out));
    }
    if (name == "get") {
      const Value* key = args.get(0, "key");
      if (key == nullptr) throw Error("get needs a key");
      if (const Value* found = recv.find(*key)) return *found;
      const Value* dflt = args.get(1, "default");
      return dflt ? *dflt : Value::none();
    }
    if (name == "copy") return Value::dict(entries);
    throw Error("dict method '" + name + "' modifies the mapping, which the sandbox does not allow");
  }
  throw Error("'" + recv.str() + "' has no method '" + name + "'");
}

Value apply_filter(const std::string& name, const Value& subject, const CallArgs& args) {
  if (name == "trim") {
    const Value* chars = opt(args, 0, "chars");
    std::string set;
    if (chars != nullptr) set = chars->as_string();
    return Value::string(strip_text(subject.str(), true, true, chars != nullptr ? &set : nullptr));
  }
  if (name == "tojson") {
    JsonOptions o;
    o.ensure_ascii = bool_arg(args, 0, "ensure_ascii", false);
    o.indent = static_cast<int>(int_arg(args, 1, "indent", -1));
    o.sort_keys = bool_arg(args, 3, "sort_keys", false);
    o.item_separator = o.indent >= 0 ? "," : ", ";
    if (const Value* seps = opt(args, 2, "separators")) {
      if (!seps->is_list() || seps->as_list().size() != 2) throw Error("separators must be a pair");
      o.item_separator = seps->as_list()[0].as_string();
      o.key_separator = seps->as_list()[1].as_string();
    }
    return Value::string(json_dumps_value(subject, o));
  }
  if (name == "dictsort") {
    if (!subject.is_dict()) throw Error("dictsort needs a mapping");
    const bool case_sensitive = bool_arg(args, 0, "case_sensitive", false);
    const Value* by_arg = opt(args, 1, "by");
    const bool by_value = by_arg != nullptr && by_arg->as_string() == "value";
    const bool reverse = bool_arg(args, 2, "reverse", false);
    auto sort_key = [&](const Pair& kv) {
      const Value& k = by_value ? kv.second : kv.first;
      if (!case_sensitive && k.is_string()) return Value::string(ascii_lower(k.as_string()));
      return k;
    };
    std::vector<Pair> entries = subject.as_dict();
    std::stable_sort(entries.begin(), entries.end(), [&](const Pair& a, const Pair& b) {
      const Value ka = sort_key(a);
      const Value kb = sort_key(b);
      return reverse ? py_less(kb, ka) : py_less(ka, kb);
    });
    std::vector<Value> out;
    for (const Pair& kv : entries) out.push_back(Value::tuple({kv.first, kv.second}));
    return Value::list(std::move(out));
  }
  if (name == "length" || name == "count") {
    if (subject.is_string()) return Value::integer(static_cast<int64_t>(code_point_count(subject.as_string())));
    if (subject.is_list()) return Value::integer(static_cast<int64_t>(subject.as_list().size()));
    if (subject.is_dict()) return Value::integer(static_cast<int64_t>(subject.as_dict().size()));
    throw Error("object has no len()");
  }
  if (name == "string") return Value::string(subject.str());
  if (name == "lower") return Value::string(ascii_lower(subject.str()));
  if (name == "upper") return Value::string(ascii_upper(subject.str()));
  if (name == "capitalize" || name == "title") {
    return call_method(subject, name, CallArgs{});
  }
  if (name == "replace") {
    const Value* old = args.get(0, "old");
    const Value* repl = args.get(1, "new");
    if (old == nullptr || repl == nullptr) throw Error("replace needs old and new");
    return Value::string(python_replace(subject.str(), old->str(), repl->str(), int_arg(args, 2, "count", -1)));
  }
  if (name == "default" || name == "d") {
    const Value* dflt = args.get(0, "default_value");
    const Value fallback = dflt ? *dflt : Value::string("");
    if (subject.is_undefined()) return fallback;
    if (bool_arg(args, 1, "boolean", false) && !subject.truthy()) return fallback;
    return subject;
  }
  if (name == "first" || name == "last") {
    const std::vector<Value> items = items_of(subject);
    if (items.empty()) return Value();
    return name == "first" ? items.front() : items.back();
  }
  if (name == "join") {
    const Value* sep = args.get(0, "d");
    const std::string separator = sep != nullptr ? sep->as_string() : std::string();
    const Value* attribute = opt(args, 1, "attribute");
    std::string out;
    bool first = true;
    for (const Value& item : items_of(subject)) {
      if (!first) out += separator;
      first = false;
      out += attribute != nullptr ? attribute_of(item, attribute->as_string()).str() : item.str();
    }
    return Value::string(std::move(out));
  }
  if (name == "items") {
    if (subject.is_undefined()) return Value::list({});
    if (!subject.is_dict()) throw Error("items needs a mapping");
    std::vector<Value> out;
    for (const Pair& kv : subject.as_dict()) out.push_back(Value::tuple({kv.first, kv.second}));
    return Value::list(std::move(out));
  }
  if (name == "list") return Value::list(items_of(subject));
  if (name == "int") {
    const Value fallback = args.get(0, "default") ? *args.get(0, "default") : Value::integer(0);
    if (subject.is_int() || subject.is_bool()) return Value::integer(subject.as_int());
    if (subject.is_float()) return Value::integer(static_cast<int64_t>(std::trunc(subject.as_float())));
    if (subject.is_string()) {
      std::string canonical;
      if (py_int(subject.as_string(), canonical)) return Value::integer(std::strtoll(canonical.c_str(), nullptr, 10));
    }
    return fallback;
  }
  if (name == "float") {
    const Value fallback = args.get(0, "default") ? *args.get(0, "default") : Value::real(0.0);
    if (subject.is_number()) return Value::real(subject.as_float());
    if (subject.is_string()) {
      double d = 0.0;
      if (py_float(subject.as_string(), d)) return Value::real(d);
    }
    return fallback;
  }
  if (name == "safe") return subject;
  if (name == "escape" || name == "e" || name == "forceescape") return Value::string(html_escape(subject.str()));
  if (name == "reverse") {
    if (subject.is_string()) {
      auto pieces = code_points(subject.as_string());
      std::string out;
      for (auto it = pieces.rbegin(); it != pieces.rend(); ++it) out += *it;
      return Value::string(std::move(out));
    }
    std::vector<Value> items = items_of(subject);
    std::reverse(items.begin(), items.end());
    return Value::list(std::move(items));
  }
  if (name == "sort") {
    std::vector<Value> items = items_of(subject);
    const bool reverse = bool_arg(args, 0, "reverse", false);
    const bool case_sensitive = bool_arg(args, 1, "case_sensitive", false);
    auto key = [&](const Value& v) {
      if (!case_sensitive && v.is_string()) return Value::string(ascii_lower(v.as_string()));
      return v;
    };
    std::stable_sort(items.begin(), items.end(), [&](const Value& a, const Value& b) {
      return reverse ? py_less(key(b), key(a)) : py_less(key(a), key(b));
    });
    return Value::list(std::move(items));
  }
  if (name == "abs") {
    if (subject.is_int() || subject.is_bool()) return Value::integer(std::llabs(subject.as_int()));
    if (subject.is_float()) return Value::real(std::fabs(subject.as_float()));
    throw Error("bad operand type for abs()");
  }
  if (name == "round") {
    const int64_t precision = int_arg(args, 0, "precision", 0);
    const double x = subject.as_float();
    if (precision == 0) return Value::real(std::nearbyint(x));
    const double factor = std::pow(10.0, static_cast<double>(precision));
    return Value::real(std::nearbyint(x * factor) / factor);
  }
  if (name == "sum") {
    Value total = Value::integer(0);
    const std::vector<Value> items = items_of(subject);
    const Value* start = args.get(0, "start");
    if (start != nullptr) total = *start;
    for (const Value& v : items) total = py_arith('+', total, v);
    return total;
  }
  if (name == "min" || name == "max") {
    const std::vector<Value> items = items_of(subject);
    if (items.empty()) return Value();
    Value best = items.front();
    for (const Value& v : items) {
      if (name == "min" ? py_less(v, best) : py_less(best, v)) best = v;
    }
    return best;
  }
  if (name == "unique") {
    std::vector<Value> out;
    for (const Value& v : items_of(subject)) {
      bool seen = false;
      for (const Value& o : out) {
        if (py_equal(o, v)) {
          seen = true;
          break;
        }
      }
      if (!seen) out.push_back(v);
    }
    return Value::list(std::move(out));
  }
  if (name == "select" || name == "reject") {
    std::vector<Value> out;
    for (const Value& v : items_of(subject)) {
      const bool keep = passes(args, 0, v);
      if (keep == (name == "select")) out.push_back(v);
    }
    return Value::list(std::move(out));
  }
  if (name == "selectattr" || name == "rejectattr") {
    const Value* attribute = args.get(0, "attr");
    if (attribute == nullptr) throw Error(name + " needs an attribute");
    std::vector<Value> out;
    for (const Value& v : items_of(subject)) {
      Value attr = attribute_of(v, attribute->as_string());
      const bool keep = passes(args, 1, attr);
      if (keep == (name == "selectattr")) out.push_back(v);
    }
    return Value::list(std::move(out));
  }
  if (name == "map") {
    const Value* attribute = opt(args, 0, "attribute");
    if (attribute == nullptr) throw Error("map is supported with an attribute= argument only");
    std::vector<Value> out;
    for (const Value& v : items_of(subject)) out.push_back(attribute_of(v, attribute->as_string()));
    return Value::list(std::move(out));
  }
  throw Error("No filter named '" + name + "'");
}

bool apply_test(const std::string& name, const Value& subject, const CallArgs& args) {
  if (name == "defined") return !subject.is_undefined();
  if (name == "undefined") return subject.is_undefined();
  if (name == "none") return subject.is_none();
  if (name == "string") return subject.is_string();
  if (name == "mapping") return subject.is_dict();
  if (name == "iterable") return subject.is_list() || subject.is_dict() || subject.is_string() || subject.is_undefined();
  if (name == "sequence") return subject.is_list() || subject.is_string() || subject.is_undefined();
  if (name == "number") return subject.is_number();
  if (name == "integer") return subject.is_int();
  if (name == "float") return subject.is_float();
  if (name == "boolean") return subject.is_bool();
  if (name == "true") return subject.is_bool() && subject.as_bool();
  if (name == "false") return subject.is_bool() && !subject.as_bool();
  if (name == "callable") return subject.is_object() && subject.as_object()->callable();
  if (name == "escaped") return false;
  if (name == "lower") return subject.is_string() && ascii_lower(subject.as_string()) == subject.as_string();
  if (name == "upper") return subject.is_string() && ascii_upper(subject.as_string()) == subject.as_string();
  if (name == "sameas") {
    const Value* other = args.get(0, "other");
    if (other == nullptr) throw Error("sameas needs an argument");
    if (subject.is_list() || subject.is_dict() || subject.is_object()) return false;
    return subject.kind() == other->kind() && py_equal(subject, *other);
  }
  if (name == "odd" || name == "even") {
    if (!subject.is_int()) return false;
    return (subject.as_int() % 2 != 0) == (name == "odd");
  }
  if (name == "divisibleby") {
    const Value* n = args.get(0, "num");
    if (n == nullptr || !n->is_int() || n->as_int() == 0) throw Error("divisibleby needs an integer");
    return subject.is_int() && subject.as_int() % n->as_int() == 0;
  }
  if (name == "in") {
    const Value* container = args.get(0, "seq");
    if (container == nullptr) throw Error("in needs a container");
    return py_contains(*container, subject);
  }
  const Value* other = args.get(0, "other");
  if (name == "eq" || name == "equalto" || name == "==") return other != nullptr && py_equal(subject, *other);
  if (name == "ne" || name == "!=") return other == nullptr || !py_equal(subject, *other);
  if (name == "lt" || name == "lessthan" || name == "<") return other != nullptr && py_less(subject, *other);
  if (name == "gt" || name == "greaterthan" || name == ">") return other != nullptr && py_less(*other, subject);
  if (name == "le" || name == "<=") return other != nullptr && !py_less(*other, subject);
  if (name == "ge" || name == ">=") return other != nullptr && !py_less(subject, *other);
  throw Error("No test named '" + name + "'");
}

bool is_global(const std::string& name) {
  return name == "range" || name == "namespace" || name == "raise_exception" || name == "strftime_now" ||
         name == "dict";
}

Value make_namespace(const CallArgs& args) {
  auto ns = std::make_shared<NamespaceObject>();
  for (const auto& kv : args.keyword) ns->setattr(kv.first, kv.second);
  return Value::object(std::move(ns));
}

Value call_global(const std::string& name, const CallArgs& args) {
  if (name == "range") {
    const Value* a = args.get(0, "start");
    const Value* b = args.get(1, "stop");
    int64_t start = 0, stop = 0, step = 1;
    if (b == nullptr) {
      if (a == nullptr) throw Error("range expected an argument");
      stop = a->as_int();
    } else {
      start = a->as_int();
      stop = b->as_int();
      if (const Value* c = args.get(2, "step")) step = c->as_int();
    }
    if (step == 0) throw Error("range() arg 3 must not be zero");
    std::vector<Value> items;
    for (int64_t i = start; step > 0 ? i < stop : i > stop; i += step) items.push_back(Value::integer(i));
    return Value::list(std::move(items));
  }
  if (name == "namespace") return make_namespace(args);
  if (name == "raise_exception") {
    const Value* msg = args.get(0, "message");
    throw Error(msg != nullptr ? msg->str() : std::string());
  }
  if (name == "strftime_now") {
    const Value* fmt = args.get(0, "format");
    if (fmt == nullptr || !fmt->is_string()) throw Error("strftime_now needs a format");
    const std::time_t now = std::time(nullptr);
    std::tm local{};
    localtime_r(&now, &local);
    char buf[512];
    const size_t n = std::strftime(buf, sizeof buf, fmt->as_string().c_str(), &local);
    return Value::string(std::string(buf, n));
  }
  if (name == "dict") {
    std::vector<Pair> items;
    for (const auto& kv : args.keyword) items.emplace_back(Value::string(kv.first), kv.second);
    return Value::dict(std::move(items));
  }
  throw Error("No function named '" + name + "'");
}

}  // namespace lx::text::jinja
