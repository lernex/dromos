#include "text/json.h"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <unordered_map>

#include "text/utf8.h"

namespace lx::text {
namespace {

constexpr int kMaxDepth = 1000;

class Parser {
 public:
  explicit Parser(std::string_view s) : s_(s) {}

  bool parse(JValue& out) {
    skip_ws();
    if (!value(out, 0)) return false;
    skip_ws();
    return i_ == s_.size();
  }

 private:
  void skip_ws() {
    while (i_ < s_.size() && (s_[i_] == ' ' || s_[i_] == '\t' || s_[i_] == '\n' || s_[i_] == '\r')) ++i_;
  }

  bool literal(std::string_view word) {
    if (s_.substr(i_, word.size()) != word) return false;
    i_ += word.size();
    return true;
  }

  bool value(JValue& out, int depth) {
    if (depth > kMaxDepth || i_ >= s_.size()) return false;
    const char c = s_[i_];
    if (c == '{') return object(out, depth);
    if (c == '[') return array(out, depth);
    if (c == '"') {
      std::string str;
      if (!string(str)) return false;
      out = JValue::string(std::move(str));
      return true;
    }
    if (c == 't' && literal("true")) { out = JValue::boolean_value(true); return true; }
    if (c == 'f' && literal("false")) { out = JValue::boolean_value(false); return true; }
    if (c == 'n' && literal("null")) { out = JValue(); return true; }
    if (c == 'N' && literal("NaN")) { out = JValue::flt(std::nan("")); return true; }
    if (c == 'I' && literal("Infinity")) { out = JValue::flt(std::numeric_limits<double>::infinity()); return true; }
    if (c == '-' && literal("-Infinity")) { out = JValue::flt(-std::numeric_limits<double>::infinity()); return true; }
    if (c == '-' || (c >= '0' && c <= '9')) return number(out);
    return false;
  }

  bool number(JValue& out) {
    const size_t start = i_;
    if (s_[i_] == '-') ++i_;
    if (i_ >= s_.size()) return false;
    if (s_[i_] == '0') {
      ++i_;
    } else if (s_[i_] >= '1' && s_[i_] <= '9') {
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    } else {
      return false;
    }
    bool is_float = false;
    if (i_ + 1 < s_.size() && s_[i_] == '.' && s_[i_ + 1] >= '0' && s_[i_ + 1] <= '9') {
      is_float = true;
      i_ += 1;
      while (i_ < s_.size() && s_[i_] >= '0' && s_[i_] <= '9') ++i_;
    }
    if (i_ < s_.size() && (s_[i_] == 'e' || s_[i_] == 'E')) {
      size_t j = i_ + 1;
      if (j < s_.size() && (s_[j] == '+' || s_[j] == '-')) ++j;
      if (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') {
        is_float = true;
        while (j < s_.size() && s_[j] >= '0' && s_[j] <= '9') ++j;
        i_ = j;
      }
    }
    const std::string token(s_.substr(start, i_ - start));
    if (is_float) {
      out = JValue::flt(std::strtod(token.c_str(), nullptr));
    } else {
      out = JValue::integer(token == "-0" ? std::string("0") : token);
    }
    return true;
  }

  static int hex_value(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  }

  bool hex4(uint32_t& v) {
    if (i_ + 4 > s_.size()) return false;
    v = 0;
    for (int k = 0; k < 4; ++k) {
      const int h = hex_value(s_[i_ + k]);
      if (h < 0) return false;
      v = v * 16 + static_cast<uint32_t>(h);
    }
    i_ += 4;
    return true;
  }

  bool string(std::string& out) {
    ++i_;  // opening quote
    while (true) {
      if (i_ >= s_.size()) return false;
      const auto c = static_cast<unsigned char>(s_[i_]);
      if (c == '"') {
        ++i_;
        return true;
      }
      if (c < 0x20) return false;  // strict: control characters must be escaped
      if (c != '\\') {
        out.push_back(static_cast<char>(c));
        ++i_;
        continue;
      }
      ++i_;
      if (i_ >= s_.size()) return false;
      const char e = s_[i_++];
      switch (e) {
        case '"': out.push_back('"'); break;
        case '\\': out.push_back('\\'); break;
        case '/': out.push_back('/'); break;
        case 'b': out.push_back('\b'); break;
        case 'f': out.push_back('\f'); break;
        case 'n': out.push_back('\n'); break;
        case 'r': out.push_back('\r'); break;
        case 't': out.push_back('\t'); break;
        case 'u': {
          uint32_t cp;
          if (!hex4(cp)) return false;
          if (cp >= 0xD800 && cp <= 0xDBFF && i_ + 6 <= s_.size() && s_[i_] == '\\' && s_[i_ + 1] == 'u') {
            const size_t save = i_;
            i_ += 2;
            uint32_t low;
            if (hex4(low) && low >= 0xDC00 && low <= 0xDFFF) {
              cp = 0x10000 + ((cp - 0xD800) << 10) + (low - 0xDC00);
            } else {
              i_ = save;  // a lone high surrogate; the next escape is read on its own
            }
          }
          utf8_append(out, cp);
          break;
        }
        default:
          return false;
      }
    }
  }

  bool array(JValue& out, int depth) {
    ++i_;
    out = JValue();
    out.type = JValue::Type::kArray;
    skip_ws();
    if (i_ < s_.size() && s_[i_] == ']') {
      ++i_;
      return true;
    }
    while (true) {
      skip_ws();
      JValue item;
      if (!value(item, depth + 1)) return false;
      out.items.push_back(std::move(item));
      skip_ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == ']') {
        ++i_;
        return true;
      }
      return false;
    }
  }

  bool object(JValue& out, int depth) {
    ++i_;
    out = JValue();
    out.type = JValue::Type::kObject;
    skip_ws();
    if (i_ < s_.size() && s_[i_] == '}') {
      ++i_;
      return true;
    }
    // Keys are looked up through an index, so a large object (the tokenizer's vocabulary) stays linear.
    std::unordered_map<std::string, size_t> index;
    while (true) {
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != '"') return false;
      std::string key;
      if (!string(key)) return false;
      skip_ws();
      if (i_ >= s_.size() || s_[i_] != ':') return false;
      ++i_;
      skip_ws();
      JValue member;
      if (!value(member, depth + 1)) return false;
      const auto found = index.find(key);
      if (found != index.end()) {
        out.values[found->second] = std::move(member);
      } else {
        index.emplace(key, out.keys.size());
        out.keys.push_back(std::move(key));
        out.values.push_back(std::move(member));
      }
      skip_ws();
      if (i_ >= s_.size()) return false;
      if (s_[i_] == ',') {
        ++i_;
        continue;
      }
      if (s_[i_] == '}') {
        ++i_;
        return true;
      }
      return false;
    }
  }

  std::string_view s_;
  size_t i_ = 0;
};

void dump_number_float(double v, std::string& out) {
  if (std::isnan(v)) {
    out += "NaN";
  } else if (std::isinf(v)) {
    out += v > 0 ? "Infinity" : "-Infinity";
  } else {
    out += py_float_repr(v);
  }
}

}  // namespace

const JValue* JValue::get(std::string_view key) const {
  if (type != Type::kObject) return nullptr;
  for (size_t k = 0; k < keys.size(); ++k) {
    if (keys[k] == key) return &values[k];
  }
  return nullptr;
}

JValue JValue::string(std::string s) {
  JValue v;
  v.type = Type::kString;
  v.text = std::move(s);
  return v;
}

JValue JValue::integer(std::string canonical_digits) {
  JValue v;
  v.type = Type::kInt;
  v.text = std::move(canonical_digits);
  return v;
}

JValue JValue::flt(double d) {
  JValue v;
  v.type = Type::kFloat;
  v.number = d;
  return v;
}

JValue JValue::boolean_value(bool b) {
  JValue v;
  v.type = Type::kBool;
  v.boolean = b;
  return v;
}

bool json_loads(std::string_view text, JValue& out) {
  Parser parser(text);
  return parser.parse(out);
}

void json_append_string(std::string& out, std::string_view s) {
  out.push_back('"');
  for (const char raw : s) {
    const auto c = static_cast<unsigned char>(raw);
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\r': out += "\\r"; break;
      case '\t': out += "\\t"; break;
      case '\b': out += "\\b"; break;
      case '\f': out += "\\f"; break;
      default:
        if (c < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof(buf), "\\u%04x", c);
          out += buf;
        } else {
          out.push_back(raw);
        }
    }
  }
  out.push_back('"');
}

void json_dumps(const JValue& value, std::string& out) {
  switch (value.type) {
    case JValue::Type::kNull: out += "null"; break;
    case JValue::Type::kBool: out += value.boolean ? "true" : "false"; break;
    case JValue::Type::kInt: out += value.text; break;
    case JValue::Type::kFloat: dump_number_float(value.number, out); break;
    case JValue::Type::kString: json_append_string(out, value.text); break;
    case JValue::Type::kArray:
      out.push_back('[');
      for (size_t k = 0; k < value.items.size(); ++k) {
        if (k) out += ", ";
        json_dumps(value.items[k], out);
      }
      out.push_back(']');
      break;
    case JValue::Type::kObject:
      out.push_back('{');
      for (size_t k = 0; k < value.keys.size(); ++k) {
        if (k) out += ", ";
        json_append_string(out, value.keys[k]);
        out += ": ";
        json_dumps(value.values[k], out);
      }
      out.push_back('}');
      break;
  }
}

std::string json_dumps(const JValue& value) {
  std::string out;
  json_dumps(value, out);
  return out;
}

std::string py_float_repr(double value) {
  if (value == 0.0) return std::signbit(value) ? "-0.0" : "0.0";
  // Shortest round-trip digits, in scientific form: d.ddde[+-]XX
  char buf[64];
  const auto res = std::to_chars(buf, buf + sizeof(buf), value, std::chars_format::scientific);
  std::string sci(buf, res.ptr);
  std::string sign;
  size_t p = 0;
  if (sci[0] == '-') {
    sign = "-";
    p = 1;
  }
  const size_t e = sci.find('e', p);
  std::string digits;
  for (size_t k = p; k < e; ++k) {
    if (sci[k] != '.') digits.push_back(sci[k]);
  }
  const int exp10 = std::atoi(sci.c_str() + e + 1);
  // Python's repr: scientific when the decimal point would sit before -3 or after 16 digits.
  const int decpt = exp10 + 1;
  std::string out = sign;
  if (decpt <= -4 || decpt > 16) {
    out.push_back(digits[0]);
    if (digits.size() > 1) {
      out.push_back('.');
      out.append(digits, 1, std::string::npos);
    }
    out.push_back('e');
    const int mag = exp10 < 0 ? -exp10 : exp10;
    out.push_back(exp10 < 0 ? '-' : '+');
    if (mag < 10) out.push_back('0');
    out += std::to_string(mag);
    return out;
  }
  const int n = static_cast<int>(digits.size());
  if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<size_t>(-decpt), '0');
    out += digits;
  } else if (decpt >= n) {
    out += digits;
    out.append(static_cast<size_t>(decpt - n), '0');
    out += ".0";
  } else {
    out.append(digits, 0, static_cast<size_t>(decpt));
    out.push_back('.');
    out.append(digits, static_cast<size_t>(decpt), std::string::npos);
  }
  return out;
}

bool json_is_finite(const JValue& value) {
  switch (value.type) {
    case JValue::Type::kFloat: return std::isfinite(value.number);
    case JValue::Type::kArray:
      for (const JValue& item : value.items) {
        if (!json_is_finite(item)) return false;
      }
      return true;
    case JValue::Type::kObject:
      for (const JValue& member : value.values) {
        if (!json_is_finite(member)) return false;
      }
      return true;
    default: return true;
  }
}

}  // namespace lx::text
