#include "runtime/tool_grammar.h"

#include <algorithm>
#include <cstring>
#include <initializer_list>
#include <string_view>
#include <stdexcept>
#include <unordered_map>

#include "text/json.h"

namespace lx::runtime {
namespace {

// The format's literal pieces (see the header).
constexpr const char* kThinkEnd = "</think>";
constexpr const char* kCallStart = "<|tool_call|><function=";
constexpr const char* kParamOpen = "<parameter=";
constexpr const char* kFuncClose = "</function>";
constexpr const char* kParamClose = "</parameter>";
constexpr const char* kValueEnd = "\n</parameter>\n";  // what follows a value, the newline included
constexpr size_t kMaxProps = 64;

enum class Mode : uint8_t {
  kThink,      // reasoning: any text up to "</think>"
  kLead,       // before a call: whitespace, then kCallStart
  kName,       // the function's name, then ">"
  kOpen,       // the newline after the function's header
  kParams,     // between arguments: kParamOpen, or kFuncClose once the required ones are in
  kPName,      // an argument's name, then ">"
  kPNl,        // the newline after an argument's header
  kValue,      // the argument's value (its kind decides how it is read)
  kTermLit,    // the value's end: kValueEnd
  kAfterCall,  // a call is complete: another kCallStart, or the end
};

enum class Kind : uint8_t {
  kRaw,      // a string argument: its text as it is
  kLiteral,  // one of a fixed list of texts (booleans, null, enums, consts)
  kJson,     // a JSON value, restricted at the top to the kinds in `allow`
};

// JSON kinds allowed at the top of a value.
constexpr uint8_t kAllowObject = 1, kAllowArray = 2, kAllowString = 4, kAllowNumber = 8, kAllowLiteral = 16;

// JSON reading states.
enum JS : uint8_t {
  kJV,     // a value is expected
  kJAOK,   // after "[": a value or "]"
  kJOK,    // after "{": a key or "}"
  kJOKEY,  // after "," in an object: a key
  kJStr,   // inside a string
  kJEsc,   // after a backslash
  kJU,     // inside a \u escape
  kJCol,   // after a key: ":"
  kJNum,   // inside a number
  kJLit,   // inside true, false or null
  kJVA,    // after a value in a container: "," or the closer
};

// Number reading states (JSON numbers, the integer form when the schema says integer).
enum NS : uint8_t {
  kNone, kNStart, kNNeg, kNZero, kNInt, kNDot, kNFrac, kNE, kNESign, kNExp1, kNExp2,
};

// A number is at most this many bytes, so its value is always finite as a double (the output parser
// falls back to a string for a value that overflows, which would break the schema's type).
constexpr uint8_t kMaxNumberBytes = 64;

bool accepting_number(uint8_t n) {
  return n == kNZero || n == kNInt || n == kNFrac || n == kNExp1 || n == kNExp2;
}

// The next number state for byte c, or 0 when c does not continue the number.
uint8_t number_next(uint8_t n, unsigned char c, bool integer) {
  const bool digit = c >= '0' && c <= '9';
  switch (n) {
    case kNStart:
      if (c == '-') return kNNeg;
      if (c == '0') return kNZero;
      if (digit) return kNInt;
      return 0;
    case kNNeg:
      if (c == '0') return kNZero;
      if (digit) return kNInt;
      return 0;
    case kNZero:
    case kNInt:
      if (digit && n == kNInt) return kNInt;
      if (integer) return 0;
      if (c == '.') return kNDot;
      if (c == 'e' || c == 'E') return kNE;
      return 0;
    case kNDot:
      return digit ? kNFrac : 0;
    case kNFrac:
      if (digit) return kNFrac;
      if (c == 'e' || c == 'E') return kNE;
      return 0;
    case kNE:
      if (c == '+' || c == '-') return kNESign;
      return digit ? kNExp1 : 0;
    case kNESign:
      return digit ? kNExp1 : 0;
    case kNExp1:
      return digit ? kNExp2 : 0;  // at most two exponent digits: |exponent| <= 99, so the value stays finite
    case kNExp2:
      return 0;
    default:
      return 0;
  }
}

bool is_space(unsigned char c) { return c == ' ' || c == '\n' || c == '\r' || c == '\t'; }
bool is_hex(unsigned char c) {
  return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F');
}

bool ends_with(const std::string& s, std::string_view p) {
  return s.size() >= p.size() && std::string_view(s).substr(s.size() - p.size()) == p;
}

// The longest suffix of `s` that is a proper prefix of one of the patterns. Matching across tokens
// only ever needs this suffix, so it is also the state a mask is cached by.
std::string tail_state(const std::string& s, std::initializer_list<const char*> patterns) {
  size_t best = 0;
  for (const char* p : patterns) {
    const size_t len = std::strlen(p);
    for (size_t k = std::min(len - 1, s.size()); k > best; --k) {
      if (s.compare(s.size() - k, k, p, k) == 0) {
        best = k;
        break;
      }
    }
  }
  return s.substr(s.size() - best);
}

struct Prop {
  std::string name;
  Kind kind = Kind::kRaw;
  uint8_t allow = 0;      // kJson: the top-level kinds
  bool integer = false;   // kJson: the top-level number is an integer
  std::vector<std::string> literals;  // kLiteral
};

struct Tool {
  std::string name;
  std::vector<Prop> props;
  uint64_t required = 0;  // bit i: props[i] must appear
};

}  // namespace

// The state of one grammar: small, copyable, and everything the mask is a function of.
struct State {
  Mode mode = Mode::kLead;
  std::string buf;          // the matcher's bytes in the current element (see tail_state)
  int tool = -1;
  uint64_t emitted = 0;     // arguments written in this call
  int prop = -1;            // the argument being written
  // JSON reading (kValue with kJson)
  std::string jstack;       // open containers, '{' or '['
  uint8_t js = kJV;
  uint8_t jnum = kNone;
  uint8_t jlen = 0;         // bytes of the JSON number being read
  uint8_t jesc = 0;         // hex digits left in a \u escape
  bool jdone = false;       // the top-level JSON value is complete
  bool jkey = false;        // the string being read is an object key
};

struct ToolGrammar::Impl {
  std::vector<Tool> tools;
  std::vector<int> names;  // tools that may be called
  bool thinking = true;
  State start;
  State state;
  std::unordered_map<std::string, std::vector<uint32_t>> cache;
  std::vector<uint32_t> out;
  const std::vector<std::string>* texts = nullptr;  // for mask(); not owned
  std::vector<bool> stop_mark;                       // scratch for mask()
};

namespace {

// Feeds one byte; false when it cannot come next (the state may then be partly changed: callers
// work on a copy).
class Stepper {
 public:
  Stepper(const std::vector<Tool>& tools, const std::vector<int>& names) : tools_(tools), names_(names) {}

  bool step(State& s, unsigned char c) const {
    switch (s.mode) {
      case Mode::kThink: {
        const std::string nb = s.buf + static_cast<char>(c);
        if (ends_with(nb, kThinkEnd)) {
          s.mode = Mode::kLead;
          s.buf.clear();
          return true;
        }
        s.buf = tail_state(nb, {kThinkEnd});
        return true;
      }
      default:
        break;
    }
    return step_format(s, c);
  }

 private:
  bool step_format(State& s, unsigned char c) const;
  bool value_raw(State& s, unsigned char c) const;
  bool value_literal(State& s, unsigned char c) const;
  bool json_step(State& s, unsigned char c, const Prop& p, bool top) const;
  bool term_step(State& s, unsigned char c) const;
  void json_value_done(State& s) const {
    if (s.jstack.empty()) s.jdone = true;
    else s.js = kJVA;
  }
  void json_close(State& s) const {
    s.jstack.pop_back();
    json_value_done(s);
  }

  const std::vector<Tool>& tools_;
  const std::vector<int>& names_;
};

bool Stepper::step_format(State& s, unsigned char c) const {
  switch (s.mode) {
    case Mode::kLead: {
      if (s.buf.empty() && is_space(c)) return true;
      const std::string nb = s.buf + static_cast<char>(c);
      if (std::strncmp(kCallStart, nb.c_str(), nb.size()) != 0) return false;
      s.buf = nb;
      if (s.buf.size() == std::strlen(kCallStart)) {
        s.mode = Mode::kName;
        s.buf.clear();
      }
      return true;
    }
    case Mode::kAfterCall: {
      const std::string nb = s.buf + static_cast<char>(c);
      if (std::strncmp(kCallStart, nb.c_str(), nb.size()) != 0) return false;
      s.buf = nb;
      if (s.buf.size() == std::strlen(kCallStart)) {
        s.mode = Mode::kName;
        s.buf.clear();
      }
      return true;
    }
    case Mode::kName: {
      if (c == '>') {
        for (int t : names_)
          if (tools_[t].name == s.buf) {
            s.tool = t;
            s.mode = Mode::kOpen;
            s.buf.clear();
            s.emitted = 0;
            return true;
          }
      }
      const std::string nb = s.buf + static_cast<char>(c);
      for (int t : names_)
        if (tools_[t].name.compare(0, nb.size(), nb) == 0 && nb.size() <= tools_[t].name.size()) {
          s.buf = nb;
          return true;
        }
      return false;
    }
    case Mode::kOpen: {
      if (c != '\n') return false;
      s.mode = Mode::kParams;
      s.buf.clear();
      return true;
    }
    case Mode::kParams: {
      const Tool& tool = tools_[s.tool];
      const bool may_open = s.emitted != (tool.props.size() == 64 ? ~0ull : ((1ull << tool.props.size()) - 1));
      const bool may_close = (s.emitted & tool.required) == tool.required;
      const std::string nb = s.buf + static_cast<char>(c);
      if (may_open && std::strncmp(kParamOpen, nb.c_str(), nb.size()) == 0) {
        s.buf = nb;
        if (s.buf.size() == std::strlen(kParamOpen)) {
          s.mode = Mode::kPName;
          s.buf.clear();
        }
        return true;
      }
      if (may_close && std::strncmp(kFuncClose, nb.c_str(), nb.size()) == 0) {
        s.buf = nb;
        if (s.buf.size() == std::strlen(kFuncClose)) {
          s.mode = Mode::kAfterCall;
          s.buf.clear();
        }
        return true;
      }
      return false;
    }
    case Mode::kPName: {
      const Tool& tool = tools_[s.tool];
      if (c == '>') {
        for (size_t i = 0; i < tool.props.size(); ++i)
          if (!(s.emitted >> i & 1) && tool.props[i].name == s.buf) {
            s.prop = static_cast<int>(i);
            s.emitted |= 1ull << i;
            s.mode = Mode::kPNl;
            s.buf.clear();
            return true;
          }
      }
      const std::string nb = s.buf + static_cast<char>(c);
      for (size_t i = 0; i < tool.props.size(); ++i)
        if (!(s.emitted >> i & 1) && tool.props[i].name.size() >= nb.size() &&
            tool.props[i].name.compare(0, nb.size(), nb) == 0) {
          s.buf = nb;
          return true;
        }
      return false;
    }
    case Mode::kPNl: {
      if (c != '\n') return false;
      s.mode = Mode::kValue;
      s.buf.clear();
      s.jstack.clear();
      s.js = kJV;
      s.jnum = kNone;
      s.jlen = 0;
      s.jdone = false;
      s.jkey = false;
      s.jesc = 0;
      return true;
    }
    case Mode::kValue: {
      const Prop& p = tools_[s.tool].props[s.prop];
      if (p.kind == Kind::kRaw) return value_raw(s, c);
      if (p.kind == Kind::kLiteral) return value_literal(s, c);
      if (json_step(s, c, p, true)) return true;
      if (!s.jdone) return false;
      // the value ended before this byte: the byte is the value's end
      s.mode = Mode::kTermLit;
      s.buf.clear();
      return term_step(s, c);
    }
    case Mode::kTermLit:
      return term_step(s, c);
    default:
      return false;
  }
}

bool Stepper::value_raw(State& s, unsigned char c) const {
  const std::string nb = s.buf + static_cast<char>(c);
  if (ends_with(nb, kValueEnd)) {
    s.mode = Mode::kParams;
    s.buf.clear();
    return true;
  }
  // A closing tag inside a value is only allowed as the value's end: its newline must follow it.
  if (s.buf == std::string("\n") + kParamClose) return false;
  if (ends_with(nb, kParamClose) && !ends_with(nb, std::string("\n") + kParamClose)) return false;
  s.buf = tail_state(nb, {kValueEnd, kParamClose});
  return true;
}

bool Stepper::value_literal(State& s, unsigned char c) const {
  const Prop& p = tools_[s.tool].props[s.prop];
  if (c == '\n') {
    for (const std::string& lit : p.literals)
      if (lit == s.buf) {
        s.mode = Mode::kTermLit;
        s.buf.clear();
        return term_step(s, c);
      }
    return false;
  }
  const std::string nb = s.buf + static_cast<char>(c);
  for (const std::string& lit : p.literals)
    if (lit.size() >= nb.size() && lit.compare(0, nb.size(), nb) == 0) {
      s.buf = nb;
      return true;
    }
  return false;
}

bool Stepper::term_step(State& s, unsigned char c) const {
  const size_t pos = s.buf.size();
  if (pos >= std::strlen(kValueEnd) || static_cast<unsigned char>(kValueEnd[pos]) != c) return false;
  s.buf.push_back(static_cast<char>(c));
  if (s.buf.size() == std::strlen(kValueEnd)) {
    s.mode = Mode::kParams;
    s.buf.clear();
  }
  return true;
}

// One byte of a JSON value (a top-level one is restricted by the property's kinds). Returns false
// when the byte cannot come next.
bool Stepper::json_step(State& s, unsigned char c, const Prop& p, bool top_level) const {
  if (s.jdone) return false;  // the caller hands such a byte to the value's end
  const bool top = s.jstack.empty();
  switch (s.js) {
    case kJV:
    case kJAOK: {
      // A top-level value starts at its first byte (the template writes json(value), with no space).
      if (is_space(c) && !(s.js == kJV && top && top_level)) return true;
      if (s.js == kJAOK && c == ']') {
        json_close(s);
        return true;
      }
      if (top && top_level) {
        const bool ok = (c == '{' && (p.allow & kAllowObject)) || (c == '[' && (p.allow & kAllowArray)) ||
                        (c == '"' && (p.allow & kAllowString)) ||
                        ((c == '-' || (c >= '0' && c <= '9')) && (p.allow & kAllowNumber)) ||
                        ((c == 't' || c == 'f' || c == 'n') && (p.allow & kAllowLiteral));
        if (!ok) return false;
      }
      if (c == '{') {
        s.jstack.push_back('{');
        s.js = kJOK;
        return true;
      }
      if (c == '[') {
        s.jstack.push_back('[');
        s.js = kJAOK;
        return true;
      }
      if (c == '"') {
        s.js = kJStr;
        s.jkey = false;
        return true;
      }
      if (c == '-' || (c >= '0' && c <= '9')) {
        s.js = kJNum;
        s.jnum = c == '-' ? kNNeg : (c == '0' ? kNZero : kNInt);
        s.jlen = 1;
        return true;
      }
      if (c == 't' || c == 'f' || c == 'n') {
        s.js = kJLit;
        s.buf.assign(1, static_cast<char>(c));
        return true;
      }
      return false;
    }
    case kJOK:
    case kJOKEY: {
      if (is_space(c)) return true;
      if (s.js == kJOK && c == '}') {
        json_close(s);
        return true;
      }
      if (c == '"') {
        s.js = kJStr;
        s.jkey = true;
        return true;
      }
      return false;
    }
    case kJStr: {
      if (c == '"') {
        if (s.jkey) s.js = kJCol;
        else json_value_done(s);
        return true;
      }
      if (c == '\\') {
        s.js = kJEsc;
        return true;
      }
      return c >= 0x20;
    }
    case kJEsc: {
      if (c == 'u') {
        s.js = kJU;
        s.jesc = 4;
        return true;
      }
      if (std::strchr("\"\\/bfnrt", c) && c != 0) {
        s.js = kJStr;
        return true;
      }
      return false;
    }
    case kJU: {
      if (!is_hex(c)) return false;
      if (--s.jesc == 0) s.js = kJStr;
      return true;
    }
    case kJCol: {
      if (is_space(c)) return true;
      if (c != ':') return false;
      s.js = kJV;
      return true;
    }
    case kJNum: {
      const uint8_t next = s.jlen < kMaxNumberBytes ? number_next(s.jnum, c, top && top_level && p.integer) : 0;
      if (next) {
        s.jnum = next;
        s.jlen++;
        return true;
      }
      if (!accepting_number(s.jnum)) return false;
      json_value_done(s);  // the number ended before this byte; the byte is read after it
      return json_step(s, c, p, top_level);
    }
    case kJLit: {
      const std::string nb = s.buf + static_cast<char>(c);
      for (const char* word : {"true", "false", "null"}) {
        const size_t n = std::strlen(word);
        if (nb.size() <= n && std::strncmp(word, nb.c_str(), nb.size()) == 0) {
          s.buf = nb;
          if (nb.size() == n) {
            s.buf.clear();
            json_value_done(s);
          }
          return true;
        }
      }
      return false;
    }
    case kJVA: {
      if (is_space(c)) return true;
      const char top_char = s.jstack.empty() ? 0 : s.jstack.back();
      if (c == ',' && top_char == '[') {
        s.js = kJV;
        return true;
      }
      if (c == ',' && top_char == '{') {
        s.js = kJOKEY;
        return true;
      }
      if (c == ']' && top_char == '[') {
        json_close(s);
        return true;
      }
      if (c == '}' && top_char == '{') {
        json_close(s);
        return true;
      }
      return false;
    }
    default:
      return false;
  }
}

}  // namespace

ToolGrammar::ToolGrammar(const ToolGrammarSpec& spec, bool thinking) : impl_(std::make_unique<Impl>()) {
  if (spec.tools.empty()) throw std::invalid_argument("tool_choice needs at least one tool");
  for (const auto& [name, schema_text] : spec.tools) {
    if (name.empty() || name.find_first_of("<>\n") != std::string::npos)
      throw std::invalid_argument("the tool name '" + name + "' cannot be called by this grammar");
    Tool tool;
    tool.name = name;
    lx::text::JValue schema;
    if (!lx::text::json_loads(schema_text, schema) || !schema.is_object())
      throw std::invalid_argument("the parameters of tool '" + name + "' are not a JSON object");
    const lx::text::JValue* props = schema.get("properties");
    if (props && !props->is_object())
      throw std::invalid_argument("the properties of tool '" + name + "' must be an object");
    if (props) {
      if (props->keys.size() > kMaxProps)
        throw std::invalid_argument("tool '" + name + "' has more than 64 arguments");
      for (size_t i = 0; i < props->keys.size(); ++i) {
        const lx::text::JValue& ps = props->values[i];
        Prop p;
        p.name = props->keys[i];
        if (p.name.empty() || p.name.find_first_of("<>\n") != std::string::npos)
          throw std::invalid_argument("argument '" + p.name + "' of tool '" + name + "' cannot be written by this grammar");
        const lx::text::JValue* type = ps.get("type");
        const lx::text::JValue* en = ps.get("enum");
        const lx::text::JValue* cst = ps.get("const");
        if (cst || (en && en->is_array())) {
          const auto& list = cst ? std::vector<lx::text::JValue>{*cst} : en->items;
          p.kind = Kind::kLiteral;
          for (const auto& v : list) {
            std::string text = v.is_string() ? v.text : lx::text::json_dumps(v);
            if (text.find('\n') != std::string::npos || text.empty())
              throw std::invalid_argument("argument '" + p.name + "' has an enum value this grammar cannot write");
            p.literals.push_back(text);
          }
        } else if (type && type->is_string() && type->text == "string") {
          p.kind = Kind::kRaw;
        } else if (type && type->is_string() && (type->text == "integer" || type->text == "number")) {
          p.kind = Kind::kJson;
          p.allow = kAllowNumber;
          p.integer = type->text == "integer";
        } else if (type && type->is_string() && type->text == "boolean") {
          p.kind = Kind::kLiteral;
          p.literals = {"true", "false"};
        } else if (type && type->is_string() && type->text == "null") {
          p.kind = Kind::kLiteral;
          p.literals = {"null"};
        } else if (type && type->is_string() && type->text == "object") {
          p.kind = Kind::kJson;
          p.allow = kAllowObject;
        } else if (type && type->is_string() && type->text == "array") {
          p.kind = Kind::kJson;
          p.allow = kAllowArray;
        } else {
          p.kind = Kind::kJson;  // any JSON value: a union of types, or no type at all
          p.allow = kAllowObject | kAllowArray | kAllowString | kAllowNumber | kAllowLiteral;
        }
        const lx::text::JValue* required = schema.get("required");
        if (required && required->is_array())
          for (const auto& r : required->items)
            if (r.is_string() && r.text == p.name) tool.required |= 1ull << tool.props.size();
        tool.props.push_back(std::move(p));
      }
    }
    impl_->tools.push_back(std::move(tool));
  }
  impl_->thinking = thinking;
  impl_->names.clear();
  const size_t callable = spec.named ? 1 : impl_->tools.size();
  for (size_t i = 0; i < callable; ++i) impl_->names.push_back(static_cast<int>(i));
  impl_->start.mode = thinking ? Mode::kThink : Mode::kLead;
  impl_->state = impl_->start;
}

ToolGrammar::~ToolGrammar() = default;

ToolGrammar::ToolGrammar(const ToolGrammar& other) : impl_(std::make_unique<Impl>(*other.impl_)) {}

ToolGrammar& ToolGrammar::operator=(const ToolGrammar& other) {
  if (this != &other) impl_ = std::make_unique<Impl>(*other.impl_);
  return *this;
}

ToolGrammar::ToolGrammar(ToolGrammar&&) noexcept = default;
ToolGrammar& ToolGrammar::operator=(ToolGrammar&&) noexcept = default;

bool ToolGrammar::accept_byte(unsigned char c) {
  State next = impl_->state;
  Stepper stepper(impl_->tools, impl_->names);
  if (!stepper.step(next, c)) return false;
  impl_->state = std::move(next);
  return true;
}

bool ToolGrammar::accept(std::string_view text) {
  State next = impl_->state;
  Stepper stepper(impl_->tools, impl_->names);
  for (unsigned char c : text)
    if (!stepper.step(next, c)) return false;
  impl_->state = std::move(next);
  return true;
}

bool ToolGrammar::allows(std::string_view text) const {
  if (text.empty()) return false;
  State next = impl_->state;
  Stepper stepper(impl_->tools, impl_->names);
  for (unsigned char c : text)
    if (!stepper.step(next, c)) return false;
  return true;
}

bool ToolGrammar::may_end() const { return impl_->state.mode == Mode::kAfterCall && impl_->state.buf.empty(); }

const std::vector<uint32_t>& ToolGrammar::mask(const std::vector<std::string>& texts, const std::vector<int32_t>& stops) {
  // the key is everything the next byte depends on
  const State& s = impl_->state;
  std::string key;
  key.push_back(static_cast<char>(s.mode));
  key.append(s.buf).push_back('\0');
  key.append(std::to_string(s.tool)).push_back(',');
  key.append(std::to_string(s.emitted)).push_back(',');
  key.append(std::to_string(s.prop)).push_back(',');
  key.append(s.jstack).push_back('\0');
  key.push_back(static_cast<char>(s.js));
  key.push_back(static_cast<char>(s.jnum));
  key.push_back(static_cast<char>(s.jlen));
  key.push_back(static_cast<char>(s.jdone));
  key.push_back(static_cast<char>(s.jkey));
  key.push_back(static_cast<char>(s.jesc));
  key.push_back(static_cast<char>(stops.size()));
  key.append(std::to_string(stops.empty() ? 0 : stops.front()));

  auto it = impl_->cache.find(key);
  if (it != impl_->cache.end()) return it->second;

  const size_t words = (texts.size() + 31) / 32;
  std::vector<uint32_t> bits(words, 0);
  const bool end_ok = may_end();
  std::vector<bool> stop_mark(texts.size(), false);
  for (int32_t id : stops)
    if (id >= 0 && static_cast<size_t>(id) < texts.size()) stop_mark[static_cast<size_t>(id)] = true;
  for (size_t i = 0; i < texts.size(); ++i) {
    const bool ok = stop_mark[i] ? end_ok : allows(texts[i]);
    if (ok) bits[i / 32] |= 1u << (i % 32);
  }
  if (impl_->cache.size() > 4096) impl_->cache.clear();
  return impl_->cache.emplace(std::move(key), std::move(bits)).first->second;
}

}  // namespace lx::runtime
