// A Jinja interpreter for chat templates, with the behaviour transformers' apply_chat_template gives
// them: jinja2's lexer whitespace rules (trim_blocks, lstrip_blocks, one trailing newline dropped),
// Python value semantics (truthiness, str and repr, comparisons and operators), the filters, tests
// and str/list/dict methods templates use, loops with the loop object and loop controls, if/elif,
// set and namespace, macros, and the globals transformers installs (raise_exception, strftime_now,
// range, namespace, dict). Templates are parsed once and rendered many times.
#pragma once

#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lx::text::jinja {

// A template error: a syntax error, an operation Python would reject, or raise_exception().
class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class Value;

// The arguments of a call: positional, then keyword.
struct CallArgs {
  std::vector<Value> positional;
  std::vector<std::pair<std::string, Value>> keyword;
  // The argument given at position `index` or by keyword `name`; nullptr when neither.
  const Value* get(size_t index, const char* name) const;
};

// Something with attributes and, maybe, calls: namespaces, loops, macros, builtins, bound methods.
class Object {
 public:
  virtual ~Object() = default;
  // The attribute `name`, or undefined.
  virtual Value getattr(const std::string& name) const;
  // Attribute assignment (namespaces only). False when the object does not accept it.
  virtual bool setattr(const std::string& name, const Value& value);
  virtual bool callable() const { return false; }
  virtual Value call(const CallArgs& args) const;
  virtual const char* type_name() const { return "object"; }
};

// A Python-like value as a template sees it. Lists and dicts are shared, as Python objects are.
class Value {
 public:
  enum class Kind : uint8_t { kUndefined, kNone, kBool, kInt, kFloat, kString, kList, kDict, kObject };

  Value() = default;  // undefined
  static Value none();
  static Value boolean(bool v);
  static Value integer(int64_t v);
  static Value real(double v);
  static Value string(std::string s);
  static Value list(std::vector<Value> items);
  static Value tuple(std::vector<Value> items);
  static Value dict(std::vector<std::pair<Value, Value>> items);
  static Value object(std::shared_ptr<Object> obj);

  Kind kind() const { return kind_; }
  bool is_undefined() const { return kind_ == Kind::kUndefined; }
  bool is_none() const { return kind_ == Kind::kNone; }
  bool is_bool() const { return kind_ == Kind::kBool; }
  bool is_int() const { return kind_ == Kind::kInt; }
  bool is_float() const { return kind_ == Kind::kFloat; }
  bool is_number() const { return kind_ == Kind::kBool || kind_ == Kind::kInt || kind_ == Kind::kFloat; }
  bool is_string() const { return kind_ == Kind::kString; }
  bool is_list() const { return kind_ == Kind::kList; }
  bool is_tuple() const { return kind_ == Kind::kList && tuple_; }
  bool is_dict() const { return kind_ == Kind::kDict; }
  bool is_object() const { return kind_ == Kind::kObject; }

  // Python truthiness.
  bool truthy() const;
  // Python str() and repr().
  std::string str() const;
  std::string repr() const;

  bool as_bool() const { return bool_; }
  int64_t as_int() const { return int_; }  // int and bool
  double as_float() const;                 // int, bool and float
  const std::string& as_string() const { return string_; }
  const std::vector<Value>& as_list() const { return *list_; }
  const std::vector<std::pair<Value, Value>>& as_dict() const { return *dict_; }
  const std::shared_ptr<Object>& as_object() const { return object_; }

  // The dict entry for `key` (Python equality), or nullptr.
  const Value* find(const Value& key) const;
  // Inserts or replaces a dict entry.
  void set_item(const Value& key, Value value);

 private:
  Kind kind_ = Kind::kUndefined;
  bool bool_ = false;
  int64_t int_ = 0;
  double float_ = 0.0;
  std::string string_;
  std::shared_ptr<std::vector<Value>> list_;
  bool tuple_ = false;  // a list of kind tuple: repr and == tell the two apart
  std::shared_ptr<std::vector<std::pair<Value, Value>>> dict_;
  std::shared_ptr<Object> object_;
};

struct TemplateImpl;  // the parsed tree (jinja.cpp)

// A parsed template.
class Template {
 public:
  // Parses the source. Throws Error on a syntax error.
  explicit Template(std::string_view source);

  // Renders with the given top-level variables (messages, tools, add_generation_prompt, ...).
  // Throws Error when the template fails, including through raise_exception().
  std::string render(const std::vector<std::pair<std::string, Value>>& variables) const;

 private:
  std::shared_ptr<const TemplateImpl> impl_;
};

}  // namespace lx::text::jinja
