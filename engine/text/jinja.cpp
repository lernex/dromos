// The Jinja interpreter: the lexer (jinja2's whitespace rules with trim_blocks and lstrip_blocks), the
// parser (jinja2's expression grammar and statements), and the evaluator (frames, loops, macros).
#include "text/jinja.h"

#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <unordered_map>
#include <utility>
#include <vector>

#include "text/jinja_internal.h"
#include "text/py.h"
#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text::jinja {

// ---- The tree ---------------------------------------------------------------------------------

struct Node;
using NodePtr = std::unique_ptr<Node>;

struct Node {
  enum class Type : uint8_t {
    // statements
    kText, kOutput, kIf, kBranch, kFor, kSet, kSetBlock, kMacro, kFilterBlock, kGeneration, kBreak, kContinue,
    // expressions
    kLiteral, kName, kSequence, kDict, kAttr, kItem, kSlice, kCall, kFilter, kTest, kNeg, kPos, kNot,
    kBinary, kCompare, kAnd, kOr, kCondExpr,
  };
  // The fields each type uses:
  //   kText: name (the text). kOutput: children[0].
  //   kIf: children (kBranch). kBranch: subject (condition, nullptr for else), body.
  //   kFor: names (targets), subject (iterable), filter (condition or nullptr), body, else_body.
  //   kSet: names (targets; "ns.attr" for an attribute), children[0] (value). kSetBlock: names, body.
  //   kMacro: name, names (parameters), defaults (nullptr for none), body.
  //   kFilterBlock: name, children and kwargs (arguments), body. kGeneration: body.
  //   kLiteral: literal. kName: name. kSequence: children. kDict: children as key, value pairs.
  //   kAttr: subject, name. kItem: subject, children[0]. kSlice: subject, children[0..2] (start, stop,
  //   step; nullptr when absent). kCall: subject (callee), children, kwargs.
  //   kFilter: subject, name, children, kwargs. kTest: subject, name, children, negated.
  //   kNeg, kPos, kNot: children[0]. kBinary: name (operator), children[0..1].
  //   kCompare: children (operands), names (operators). kAnd, kOr: children[0..1].
  //   kCondExpr: children[0] (value), children[1] (condition), children[2] (else, or nullptr).
  Type type;
  std::string name;
  Value literal;
  std::vector<NodePtr> children;
  std::vector<std::string> names;
  std::vector<NodePtr> defaults;
  std::vector<std::pair<std::string, NodePtr>> kwargs;
  NodePtr subject;
  NodePtr filter;
  std::vector<NodePtr> body;
  std::vector<NodePtr> else_body;
  bool negated = false;
  bool tuple = false;  // kSequence written with parentheses

  explicit Node(Type t) : type(t) {}
};

struct TemplateImpl {
  std::vector<NodePtr> body;
};

namespace {

// ---- Lexer ------------------------------------------------------------------------------------

struct Token {
  enum class Kind : uint8_t { kName, kString, kNumber, kOp };
  Kind kind;
  std::string text;
};

struct Piece {
  enum class Kind : uint8_t { kData, kVar, kBlock };
  Kind kind;
  std::string data;
  std::vector<Token> tokens;
};

[[noreturn]] void syntax(const std::string& what) { throw Error("template syntax error: " + what); }

uint32_t decode_at(const std::string& s, size_t i, size_t& len) { return utf8_decode(s, i, len); }

bool space_at(const std::string& s, size_t i, size_t& len) {
  const unsigned char c = static_cast<unsigned char>(s[i]);
  if (c < 0x80) {
    len = 1;
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v' || (c >= 0x1c && c <= 0x1f);
  }
  const uint32_t cp = decode_at(s, i, len);
  return uni::is_py_space(cp);
}

// The source as jinja2 sees it: CRLF and CR become LF, and one trailing newline is dropped.
std::string normalize(std::string_view in) {
  std::string out;
  out.reserve(in.size());
  for (size_t i = 0; i < in.size(); ++i) {
    if (in[i] == '\r') {
      out.push_back('\n');
      if (i + 1 < in.size() && in[i + 1] == '\n') ++i;
    } else {
      out.push_back(in[i]);
    }
  }
  if (!out.empty() && out.back() == '\n') out.pop_back();
  return out;
}

// Skips whitespace (Python's str.isspace) from pos; returns the first position after it.
size_t skip_space(const std::string& s, size_t pos) {
  while (pos < s.size()) {
    size_t len = 1;
    if (!space_at(s, pos, len)) break;
    pos += len;
  }
  return pos;
}

// jinja2's lstrip_blocks: whitespace between the start of a line and a block tag goes.
std::string lstrip_block(std::string data, bool line_start) {
  const size_t nl = data.rfind('\n');
  const size_t l = nl == std::string::npos ? 0 : nl + 1;
  if (l > 0 || line_start) {
    if (skip_space(data, l) == data.size()) data.resize(l);
  }
  return data;
}

bool ident_start(unsigned char c) { return (c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || c == '_' || c >= 0x80; }
bool ident_part(unsigned char c) { return ident_start(c) || (c >= '0' && c <= '9'); }

int hex_value(char c) {
  if (c >= '0' && c <= '9') return c - '0';
  if (c >= 'a' && c <= 'f') return c - 'a' + 10;
  if (c >= 'A' && c <= 'F') return c - 'A' + 10;
  return -1;
}

// A string literal starting at src[p] (a quote). Returns the decoded text; p moves past the quote.
std::string lex_string(const std::string& src, size_t& p) {
  const char quote = src[p++];
  std::string out;
  while (p < src.size() && src[p] != quote) {
    if (src[p] != '\\' || p + 1 >= src.size()) {
      out.push_back(src[p++]);
      continue;
    }
    const char e = src[p + 1];
    p += 2;
    switch (e) {
      case 'n': out.push_back('\n'); break;
      case 't': out.push_back('\t'); break;
      case 'r': out.push_back('\r'); break;
      case 'b': out.push_back('\b'); break;
      case 'f': out.push_back('\f'); break;
      case 'v': out.push_back('\v'); break;
      case 'a': out.push_back('\a'); break;
      case '0': out.push_back('\0'); break;
      case '\\': out.push_back('\\'); break;
      case '\'': out.push_back('\''); break;
      case '"': out.push_back('"'); break;
      case 'x':
      case 'u':
      case 'U': {
        const size_t digits = e == 'x' ? 2 : e == 'u' ? 4 : 8;
        uint32_t cp = 0;
        size_t k = 0;
        for (; k < digits && p + k < src.size() && hex_value(src[p + k]) >= 0; ++k) {
          cp = cp * 16 + static_cast<uint32_t>(hex_value(src[p + k]));
        }
        if (k != digits) syntax("bad escape in string");
        p += digits;
        utf8_append(out, cp);
        break;
      }
      default:
        out.push_back('\\');
        out.push_back(e);
    }
  }
  if (p >= src.size()) syntax("unterminated string literal");
  ++p;  // the closing quote
  return out;
}

// The tokens of one tag from src[p] (just after "{%" or "{{" and any modifier) to its closing
// delimiter. end_pos is the position after the delimiter.
std::vector<Token> lex_tag(const std::string& src, size_t p, bool is_var, size_t& end_pos, bool& strip_right,
                           bool& plus_right) {
  std::vector<Token> tokens;
  const char* close = is_var ? "}}" : "%}";
  const char* dash_close = is_var ? "-}}" : "-%}";
  const char* plus_close = is_var ? "+}}" : "+%}";
  int depth = 0;
  while (true) {
    p = skip_space(src, p);
    if (p >= src.size()) syntax("unexpected end of template inside a tag");
    if (depth == 0) {
      if (src.compare(p, 3, dash_close) == 0) {
        strip_right = true;
        end_pos = p + 3;
        return tokens;
      }
      if (src.compare(p, 3, plus_close) == 0) {
        plus_right = true;
        end_pos = p + 3;
        return tokens;
      }
      if (src.compare(p, 2, close) == 0) {
        end_pos = p + 2;
        return tokens;
      }
    }
    const unsigned char c = static_cast<unsigned char>(src[p]);
    if (ident_start(c)) {
      size_t q = p;
      while (q < src.size() && ident_part(static_cast<unsigned char>(src[q]))) {
        size_t len = 1;
        if (static_cast<unsigned char>(src[q]) >= 0x80 && space_at(src, q, len)) break;
        q += static_cast<unsigned char>(src[q]) >= 0x80 ? len : 1;
      }
      tokens.push_back(Token{Token::Kind::kName, src.substr(p, q - p)});
      p = q;
      continue;
    }
    if (c >= '0' && c <= '9') {
      size_t q = p;
      while (q < src.size() && ((src[q] >= '0' && src[q] <= '9') || src[q] == '_')) ++q;
      if (q + 1 < src.size() && src[q] == '.' && src[q + 1] >= '0' && src[q + 1] <= '9') {
        ++q;
        while (q < src.size() && ((src[q] >= '0' && src[q] <= '9') || src[q] == '_')) ++q;
      }
      if (q < src.size() && (src[q] == 'e' || src[q] == 'E')) {
        size_t r = q + 1;
        if (r < src.size() && (src[r] == '+' || src[r] == '-')) ++r;
        if (r < src.size() && src[r] >= '0' && src[r] <= '9') {
          q = r;
          while (q < src.size() && src[q] >= '0' && src[q] <= '9') ++q;
        }
      }
      std::string text;
      for (size_t k = p; k < q; ++k) {
        if (src[k] != '_') text.push_back(src[k]);
      }
      tokens.push_back(Token{Token::Kind::kNumber, text});
      p = q;
      continue;
    }
    if (c == '\'' || c == '"') {
      tokens.push_back(Token{Token::Kind::kString, lex_string(src, p)});
      continue;
    }
    static const char* two[] = {"**", "//", "==", "!=", "<=", ">="};
    bool matched = false;
    for (const char* op : two) {
      if (src.compare(p, 2, op) == 0) {
        tokens.push_back(Token{Token::Kind::kOp, op});
        p += 2;
        matched = true;
        break;
      }
    }
    if (matched) continue;
    if (std::strchr("+-*/%~()[]{},.:|=<>", static_cast<char>(c)) == nullptr || c == 0) {
      syntax(std::string("unexpected character '") + static_cast<char>(c) + "'");
    }
    if (c == '{') ++depth;
    if (c == '}' && depth > 0) --depth;
    tokens.push_back(Token{Token::Kind::kOp, std::string(1, static_cast<char>(c))});
    ++p;
  }
}

std::vector<Piece> lex_template(std::string_view raw) {
  const std::string src = normalize(raw);
  const size_t n = src.size();
  std::vector<Piece> pieces;
  size_t pos = 0;
  auto add_data = [&](std::string data) {
    if (data.empty()) return;
    Piece piece{Piece::Kind::kData, std::move(data), {}};
    pieces.push_back(std::move(piece));
  };
  while (true) {
    size_t start = std::string::npos;
    for (size_t i = pos; i + 1 < n; ++i) {
      if (src[i] == '{' && (src[i + 1] == '%' || src[i + 1] == '{' || src[i + 1] == '#')) {
        start = i;
        break;
      }
    }
    if (start == std::string::npos) {
      add_data(src.substr(pos));
      break;
    }
    std::string data = src.substr(pos, start - pos);
    const char kind = src[start + 1];
    size_t p = start + 2;
    bool strip_left = false;
    bool no_lstrip = false;
    if (p < n && src[p] == '-') {
      strip_left = true;
      ++p;
    } else if (p < n && src[p] == '+') {
      no_lstrip = true;
      ++p;
    }
    if (strip_left) {
      data.resize(py_last_nonspace_end(data));
    } else if (kind != '{' && !no_lstrip) {
      data = lstrip_block(std::move(data), pos == 0 || src[pos - 1] == '\n');
    }
    add_data(std::move(data));

    bool strip_right = false;
    bool plus_right = false;
    size_t end = 0;
    const bool block_like = kind != '{';
    if (kind == '#') {
      const size_t close = src.find("#}", p);
      if (close == std::string::npos) syntax("unexpected end of template inside a comment");
      if (close > p && src[close - 1] == '-') strip_right = true;
      if (close > p && src[close - 1] == '+') plus_right = true;
      end = close + 2;
      pos = end;
    } else {
      std::vector<Token> tokens = lex_tag(src, p, kind == '{', end, strip_right, plus_right);
      pos = end;
      if (kind == '%' && tokens.size() == 1 && tokens[0].kind == Token::Kind::kName && tokens[0].text == "raw") {
        // {% raw %} ... {% endraw %}: the text between is data, taken as it is.
        if (strip_right) {
          pos = skip_space(src, pos);
        } else if (!plus_right && pos < n && src[pos] == '\n') {
          ++pos;
        }
        size_t scan = pos;
        size_t endraw_begin = std::string::npos;
        size_t endraw_end = 0;
        bool endraw_strip = false;
        bool endraw_strip_after = false;
        while (true) {
          const size_t j = src.find("{%", scan);
          if (j == std::string::npos) syntax("raw block not closed");
          size_t k = skip_space(src, j + 2 + ((j + 2 < n && (src[j + 2] == '-' || src[j + 2] == '+')) ? 1 : 0));
          if (src.compare(k, 6, "endraw") == 0) {
            size_t after = k + 6;
            after = skip_space(src, after);
            if (after < n && src[after] == '-') {
              endraw_strip = true;
              after = skip_space(src, after + 1);
            }
            if (src.compare(after, 2, "%}") == 0) {
              endraw_begin = j;
              endraw_end = after + 2;
              break;
            }
            if (src.compare(after, 3, "-%}") == 0) {
              endraw_begin = j;
              endraw_end = after + 3;
              endraw_strip_after = true;
              break;
            }
          }
          scan = j + 2;
        }
        std::string raw_text = src.substr(pos, endraw_begin - pos);
        if (endraw_strip) raw_text.resize(py_last_nonspace_end(raw_text));
        add_data(std::move(raw_text));
        pos = endraw_end;
        if (endraw_strip_after) {
          pos = skip_space(src, pos);
        } else if (pos < n && src[pos] == '\n') {
          ++pos;  // trim_blocks after {% endraw %}
        }
        continue;
      }
      Piece piece{kind == '{' ? Piece::Kind::kVar : Piece::Kind::kBlock, {}, std::move(tokens)};
      pieces.push_back(std::move(piece));
    }
    if (strip_right) {
      pos = skip_space(src, pos);
    } else if (block_like && !plus_right && pos < n && src[pos] == '\n') {
      ++pos;  // trim_blocks
    }
    if (pos >= n) break;
  }
  return pieces;
}

// ---- Parser -----------------------------------------------------------------------------------

bool is_keyword(const std::string& s) {
  static const char* words[] = {"and", "or", "not", "in", "is", "if", "else"};
  for (const char* w : words) {
    if (s == w) return true;
  }
  return false;
}

NodePtr make(Node::Type type) { return std::make_unique<Node>(type); }

class ExprParser {
 public:
  ExprParser(const std::vector<Token>& tokens, size_t index = 0) : t_(tokens), i_(index) {}

  size_t index() const { return i_; }
  bool at_end() const { return i_ >= t_.size(); }
  const Token* peek(size_t k = 0) const { return i_ + k < t_.size() ? &t_[i_ + k] : nullptr; }

  bool peek_op(const char* op, size_t k = 0) const {
    const Token* tk = peek(k);
    return tk && tk->kind == Token::Kind::kOp && tk->text == op;
  }
  bool peek_name(const char* name, size_t k = 0) const {
    const Token* tk = peek(k);
    return tk && tk->kind == Token::Kind::kName && tk->text == name;
  }
  bool accept_op(const char* op) {
    if (!peek_op(op)) return false;
    ++i_;
    return true;
  }
  bool accept_name(const char* name) {
    if (!peek_name(name)) return false;
    ++i_;
    return true;
  }
  void expect_op(const char* op) {
    if (!accept_op(op)) syntax(std::string("expected '") + op + "'");
  }
  std::string expect_name() {
    const Token* tk = peek();
    if (!tk || tk->kind != Token::Kind::kName) syntax("expected a name");
    ++i_;
    return tk->text;
  }

  NodePtr parse_expression(bool condexpr = true) {
    if (!condexpr) return parse_or();
    NodePtr e = parse_or();
    while (accept_name("if")) {
      NodePtr cond = parse_or();
      NodePtr other;
      if (accept_name("else")) other = parse_expression(true);
      NodePtr node = make(Node::Type::kCondExpr);
      node->children.push_back(std::move(e));
      node->children.push_back(std::move(cond));
      node->children.push_back(std::move(other));
      e = std::move(node);
    }
    return e;
  }

 private:
  NodePtr binary(Node::Type type, std::string op, NodePtr left, NodePtr right) {
    NodePtr node = make(type);
    node->name = std::move(op);
    node->children.push_back(std::move(left));
    node->children.push_back(std::move(right));
    return node;
  }

  NodePtr parse_or() {
    NodePtr left = parse_and();
    while (accept_name("or")) left = binary(Node::Type::kOr, "or", std::move(left), parse_and());
    return left;
  }

  NodePtr parse_and() {
    NodePtr left = parse_not();
    while (accept_name("and")) left = binary(Node::Type::kAnd, "and", std::move(left), parse_not());
    return left;
  }

  NodePtr parse_not() {
    if (accept_name("not")) {
      NodePtr node = make(Node::Type::kNot);
      node->children.push_back(parse_not());
      return node;
    }
    return parse_compare();
  }

  NodePtr parse_compare() {
    NodePtr first = parse_math1();
    std::vector<std::string> ops;
    std::vector<NodePtr> operands;
    while (true) {
      std::string op;
      for (const char* candidate : {"==", "!=", "<=", ">=", "<", ">"}) {
        if (peek_op(candidate)) {
          op = candidate;
          break;
        }
      }
      if (!op.empty()) {
        ++i_;
      } else if (accept_name("in")) {
        op = "in";
      } else if (peek_name("not") && peek_name("in", 1)) {
        i_ += 2;
        op = "not in";
      } else {
        break;
      }
      ops.push_back(op);
      operands.push_back(parse_math1());
    }
    if (ops.empty()) return first;
    NodePtr node = make(Node::Type::kCompare);
    node->children.push_back(std::move(first));
    for (auto& operand : operands) node->children.push_back(std::move(operand));
    node->names = std::move(ops);
    return node;
  }

  NodePtr parse_math1() {
    NodePtr left = parse_concat();
    while (peek_op("+") || peek_op("-")) {
      std::string op = t_[i_++].text;
      left = binary(Node::Type::kBinary, op, std::move(left), parse_concat());
    }
    return left;
  }

  NodePtr parse_concat() {
    NodePtr left = parse_math2();
    while (accept_op("~")) left = binary(Node::Type::kBinary, "~", std::move(left), parse_math2());
    return left;
  }

  NodePtr parse_math2() {
    NodePtr left = parse_pow();
    while (peek_op("*") || peek_op("/") || peek_op("//") || peek_op("%")) {
      std::string op = t_[i_++].text;
      left = binary(Node::Type::kBinary, op, std::move(left), parse_pow());
    }
    return left;
  }

  NodePtr parse_pow() {
    NodePtr left = parse_unary(true);
    while (accept_op("**")) left = binary(Node::Type::kBinary, "**", std::move(left), parse_unary(true));
    return left;
  }

 public:
  NodePtr parse_unary(bool with_filter) {
    NodePtr node;
    if (accept_op("-")) {
      node = make(Node::Type::kNeg);
      node->children.push_back(parse_unary(false));
    } else if (accept_op("+")) {
      node = make(Node::Type::kPos);
      node->children.push_back(parse_unary(false));
    } else {
      node = parse_primary();
    }
    node = parse_postfix(std::move(node));
    if (with_filter) node = parse_filter_expr(std::move(node));
    return node;
  }

 private:
  NodePtr parse_primary() {
    const Token* tk = peek();
    if (!tk) syntax("unexpected end of expression");
    if (tk->kind == Token::Kind::kName) {
      ++i_;
      const std::string& w = tk->text;
      NodePtr node = make(Node::Type::kLiteral);
      if (w == "true" || w == "True") {
        node->literal = Value::boolean(true);
      } else if (w == "false" || w == "False") {
        node->literal = Value::boolean(false);
      } else if (w == "none" || w == "None") {
        node->literal = Value::none();
      } else {
        if (is_keyword(w)) syntax("unexpected keyword '" + w + "'");
        node = make(Node::Type::kName);
        node->name = w;
      }
      return node;
    }
    if (tk->kind == Token::Kind::kString) {
      std::string text;
      while (peek() && peek()->kind == Token::Kind::kString) text += t_[i_++].text;
      NodePtr node = make(Node::Type::kLiteral);
      node->literal = Value::string(std::move(text));
      return node;
    }
    if (tk->kind == Token::Kind::kNumber) {
      ++i_;
      NodePtr node = make(Node::Type::kLiteral);
      const std::string& text = tk->text;
      if (text.find_first_of(".eE") == std::string::npos) {
        errno = 0;
        const long long v = std::strtoll(text.c_str(), nullptr, 10);
        node->literal = errno == ERANGE ? Value::real(std::strtod(text.c_str(), nullptr)) : Value::integer(v);
      } else {
        node->literal = Value::real(std::strtod(text.c_str(), nullptr));
      }
      return node;
    }
    if (accept_op("(")) {
      if (accept_op(")")) return make_sequence({}, true);
      NodePtr first = parse_expression(true);
      if (!peek_op(",")) {
        expect_op(")");
        return first;
      }
      std::vector<NodePtr> items;
      items.push_back(std::move(first));
      while (accept_op(",")) {
        if (peek_op(")")) break;
        items.push_back(parse_expression(true));
      }
      expect_op(")");
      return make_sequence(std::move(items), true);
    }
    if (accept_op("[")) {
      std::vector<NodePtr> items;
      while (!peek_op("]")) {
        items.push_back(parse_expression(true));
        if (!accept_op(",")) break;
      }
      expect_op("]");
      return make_sequence(std::move(items));
    }
    if (accept_op("{")) {
      NodePtr node = make(Node::Type::kDict);
      while (!peek_op("}")) {
        node->children.push_back(parse_expression(true));
        expect_op(":");
        node->children.push_back(parse_expression(true));
        if (!accept_op(",")) break;
      }
      expect_op("}");
      return node;
    }
    syntax("unexpected token '" + tk->text + "'");
  }

  static NodePtr make_sequence(std::vector<NodePtr> items, bool tuple = false) {
    NodePtr node = make(Node::Type::kSequence);
    node->children = std::move(items);
    node->tuple = tuple;
    return node;
  }

  NodePtr parse_postfix(NodePtr node) {
    while (true) {
      if (accept_op(".")) {
        NodePtr attr = make(Node::Type::kAttr);
        attr->subject = std::move(node);
        attr->name = expect_name();
        node = std::move(attr);
      } else if (accept_op("[")) {
        NodePtr start_expr, stop_expr, step_expr;
        bool is_slice = false;
        if (!peek_op(":")) start_expr = parse_expression(true);
        if (accept_op(":")) {
          is_slice = true;
          if (!peek_op(":") && !peek_op("]")) stop_expr = parse_expression(true);
          if (accept_op(":") && !peek_op("]")) step_expr = parse_expression(true);
        }
        expect_op("]");
        NodePtr sub = make(is_slice ? Node::Type::kSlice : Node::Type::kItem);
        sub->subject = std::move(node);
        if (is_slice) {
          sub->children.push_back(std::move(start_expr));
          sub->children.push_back(std::move(stop_expr));
          sub->children.push_back(std::move(step_expr));
        } else {
          sub->children.push_back(std::move(start_expr));
        }
        node = std::move(sub);
      } else if (accept_op("(")) {
        NodePtr call = make(Node::Type::kCall);
        call->subject = std::move(node);
        parse_arguments(call->children, call->kwargs);
        node = std::move(call);
      } else {
        break;
      }
    }
    return node;
  }

 public:
  // The arguments of a call, after its "(": positional ones and name=value ones, up to ")".
  void parse_arguments(std::vector<NodePtr>& positional, std::vector<std::pair<std::string, NodePtr>>& keyword) {
    while (!peek_op(")")) {
      if (peek() && peek()->kind == Token::Kind::kName && peek_op("=", 1)) {
        std::string key = expect_name();
        expect_op("=");
        keyword.emplace_back(std::move(key), parse_expression(true));
      } else {
        positional.push_back(parse_expression(true));
      }
      if (!accept_op(",")) break;
    }
    expect_op(")");
  }

 private:
  NodePtr parse_filter_expr(NodePtr node) {
    while (true) {
      if (accept_op("|")) {
        NodePtr f = make(Node::Type::kFilter);
        f->subject = std::move(node);
        f->name = expect_name();
        if (accept_op("(")) parse_arguments(f->children, f->kwargs);
        node = std::move(f);
      } else if (accept_name("is")) {
        NodePtr test = make(Node::Type::kTest);
        test->negated = accept_name("not");
        test->subject = std::move(node);
        test->name = expect_name();
        if (accept_op("(")) {
          std::vector<std::pair<std::string, NodePtr>> kw;
          parse_arguments(test->children, kw);
          if (!kw.empty()) syntax("keyword arguments to a test are not supported");
        } else if (starts_test_argument()) {
          test->children.push_back(parse_unary(false));
        }
        node = std::move(test);
      } else {
        break;
      }
    }
    return node;
  }

  bool starts_test_argument() const {
    const Token* tk = peek();
    if (!tk) return false;
    if (tk->kind == Token::Kind::kName) return !is_keyword(tk->text);
    if (tk->kind == Token::Kind::kNumber || tk->kind == Token::Kind::kString) return true;
    return tk->kind == Token::Kind::kOp && (tk->text == "-" || tk->text == "(" || tk->text == "[" ||
                                             tk->text == "{");
  }

  const std::vector<Token>& t_;
  size_t i_;
};

class Parser {
 public:
  explicit Parser(std::vector<Piece> pieces) : pieces_(std::move(pieces)) {}

  std::vector<NodePtr> parse() {
    size_t idx = 0;
    std::string found;
    std::vector<Token> found_tokens;
    std::vector<NodePtr> body = parse_body(idx, {}, found, found_tokens);
    return body;
  }

 private:
  // Statements up to a terminator tag (named in `terminators`, reported in found/found_tokens) or,
  // with no terminators, to the end.
  std::vector<NodePtr> parse_body(size_t& idx, const std::vector<std::string>& terminators, std::string& found,
                                  std::vector<Token>& found_tokens) {
    std::vector<NodePtr> body;
    while (idx < pieces_.size()) {
      Piece& piece = pieces_[idx];
      if (piece.kind == Piece::Kind::kData) {
        NodePtr text = make(Node::Type::kText);
        text->name = piece.data;
        body.push_back(std::move(text));
        ++idx;
        continue;
      }
      if (piece.kind == Piece::Kind::kVar) {
        ExprParser ep(piece.tokens);
        NodePtr out = make(Node::Type::kOutput);
        out->children.push_back(ep.parse_expression(true));
        if (!ep.at_end()) syntax("unexpected token in expression");
        body.push_back(std::move(out));
        ++idx;
        continue;
      }
      const std::vector<Token>& tokens = piece.tokens;
      if (tokens.empty() || tokens[0].kind != Token::Kind::kName) syntax("a block tag needs a keyword");
      const std::string keyword = tokens[0].text;
      for (const auto& term : terminators) {
        if (keyword == term) {
          found = keyword;
          found_tokens = tokens;
          ++idx;
          return body;
        }
      }
      ++idx;
      body.push_back(parse_statement(keyword, tokens, idx));
    }
    if (!terminators.empty()) syntax("unexpected end of template, expected '" + terminators.back() + "'");
    return body;
  }

  std::vector<NodePtr> parse_block(size_t& idx, const std::string& end) {
    std::string found;
    std::vector<Token> found_tokens;
    std::vector<NodePtr> body = parse_body(idx, {end}, found, found_tokens);
    return body;
  }

  NodePtr parse_statement(const std::string& keyword, const std::vector<Token>& tokens, size_t& idx) {
    if (keyword == "if") {
      NodePtr node = make(Node::Type::kIf);
      ExprParser head(tokens, 1);
      NodePtr cond = head.parse_expression(true);
      if (!head.at_end()) syntax("unexpected token after 'if'");
      std::string found;
      std::vector<Token> found_tokens;
      NodePtr branch = make(Node::Type::kBranch);
      branch->subject = std::move(cond);
      branch->body = parse_body(idx, {"elif", "else", "endif"}, found, found_tokens);
      node->children.push_back(std::move(branch));
      while (found != "endif") {
        NodePtr next = make(Node::Type::kBranch);
        if (found == "elif") {
          ExprParser ep(found_tokens, 1);
          next->subject = ep.parse_expression(true);
          if (!ep.at_end()) syntax("unexpected token after 'elif'");
          next->body = parse_body(idx, {"elif", "else", "endif"}, found, found_tokens);
        } else {
          next->body = parse_body(idx, {"endif"}, found, found_tokens);
          found = "endif";
        }
        node->children.push_back(std::move(next));
      }
      return node;
    }
    if (keyword == "for") {
      NodePtr node = make(Node::Type::kFor);
      ExprParser head(tokens, 1);
      do {
        node->names.push_back(head.expect_name());
      } while (head.accept_op(","));
      if (!head.accept_name("in")) syntax("expected 'in' in a for loop");
      node->subject = head.parse_expression(false);
      if (head.accept_name("if")) node->filter = head.parse_expression(false);
      if (!head.at_end()) syntax("unsupported syntax in a for loop");
      std::string found;
      std::vector<Token> found_tokens;
      node->body = parse_body(idx, {"else", "endfor"}, found, found_tokens);
      if (found == "else") node->else_body = parse_body(idx, {"endfor"}, found, found_tokens);
      return node;
    }
    if (keyword == "set") {
      ExprParser head(tokens, 1);
      std::vector<std::string> targets;
      do {
        std::string target = head.expect_name();
        if (head.accept_op(".")) target += "." + head.expect_name();
        targets.push_back(std::move(target));
      } while (head.accept_op(","));
      if (head.accept_op("=")) {
        NodePtr node = make(Node::Type::kSet);
        node->names = std::move(targets);
        node->children.push_back(head.parse_expression(true));
        if (!head.at_end()) syntax("unexpected token after 'set'");
        return node;
      }
      if (!head.at_end()) syntax("unsupported syntax in 'set'");
      NodePtr node = make(Node::Type::kSetBlock);
      node->names = std::move(targets);
      node->body = parse_block(idx, "endset");
      return node;
    }
    if (keyword == "macro") {
      NodePtr node = make(Node::Type::kMacro);
      ExprParser head(tokens, 1);
      node->name = head.expect_name();
      head.expect_op("(");
      while (!head.peek_op(")")) {
        node->names.push_back(head.expect_name());
        NodePtr dflt;
        if (head.accept_op("=")) dflt = head.parse_expression(true);
        node->defaults.push_back(std::move(dflt));
        if (!head.accept_op(",")) break;
      }
      head.expect_op(")");
      if (!head.at_end()) syntax("unexpected token in a macro definition");
      node->body = parse_block(idx, "endmacro");
      return node;
    }
    if (keyword == "filter") {
      NodePtr node = make(Node::Type::kFilterBlock);
      ExprParser head(tokens, 1);
      node->name = head.expect_name();
      if (head.accept_op("(")) {
        std::vector<std::pair<std::string, NodePtr>> kw;
        head.parse_arguments(node->children, kw);
        node->kwargs = std::move(kw);
      }
      node->body = parse_block(idx, "endfilter");
      return node;
    }
    if (keyword == "generation") {
      NodePtr node = make(Node::Type::kGeneration);
      node->body = parse_block(idx, "endgeneration");
      return node;
    }
    if (keyword == "break") return make(Node::Type::kBreak);
    if (keyword == "continue") return make(Node::Type::kContinue);
    syntax("unsupported tag '" + keyword + "'");
  }

  std::vector<Piece> pieces_;
};

// ---- Evaluator --------------------------------------------------------------------------------

struct Frame {
  std::unordered_map<std::string, Value> vars;
  std::shared_ptr<Frame> parent;
};

const Value* lookup(const std::shared_ptr<Frame>& frame, const std::string& name) {
  for (Frame* f = frame.get(); f != nullptr; f = f->parent.get()) {
    auto it = f->vars.find(name);
    if (it != f->vars.end()) return &it->second;
  }
  return nullptr;
}

// A global function (range, namespace, raise_exception, ...) as a value.
class GlobalFunction : public Object {
 public:
  explicit GlobalFunction(std::string name) : name_(std::move(name)) {}
  bool callable() const override { return true; }
  Value call(const CallArgs& args) const override { return call_global(name_, args); }
  const char* type_name() const override { return "function"; }

 private:
  std::string name_;
};

// The loop object of one iteration of a for loop.
class LoopObject : public Object {
 public:
  LoopObject(std::shared_ptr<const std::vector<Value>> items, size_t index0)
      : items_(std::move(items)), index0_(index0) {}

  Value getattr(const std::string& name) const override {
    const int64_t length = static_cast<int64_t>(items_->size());
    const int64_t i = static_cast<int64_t>(index0_);
    if (name == "index") return Value::integer(i + 1);
    if (name == "index0") return Value::integer(i);
    if (name == "revindex") return Value::integer(length - i);
    if (name == "revindex0") return Value::integer(length - i - 1);
    if (name == "first") return Value::boolean(i == 0);
    if (name == "last") return Value::boolean(i == length - 1);
    if (name == "length") return Value::integer(length);
    if (name == "depth") return Value::integer(1);
    if (name == "depth0") return Value::integer(0);
    if (name == "previtem") return i > 0 ? (*items_)[index0_ - 1] : Value();
    if (name == "nextitem") return i + 1 < length ? (*items_)[index0_ + 1] : Value();
    if (name == "cycle") return Value::object(std::make_shared<Cycler>(index0_));
    return Value();
  }
  const char* type_name() const override { return "LoopContext"; }

 private:
  class Cycler : public Object {
   public:
    explicit Cycler(size_t index0) : index0_(index0) {}
    bool callable() const override { return true; }
    Value call(const CallArgs& args) const override {
      if (args.positional.empty()) throw Error("no items for cycling");
      return args.positional[index0_ % args.positional.size()];
    }
    const char* type_name() const override { return "function"; }

   private:
    size_t index0_;
  };

  std::shared_ptr<const std::vector<Value>> items_;
  size_t index0_;
};

class Interp;

// A macro: the definition's body, closed over the frame it was defined in.
class Macro : public Object {
 public:
  Macro(std::shared_ptr<const TemplateImpl> impl, const Node* def, std::shared_ptr<Frame> scope)
      : impl_(std::move(impl)), def_(def), scope_(std::move(scope)) {}
  bool callable() const override { return true; }
  Value call(const CallArgs& args) const override;
  const char* type_name() const override { return "macro"; }

 private:
  std::shared_ptr<const TemplateImpl> impl_;
  const Node* def_;
  std::shared_ptr<Frame> scope_;
};

class Interp {
 public:
  explicit Interp(std::shared_ptr<const TemplateImpl> impl) : impl_(std::move(impl)) {}

  std::string out;

  void exec_body(const std::vector<NodePtr>& body, const std::shared_ptr<Frame>& frame) {
    for (const NodePtr& node : body) {
      exec(*node, frame);
      if (flow_ != Flow::kNormal) return;
    }
  }

  Value eval(const Node& node, const std::shared_ptr<Frame>& frame) {
    switch (node.type) {
      case Node::Type::kLiteral:
        return node.literal;
      case Node::Type::kName: {
        if (const Value* v = lookup(frame, node.name)) return *v;
        if (is_global(node.name)) return Value::object(std::make_shared<GlobalFunction>(node.name));
        return Value();
      }
      case Node::Type::kSequence: {
        std::vector<Value> items;
        items.reserve(node.children.size());
        for (const NodePtr& c : node.children) items.push_back(eval(*c, frame));
        return node.tuple ? Value::tuple(std::move(items)) : Value::list(std::move(items));
      }
      case Node::Type::kDict: {
        std::vector<std::pair<Value, Value>> items;
        for (size_t i = 0; i + 1 < node.children.size(); i += 2) {
          Value key = eval(*node.children[i], frame);
          Value value = eval(*node.children[i + 1], frame);
          bool replaced = false;
          for (auto& kv : items) {
            if (py_equal(kv.first, key)) {
              kv.second = value;
              replaced = true;
              break;
            }
          }
          if (!replaced) items.emplace_back(std::move(key), std::move(value));
        }
        return Value::dict(std::move(items));
      }
      case Node::Type::kAttr:
        return py_attr(eval(*node.subject, frame), node.name);
      case Node::Type::kItem:
        return py_item(eval(*node.subject, frame), eval(*node.children[0], frame));
      case Node::Type::kSlice: {
        Value obj = eval(*node.subject, frame);
        Value start, stop, step;
        if (node.children[0]) start = eval(*node.children[0], frame);
        if (node.children[1]) stop = eval(*node.children[1], frame);
        if (node.children[2]) step = eval(*node.children[2], frame);
        return py_slice(obj, node.children[0] ? &start : nullptr, node.children[1] ? &stop : nullptr,
                        node.children[2] ? &step : nullptr);
      }
      case Node::Type::kCall: {
        Value callee = eval(*node.subject, frame);
        CallArgs args = eval_args(node.children, node.kwargs, frame);
        if (!callee.is_object() || !callee.as_object()->callable()) {
          throw Error("'" + callee.str() + "' is not callable");
        }
        return callee.as_object()->call(args);
      }
      case Node::Type::kFilter: {
        Value subject = eval(*node.subject, frame);
        CallArgs args = eval_args(node.children, node.kwargs, frame);
        return apply_filter(node.name, subject, args);
      }
      case Node::Type::kTest: {
        Value subject = eval(*node.subject, frame);
        CallArgs args = eval_args(node.children, {}, frame);
        const bool result = apply_test(node.name, subject, args);
        return Value::boolean(node.negated ? !result : result);
      }
      case Node::Type::kNeg: {
        Value v = eval(*node.children[0], frame);
        if (v.is_int() || v.is_bool()) return Value::integer(-v.as_int());
        if (v.is_float()) return Value::real(-v.as_float());
        throw Error("bad operand type for unary -");
      }
      case Node::Type::kPos: {
        Value v = eval(*node.children[0], frame);
        if (v.is_int() || v.is_bool()) return Value::integer(v.as_int());
        if (v.is_float()) return v;
        throw Error("bad operand type for unary +");
      }
      case Node::Type::kNot:
        return Value::boolean(!eval(*node.children[0], frame).truthy());
      case Node::Type::kBinary: {
        Value a = eval(*node.children[0], frame);
        Value b = eval(*node.children[1], frame);
        const std::string& op = node.name;
        if (op == "~") return py_arith('~', a, b);
        if (op == "//") return py_arith('d', a, b);
        if (op == "**") return py_arith('p', a, b);
        return py_arith(op[0], a, b);
      }
      case Node::Type::kCompare: {
        Value left = eval(*node.children[0], frame);
        for (size_t i = 0; i < node.names.size(); ++i) {
          Value right = eval(*node.children[i + 1], frame);
          if (!compare(node.names[i], left, right)) return Value::boolean(false);
          left = std::move(right);
        }
        return Value::boolean(true);
      }
      case Node::Type::kAnd: {
        Value left = eval(*node.children[0], frame);
        if (!left.truthy()) return left;
        return eval(*node.children[1], frame);
      }
      case Node::Type::kOr: {
        Value left = eval(*node.children[0], frame);
        if (left.truthy()) return left;
        return eval(*node.children[1], frame);
      }
      case Node::Type::kCondExpr: {
        if (eval(*node.children[1], frame).truthy()) return eval(*node.children[0], frame);
        if (node.children[2]) return eval(*node.children[2], frame);
        return Value();
      }
      default:
        throw Error("internal: statement used as an expression");
    }
  }

 private:
  enum class Flow : uint8_t { kNormal, kBreak, kContinue };

  static bool compare(const std::string& op, const Value& a, const Value& b) {
    if (op == "==") return py_equal(a, b);
    if (op == "!=") return !py_equal(a, b);
    if (op == "<") return py_less(a, b);
    if (op == ">") return py_less(b, a);
    if (op == "<=") return !py_less(b, a);
    if (op == ">=") return !py_less(a, b);
    if (op == "in") return py_contains(b, a);
    if (op == "not in") return !py_contains(b, a);
    throw Error("internal: unknown comparison " + op);
  }

  CallArgs eval_args(const std::vector<NodePtr>& positional,
                     const std::vector<std::pair<std::string, NodePtr>>& keyword,
                     const std::shared_ptr<Frame>& frame) {
    CallArgs args;
    for (const NodePtr& p : positional) args.positional.push_back(eval(*p, frame));
    for (const auto& kv : keyword) args.keyword.emplace_back(kv.first, eval(*kv.second, frame));
    return args;
  }

  void assign(const std::string& target, Value value, const std::shared_ptr<Frame>& frame) {
    const size_t dot = target.find('.');
    if (dot == std::string::npos) {
      frame->vars[target] = std::move(value);
      return;
    }
    const Value* ns = lookup(frame, target.substr(0, dot));
    if (ns == nullptr || !ns->is_object()) throw Error("cannot assign an attribute of '" + target.substr(0, dot) + "'");
    if (!ns->as_object()->setattr(target.substr(dot + 1), value)) throw Error("attribute assignment is not supported");
  }

  void assign_targets(const std::vector<std::string>& names, Value value, const std::shared_ptr<Frame>& frame) {
    if (names.size() == 1) {
      assign(names[0], std::move(value), frame);
      return;
    }
    if (!value.is_list() || value.as_list().size() != names.size()) throw Error("cannot unpack the value");
    for (size_t i = 0; i < names.size(); ++i) assign(names[i], value.as_list()[i], frame);
  }

  std::vector<Value> iterate(const Value& v) {
    switch (v.kind()) {
      case Value::Kind::kList:
        return v.as_list();
      case Value::Kind::kDict: {
        std::vector<Value> keys;
        for (const auto& kv : v.as_dict()) keys.push_back(kv.first);
        return keys;
      }
      case Value::Kind::kString: {
        std::vector<Value> chars;
        const std::string& s = v.as_string();
        size_t i = 0;
        while (i < s.size()) {
          size_t len = 1;
          utf8_decode(s, i, len);
          chars.push_back(Value::string(s.substr(i, len)));
          i += len;
        }
        return chars;
      }
      case Value::Kind::kUndefined:
        return {};
      default:
        throw Error("'" + v.str() + "' is not iterable");
    }
  }

  void exec_for(const Node& node, const std::shared_ptr<Frame>& frame) {
    std::vector<Value> items = iterate(eval(*node.subject, frame));
    if (node.filter) {
      std::vector<Value> kept;
      for (const Value& item : items) {
        auto probe = std::make_shared<Frame>();
        probe->parent = frame;
        assign_targets(node.names, item, probe);
        if (eval(*node.filter, probe).truthy()) kept.push_back(item);
      }
      items = std::move(kept);
    }
    if (items.empty()) {
      exec_body(node.else_body, frame);
      return;
    }
    auto shared_items = std::make_shared<const std::vector<Value>>(std::move(items));
    for (size_t i = 0; i < shared_items->size(); ++i) {
      auto iter = std::make_shared<Frame>();
      iter->parent = frame;
      assign_targets(node.names, (*shared_items)[i], iter);
      iter->vars["loop"] = Value::object(std::make_shared<LoopObject>(shared_items, i));
      exec_body(node.body, iter);
      if (flow_ == Flow::kBreak) {
        flow_ = Flow::kNormal;
        break;
      }
      if (flow_ == Flow::kContinue) flow_ = Flow::kNormal;
    }
  }

  // Renders a body into a string, leaving the output as it was.
  std::string capture(const std::vector<NodePtr>& body, const std::shared_ptr<Frame>& frame) {
    const size_t mark = out.size();
    exec_body(body, frame);
    std::string text = out.substr(mark);
    out.resize(mark);
    return text;
  }

  void exec(const Node& node, const std::shared_ptr<Frame>& frame) {
    switch (node.type) {
      case Node::Type::kText:
        out += node.name;
        return;
      case Node::Type::kOutput:
        out += eval(*node.children[0], frame).str();
        return;
      case Node::Type::kIf:
        for (const NodePtr& branch : node.children) {
          if (!branch->subject || eval(*branch->subject, frame).truthy()) {
            exec_body(branch->body, frame);
            return;
          }
        }
        return;
      case Node::Type::kFor:
        exec_for(node, frame);
        return;
      case Node::Type::kSet:
        assign_targets(node.names, eval(*node.children[0], frame), frame);
        return;
      case Node::Type::kSetBlock:
        assign_targets(node.names, Value::string(capture(node.body, frame)), frame);
        return;
      case Node::Type::kMacro:
        frame->vars[node.name] = Value::object(std::make_shared<Macro>(impl_, &node, frame));
        return;
      case Node::Type::kFilterBlock: {
        std::string text = capture(node.body, frame);
        CallArgs args = eval_args(node.children, node.kwargs, frame);
        out += apply_filter(node.name, Value::string(std::move(text)), args).str();
        return;
      }
      case Node::Type::kGeneration:
        exec_body(node.body, frame);
        return;
      case Node::Type::kBreak:
        flow_ = Flow::kBreak;
        return;
      case Node::Type::kContinue:
        flow_ = Flow::kContinue;
        return;
      default:
        throw Error("internal: expression used as a statement");
    }
  }

  std::shared_ptr<const TemplateImpl> impl_;
  Flow flow_ = Flow::kNormal;
};

Value Macro::call(const CallArgs& args) const {
  auto frame = std::make_shared<Frame>();
  frame->parent = scope_;
  Interp interp(impl_);
  for (size_t i = 0; i < def_->names.size(); ++i) {
    const std::string& param = def_->names[i];
    if (const Value* given = args.get(i, param.c_str())) {
      frame->vars[param] = *given;
    } else if (def_->defaults[i]) {
      frame->vars[param] = interp.eval(*def_->defaults[i], frame);
    } else {
      frame->vars[param] = Value();
    }
  }
  interp.exec_body(def_->body, frame);
  return Value::string(std::move(interp.out));
}

}  // namespace

Template::Template(std::string_view source) {
  auto impl = std::make_shared<TemplateImpl>();
  impl->body = Parser(lex_template(source)).parse();
  impl_ = std::move(impl);
}

std::string Template::render(const std::vector<std::pair<std::string, Value>>& variables) const {
  Interp interp(impl_);
  auto frame = std::make_shared<Frame>();
  for (const auto& kv : variables) frame->vars[kv.first] = kv.second;
  interp.exec_body(impl_->body, frame);
  return std::move(interp.out);
}

}  // namespace lx::text::jinja
