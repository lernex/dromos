// OutputParser: the Metis reasoning and tool-call format, incrementally.
//
// The behaviour is that of the reference parser (vLLM's metis parser, DelegatingParser over the
// metis ParserEngine), whole-text, which the golden tests compare against:
//
//   1. Reasoning (thinking mode only). Text up to the first "</think>" (its newline included) is
//      reasoning; "<think>" markers are dropped. A "<|tool_call|>" in the reasoning ends it, and
//      the marker starts the content. The reasoning is right-trimmed.
//   2. Content and tool calls. From the content start: text is content, "<|tool_call|>" or
//      "<function=" opens a call, "<function=NAME>" names it, "<parameter=K>V</parameter>" blocks
//      give its arguments, "</function>" closes it, and anything between calls is dropped. A call
//      is kept when its name (after strip) is non-empty. Content is kept as is, except that its
//      trailing whitespace is dropped when a call is kept.
//   3. Special tokens that are not markup (<|fim_suffix|> and friends) are removed everywhere.
//
// Streaming emits the same text the whole-text parse gives, in pieces: trailing whitespace is
// held until it is known whether it stays, and a call is emitted when its closing tag arrives (or
// at finish() when the output ends inside it).
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#include "text/json.h"
#include "text/py.h"
#include "text/tokenizer.h"
#include "text/tool_args.h"
#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text {
namespace {

enum class Kind : uint8_t {
  kThinkStart,
  kThinkEnd,
  kToolStart,
  kFuncPrefix,
  kFuncEnd,
  kParamStart,
  kParamEnd,
  kCloseAngle,
  kDrop,
};

struct Literal {
  std::string text;
  Kind kind;
};

// Longest match wins at each position. A literal that is only a prefix of the unlexed text waits
// for more text (unless the stream has ended), so chunking never changes the result.
const std::vector<Literal>& literals() {
  static const std::vector<Literal> table = {
      {"<think>\n", Kind::kThinkStart},
      {"<think>", Kind::kThinkStart},
      {"</think>\n", Kind::kThinkEnd},
      {"</think>", Kind::kThinkEnd},
      {"<|tool_call|>", Kind::kToolStart},
      {"<function=", Kind::kFuncPrefix},
      {"</function>", Kind::kFuncEnd},
      {"<parameter=", Kind::kParamStart},
      {"</parameter>", Kind::kParamEnd},
      {">", Kind::kCloseAngle},
      // The Metis special tokens that carry no markup; the reference parser drops their text.
      {"<|fim_suffix|>", Kind::kDrop},
      {"<|padding|>", Kind::kDrop},
      {"<|endoftext|>", Kind::kDrop},
      {"<|fim_prefix|>", Kind::kDrop},
      {"<|fim_middle|>", Kind::kDrop},
      {"<|tool_result|>", Kind::kDrop},
  };
  return table;
}

const std::unordered_set<std::string>& proper_prefixes() {
  static const std::unordered_set<std::string> prefixes = [] {
    std::unordered_set<std::string> set;
    for (const Literal& lit : literals()) {
      for (size_t n = 1; n < lit.text.size(); ++n) set.insert(lit.text.substr(0, n));
    }
    return set;
  }();
  return prefixes;
}

size_t longest_literal() {
  static const size_t longest = [] {
    size_t n = 0;
    for (const Literal& lit : literals()) n = lit.text.size() > n ? lit.text.size() : n;
    return n;
  }();
  return longest;
}

}  // namespace

struct OutputParserState {
  struct Call {
    std::string name;  // the text between "<function=" and ">"
    std::string body;  // everything after ">" up to "</function>", markers included
  };

  // The stream is in the reasoning block (thinking mode, before its end).
  bool reasoning_phase = false;

  enum class Tool : uint8_t { kContent, kPreamble, kName, kArgs, kBetween };
  Tool tool = Tool::kContent;

  JValue tools;  // the request's tools array, or null
  std::vector<Call> calls;
  size_t finalized = 0;  // calls[0, finalized) have been emitted or dropped
  bool accepted = false;  // a call with a name has been emitted

  std::string pending;          // text not lexed yet
  std::string reasoning_hold;   // trailing whitespace of the reasoning, not yet known to be inside it
  std::string content_hold;     // content not yet known to be final (see emit_content)
  bool content_started = false;
  // Content that starts with whitespace is held whole until a call is kept (which strips it) or the
  // output ends (which keeps it). Other content streams, minus its trailing whitespace.
  bool content_holding = false;
  bool content_nonspace = false;

  OutputParser::Delta out;
};

namespace {

void emit_reasoning(OutputParserState& st, std::string_view text) {
  std::string combined = st.reasoning_hold + std::string(text);
  const size_t keep = py_last_nonspace_end(combined);
  st.out.reasoning.append(combined, 0, keep);
  st.reasoning_hold = combined.substr(keep);
}

bool first_is_space(std::string_view s) {
  size_t len = 1;
  return !s.empty() && uni::is_py_space(utf8_decode(s, 0, len));
}

void emit_content(OutputParserState& st, std::string_view text) {
  if (!st.content_started && !text.empty()) {
    st.content_started = true;
    st.content_holding = first_is_space(text);
  }
  if (py_last_nonspace_end(text) > 0) st.content_nonspace = true;
  if (st.content_holding) {
    st.content_hold.append(text);
    return;
  }
  std::string combined = st.content_hold + std::string(text);
  const size_t keep = py_last_nonspace_end(combined);
  st.out.content.append(combined, 0, keep);
  st.content_hold = combined.substr(keep);
}

// The reasoning ends: whatever whitespace it ended with is not part of it.
void end_reasoning(OutputParserState& st) { st.reasoning_hold.clear(); }

// A call's tool-call emission, as the reference's whole-text build (_build_extracted_result) does it.
void finalize(OutputParserState& st, const OutputParserState::Call& call) {
  const std::string name_text = call.name.empty() ? name_from_body(call.body) : call.name;
  if (name_text.empty() && call.body.empty()) return;
  std::string name(py_strip(name_text));
  std::string args;
  const bool body_has_text = !py_strip(call.body).empty();
  if (name.empty() && body_has_text) {
    std::string json_name, json_args;
    if (name_and_args_from_json(call.body, json_name, json_args)) {
      name = json_name;
      args = json_args;
    } else {
      name.clear();
    }
  } else if (body_has_text) {
    args = convert_parameters(call.body);
  } else {
    args = "{}";
  }
  if (name.empty()) return;  // no name, no call
  args = fix_arg_types(args, name, st.tools);
  st.out.tool_calls.push_back(ToolCall{"", name, args});
  st.accepted = true;
  // A kept call strips the content on both sides: what is held is emitted stripped, and the
  // trailing whitespace held for the streamed part is dropped.
  if (st.content_holding) {
    st.out.content.append(py_strip(st.content_hold));
    st.content_holding = false;
  }
  st.content_hold.clear();
}

void finalize_through(OutputParserState& st, size_t count) {
  while (st.finalized < count) {
    finalize(st, st.calls[st.finalized]);
    ++st.finalized;
  }
}

void on_text(OutputParserState& st, std::string_view text) {
  if (st.reasoning_phase) {
    emit_reasoning(st, text);
    return;
  }
  switch (st.tool) {
    case OutputParserState::Tool::kContent:
      emit_content(st, text);
      return;
    case OutputParserState::Tool::kName:
      st.calls.back().name.append(text);
      return;
    case OutputParserState::Tool::kArgs:
      st.calls.back().body.append(text);
      return;
    case OutputParserState::Tool::kPreamble:
    case OutputParserState::Tool::kBetween:
      return;  // text between a call's markers is dropped
  }
}

void on_terminal(OutputParserState& st, Kind kind, const std::string& text) {
  if (kind == Kind::kDrop) return;
  using Tool = OutputParserState::Tool;
  if (st.reasoning_phase) {
    if (kind == Kind::kThinkStart) return;
    if (kind == Kind::kThinkEnd) {
      end_reasoning(st);
      st.reasoning_phase = false;
      return;
    }
    if (kind != Kind::kToolStart) {
      on_text(st, text);
      return;
    }
    // A call straight out of the reasoning ends it, and the marker opens the call below.
    end_reasoning(st);
    st.reasoning_phase = false;
  }
  switch (st.tool) {
    case Tool::kContent:
      if (kind == Kind::kToolStart) {
        st.calls.emplace_back();
        st.tool = Tool::kPreamble;
        return;
      }
      if (kind == Kind::kFuncPrefix) {
        st.calls.emplace_back();
        st.tool = Tool::kName;
        return;
      }
      break;
    case Tool::kPreamble:
      if (kind == Kind::kFuncPrefix) {
        st.tool = Tool::kName;
        return;
      }
      break;
    case Tool::kName:
      if (kind == Kind::kCloseAngle) {
        st.tool = Tool::kArgs;
        return;
      }
      if (kind == Kind::kFuncEnd) {
        st.tool = Tool::kBetween;
        finalize_through(st, st.calls.size());
        return;
      }
      break;
    case Tool::kArgs:
      if (kind == Kind::kParamStart || kind == Kind::kParamEnd) {
        st.calls.back().body.append(text);
        return;
      }
      if (kind == Kind::kFuncEnd) {
        st.tool = Tool::kBetween;
        finalize_through(st, st.calls.size());
        return;
      }
      break;
    case Tool::kBetween:
      if (kind == Kind::kToolStart) {
        st.calls.emplace_back();
        st.tool = Tool::kPreamble;
        return;
      }
      if (kind == Kind::kFuncPrefix) {
        st.calls.emplace_back();
        st.tool = Tool::kName;
        return;
      }
      break;
  }
  // No transition for this marker here: it is ordinary text in the current state.
  on_text(st, text);
}

void lex(OutputParserState& st, bool final_chunk) {
  const auto& table = literals();
  const size_t max_len = longest_literal();
  // Until the stream ends, only whole code points are lexed: the bytes of a character that a chunk
  // boundary splits wait in pending for the rest of it.
  const size_t limit = final_chunk ? st.pending.size() : st.pending.size() - utf8_incomplete_tail(st.pending);
  size_t pos = 0;
  while (pos < limit) {
    const std::string_view rest(st.pending.data() + pos, limit - pos);
    if (!final_chunk && rest.size() <= max_len && proper_prefixes().count(std::string(rest)) != 0) {
      break;  // could still become a literal: wait for more text
    }
    const Literal* best = nullptr;
    for (const Literal& lit : table) {
      if (rest.size() >= lit.text.size() && rest.compare(0, lit.text.size(), lit.text) == 0 &&
          (best == nullptr || lit.text.size() > best->text.size())) {
        best = &lit;
      }
    }
    if (best != nullptr) {
      on_terminal(st, best->kind, best->text);
      pos += best->text.size();
      continue;
    }
    // Plain text up to the next place a literal could start ('<' or '>', both one byte).
    size_t next = rest.find_first_of("<>", 1);
    if (next == std::string_view::npos) next = rest.size();
    on_text(st, rest.substr(0, next));
    pos += next;
  }
  st.pending.erase(0, pos);
}

OutputParser::Delta take(OutputParserState& st) {
  OutputParser::Delta out = std::move(st.out);
  st.out = OutputParser::Delta{};
  return out;
}

}  // namespace

OutputParser::OutputParser(bool thinking) : OutputParser(thinking, std::string_view()) {}

OutputParser::OutputParser(bool thinking, std::string_view tools_json) : state_(std::make_unique<OutputParserState>()) {
  state_->reasoning_phase = thinking;
  if (!tools_json.empty()) {
    JValue tools;
    if (json_loads(tools_json, tools) && tools.is_array()) state_->tools = std::move(tools);
  }
}

OutputParser::~OutputParser() = default;

OutputParser::Delta OutputParser::push(std::string_view text) {
  state_->pending.append(text);
  lex(*state_, false);
  return take(*state_);
}

OutputParser::Delta OutputParser::finish() {
  OutputParserState& st = *state_;
  lex(st, true);
  if (st.reasoning_phase) {
    end_reasoning(st);
    st.reasoning_phase = false;
  }
  finalize_through(st, st.calls.size());
  // Without a kept call the content stays as it is, whitespace included. Whitespace-only content is
  // dropped when the output has a call that was not kept, as the reference does.
  if (!st.accepted && (st.content_nonspace || st.calls.empty())) st.out.content.append(st.content_hold);
  st.content_hold.clear();
  return take(st);
}

}  // namespace lx::text
