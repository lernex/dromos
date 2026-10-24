// Grammar-constrained tool calls: the output of a request that must call a tool (tool_choice
// "required", or a named function) is restricted, token by token, to text the tool-call format
// accepts. The format is the Metis one the chat template teaches and the output parser reads:
//
//   <|tool_call|><function=NAME>\n
//   <parameter=KEY>\nVALUE\n</parameter>\n          (one block per argument, any order, each once)
//   </function>
//
// Required arguments must all appear. A string argument is raw text (the template writes it so);
// an integer, number, boolean, null, enum, object or array argument is JSON text of its type (an
// object or array is checked as JSON, not against its nested schema). The reasoning comes first in
// thinking mode and is not constrained. A call that is started is always finished, so the output
// always parses into the calls the schema allows.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace lx::runtime {

struct ToolGrammarSpec {
  bool named = false;  // true: only the first tool may be called (tool_choice names it)
  std::vector<std::pair<std::string, std::string>> tools;  // (name, parameters schema as JSON text)
};

class ToolGrammar {
 public:
  // Throws std::invalid_argument naming what the grammar cannot express (a schema it does not read).
  ToolGrammar(const ToolGrammarSpec& spec, bool thinking);
  ~ToolGrammar();
  ToolGrammar(const ToolGrammar& other);
  ToolGrammar& operator=(const ToolGrammar& other);
  ToolGrammar(ToolGrammar&&) noexcept;
  ToolGrammar& operator=(ToolGrammar&&) noexcept;

  // Consumes one byte of output. False (and no change) when it cannot come next.
  bool accept_byte(unsigned char c);
  // Consumes the text of one token. False (and no change) when any of its bytes cannot come next.
  bool accept(std::string_view text);
  // Whether a token with this text could come next (no change).
  bool allows(std::string_view text) const;
  // Whether the output may end now: at least one complete call, and nothing half-written.
  bool may_end() const;

  // The allowed tokens as bits, one per vocabulary id: texts[i] is token i's text (empty for none).
  // A stop id is allowed exactly when the output may end. The result is cached by state; the
  // reference stays valid until the next call.
  const std::vector<uint32_t>& mask(const std::vector<std::string>& texts, const std::vector<int32_t>& stops);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lx::runtime
