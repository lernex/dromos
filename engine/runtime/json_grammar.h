// Grammar-constrained JSON: the output of a request with response_format json_object or json_schema
// is restricted, token by token, to text that is a JSON value of that shape. The grammar is a
// pushdown automaton over UTF-8 bytes: a stack of frames, one per open container or scalar, that
// the schema tells apart. A byte can be consumed by the top frame, or complete it and hand the byte
// to the frame below; where the schema allows more than one reading, the automaton holds a set of
// stacks, and a byte keeps the stacks it can continue.
//
// Masks. The allowed tokens for a state are the tokens whose bytes the automaton accepts. For each
// top frame, the tokens that stay within that frame (never complete it) depend on the frame alone:
// they are computed once per frame state by a walk of the vocabulary's byte trie and cached. The
// tokens that complete the frame depend on the stack below it; they are checked against the real
// stack, and the result is cached per (frame state, stack below).
//
// Whitespace is compact: no space anywhere except one optional space after ':' and after ','.
// Object keys are written in schema order, optional ones may be left out, and additionalProperties
// must be false (or absent): a closed object writes its declared properties only.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "text/json.h"

namespace lx::runtime {

// A request's response_format, as the server reads it: json_object (any JSON object) or json_schema
// (the schema as order-preserving JSON text, from lx::text::json_dumps).
struct JsonFormat {
  bool object = false;
  std::string schema;      // JSON text; empty when object
  bool thinking = false;   // the reply's reasoning comes first, unconstrained, up to "</think>"
};

// The vocabulary as a byte trie: every token's raw bytes, for walking the grammar over all tokens.
class VocabIndex {
 public:
  struct Node {
    int32_t parent = -1;
    uint8_t byte = 0;
    std::vector<std::pair<uint8_t, int32_t>> children;  // sorted by byte
    std::vector<int32_t> tokens;                         // ids whose bytes end at this node
  };

  // token_bytes[i] is token i's raw bytes; an empty string is a token that is never allowed.
  explicit VocabIndex(const std::vector<std::string>& token_bytes);

  int32_t vocab() const { return vocab_; }
  size_t words() const { return words_; }                    // 32-bit mask words per mask
  const std::vector<Node>& nodes() const { return nodes_; }  // node 0 is the root
  const std::string& bytes(int32_t id) const { return bytes_[static_cast<size_t>(id)]; }

 private:
  int32_t vocab_ = 0;
  size_t words_ = 0;
  std::vector<std::string> bytes_;
  std::vector<Node> nodes_;
};

// A compiled schema: the node graph and the mask caches. Shared by every request with the same
// format; its caches fill on the engine thread only.
class JsonProgram;

// Compiles a format. Throws std::invalid_argument naming the keyword or construct it cannot express.
std::shared_ptr<JsonProgram> compile_json(const JsonFormat& format);

// One request's grammar: the set of stacks its output could be in.
class JsonGrammar {
 public:
  JsonGrammar(std::shared_ptr<JsonProgram> program, const VocabIndex& vocab);
  ~JsonGrammar();
  JsonGrammar(const JsonGrammar&) = delete;
  JsonGrammar& operator=(const JsonGrammar&) = delete;

  // Consumes the bytes of one token. False, with no change, when they cannot come next.
  bool accept(std::string_view bytes);
  // Whether the output so far is a complete JSON value (the reasoning, if any, is closed).
  bool may_end() const;
  // The allowed tokens as bits, one per vocabulary id. A stop id is allowed exactly when the output
  // may end. The reference stays valid until the next accept().
  const std::vector<uint32_t>& mask(const std::vector<int32_t>& stops);
  // True when nothing but a stop can come next: the value is complete and nothing follows it.
  bool finished(const std::vector<int32_t>& stops);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace lx::runtime
