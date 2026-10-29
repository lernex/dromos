#include <doctest/doctest.h>

#include <algorithm>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "runtime/json_grammar.h"
#include "text/json.h"

using lx::runtime::compile_json;
using lx::runtime::JsonFormat;
using lx::runtime::JsonGrammar;
using lx::runtime::JsonProgram;
using lx::runtime::VocabIndex;

namespace {

// Every byte as its own token, then multi-byte tokens that JSON output is written in, a token that is half of a
// UTF-8 character (the byte 0xC3 above is the first half of "é"), an empty token (never allowed) and a stop token.
std::vector<std::string> fake_vocab() {
  std::vector<std::string> v;
  for (int c = 0; c < 256; ++c) v.push_back(std::string(1, static_cast<char>(c)));
  for (const char* t : {"{\"", "\":\"", "\",\"", "\"}", "\":", "title", "year", "tags", "true", "false", "null",
                        "123", "45", "\"]", "]}", "},", "[\"", "</think>", "é", "\\\"", "\\n", " \"", ": ", ", ",
                        "ab", "<|end|>"})
    v.push_back(t);
  v.push_back("");
  return v;
}

constexpr int32_t kStop = 256 + 25;  // the index of "<|end|>" in fake_vocab()

const VocabIndex& vocab() {
  static const VocabIndex index(fake_vocab());
  return index;
}

std::vector<int32_t> stops() { return {kStop}; }

std::shared_ptr<JsonProgram> program(const std::string& schema, bool thinking = false) {
  return compile_json(JsonFormat{false, schema, thinking});
}

std::shared_ptr<JsonProgram> object_program() { return compile_json(JsonFormat{true, "", false}); }

// Whether the text is a complete value of the program: every byte accepted, and the output may end after it.
bool complete(const std::shared_ptr<JsonProgram>& p, const std::string& text) {
  JsonGrammar g(p, vocab());
  return g.accept(text) && g.may_end();
}

// Whether the text can be a prefix of the output: every byte accepted (the value may still be unfinished).
bool prefix_ok(const std::shared_ptr<JsonProgram>& p, const std::string& text) {
  JsonGrammar g(p, vocab());
  return g.accept(text);
}

const char* kRivals = R"({
  "type": "object",
  "properties": {
    "title": {"type": "string"},
    "year": {"type": "integer"},
    "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 5}
  },
  "required": ["title", "year", "tags"],
  "additionalProperties": false
})";

// The mask, by brute force: a fresh grammar for each token, replaying the prefix of token ids first.
std::vector<uint32_t> brute_mask(const std::shared_ptr<JsonProgram>& p, const std::vector<int32_t>& prefix) {
  const VocabIndex& v = vocab();
  std::vector<uint32_t> bits(v.words(), 0);
  for (int32_t id = 0; id < v.vocab(); ++id) {
    if (id == kStop || v.bytes(id).empty()) continue;
    JsonGrammar g(p, v);
    for (int32_t t : prefix) REQUIRE(g.accept(v.bytes(t)));
    if (g.accept(v.bytes(id))) bits[static_cast<size_t>(id) / 32] |= 1u << (static_cast<size_t>(id) % 32);
  }
  JsonGrammar g(p, v);
  for (int32_t t : prefix) REQUIRE(g.accept(v.bytes(t)));
  if (g.may_end()) bits[static_cast<size_t>(kStop) / 32] |= 1u << (static_cast<size_t>(kStop) % 32);
  return bits;
}

// Ids of the tokens (other than stops) whose bits are set.
std::vector<int32_t> allowed_ids(const std::vector<uint32_t>& bits) {
  std::vector<int32_t> out;
  for (int32_t id = 0; id < vocab().vocab(); ++id)
    if (bits[static_cast<size_t>(id) / 32] >> (static_cast<size_t>(id) % 32) & 1u) out.push_back(id);
  return out;
}

std::string text_of(const std::vector<int32_t>& ids) {
  std::string out;
  for (int32_t id : ids) out += vocab().bytes(id);
  return out;
}

}  // namespace

TEST_CASE("the RIVALS schema accepts its replies and refuses the rest") {
  const auto p = program(kRivals);
  // what the schema allows
  CHECK(complete(p, R"({"title":"Ada","year":1843,"tags":["math","history"]})"));
  CHECK(complete(p, R"({"title":"x","year":0,"tags":[]})"));
  CHECK(complete(p, R"({"title":"Müller","year":-5,"tags":["a\"b"]})"));
  CHECK(complete(p, R"({"title":"x","year":1,"tags":["a","b","c","d","e"]})"));
  // one optional space after ":" and after ","
  CHECK(complete(p, R"({"title": "x", "year": 2, "tags": []})"));
  // what it does not: key order, missing required keys, the wrong types, too many tags, whitespace elsewhere
  CHECK_FALSE(complete(p, R"({"year":1,"title":"x","tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1.5,"tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1,"tags":["a","b","c","d","e","f"]})"));
  CHECK_FALSE(complete(p, R"({ "title":"x","year":1,"tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":  "x","year":1,"tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1,"tags":[]} )"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":01,"tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":"bad \x escape","year":1,"tags":[]})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1,"tags":[],"extra":true})"));
  CHECK_FALSE(complete(p, R"({"title":"x","year":1,"tags":[],})"));
  // a prefix of a reply may stop early (not complete), and a value that starts wrong is refused outright
  CHECK(prefix_ok(p, R"({"title":"Ad)"));
  CHECK_FALSE(prefix_ok(p, R"([)"));
}

TEST_CASE("the UTF-8 in strings must be whole characters") {
  const auto p = program(R"({"type":"string"})");
  CHECK(complete(p, "\"caf\xC3\xA9\""));                  // é as two bytes
  CHECK(complete(p, "\"\xF0\x9F\x98\x80\""));             // a four-byte character
  CHECK_FALSE(complete(p, "\"\x80\""));                   // a continuation byte alone
  CHECK_FALSE(complete(p, "\"\xC0\xAF\""));               // an overlong encoding
  CHECK_FALSE(complete(p, "\"\xED\xA0\x80\""));           // a surrogate, which UTF-8 does not allow
  CHECK_FALSE(complete(p, "\"\x01\""));                   // control characters must be escaped
  CHECK(complete(p, R"("é\n\t")"));
  CHECK_FALSE(complete(p, R"("\u00g0")"));
}

TEST_CASE("unsupported keywords are refused, and the message names them") {
  const std::vector<std::pair<std::string, std::string>> cases = {
      {R"({"type":"string","pattern":"a+"})", "pattern"},
      {R"({"type":"string","minLength":2})", "minLength"},
      {R"({"type":"string","format":"date"})", "format"},
      {R"({"type":"integer","minimum":0})", "minimum"},
      {R"({"type":"array","uniqueItems":true})", "uniqueItems"},
      {R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":{"type":"string"}})",
       "additionalProperties"},
      {R"({"type":"object","properties":{"a":{"type":"string"}},"additionalProperties":true})",
       "additionalProperties"},
      {R"({"allOf":[{"type":"string"}]})", "allOf"},
      {R"({"not":{"type":"string"}})", "not"},
      {R"({"$ref":"https://example.com/x.json"})", "$ref"},
      {R"({"type":"object","properties":{"a":{"type":"string"}},"required":["b"],"additionalProperties":false})",
       "required"},
      {R"({"type":"array","items":[{"type":"string"}]})", "items"},
      {R"({"type":"string","contains":{"const":"x"}})", "contains"},
      {R"({"oneOf":[{"type":"string"},{"type":"string","const":"x"}]})", "oneOf"},
      {R"({"type":"widget"})", "widget"},
  };
  for (const auto& [schema, keyword] : cases) {
    INFO(schema);
    bool refused = false;
    std::string message;
    try {
      program(schema);
    } catch (const std::invalid_argument& e) {
      refused = true;
      message = e.what();
    }
    CHECK(refused);
    CHECK(message.find(keyword) != std::string::npos);
  }
}

TEST_CASE("a closed object with optional keys writes its declared keys only, in order") {
  const auto p = program(R"({"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"boolean"},"c":{"type":"null"}}})");
  CHECK(complete(p, "{}"));
  CHECK(complete(p, R"({"a":1})"));
  CHECK(complete(p, R"({"b":true,"c":null})"));
  CHECK(complete(p, R"({"a":1,"b":false,"c":null})"));
  CHECK_FALSE(complete(p, R"({"b":true,"a":1})"));
  CHECK_FALSE(complete(p, R"({"d":1})"));
}

TEST_CASE("required keys must all come before the object closes") {
  const auto p = program(R"({"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["b"],"additionalProperties":false})");
  CHECK(complete(p, R"({"b":"x"})"));
  CHECK(complete(p, R"({"a":"x","b":"y"})"));
  CHECK_FALSE(complete(p, "{}"));
  CHECK_FALSE(complete(p, R"({"a":"x"})"));
}

TEST_CASE("json_object takes any object and nothing else") {
  const auto p = object_program();
  CHECK(complete(p, R"({"a":[1,{"b":null}],"c":"d"})"));
  CHECK(complete(p, "{}"));
  CHECK(complete(p, R"({"x":{"y":[true,false,null,-0.5e+3]}})"));
  CHECK_FALSE(complete(p, "[1]"));
  CHECK_FALSE(complete(p, R"("text")"));
  CHECK_FALSE(complete(p, R"({"a":1,})"));
  CHECK_FALSE(complete(p, R"({"a":01})"));
}

TEST_CASE("enum, const, anyOf and nullable types") {
  CHECK(complete(program(R"({"enum":["red","green",3,null]})"), R"("green")"));
  CHECK(complete(program(R"({"enum":["red","green",3,null]})"), "3"));
  CHECK_FALSE(complete(program(R"({"enum":["red","green"]})"), R"("blue")"));
  CHECK(complete(program(R"({"const":"x"})"), R"("x")"));
  CHECK_FALSE(complete(program(R"({"const":"x"})"), R"("y")"));
  const auto nullable = program(R"({"type":["string","null"]})");
  CHECK(complete(nullable, "null"));
  CHECK(complete(nullable, R"("hi")"));
  CHECK_FALSE(complete(nullable, "12"));
  const auto any_of = program(R"({"anyOf":[{"type":"string"},{"type":"null"},{"type":"array","items":{"type":"integer"},"maxItems":3}]})");
  CHECK(complete(any_of, "[1,2]"));
  CHECK(complete(any_of, "null"));
  CHECK_FALSE(complete(any_of, "[1,2,3,4]"));
  CHECK_FALSE(complete(any_of, "true"));
  const auto one_of = program(R"({"oneOf":[{"const":"a"},{"const":"b"}]})");
  CHECK(complete(one_of, R"("b")"));
  CHECK_FALSE(complete(one_of, R"("c")"));
  const auto enum_type = program(R"({"type":"integer","enum":[1,2.5,3]})");
  CHECK(complete(enum_type, "3"));
  CHECK_FALSE(complete(enum_type, "2.5"));
}

TEST_CASE("recursive references nest to any depth and end") {
  const auto p = program(R"({
    "type": "object",
    "properties": {"kind": {"enum": ["cat", "dog"]}, "self": {"$ref": "#/$defs/node"}},
    "required": ["kind"],
    "additionalProperties": false,
    "$defs": {"node": {"type": "object", "properties": {"v": {"type": ["number", "null"]}, "next": {"$ref": "#/$defs/node"}}, "additionalProperties": false}}
  })");
  CHECK(complete(p, R"({"kind":"cat"})"));
  CHECK(complete(p, R"({"kind":"dog","self":{"v":1.5,"next":{"v":null}}})"));
  CHECK_FALSE(complete(p, R"({"kind":"cow"})"));
  CHECK_FALSE(complete(p, R"({"kind":"cat","self":{"v":"x"}})"));
}

TEST_CASE("prefixItems come first, then items, within the length bounds") {
  const auto p = program(R"({"type":"array","prefixItems":[{"type":"boolean"},{"const":"x"}],"items":{"type":"number"}})");
  CHECK(complete(p, "[]"));
  CHECK(complete(p, "[true]"));
  CHECK(complete(p, R"([true,"x",1.5,-2e3])"));
  CHECK_FALSE(complete(p, R"(["x",true])"));
  CHECK_FALSE(complete(p, R"([true,"y"])"));
  const auto counted = program(R"({"type":"array","items":{"type":"integer"},"minItems":2,"maxItems":3})");
  CHECK_FALSE(complete(counted, "[]"));
  CHECK_FALSE(complete(counted, "[1]"));
  CHECK(complete(counted, "[1,2]"));
  CHECK(complete(counted, "[1,2,3]"));
  CHECK_FALSE(complete(counted, "[1,2,3,4]"));
}

TEST_CASE("the thinking prefix is free text up to </think>, then the value") {
  const auto p = program(R"({"type":"object","properties":{"a":{"type":"integer"}},"additionalProperties":false})", true);
  CHECK(complete(p, "hmm, let me think {[ ]} </think>{\"a\":1}"));
  CHECK_FALSE(complete(p, "hmm </think>{\"a\":1} "));
  CHECK(prefix_ok(p, "still thinking, no end yet"));
  CHECK_FALSE(complete(p, "still thinking, no end yet"));
}

TEST_CASE("the mask allows exactly the tokens the automaton accepts, at every state of random replies") {
  const std::vector<std::pair<std::string, bool>> cases = {
      {kRivals, false},
      {R"({"anyOf":[{"type":"string"},{"type":"null"},{"type":"array","items":{"type":"integer"},"maxItems":3}]})", false},
      {R"({"type":"object","properties":{"a":{"type":"array","prefixItems":[{"const":"x"}],"items":{"type":"number"}}}})", true},
  };
  for (const auto& [schema, thinking] : cases) {
    INFO(schema);
    const auto p = program(schema, thinking);
    std::mt19937 rng(20261028);
    for (int walk = 0; walk < 2; ++walk) {
      JsonGrammar g(p, vocab());
      std::vector<int32_t> prefix;
      for (int step = 0; step < 24; ++step) {
        const std::vector<uint32_t> got = g.mask(stops());
        const std::vector<uint32_t> want = brute_mask(p, prefix);
        REQUIRE(got == want);
        const std::vector<int32_t> ids = allowed_ids(got);
        std::vector<int32_t> choices;
        for (int32_t id : ids)
          if (id != kStop) choices.push_back(id);
        if (choices.empty()) break;
        const int32_t pick = choices[rng() % choices.size()];
        REQUIRE(g.accept(vocab().bytes(pick)));
        prefix.push_back(pick);
      }
    }
  }
}

TEST_CASE("random replies parse as JSON and satisfy the schema") {
  const auto p = program(kRivals);
  std::mt19937 rng(7);
  for (int walk = 0; walk < 40; ++walk) {
    JsonGrammar g(p, vocab());
    std::vector<int32_t> ids;
    for (int step = 0; step < 2000; ++step) {
      const std::vector<uint32_t> got = g.mask(stops());
      std::vector<int32_t> choices;
      for (int32_t id : allowed_ids(got))
        if (id != kStop) choices.push_back(id);
      if (g.may_end() && (rng() % 4 == 0 || choices.empty())) break;
      REQUIRE(!choices.empty());
      const int32_t pick = choices[rng() % choices.size()];
      REQUIRE(g.accept(vocab().bytes(pick)));
      ids.push_back(pick);
    }
    REQUIRE(g.may_end());
    const std::string text = text_of(ids);
    lx::text::JValue value;
    INFO(text);
    REQUIRE(lx::text::json_loads(text, value));
    REQUIRE(value.is_object());
    REQUIRE(value.keys.size() == 3);
    CHECK(value.keys[0] == "title");
    CHECK(value.keys[1] == "year");
    CHECK(value.keys[2] == "tags");
    CHECK(value.values[0].is_string());
    CHECK((value.values[1].type == lx::text::JValue::Type::kInt));
    CHECK(value.values[2].is_array());
    CHECK(value.values[2].items.size() <= 5);
  }
}

TEST_CASE("a finished value allows only the stop token") {
  const auto p = program(R"({"type":"object","properties":{"a":{"type":"boolean"}},"additionalProperties":false})");
  JsonGrammar g(p, vocab());
  REQUIRE(g.accept(R"({"a":true})"));
  const std::vector<uint32_t> bits = g.mask(stops());
  const std::vector<int32_t> ids = allowed_ids(bits);
  REQUIRE(ids.size() == 1);
  CHECK(ids[0] == kStop);
}

TEST_CASE("a rejected token leaves the grammar unchanged") {
  const auto p = program(kRivals);
  JsonGrammar g(p, vocab());
  REQUIRE(g.accept(R"({"title":"x")"));
  const std::vector<uint32_t> before = g.mask(stops());
  CHECK_FALSE(g.accept(R"(,"year":1.5)"));
  CHECK(g.mask(stops()) == before);
  REQUIRE(g.accept(R"(,"year":1,"tags":[]})"));
  CHECK(g.may_end());
}

TEST_CASE("a clone is in the same state and advances on its own") {
  auto p = program(R"({"type":"object","properties":{"title":{"type":"string"}},"required":["title"]})");
  JsonGrammar g(p, vocab());
  REQUIRE(g.accept("{\""));
  auto copy = g.clone();
  CHECK(copy->mask(stops()) == g.mask(stops()));
  // The copy takes the rest of a value; the original is where it was.
  REQUIRE(copy->accept("title\":\"ab\"}"));
  CHECK(copy->may_end());
  CHECK_FALSE(g.may_end());
  CHECK(g.accept("title"));
  CHECK(copy->mask(stops()) != g.mask(stops()));
}
