#include <doctest/doctest.h>

#include <random>
#include <set>
#include <string>
#include <vector>

#include "runtime/tool_grammar.h"
#include "text/json.h"
#include "text/tokenizer.h"

using lx::runtime::ToolGrammar;
using lx::runtime::ToolGrammarSpec;

namespace {

// The printable ASCII bytes and newline, which is all the walks below choose from.
std::vector<unsigned char> ascii() {
  std::vector<unsigned char> out{'\n'};
  for (unsigned char c = 32; c < 127; ++c) out.push_back(c);
  return out;
}

// One random output the grammar allows, ended as soon as it may end (with a little luck) or at a
// length limit that the walk then closes with the first ending it can find.
std::string walk(ToolGrammar& g, std::mt19937& rng, size_t limit = 4000) {
  std::string out;
  const auto candidates = ascii();
  // past the limit, the walk closes what is open: the first of these the grammar accepts
  // The value closer comes first. A raw value takes a quote or a bracket as ordinary text, so only its
  // own ending closes it, and a JSON string takes the tag closers as text, so the quote closes it: each
  // closer is tried until the grammar takes one, and the first that is taken is the one that ends a value.
  const std::vector<std::string> closers = {"\n</parameter>\n", "\"", "0", "]", "}", "\n</function>", "</function>"};
  while (true) {
    REQUIRE(out.size() < 20 * limit);  // a walk that cannot close fails here, not by exhausting memory
    if (out.size() > limit) {
      bool closed = false;
      for (const std::string& text : closers) {
        if (!g.allows(text)) continue;
        REQUIRE(g.accept(text));
        out += text;
        closed = true;
        break;
      }
      if (closed) {
        if (g.may_end()) break;
        continue;
      }
    }
    std::vector<unsigned char> ok;
    for (unsigned char c : candidates)
      if (g.allows(std::string(1, static_cast<char>(c)))) ok.push_back(c);
    if (g.may_end() && (ok.empty() || rng() % 3 == 0)) break;
    REQUIRE(!ok.empty());
    const unsigned char c = ok[rng() % ok.size()];
    REQUIRE(g.accept_byte(c));
    out.push_back(static_cast<char>(c));
  }
  return out;
}

std::string tools_json(const std::vector<std::pair<std::string, std::string>>& tools) {
  std::string out = "[";
  for (size_t i = 0; i < tools.size(); ++i) {
    if (i) out += ",";
    out += "{\"type\":\"function\",\"function\":{\"name\":\"" + tools[i].first + "\",\"parameters\":" + tools[i].second + "}}";
  }
  return out + "]";
}

// Parses the output with the output parser, and checks every call: the name is one the grammar may
// call, every required argument is there, and each argument has the JSON type its schema names.
void check_calls(const std::string& out, const std::vector<std::pair<std::string, std::string>>& tools,
                 const std::set<std::string>& allowed, bool named_one) {
  lx::text::OutputParser parser(false, tools_json(tools));
  auto pushed = parser.push(out);
  auto fin = parser.finish();
  std::vector<lx::text::ToolCall> calls = pushed.tool_calls;
  calls.insert(calls.end(), fin.tool_calls.begin(), fin.tool_calls.end());
  REQUIRE(!calls.empty());
  (void)named_one;
  for (const auto& call : calls) {
    CAPTURE(out);  // shown only when a check below fails
    CAPTURE(call.arguments_json);
    CHECK(allowed.count(call.name) == 1);
    lx::text::JValue schema, args;
    const std::string* schema_text = nullptr;
    for (const auto& [name, text] : tools)
      if (name == call.name) schema_text = &text;
    REQUIRE(schema_text != nullptr);
    REQUIRE(lx::text::json_loads(*schema_text, schema));
    REQUIRE(lx::text::json_loads(call.arguments_json, args));
    REQUIRE(args.is_object());
    const lx::text::JValue* props = schema.get("properties");
    const lx::text::JValue* required = schema.get("required");
    if (required)
      for (const auto& r : required->items) CHECK(args.get(r.text) != nullptr);
    for (size_t i = 0; i < args.keys.size(); ++i) {
      const lx::text::JValue* ps = props ? props->get(args.keys[i]) : nullptr;
      REQUIRE(ps != nullptr);  // only declared arguments
      const lx::text::JValue* type = ps->get("type");
      const lx::text::JValue& v = args.values[i];
      if (ps->get("enum")) {
        bool found = false;
        for (const auto& e : ps->get("enum")->items)
          found = found || lx::text::json_dumps(e) == lx::text::json_dumps(v);
        CHECK(found);
      }
      if (!type || !type->is_string()) continue;
      const std::string& t = type->text;
      if (t == "string") CHECK(v.is_string());
      if (t == "integer") CHECK(v.type == lx::text::JValue::Type::kInt);
      if (t == "number") CHECK((v.type == lx::text::JValue::Type::kInt || v.type == lx::text::JValue::Type::kFloat));
      if (t == "boolean") CHECK(v.type == lx::text::JValue::Type::kBool);
      if (t == "object") CHECK(v.is_object());
      if (t == "array") CHECK(v.is_array());
    }
  }
}

const std::string kWeather =
    R"({"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]})";
const std::string kMixed =
    R"({"type":"object","properties":{"ratio":{"type":"number"},"ok":{"type":"boolean"},"mode":{"type":"string","enum":["fast","slow"]}},"required":["ratio","mode"]})";
const std::string kNested =
    R"({"type":"object","properties":{"filters":{"type":"object"},"tags":{"type":"array"},"note":{"type":"null"}},"required":["filters"]})";
const std::string kNoArgs = R"({"type":"object","properties":{}})";
const std::string kLevel = R"({"type":"object","properties":{"level":{"enum":[1,2,3]}},"required":["level"]})";
const std::string kQuery = R"({"type":"object","properties":{"q":{"type":"string"}},"required":["q"]})";
const std::string kCount =
    R"({"type":"object","properties":{"n":{"type":"integer"},"x":{"type":"number"}},"required":["n"]})";
const std::string kLookup =
    R"({"type":"object","properties":{"id":{"type":"integer"},"name":{"type":"string"}},"required":["id","name"]})";

}  // namespace

TEST_CASE("the grammar accepts exactly the format's calls, and every walk parses into typed calls") {
  struct Case {
    const char* label;
    ToolGrammarSpec spec;
    bool thinking;
  };
  const std::vector<Case> cases = {
      {"string and integer, one required", {false, {{"get_weather", kWeather}}}, false},
      {"number, boolean and a string enum", {false, {{"mix", kMixed}}}, false},
      {"object and array arguments, null", {false, {{"search", kNested}}}, false},
      {"no arguments at all", {false, {{"ping", kNoArgs}}}, false},
      {"integer enum", {false, {{"set_level", kLevel}}}, false},
      {"two tools, any may be called", {false, {{"query", kQuery}, {"count", kCount}}}, false},
      {"named: only the first of two", {true, {{"lookup", kLookup}, {"query", kQuery}}}, false},
      {"thinking first, then the call", {false, {{"get_weather", kWeather}}}, true},
  };
  for (const Case& c : cases) {
    SUBCASE(c.label) {
      std::mt19937 rng(7);
      std::set<std::string> allowed;
      for (const auto& t : c.spec.tools) allowed.insert(t.first);
      if (c.spec.named) allowed = {c.spec.tools.front().first};
      for (int i = 0; i < 40; ++i) {
        ToolGrammar g(c.spec, c.thinking);
        std::string out;
        if (c.thinking) {
          for (char ch : std::string("Let me check the tool.")) REQUIRE(g.accept_byte(static_cast<unsigned char>(ch)));
          out = "Let me check the tool.</think>";
          REQUIRE(g.accept("</think>"));
        }
        out += walk(g, rng);
        check_calls(out, c.spec.tools, allowed, c.spec.named);
      }
    }
  }
}

TEST_CASE("the grammar refuses what the format does not allow") {
  ToolGrammarSpec spec{false, {{"get_weather", kWeather}}};
  ToolGrammar g(spec, false);
  CHECK(g.allows("<|tool_call|><function=get_weather>\n"));
  CHECK_FALSE(g.allows("Sure, here is the weather."));  // plain content is not a call
  CHECK_FALSE(g.allows("<|tool_call|><function=unknown>\n"));
  CHECK_FALSE(g.may_end());

  ToolGrammar call(spec, false);
  REQUIRE(call.accept("<|tool_call|><function=get_weather>\n"));
  CHECK(call.accept("<parameter=days>\n"));            // optional arguments may come first, in any order
  ToolGrammar named(ToolGrammarSpec{true, {{"lookup", kLookup}, {"query", kQuery}}}, false);
  CHECK_FALSE(named.allows("<|tool_call|><function=query>\n"));  // only the named function may be called
  CHECK(named.allows("<|tool_call|><function=lookup>\n"));

  ToolGrammar typed(ToolGrammarSpec{false, {{"count", kCount}}}, false);
  REQUIRE(typed.accept("<|tool_call|><function=count>\n<parameter=n>\n"));
  CHECK_FALSE(typed.accept("1.5"));                       // an integer is not 1.5
  CHECK(typed.accept("7"));
  CHECK_FALSE(typed.accept("\n</parameter>\nx"));         // after the value only the closing tag may follow

  ToolGrammar raw(ToolGrammarSpec{false, {{"query", kQuery}}}, false);
  REQUIRE(raw.accept("<|tool_call|><function=query>\n<parameter=q>\n"));
  CHECK_FALSE(raw.accept("a </parameter> b"));            // a string may not contain its own closing tag
}

TEST_CASE("a required argument must be written before the call may close") {
  ToolGrammar g(ToolGrammarSpec{false, {{"get_weather", kWeather}}}, false);
  REQUIRE(g.accept("<|tool_call|><function=get_weather>\n"));
  CHECK_FALSE(g.allows("</function>"));
  REQUIRE(g.accept("<parameter=city>\nOslo\n</parameter>\n"));
  CHECK(g.allows("</function>"));
  CHECK(g.allows("<parameter=days>\n"));
}

TEST_CASE("the token mask allows what the state allows, and stop ids only when the output may end") {
  ToolGrammar g(ToolGrammarSpec{false, {{"get_weather", kWeather}}}, false);
  std::vector<std::string> texts = {"<|tool_call|><function=", "hello", "", "xx"};
  const std::vector<int32_t> stops = {3};
  const auto& mask = g.mask(texts, stops);
  CHECK((mask[0] & 1u) != 0);   // the call starts
  CHECK((mask[0] & 2u) == 0);
  CHECK((mask[0] & 4u) == 0);   // an empty token is never allowed
  CHECK((mask[0] & 8u) == 0);   // the stop id, before any call
}
