// The Jinja interpreter against jinja2 under transformers' environment (tools/text/golden_jinja.py):
// whitespace control, value printing, operators, filters, tests, loops, namespaces, macros, and errors.
#include <cstdio>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "support.h"
#include "text/json.h"
#include "text/jinja.h"
#include "text/jinja_internal.h"

using lx::text::JValue;
using lx::text::jinja::Template;
using lx::text::jinja::Value;
using lx::text::test::for_each_json_line;
using lx::text::test::golden_dir;
using lx::text::test::str_of;

TEST_CASE("jinja interpreter matches jinja2 under transformers' environment") {
  if (golden_dir().empty()) {
    std::printf("jinja tests skipped: set LX_TEXT_GOLDEN\n");
    return;
  }
  size_t cases = 0, mismatches = 0;
  const bool read = for_each_json_line(golden_dir() + "/jinja.jsonl", [&](const JValue& c) {
    ++cases;
    const std::string name = str_of(c, "name");
    std::vector<std::pair<std::string, Value>> vars;
    const JValue* values = c.get("vars");
    for (size_t i = 0; i < values->keys.size(); ++i) {
      vars.emplace_back(values->keys[i], lx::text::jinja::value_from_json(values->values[i]));
    }
    const JValue* error = c.get("error");
    std::string got;
    bool threw = false;
    std::string message;
    try {
      Template tpl(str_of(c, "template"));
      got = tpl.render(vars);
    } catch (const std::exception& e) {
      threw = true;
      message = e.what();
    }
    if (error != nullptr) {
      // The golden error ends with jinja2's message; a raise_exception carries it verbatim.
      const std::string golden = error->text;
      // raise_exception() carries its message verbatim, so that message must come out. Other errors
      // (undefined attributes, syntax errors, bad calls) must throw; their wording is jinja2's.
      const bool raised = golden.rfind("TemplateError: ", 0) == 0;
      const bool same = threw && (!raised || golden.find(message) != std::string::npos);
      if (!same) {
        if (mismatches < 5) {
          std::printf("%s: expected an error \"%s\", got %s\n", name.c_str(), golden.c_str(),
                      threw ? ("\"" + message + "\"").c_str() : "no error");
        }
        ++mismatches;
      }
      return;
    }
    const std::string want = str_of(c, "expected");
    if (threw || got != want) {
      if (mismatches < 5) {
        std::printf("%s\n  template %s\n  want [%s]\n  got  [%s]%s\n", name.c_str(),
                    str_of(c, "template").c_str(), want.c_str(), got.c_str(),
                    threw ? (" threw: " + message).c_str() : "");
      }
      ++mismatches;
    }
  });
  std::printf("jinja cases %zu, mismatches %zu\n", cases, mismatches);
  CHECK(read);
  CHECK(cases >= 70);
  CHECK(mismatches == 0);
}
