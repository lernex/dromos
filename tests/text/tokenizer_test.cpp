// Tokenizer parity with transformers: encode (with and without special tokens), decode, streaming,
// the vocabulary facts and stop ids, and the encode speed. The goldens come from
// tools/text/golden_tokenizer.py through LX_TEXT_GOLDEN; the model folder through LX_TEXT_MODEL.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "support.h"
#include "text/json.h"
#include "text/tokenizer.h"
#include "text/utf8.h"

using lx::text::JValue;
using lx::text::Tokenizer;
using lx::text::test::for_each_json_line;
using lx::text::test::from_hex;
using lx::text::test::golden_dir;
using lx::text::test::model_dir;
using lx::text::test::str_of;

namespace {

std::vector<int32_t> ids_of(const JValue* array) {
  std::vector<int32_t> ids;
  if (array == nullptr) return ids;
  ids.reserve(array->items.size());
  for (const JValue& v : array->items) ids.push_back(static_cast<int32_t>(std::stol(v.text)));
  return ids;
}

bool is_prefix_of(const std::string& prefix, const std::string& whole) {
  return prefix.size() <= whole.size() && whole.compare(0, prefix.size(), prefix) == 0;
}

struct Skip {
  bool skip;
  Skip() : skip(model_dir().empty() || golden_dir().empty()) {
    if (skip) std::printf("text tokenizer tests skipped: set LX_TEXT_MODEL and LX_TEXT_GOLDEN\n");
  }
};

}  // namespace

TEST_CASE("tokenizer encode matches transformers ids") {
  Skip skip;
  if (skip.skip) return;
  auto tok = Tokenizer::load(model_dir());
  size_t cases = 0, mismatches = 0, nospecial = 0, hex = 0, roundtrip = 0;
  const bool read = for_each_json_line(golden_dir() + "/tokenizer/encode.jsonl", [&](const JValue& c) {
    ++cases;
    const std::string text = str_of(c, "text");
    const auto want = ids_of(c.get("ids"));
    const auto got = tok->encode(text, true);
    if (got != want) {
      if (mismatches < 5) std::printf("encode mismatch on case %zu (%zu bytes)\n", cases, text.size());
      ++mismatches;
    }
    if (tok->decode(got, false) == text) {
      ++roundtrip;
    }
    const auto want_ns = ids_of(c.get("ids_nospecial"));
    if (tok->encode(text, false) != want_ns) {
      if (nospecial < 5) std::printf("no-special mismatch on case %zu\n", cases);
      ++nospecial;
    }
    if (c.get("hex") != nullptr) {
      const std::string raw = from_hex(str_of(c, "hex"));
      if (tok->encode(raw, true) != want) {
        if (hex < 5) std::printf("invalid-bytes mismatch on case %zu\n", cases);
        ++hex;
      }
    }
  });
  CHECK(read);
  std::printf("encode cases %zu, id mismatches %zu, no-special mismatches %zu, invalid-byte mismatches %zu, "
              "round trips %zu\n", cases, mismatches, nospecial, hex, roundtrip);
  CHECK(cases >= 20000);
  CHECK(mismatches == 0);
  CHECK(nospecial == 0);
  CHECK(hex == 0);
  CHECK(roundtrip == cases);
}

TEST_CASE("tokenizer decode matches transformers, and streaming never changes emitted text") {
  Skip skip;
  if (skip.skip) return;
  auto tok = Tokenizer::load(model_dir());
  size_t cases = 0, decode_bad = 0, skip_bad = 0, stream_bad = 0, prefix_bad = 0, utf8_bad = 0;
  for_each_json_line(golden_dir() + "/tokenizer/decode.jsonl", [&](const JValue& c) {
    ++cases;
    const auto ids = ids_of(c.get("ids"));
    const std::string text = str_of(c, "text");
    const std::string text_skip = str_of(c, "text_skip");
    if (tok->decode(ids, false) != text) ++decode_bad;
    if (tok->decode(ids, true) != text_skip) ++skip_bad;
    for (bool skip_special : {false, true}) {
      const std::string& want = skip_special ? text_skip : text;
      auto st = tok->stream(skip_special);
      std::string acc;
      for (int32_t id : ids) {
        const std::string piece = st->push(id);
        if (!lx::text::utf8_valid(piece)) ++utf8_bad;
        acc += piece;
        if (!is_prefix_of(acc, want)) ++prefix_bad;
      }
      acc += st->flush();
      if (acc != want) ++stream_bad;
    }
  });
  std::printf("decode cases %zu, decode mismatches %zu, skip mismatches %zu, stream mismatches %zu, "
              "stream prefix violations %zu, invalid pieces %zu\n",
              cases, decode_bad, skip_bad, stream_bad, prefix_bad, utf8_bad);
  CHECK(cases >= 30000);
  CHECK(decode_bad == 0);
  CHECK(skip_bad == 0);
  CHECK(stream_bad == 0);
  CHECK(prefix_bad == 0);
  CHECK(utf8_bad == 0);
}

TEST_CASE("tokenizer vocabulary, special tokens and stop ids match the folder") {
  Skip skip;
  if (skip.skip) return;
  auto tok = Tokenizer::load(model_dir());
  JValue meta;
  REQUIRE(lx::text::json_loads(lx::text::test::read_file(golden_dir() + "/tokenizer/meta.json"), meta));
  CHECK(tok->vocab_size() == std::stoi(meta.get("vocab_size")->text));
  for (const JValue& t : meta.get("added_tokens")->items) {
    const int32_t id = static_cast<int32_t>(std::stol(t.get("id")->text));
    CHECK(tok->token_id(str_of(t, "content")) == id);
    CHECK(tok->is_special(id) == t.get("special")->boolean);
  }
  std::vector<int32_t> want_stops;
  for (const JValue& v : meta.get("generation_eos_token_ids")->items) {
    want_stops.push_back(static_cast<int32_t>(std::stol(v.text)));
  }
  std::vector<int32_t> stops = tok->stop_ids();
  std::sort(stops.begin(), stops.end());
  std::sort(want_stops.begin(), want_stops.end());
  CHECK(stops == want_stops);
  CHECK(tok->token_id("<definitely-not-a-token>") == -1);
}

TEST_CASE("encode speed, single thread") {
  Skip skip;
  if (skip.skip) return;
  auto tok = Tokenizer::load(model_dir());
  const std::string text = lx::text::test::read_file(golden_dir() + "/tokenizer/speed.txt");
  double best = 0.0;
  size_t total_ids = 0;
  for (int run = 0; run < 3; ++run) {
    const auto start = std::chrono::steady_clock::now();
    const auto ids = tok->encode(text, true);
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    total_ids = ids.size();
    best = std::max(best, static_cast<double>(text.size()) / seconds / 1e6);
  }
  std::printf("encode speed: %.1f MB/s single-threaded (%zu bytes, %zu tokens, best of 3)\n", best, text.size(),
              total_ids);
  CHECK(best > 0.0);
}

TEST_CASE("a reply that stops inside a character ends without U+FFFD") {
  // The model folder alone is enough: no golden needed. Some vocabulary entries are one byte of a multi-byte
  // character; alone they decode to U+FFFD, and a reply cut right after one must not show it.
  if (model_dir().empty()) {
    std::printf("reply-end test skipped: set LX_TEXT_MODEL\n");
    return;
  }
  auto tok = Tokenizer::load(model_dir());
  const std::string replacement = "\xEF\xBF\xBD";
  int32_t partial = -1;
  for (int32_t id = 0; id < tok->vocab_size() && partial < 0; ++id) {
    if (tok->is_special(id) || tok->decode({id}, false).find(replacement) == std::string::npos) continue;
    auto probe = tok->stream(false);
    // A lead byte: push() holds it back and finish() drops it, so nothing is shown at all. (A complete character
    // is shown by push(), and a stray continuation byte by push() as U+FFFD.)
    if (probe->push(id).empty() && probe->finish().empty()) partial = id;
  }
  REQUIRE(partial >= 0);
  auto cut = tok->stream(false);
  CHECK(cut->push(partial).empty());
  CHECK(cut->finish().find(replacement) == std::string::npos);
  auto flushed = tok->stream(false);
  flushed->push(partial);
  CHECK(flushed->flush() == replacement);  // flush keeps the transformers decode: the replacement is still there

  // A finished reply is unchanged: the text before the cut is what the stream showed, and finish adds nothing.
  const auto ids = tok->encode("a\xE4\xBD\xA0" "b", false);
  auto whole = tok->stream(false);
  std::string acc;
  for (int32_t id : ids) acc += whole->push(id);
  acc += whole->finish();
  CHECK(acc == "a\xE4\xBD\xA0" "b");
}
