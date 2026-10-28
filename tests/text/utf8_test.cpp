// UTF-8 at the end of a reply. A reply cut by max_tokens (or the context, or a stop) can stop inside a multi-byte
// character; the characters before it are shown, the unfinished bytes are dropped, and no U+FFFD is added for them.
// Invalid bytes elsewhere keep utf8_lossy's replacement (one U+FFFD per maximal invalid subpart).
#include <string>

#include "doctest/doctest.h"
#include "text/utf8.h"

using lx::text::utf8_incomplete_tail;
using lx::text::utf8_lossy;
using lx::text::utf8_lossy_final;

namespace {

const char* kReplacement = "\xEF\xBF\xBD";  // U+FFFD

}  // namespace

TEST_CASE("a character cut off at the end of a reply is dropped, not replaced") {
  // "你" is E4 BD A0; "😀" is F0 9F 98 80.
  CHECK(utf8_lossy_final("ok \xE4\xBD") == "ok ");          // two of three bytes
  CHECK(utf8_lossy_final("ok \xE4") == "ok ");              // one byte of three
  CHECK(utf8_lossy_final("ok \xF0\x9F\x98") == "ok ");      // three of four
  CHECK(utf8_lossy_final("ok \xF0\x9F") == "ok ");          // two of four
  CHECK(utf8_lossy_final("\xE4\xBD") == "");                // nothing else at all
  CHECK(utf8_lossy_final("\xC3") == "");                    // a two-byte lead alone
  CHECK(utf8_lossy_final("\xE4\xBD\xA0") == "\xE4\xBD\xA0"); // a complete character is kept
  CHECK(utf8_lossy_final("\xF0\x9F\x98\x80") == "\xF0\x9F\x98\x80");
}

TEST_CASE("a finished reply ends exactly as utf8_lossy would") {
  for (const char* text : {"", "plain", "\xE4\xBD\xA0\xE5\xA5\xBD", "mixed \xF0\x9F\x98\x80 text"}) {
    CHECK(utf8_lossy_final(text) == utf8_lossy(text));
    CHECK(utf8_incomplete_tail(text) == 0);
  }
}

TEST_CASE("invalid bytes before the end keep their replacement") {
  // A lead byte that never starts a character.
  CHECK(utf8_lossy_final("\xFF") == kReplacement);
  // E4 BD followed by a plain letter: an invalid subpart in the middle, replaced once, then the letter.
  CHECK(utf8_lossy_final("\xE4\xBD" "A") == std::string(kReplacement) + "A");
  // E0 must be followed by A0..BF, so E0 80 is two invalid subparts at the end, not an unfinished character.
  CHECK(utf8_lossy_final("\xE0\x80") == std::string(kReplacement) + kReplacement);
  // An unfinished character after a replaced one: the replacement stays, the unfinished bytes go.
  CHECK(utf8_lossy_final("\xFF" "\xE4\xBD") == kReplacement);
}

TEST_CASE("a stray continuation byte at the end is replaced, not dropped") {
  // 0x80 can never begin a character, so it is not an unfinished one.
  CHECK(utf8_lossy_final("ab\x80") == std::string("ab") + kReplacement);
}
