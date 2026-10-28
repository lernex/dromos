// UTF-8 helpers for the text stack. Replacement follows the WHATWG and Rust (from_utf8_lossy)
// rules that Python's errors="replace" and the tokenizers crate share: one U+FFFD per maximal
// invalid subpart.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>

namespace lx::text {

// Decodes the code point starting at s[i]. The text must be valid UTF-8; len receives its size.
uint32_t utf8_decode(std::string_view s, size_t i, size_t& len);

// Appends a code point. Surrogates are written as three bytes (WTF-8), which is what a lone
// surrogate from a JSON escape looks like in memory.
void utf8_append(std::string& out, uint32_t cp);

bool utf8_valid(std::string_view s);

// Valid UTF-8 with every invalid subpart replaced by U+FFFD.
std::string utf8_lossy(std::string_view s);

// utf8_lossy for the end of a reply: an unfinished character at the very end is dropped, not replaced. The reply
// stopped inside it (max_tokens, the context, a stop), so no U+FFFD belongs there. Everything before it is utf8_lossy.
std::string utf8_lossy_final(std::string_view s);

// Length of the trailing bytes that a later byte could still complete into one code point (at
// most three). Bytes before them are final: decoding them does not depend on what follows.
size_t utf8_incomplete_tail(std::string_view s);

}  // namespace lx::text
