#include "text/pretokenize.h"

#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text {
namespace {

enum class Cls { kSpace, kLetter, kNumber, kOther };

struct Scanner {
  std::string_view s;

  Cls cls_at(size_t i, size_t& len) const {
    const uint32_t cp = utf8_decode(s, i, len);
    if (uni::is_regex_space(cp)) return Cls::kSpace;
    if (uni::is_letter(cp)) return Cls::kLetter;
    if (uni::is_number(cp)) return Cls::kNumber;
    return Cls::kOther;
  }

  // End of the run of characters of class c that starts at i (i itself is of class c).
  size_t run_end(size_t i, Cls c) const {
    size_t len = 0;
    while (i < s.size() && cls_at(i, len) == c) i += len;
    return i;
  }

  // The match length of the pattern at i. Every character matches something, so it is never 0.
  size_t match_at(size_t i) const {
    size_t len = 0;
    // 's|'t|'re|'ve|'m|'ll|'d
    if (s[i] == '\'' && i + 1 < s.size()) {
      const char a = s[i + 1];
      if (a == 's' || a == 't' || a == 'm' || a == 'd') return 2;
      if (i + 2 < s.size()) {
        const char b = s[i + 2];
        if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) return 3;
      }
    }
    // ` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`: an optional single space, then a run of one class.
    const bool space = s[i] == ' ';
    const size_t j = space ? i + 1 : i;
    if (j < s.size()) {
      const Cls c = cls_at(j, len);
      if (c == Cls::kLetter || c == Cls::kNumber || c == Cls::kOther) {
        return run_end(j, c) - i;
      }
    }
    // `\s+(?!\S)` then `\s+`: a run of whitespace, minus its last character when text follows it.
    if (cls_at(i, len) == Cls::kSpace) {
      size_t end = i;
      size_t last_len = 0;
      size_t count = 0;
      while (end < s.size()) {
        size_t l = 0;
        if (cls_at(end, l) != Cls::kSpace) break;
        last_len = l;
        end += l;
        ++count;
      }
      if (end == s.size()) return end - i;
      if (count >= 2) return end - last_len - i;
      return end - i;
    }
    return len;  // unreachable for valid input: a lone space is handled above
  }
};

}  // namespace

void pretokenize(std::string_view s, std::vector<std::pair<uint32_t, uint32_t>>& pieces) {
  Scanner scanner{s};
  size_t i = 0;
  while (i < s.size()) {
    const size_t n = scanner.match_at(i);  // at least one character: every character matches something
    pieces.emplace_back(static_cast<uint32_t>(i), static_cast<uint32_t>(i + n));
    i += n;
  }
}

}  // namespace lx::text
