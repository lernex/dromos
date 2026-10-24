#include "text/unicode.h"

#include <cstddef>

namespace lx::text::uni {
namespace {

struct Range {
  uint32_t lo, hi;
};

struct DigitRun {
  uint32_t lo, hi;
  uint32_t zero_value;  // the digit value of lo; the run rises by one with the code point
};

#include "unicode_tables.inc"

template <size_t N>
bool in_ranges(const Range (&table)[N], uint32_t cp) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < table[mid].lo) {
      hi = mid;
    } else if (cp > table[mid].hi) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

}  // namespace

bool is_letter(uint32_t cp) {
  if (cp < 0x80) return (cp | 0x20) >= 'a' && (cp | 0x20) <= 'z';
  return in_ranges(kLetter, cp);
}

bool is_number(uint32_t cp) {
  if (cp < 0x80) return cp >= '0' && cp <= '9';
  return in_ranges(kNumber, cp);
}

bool is_regex_space(uint32_t cp) {
  if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D);
  return in_ranges(kRegexSpace, cp);
}

bool is_py_space(uint32_t cp) {
  if (cp < 0x80) return cp == ' ' || (cp >= 0x09 && cp <= 0x0D) || (cp >= 0x1C && cp <= 0x1F);
  return in_ranges(kPySpace, cp);
}

int decimal_digit(uint32_t cp) {
  if (cp < 0x80) return (cp >= '0' && cp <= '9') ? int(cp - '0') : -1;
  size_t lo = 0, hi = sizeof(kDecimal) / sizeof(kDecimal[0]);
  while (lo < hi) {
    const size_t mid = (lo + hi) / 2;
    if (cp < kDecimal[mid].lo) {
      hi = mid;
    } else if (cp > kDecimal[mid].hi) {
      lo = mid + 1;
    } else {
      return int(kDecimal[mid].zero_value + (cp - kDecimal[mid].lo));
    }
  }
  return -1;
}

}  // namespace lx::text::uni
