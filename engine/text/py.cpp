#include "text/py.h"

#include <cmath>
#include <cstdlib>

#include "text/unicode.h"
#include "text/utf8.h"

namespace lx::text {
namespace {

// The text with every Unicode decimal digit written as its ASCII digit. False when something
// else (a letter, a symbol) is in it, which int() and float() would reject anyway.
bool to_ascii_digits(std::string_view s, std::string& out) {
  out.clear();
  size_t i = 0;
  while (i < s.size()) {
    const auto b = static_cast<unsigned char>(s[i]);
    if (b < 0x80) {
      out.push_back(static_cast<char>(b));
      ++i;
      continue;
    }
    size_t len = 0;
    const uint32_t cp = utf8_decode(s, i, len);
    const int digit = uni::decimal_digit(cp);
    if (digit < 0) return false;
    out.push_back(static_cast<char>('0' + digit));
    i += len;
  }
  return true;
}

// Reads digits with single underscores between them. Returns the digit count (0 when none).
size_t read_digits(const std::string& s, size_t& i, std::string& digits) {
  size_t count = 0;
  while (i < s.size()) {
    const char c = s[i];
    if (c >= '0' && c <= '9') {
      digits.push_back(c);
      ++count;
      ++i;
      continue;
    }
    // An underscore is allowed only between two digits.
    if (c == '_' && count > 0 && i + 1 < s.size() && s[i + 1] >= '0' && s[i + 1] <= '9') {
      ++i;
      continue;
    }
    break;
  }
  return count;
}

}  // namespace

std::string_view py_strip(std::string_view s) {
  size_t begin = 0;
  while (begin < s.size()) {
    size_t len = 0;
    const uint32_t cp = utf8_decode(s, begin, len);
    if (!uni::is_py_space(cp)) break;
    begin += len;
  }
  size_t end = s.size();
  while (end > begin) {
    // Step back over one code point: continuation bytes are 10xxxxxx.
    size_t start = end - 1;
    while (start > begin && (static_cast<unsigned char>(s[start]) & 0xC0) == 0x80) --start;
    size_t len = 0;
    const uint32_t cp = utf8_decode(s, start, len);
    if (!uni::is_py_space(cp)) break;
    end = start;
  }
  return s.substr(begin, end - begin);
}

size_t py_last_nonspace_end(std::string_view s) {
  size_t end = 0;
  size_t i = 0;
  while (i < s.size()) {
    size_t len = 0;
    const uint32_t cp = utf8_decode(s, i, len);
    i += len;
    if (!uni::is_py_space(cp)) end = i;
  }
  return end;
}

std::string py_lower_ascii(std::string_view s) {
  std::string out(s);
  for (char& c : out) {
    if (c >= 'A' && c <= 'Z') c = static_cast<char>(c - 'A' + 'a');
  }
  return out;
}

bool py_int(std::string_view s, std::string& canonical) {
  std::string text;
  if (!to_ascii_digits(py_strip(s), text)) return false;
  size_t i = 0;
  bool negative = false;
  if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
    negative = text[i] == '-';
    ++i;
  }
  std::string digits;
  if (read_digits(text, i, digits) == 0 || i != text.size()) return false;
  size_t first = 0;
  while (first + 1 < digits.size() && digits[first] == '0') ++first;
  canonical = digits.substr(first);
  if (canonical == "0") {
    negative = false;
  }
  if (negative) canonical.insert(canonical.begin(), '-');
  return true;
}

bool py_float(std::string_view s, double& out) {
  std::string text;
  if (!to_ascii_digits(py_strip(s), text)) return false;
  size_t i = 0;
  bool negative = false;
  if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
    negative = text[i] == '-';
    ++i;
  }
  // Infinities and NaN, in any case.
  const std::string word = py_lower_ascii(std::string_view(text).substr(i));
  if (word == "inf" || word == "infinity") {
    out = negative ? -HUGE_VAL : HUGE_VAL;
    return true;
  }
  if (word == "nan") {
    out = NAN;
    return true;
  }
  std::string mantissa;
  const size_t int_digits = read_digits(text, i, mantissa);
  size_t frac_digits = 0;
  if (i < text.size() && text[i] == '.') {
    ++i;
    mantissa.push_back('.');
    frac_digits = read_digits(text, i, mantissa);
  }
  if (int_digits == 0 && frac_digits == 0) return false;
  if (i < text.size() && (text[i] == 'e' || text[i] == 'E')) {
    ++i;
    mantissa.push_back('e');
    if (i < text.size() && (text[i] == '+' || text[i] == '-')) {
      mantissa.push_back(text[i]);
      ++i;
    }
    if (read_digits(text, i, mantissa) == 0) return false;
  }
  if (i != text.size()) return false;
  out = std::strtod(mantissa.c_str(), nullptr);
  if (negative) out = -out;
  return true;
}

}  // namespace lx::text
