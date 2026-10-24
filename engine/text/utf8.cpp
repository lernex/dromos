#include "text/utf8.h"

namespace lx::text {
namespace {

// The bytes a valid sequence may carry at each position, as WHATWG defines them: the lead byte
// fixes the length, and the second byte's range is narrower for E0, ED, F0 and F4.
struct Lead {
  int extra;            // continuation bytes after the lead
  uint8_t lo, hi;       // range of the first continuation byte
};

bool lead_info(uint8_t b, Lead& out) {
  if (b >= 0xC2 && b <= 0xDF) { out = {1, 0x80, 0xBF}; return true; }
  if (b == 0xE0) { out = {2, 0xA0, 0xBF}; return true; }
  if (b >= 0xE1 && b <= 0xEC) { out = {2, 0x80, 0xBF}; return true; }
  if (b == 0xED) { out = {2, 0x80, 0x9F}; return true; }
  if (b >= 0xEE && b <= 0xEF) { out = {2, 0x80, 0xBF}; return true; }
  if (b == 0xF0) { out = {3, 0x90, 0xBF}; return true; }
  if (b >= 0xF1 && b <= 0xF3) { out = {3, 0x80, 0xBF}; return true; }
  if (b == 0xF4) { out = {3, 0x80, 0x8F}; return true; }
  return false;
}

}  // namespace

uint32_t utf8_decode(std::string_view s, size_t i, size_t& len) {
  const auto b0 = static_cast<uint8_t>(s[i]);
  if (b0 < 0x80) {
    len = 1;
    return b0;
  }
  // A sequence cut off by the end of s (a chunk boundary, in streaming) is one replacement
  // character over the bytes that are there, so nothing past the end is read.
  const size_t need = b0 < 0xE0 ? 2 : b0 < 0xF0 ? 3 : 4;
  if (i + need > s.size()) {
    len = s.size() - i;
    return 0xFFFD;
  }
  if (need == 2) {
    len = 2;
    return (uint32_t(b0 & 0x1F) << 6) | (uint8_t(s[i + 1]) & 0x3F);
  }
  if (need == 3) {
    len = 3;
    return (uint32_t(b0 & 0x0F) << 12) | ((uint8_t(s[i + 1]) & 0x3F) << 6) | (uint8_t(s[i + 2]) & 0x3F);
  }
  len = 4;
  return (uint32_t(b0 & 0x07) << 18) | ((uint8_t(s[i + 1]) & 0x3F) << 12) |
         ((uint8_t(s[i + 2]) & 0x3F) << 6) | (uint8_t(s[i + 3]) & 0x3F);
}

void utf8_append(std::string& out, uint32_t cp) {
  if (cp < 0x80) {
    out.push_back(static_cast<char>(cp));
  } else if (cp < 0x800) {
    out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else if (cp < 0x10000) {
    out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  } else {
    out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
    out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
  }
}

bool utf8_valid(std::string_view s) {
  size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) {
      ++i;
      continue;
    }
    Lead lead;
    if (!lead_info(b0, lead)) return false;
    if (i + 1 + lead.extra > s.size()) return false;
    for (int k = 0; k < lead.extra; ++k) {
      const auto b = static_cast<uint8_t>(s[i + 1 + k]);
      const uint8_t lo = k == 0 ? lead.lo : 0x80;
      const uint8_t hi = k == 0 ? lead.hi : 0xBF;
      if (b < lo || b > hi) return false;
    }
    i += 1 + lead.extra;
  }
  return true;
}

std::string utf8_lossy(std::string_view s) {
  std::string out;
  out.reserve(s.size());
  size_t i = 0;
  while (i < s.size()) {
    const auto b0 = static_cast<uint8_t>(s[i]);
    if (b0 < 0x80) {
      out.push_back(static_cast<char>(b0));
      ++i;
      continue;
    }
    Lead lead;
    if (!lead_info(b0, lead)) {
      out += "\xEF\xBF\xBD";
      ++i;
      continue;
    }
    // Consume the continuation bytes that are valid; the first bad one (or the end) ends the
    // subpart, and it is not consumed.
    size_t j = i + 1;
    bool ok = true;
    for (int k = 0; k < lead.extra; ++k) {
      if (j >= s.size()) { ok = false; break; }
      const auto b = static_cast<uint8_t>(s[j]);
      const uint8_t lo = k == 0 ? lead.lo : 0x80;
      const uint8_t hi = k == 0 ? lead.hi : 0xBF;
      if (b < lo || b > hi) { ok = false; break; }
      ++j;
    }
    if (ok) {
      out.append(s.substr(i, j - i));
    } else {
      out += "\xEF\xBF\xBD";
    }
    i = j;
  }
  return out;
}

size_t utf8_incomplete_tail(std::string_view s) {
  const size_t n = s.size();
  for (size_t back = 1; back <= 3 && back <= n; ++back) {
    const auto b = static_cast<uint8_t>(s[n - back]);
    if ((b & 0xC0) == 0x80) continue;  // continuation byte: look further back for its lead
    if (b < 0x80) return 0;
    Lead lead;
    if (!lead_info(b, lead)) return 0;
    const auto have = static_cast<int>(back);  // bytes of this sequence present so far, lead included
    if (have >= lead.extra + 1) return 0;     // complete (or the lead can never be completed)
    // The bytes present must be a valid prefix of the sequence.
    for (int k = 0; k < have - 1; ++k) {
      const auto c = static_cast<uint8_t>(s[n - back + 1 + k]);
      const uint8_t lo = k == 0 ? lead.lo : 0x80;
      const uint8_t hi = k == 0 ? lead.hi : 0xBF;
      if (c < lo || c > hi) return 0;
    }
    return have;
  }
  return 0;
}

}  // namespace lx::text
