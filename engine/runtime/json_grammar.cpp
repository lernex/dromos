#include "runtime/json_grammar.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <unordered_map>

namespace lx::runtime {

namespace jx {

using lx::text::JValue;

enum class NK : uint8_t { kObject, kArray, kString, kNumber, kInteger, kLiteral, kUnion };

// JSON kinds a node can start with, one bit each.
constexpr uint8_t kTObject = 1, kTArray = 2, kTString = 4, kTNumber = 8, kTBool = 16, kTNull = 32;

constexpr size_t kMaxProps = 64;
constexpr size_t kMaxLiterals = 64;
constexpr size_t kMaxNodes = 16384;
constexpr uint32_t kMaxCount = 65535;
constexpr uint32_t kMaxNumberBytes = 64;   // a number is at most this many bytes (its value stays finite)
constexpr size_t kMaxConfigs = 256;        // stacks a single output may be in at once

// One schema node: what a value of that schema can be, and what each kind of container writes.
struct Node {
  NK kind = NK::kUnion;
  uint8_t types = 0;  // JSON kinds the node can start with
  // object: closed (keys, props, required) or open (any string keys, values of `value`)
  bool open = false;
  int value = -1;
  std::vector<std::string> keys;  // escaped key text without quotes, in schema order
  std::vector<int> props;         // value node of each key
  uint64_t required = 0;          // bit i: key i must appear
  uint32_t last_required1 = 0;    // 1 + the highest required key index (0 when none)
  // array
  std::vector<int> prefix;
  int items = -1;                 // node past the prefix; -1 when nothing may follow the prefix
  uint32_t min_items = 0;
  int64_t max_items = -1;         // -1: unbounded
  // literal: each value's complete JSON text, written compactly
  std::vector<std::string> literals;
  // union: any one of the alternatives
  std::vector<int> alts;
};

// Frame kinds. A frame is one open container or scalar; the stack's top is the innermost.
enum Kind : uint8_t {
  kWrap,   // the whole reply: waits for its value (kWait), then is complete (kDone)
  kThink,  // the reasoning before the reply: any bytes up to "</think>"
  kObj,
  kArr,
  kStr,
  kNum,
  kLit,
  kPop,    // marker: a frame below the local base was completed by a byte (see advance_local)
};
enum : uint8_t { kWait = 0, kDone = 1 };
enum : uint8_t { kOOpen, kOFirst, kONext, kONext2, kOKey, kOKeyEnd, kOColon, kOColon2, kOAfter };
enum : uint8_t { kAOpen, kAFirst, kANext, kANext2, kAAfter };
enum : uint8_t { kSOpen, kSBody, kSEsc, kSHex, kSUtf };
// number states (aux of a kNum frame)
enum : uint8_t { kNone, kNStart, kNNeg, kNZero, kNInt, kNDot, kNFrac, kNE, kNESign, kNExp1, kNExp2 };

struct Frame {
  uint16_t node = 0;
  uint8_t kind = kWrap;
  uint8_t phase = 0;
  uint16_t n = 0;     // object: 1 + the last key's index; array: elements so far; literal: bytes matched; think: tail length
  uint32_t aux = 0;   // object: key bytes matched; number: number state; string: utf-8 / escape state
  uint64_t mask = 0;  // object: candidate keys; literal: candidate literals; think: tail bytes
  bool operator==(const Frame& o) const {
    return node == o.node && kind == o.kind && phase == o.phase && n == o.n && aux == o.aux && mask == o.mask;
  }
};

using Stack = std::vector<Frame>;

// Appends a frame's state, as a fixed 18-byte key.
void put(std::string& out, const Frame& f) {
  char b[18];
  std::memcpy(b, &f.node, 2);
  b[2] = static_cast<char>(f.kind);
  b[3] = static_cast<char>(f.phase);
  std::memcpy(b + 4, &f.n, 2);
  std::memcpy(b + 6, &f.aux, 4);
  std::memcpy(b + 10, &f.mask, 8);
  out.append(b, sizeof b);
}

uint64_t low_bits(size_t k) { return k >= 64 ? ~0ull : (1ull << k) - 1; }

bool is_hex(uint8_t c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F'); }

// The number reading state after byte c, or 0 when c does not continue the number.
uint8_t number_next(uint8_t n, uint8_t c, bool integer) {
  const bool digit = c >= '0' && c <= '9';
  switch (n) {
    case kNStart:
      if (c == '-') return kNNeg;
      if (c == '0') return kNZero;
      if (digit) return kNInt;
      return 0;
    case kNNeg:
      if (c == '0') return kNZero;
      if (digit) return kNInt;
      return 0;
    case kNZero:
    case kNInt:
      if (digit && n == kNInt) return kNInt;
      if (integer) return 0;
      if (c == '.') return kNDot;
      if (c == 'e' || c == 'E') return kNE;
      return 0;
    case kNDot:
      return digit ? kNFrac : 0;
    case kNFrac:
      if (digit) return kNFrac;
      if (c == 'e' || c == 'E') return kNE;
      return 0;
    case kNE:
      if (c == '+' || c == '-') return kNESign;
      return digit ? kNExp1 : 0;
    case kNESign:
      return digit ? kNExp1 : 0;
    case kNExp1:
      return digit ? kNExp2 : 0;  // at most two exponent digits: the value stays finite
    default:
      return 0;
  }
}

Stack below(const Stack& s) { return Stack(s.begin(), s.end() - 1); }

bool is_marker(const Stack& s) { return s.size() == 1 && s[0].kind == kPop; }

// Whether a value can end (its closing byte is not part of it) and whether an object or array may close.
bool may_close(const Node& nd, uint32_t n) { return nd.open || n >= nd.last_required1; }
bool can_more(const Node& nd, uint32_t n) { return nd.open || n < nd.keys.size(); }

// The keys that may come next: indices from n (one past the last key written) up to the first required key
// at or after n. Writing a later key would skip a required one.
uint64_t range_bits(const Node& nd, uint32_t n) {
  const size_t count = nd.keys.size();
  if (n >= count) return 0;
  const uint64_t rest = nd.required & ~low_bits(n);
  const size_t last = rest ? static_cast<size_t>(__builtin_ctzll(rest)) : count - 1;
  return low_bits(last + 1) & ~low_bits(n);
}

bool can_add(const Node& nd, uint32_t n) {
  const bool has_item = n < nd.prefix.size() || nd.items >= 0;
  return has_item && (nd.max_items < 0 || static_cast<int64_t>(n) < nd.max_items);
}

// The count after one more element. Past the prefix and the minimum, the count only matters as "more
// than that", so it saturates there: the state, and the mask cached for it, stay the same.
uint32_t bump(const Node& nd, uint32_t n) {
  const uint32_t cap = nd.max_items < 0
                           ? std::max<uint32_t>(static_cast<uint32_t>(nd.prefix.size()), nd.min_items)
                           : static_cast<uint32_t>(nd.max_items);
  return std::min(n + 1, cap);
}

// The byte machine. Feeds one byte to a stack and appends every stack it leaves (an empty stack is
// the whole reply complete). With `local`, a byte that completes the stack's bottom frame, and so
// needs the stack below it, yields a kPop marker instead of nothing.
class Machine {
 public:
  explicit Machine(const std::vector<Node>& nodes) : nodes_(nodes) {}

  void feed(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    if (s.empty()) return;
    consume(s, c, out, local);
    if (accepting(s.back())) {
      Stack rest = below(s);
      if (rest.empty()) {
        if (local) out.push_back(Stack{Frame{0, kPop, 0, 0, 0, 0}});
      } else {
        feed(rest, c, out, local);
      }
    }
  }

  // Whether the frame can be complete when the next byte is not its own.
  bool accepting(const Frame& f) const {
    switch (f.kind) {
      case kNum:
        return f.aux == kNZero || f.aux == kNInt || f.aux == kNFrac || f.aux == kNExp1 || f.aux == kNExp2;
      case kLit: {
        const auto& lits = nodes_[f.node].literals;
        for (size_t i = 0; i < lits.size(); ++i)
          if ((f.mask >> i & 1) && lits[i].size() == f.n) return true;
        return false;
      }
      case kWrap:
        return f.phase == kDone;
      default:
        return false;
    }
  }

  Frame initial(int node) const {
    const Node& nd = nodes_[static_cast<size_t>(node)];
    Frame f;
    f.node = static_cast<uint16_t>(node);
    switch (nd.kind) {
      case NK::kObject: f.kind = kObj; f.phase = kOOpen; break;
      case NK::kArray: f.kind = kArr; f.phase = kAOpen; break;
      case NK::kString: f.kind = kStr; f.phase = kSOpen; break;
      case NK::kNumber:
      case NK::kInteger: f.kind = kNum; f.aux = kNStart; break;
      case NK::kLiteral: f.kind = kLit; f.mask = low_bits(nd.literals.size()); break;
      case NK::kUnion: break;  // never a frame: start_value expands it
    }
    return f;
  }

 private:
  void consume(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& top = s.back();
    switch (top.kind) {
      case kWrap:
        if (top.phase == kWait) {
          Stack ns = s;
          ns.back().phase = kDone;
          start_value(ns, top.node, c, out, local);
        }
        return;
      case kThink:
        consume_think(s, c, out);
        return;
      case kObj:
        consume_obj(s, c, out, local);
        return;
      case kArr:
        consume_arr(s, c, out, local);
        return;
      case kStr:
        consume_str(s, c, out);
        return;
      case kNum:
        consume_num(s, c, out);
        return;
      case kLit:
        consume_lit(s, c, out);
        return;
      default:
        return;
    }
  }

  // A value of `node` starts with byte c. `parent` is the stack below the value (already advanced past it).
  void start_value(const Stack& parent, int node, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Node& nd = nodes_[static_cast<size_t>(node)];
    if (nd.kind == NK::kUnion) {
      for (int alt : nd.alts) start_value(parent, alt, c, out, local);
      return;
    }
    Stack t = parent;
    t.push_back(initial(node));
    feed(t, c, out, local);
  }

  void consume_think(const Stack& s, uint8_t c, std::vector<Stack>& out) const {
    static const std::string kEnd = "</think>";
    const Frame& f = s.back();
    std::string cand;
    for (uint32_t i = 0; i < f.n; ++i) cand.push_back(static_cast<char>((f.mask >> (8 * i)) & 0xff));
    cand.push_back(static_cast<char>(c));
    if (cand == kEnd) {
      out.push_back(below(s));
      return;
    }
    size_t best = 0;
    for (size_t k = std::min<size_t>(7, cand.size()); k > 0; --k)
      if (kEnd.compare(0, k, cand, cand.size() - k, k) == 0) {
        best = k;
        break;
      }
    Stack ns = s;
    Frame& g = ns.back();
    g.n = static_cast<uint16_t>(best);
    g.mask = 0;
    for (size_t i = 0; i < best; ++i)
      g.mask |= static_cast<uint64_t>(static_cast<uint8_t>(cand[cand.size() - best + i])) << (8 * i);
    out.push_back(std::move(ns));
  }

  void consume_obj(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& f = s.back();
    const Node& nd = nodes_[f.node];
    const size_t count = nd.keys.size();
    switch (f.phase) {
      case kOOpen:
        if (c == '{') {
          Stack ns = s;
          ns.back().phase = kOFirst;
          out.push_back(std::move(ns));
        }
        return;
      case kOFirst:
        if (c == '}' && may_close(nd, f.n)) {
          out.push_back(below(s));
          return;
        }
        if (c == '"') key_start(s, c, out, local);
        return;
      case kONext:
        if (c == ' ') {
          Stack ns = s;
          ns.back().phase = kONext2;
          out.push_back(std::move(ns));
          return;
        }
        if (c == '"') key_start(s, c, out, local);
        return;
      case kONext2:
        if (c == '"') key_start(s, c, out, local);
        return;
      case kOKey: {
        if (c == '"') {
          uint64_t sel = 0;
          for (size_t i = 0; i < count; ++i)
            if ((f.mask >> i & 1) && nd.keys[i].size() == f.aux) sel |= 1ull << i;
          if (sel == 0 || (sel & (sel - 1))) return;  // no key, or two that cannot both be complete here
          Stack ns = s;
          Frame& g = ns.back();
          g.phase = kOKeyEnd;
          g.n = static_cast<uint16_t>(__builtin_ctzll(sel) + 1);
          g.aux = 0;
          g.mask = 0;
          out.push_back(std::move(ns));
          return;
        }
        uint64_t next = 0;
        for (size_t i = 0; i < count; ++i)
          if ((f.mask >> i & 1) && nd.keys[i].size() > f.aux && static_cast<uint8_t>(nd.keys[i][f.aux]) == c)
            next |= 1ull << i;
        if (!next) return;
        Stack ns = s;
        ns.back().aux = f.aux + 1;
        ns.back().mask = next;
        out.push_back(std::move(ns));
        return;
      }
      case kOKeyEnd:
        if (c == ':') {
          Stack ns = s;
          ns.back().phase = kOColon;
          out.push_back(std::move(ns));
        }
        return;
      case kOColon:
        if (c == ' ') {
          Stack ns = s;
          ns.back().phase = kOColon2;
          out.push_back(std::move(ns));
          return;
        }
        object_value(s, c, out, local);
        return;
      case kOColon2:
        object_value(s, c, out, local);
        return;
      case kOAfter:
        if (c == ',' && can_more(nd, f.n)) {
          Stack ns = s;
          ns.back().phase = kONext;
          out.push_back(std::move(ns));
          return;
        }
        if (c == '}' && may_close(nd, f.n)) out.push_back(below(s));
        return;
      default:
        return;
    }
  }

  // A key starts with the quote c. Closed objects match the key bytes against the candidates; open ones read
  // a string frame for the key, whose completion returns here expecting ':'.
  void key_start(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& f = s.back();
    const Node& nd = nodes_[f.node];
    if (nd.open) {
      Stack ns = s;
      ns.back().phase = kOKeyEnd;
      Frame str;
      str.node = f.node;
      str.kind = kStr;
      str.phase = kSOpen;
      ns.push_back(str);
      feed(ns, c, out, local);
      return;
    }
    const uint64_t range = range_bits(nd, f.n);
    if (!range) return;
    Stack ns = s;
    Frame& g = ns.back();
    g.phase = kOKey;
    g.aux = 0;
    g.mask = range;
    out.push_back(std::move(ns));
  }

  // The value of the object's current key starts with c.
  void object_value(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& f = s.back();
    const Node& nd = nodes_[f.node];
    const int value = nd.open ? nd.value : nd.props[f.n - 1];
    Stack ns = s;
    ns.back().phase = kOAfter;
    start_value(ns, value, c, out, local);
  }

  void consume_arr(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& f = s.back();
    const Node& nd = nodes_[f.node];
    switch (f.phase) {
      case kAOpen:
        if (c == '[') {
          Stack ns = s;
          ns.back().phase = kAFirst;
          out.push_back(std::move(ns));
        }
        return;
      case kAFirst:
        if (c == ']' && f.n >= nd.min_items) {
          out.push_back(below(s));
          return;
        }
        array_element(s, c, out, local);
        return;
      case kANext:
        if (c == ' ') {
          Stack ns = s;
          ns.back().phase = kANext2;
          out.push_back(std::move(ns));
          return;
        }
        array_element(s, c, out, local);
        return;
      case kANext2:
        array_element(s, c, out, local);
        return;
      case kAAfter:
        if (c == ',' && can_add(nd, f.n)) {
          Stack ns = s;
          ns.back().phase = kANext;
          out.push_back(std::move(ns));
          return;
        }
        if (c == ']' && f.n >= nd.min_items) out.push_back(below(s));
        return;
      default:
        return;
    }
  }

  void array_element(const Stack& s, uint8_t c, std::vector<Stack>& out, bool local) const {
    const Frame& f = s.back();
    const Node& nd = nodes_[f.node];
    if (!can_add(nd, f.n)) return;
    const int node = f.n < nd.prefix.size() ? nd.prefix[f.n] : nd.items;
    Stack ns = s;
    ns.back().phase = kAAfter;
    ns.back().n = static_cast<uint16_t>(bump(nd, f.n));
    start_value(ns, node, c, out, local);
  }

  // The string body. Its closing quote completes the frame, which leaves the stack at once.
  void consume_str(const Stack& s, uint8_t c, std::vector<Stack>& out) const {
    const Frame& f = s.back();
    auto with_phase = [&](uint8_t phase, uint32_t aux) {
      Stack ns = s;
      ns.back().phase = phase;
      ns.back().aux = aux;
      out.push_back(std::move(ns));
    };
    switch (f.phase) {
      case kSOpen:
        if (c == '"') with_phase(kSBody, 0);
        return;
      case kSBody: {
        if (c == '"') {
          out.push_back(below(s));
          return;
        }
        if (c == '\\') {
          with_phase(kSEsc, 0);
          return;
        }
        if (c < 0x20) return;
        if (c < 0x80) {
          out.push_back(s);
          return;
        }
        uint32_t rem = 0, lo = 0x80, hi = 0xBF;  // continuation bytes still owed, and the first one's range
        if (c >= 0xC2 && c <= 0xDF) rem = 1;
        else if (c == 0xE0) { rem = 2; lo = 0xA0; }
        else if (c >= 0xE1 && c <= 0xEC) rem = 2;
        else if (c == 0xED) { rem = 2; hi = 0x9F; }
        else if (c >= 0xEE && c <= 0xEF) rem = 2;
        else if (c == 0xF0) { rem = 3; lo = 0x90; }
        else if (c >= 0xF1 && c <= 0xF3) rem = 3;
        else if (c == 0xF4) { rem = 3; hi = 0x8F; }
        else return;  // 0x80..0xC1, 0xF5..0xFF cannot start a character
        with_phase(kSUtf, rem | (lo << 8) | (hi << 16));
        return;
      }
      case kSEsc:
        if (c == 'u') {
          with_phase(kSHex, 4);
          return;
        }
        switch (c) {
          case '"': case '\\': case '/': case 'b': case 'f': case 'n': case 'r': case 't':
            with_phase(kSBody, 0);
            return;
          default:
            return;
        }
      case kSHex:
        if (!is_hex(c)) return;
        if (f.aux == 1) with_phase(kSBody, 0);
        else with_phase(kSHex, f.aux - 1);
        return;
      case kSUtf: {
        const uint32_t rem = f.aux & 0xff, lo = (f.aux >> 8) & 0xff, hi = (f.aux >> 16) & 0xff;
        if (c < lo || c > hi) return;
        if (rem == 1) with_phase(kSBody, 0);
        else with_phase(kSUtf, (rem - 1) | (0x80u << 8) | (0xBFu << 16));
        return;
      }
      default:
        return;
    }
  }

  void consume_num(const Stack& s, uint8_t c, std::vector<Stack>& out) const {
    const Frame& f = s.back();
    if (f.n >= kMaxNumberBytes) return;
    const bool integer = nodes_[f.node].kind == NK::kInteger;
    const uint8_t next = number_next(static_cast<uint8_t>(f.aux), c, integer);
    if (!next) return;
    Stack ns = s;
    ns.back().aux = next;
    ns.back().n = static_cast<uint16_t>(f.n + 1);
    out.push_back(std::move(ns));
  }

  void consume_lit(const Stack& s, uint8_t c, std::vector<Stack>& out) const {
    const Frame& f = s.back();
    const auto& lits = nodes_[f.node].literals;
    uint64_t next = 0;
    for (size_t i = 0; i < lits.size(); ++i)
      if ((f.mask >> i & 1) && lits[i].size() > f.n && static_cast<uint8_t>(lits[i][f.n]) == c) next |= 1ull << i;
    if (!next) return;
    Stack ns = s;
    ns.back().n = static_cast<uint16_t>(f.n + 1);
    ns.back().mask = next;
    out.push_back(std::move(ns));
  }

  const std::vector<Node>& nodes_;
};

// Sets the bits of `ids` in a mask.
void mark(std::vector<uint32_t>& bits, const std::vector<int32_t>& ids) {
  for (int32_t id : ids) bits[static_cast<size_t>(id) / 32] |= 1u << (static_cast<size_t>(id) % 32);
}

void or_into(std::vector<uint32_t>& out, const std::vector<uint32_t>& in) {
  for (size_t i = 0; i < out.size(); ++i) out[i] |= in[i];
}

void dedupe(std::vector<Stack>& v) {
  std::vector<Stack> keep;
  keep.reserve(v.size());
  for (Stack& s : v) {
    if (std::find(keep.begin(), keep.end(), s) == keep.end()) keep.push_back(std::move(s));
  }
  v.swap(keep);
}

// Every stack the whole output can be in after byte b.
std::vector<Stack> advance_full(const Machine& m, const std::vector<Stack>& S, uint8_t b) {
  std::vector<Stack> out;
  for (const Stack& s : S)
    if (!s.empty()) m.feed(s, b, out, false);
  dedupe(out);
  return out;
}

// The local stacks after byte b. A stack that completes the base (A: the base was completed by this byte)
// or needs the stack below it (B: the base's pop hands byte b to the parent) is reported, not followed.
void advance_local(const Machine& m, const std::vector<Stack>& S, uint8_t b, std::vector<Stack>& next, bool& A,
                   bool& B) {
  std::vector<Stack> tmp;
  for (const Stack& s : S) {
    tmp.clear();
    m.feed(s, b, tmp, true);
    for (Stack& r : tmp) {
      if (r.empty()) A = true;
      else if (is_marker(r)) B = true;
      else next.push_back(std::move(r));
    }
  }
  dedupe(next);
}

// What one frame state's own transitions do, for every token: accepted outright (its bytes never leave the
// frame), or undecided (it leaves the frame at some byte, so the rest of the token needs the real stack).
struct Entry {
  std::vector<uint32_t> accept;   // tokens that are allowed whatever lies below the frame
  std::vector<int32_t> frontier;  // trie nodes where the frame is left: the tokens below them are checked per stack
};

void local_dfs(const VocabIndex& v, const Machine& m, int32_t node, const std::vector<Stack>& S, Entry& e) {
  const auto& nodes = v.nodes();
  for (const auto& [byte, child] : nodes[static_cast<size_t>(node)].children) {
    std::vector<Stack> next;
    bool A = false, B = false;
    advance_local(m, S, byte, next, A, B);
    const auto& tokens = nodes[static_cast<size_t>(child)].tokens;
    if (A || B) {
      if (A || !next.empty()) mark(e.accept, tokens);
      e.frontier.push_back(child);
      continue;
    }
    if (next.empty()) continue;
    mark(e.accept, tokens);
    local_dfs(v, m, child, next, e);
  }
}

void full_dfs(const VocabIndex& v, const Machine& m, int32_t node, const std::vector<Stack>& S,
              std::vector<uint32_t>& bits) {
  const auto& nodes = v.nodes();
  for (const auto& [byte, child] : nodes[static_cast<size_t>(node)].children) {
    std::vector<Stack> next = advance_full(m, S, byte);
    if (next.empty()) continue;
    mark(bits, nodes[static_cast<size_t>(child)].tokens);
    full_dfs(v, m, child, next, bits);
  }
}

// The tokens below the frontier nodes that a stack allows, by walking those subtrees with the stack's real
// content: the path to the node is replayed first.
std::vector<uint32_t> frontier_bits(const VocabIndex& v, const Machine& m, const std::vector<int32_t>& frontier,
                                    const Stack& cfg) {
  std::vector<uint32_t> bits(v.words(), 0);
  const auto& nodes = v.nodes();
  for (int32_t target : frontier) {
    std::vector<uint8_t> path;
    for (int32_t n = target; n != 0; n = nodes[static_cast<size_t>(n)].parent)
      path.push_back(nodes[static_cast<size_t>(n)].byte);
    std::reverse(path.begin(), path.end());
    std::vector<Stack> S{cfg};
    for (uint8_t b : path) {
      S = advance_full(m, S, b);
      if (S.empty()) break;
    }
    if (S.empty()) continue;
    mark(bits, nodes[static_cast<size_t>(target)].tokens);
    full_dfs(v, m, target, S, bits);
  }
  return bits;
}

bool may_end_configs(const Machine& m, const std::vector<Stack>& configs) {
  for (const Stack& cfg : configs) {
    Stack s = cfg;
    while (!s.empty() && m.accepting(s.back())) s.pop_back();
    if (s.empty()) return true;
  }
  return false;
}

// ---- schema compilation ----------------------------------------------------------------------

bool annotation_key(const std::string& k) {
  static const std::set<std::string> kAnnotations = {
      "$schema", "$id", "$comment", "title", "description", "examples", "example",
      "default", "deprecated", "readOnly", "writeOnly", "$defs", "definitions"};
  return kAnnotations.count(k) != 0;
}

std::string escape_key(const std::string& key) {
  std::string quoted;
  lx::text::json_append_string(quoted, key);
  return quoted.substr(1, quoted.size() - 2);
}

// The complete JSON text of a value, written compactly (no spaces).
void compact(const JValue& v, std::string& out) {
  switch (v.type) {
    case JValue::Type::kNull: out += "null"; break;
    case JValue::Type::kBool: out += v.boolean ? "true" : "false"; break;
    case JValue::Type::kInt: out += v.text; break;
    case JValue::Type::kFloat: out += lx::text::py_float_repr(v.number); break;
    case JValue::Type::kString: lx::text::json_append_string(out, v.text); break;
    case JValue::Type::kArray:
      out.push_back('[');
      for (size_t i = 0; i < v.items.size(); ++i) {
        if (i) out.push_back(',');
        compact(v.items[i], out);
      }
      out.push_back(']');
      break;
    case JValue::Type::kObject:
      out.push_back('{');
      for (size_t i = 0; i < v.keys.size(); ++i) {
        if (i) out.push_back(',');
        lx::text::json_append_string(out, v.keys[i]);
        out.push_back(':');
        compact(v.values[i], out);
      }
      out.push_back('}');
      break;
  }
}

uint8_t value_bit(const JValue& v) {
  switch (v.type) {
    case JValue::Type::kObject: return kTObject;
    case JValue::Type::kArray: return kTArray;
    case JValue::Type::kString: return kTString;
    case JValue::Type::kBool: return kTBool;
    case JValue::Type::kNull: return kTNull;
    default: return kTNumber;
  }
}

bool integral(const JValue& v) {
  return v.type == JValue::Type::kInt ||
         (v.type == JValue::Type::kFloat && std::isfinite(v.number) && std::floor(v.number) == v.number);
}

bool type_allows(const std::vector<std::string>& types, const JValue& v) {
  for (const std::string& t : types) {
    if (t == "object" && v.type == JValue::Type::kObject) return true;
    if (t == "array" && v.type == JValue::Type::kArray) return true;
    if (t == "string" && v.type == JValue::Type::kString) return true;
    if (t == "boolean" && v.type == JValue::Type::kBool) return true;
    if (t == "null" && v.type == JValue::Type::kNull) return true;
    if (t == "number" && (v.type == JValue::Type::kInt || v.type == JValue::Type::kFloat)) return true;
    if (t == "integer" && integral(v)) return true;
  }
  return false;
}

uint32_t count_keyword(const JValue& s, const char* name, uint32_t fallback, bool& present) {
  const JValue* v = s.get(name);
  present = v != nullptr;
  if (!v) return fallback;
  if (v->type != JValue::Type::kInt || v->text.size() > 6 || std::stoul(v->text) > kMaxCount)
    throw std::invalid_argument(std::string("response_format: ") + name + " must be an integer from 0 to 65535");
  return static_cast<uint32_t>(std::stoul(v->text));
}

class Compiler {
 public:
  Compiler(const JValue& doc, std::vector<Node>& nodes) : doc_(doc), nodes_(nodes) {}

  int compile(const JValue* s) {
    for (;;) {
      if (s->type == JValue::Type::kBool) {
        if (s->boolean) return any();
        throw std::invalid_argument("response_format: the schema false matches nothing");
      }
      if (s->type != JValue::Type::kObject) throw std::invalid_argument("response_format: a schema must be an object or a boolean");
      const JValue* ref = s->get("$ref");
      if (!ref) break;
      if (!ref->is_string()) throw std::invalid_argument("response_format: $ref must be a string");
      for (const std::string& k : s->keys)
        if (k != "$ref" && !annotation_key(k))
          throw std::invalid_argument("response_format: '$ref' next to '" + k + "' is not supported");
      s = resolve(ref->text);
    }
    auto it = memo_.find(s);
    if (it != memo_.end()) return it->second;
    const int idx = add();
    memo_[s] = idx;
    fill(idx, *s);
    return idx;
  }

  int any() {
    if (any_ >= 0) return any_;
    const int idx = add();
    any_ = idx;
    const int obj = add();
    nodes_[obj].kind = NK::kObject;
    nodes_[obj].open = true;
    nodes_[obj].value = idx;
    nodes_[obj].types = kTObject;
    const int arr = add();
    nodes_[arr].kind = NK::kArray;
    nodes_[arr].items = idx;
    nodes_[arr].types = kTArray;
    const int str = add();
    nodes_[str].kind = NK::kString;
    nodes_[str].types = kTString;
    const int num = add();
    nodes_[num].kind = NK::kNumber;
    nodes_[num].types = kTNumber;
    const int bools = add();
    nodes_[bools].kind = NK::kLiteral;
    nodes_[bools].literals = {"true", "false"};
    nodes_[bools].types = kTBool;
    const int nul = add();
    nodes_[nul].kind = NK::kLiteral;
    nodes_[nul].literals = {"null"};
    nodes_[nul].types = kTNull;
    nodes_[idx].kind = NK::kUnion;
    nodes_[idx].alts = {obj, arr, str, num, bools, nul};
    nodes_[idx].types = kTObject | kTArray | kTString | kTNumber | kTBool | kTNull;
    return idx;
  }

 private:
  int add() {
    if (nodes_.size() >= kMaxNodes) throw std::invalid_argument("response_format: the schema is too large");
    nodes_.emplace_back();
    return static_cast<int>(nodes_.size()) - 1;
  }

  const JValue* resolve(const std::string& ref) const {
    if (ref.empty() || ref[0] != '#')
      throw std::invalid_argument("response_format: $ref '" + ref + "' is not supported: only JSON pointers into this schema");
    const JValue* cur = &doc_;
    size_t pos = 1;
    while (pos < ref.size()) {
      if (ref[pos] != '/') throw std::invalid_argument("response_format: $ref '" + ref + "' is not a JSON pointer");
      const size_t end = ref.find('/', pos + 1);
      std::string seg = ref.substr(pos + 1, end == std::string::npos ? std::string::npos : end - pos - 1);
      std::string key;
      for (size_t i = 0; i < seg.size(); ++i) {
        if (seg[i] == '~' && i + 1 < seg.size() && seg[i + 1] == '1') { key.push_back('/'); ++i; }
        else if (seg[i] == '~' && i + 1 < seg.size() && seg[i + 1] == '0') { key.push_back('~'); ++i; }
        else key.push_back(seg[i]);
      }
      const JValue* next = nullptr;
      if (cur->is_object()) next = cur->get(key);
      else if (cur->is_array() && !key.empty() && key.size() < 7 && key.find_first_not_of("0123456789") == std::string::npos)
        if (std::stoul(key) < cur->items.size()) next = &cur->items[std::stoul(key)];
      if (!next) throw std::invalid_argument("response_format: $ref '" + ref + "' does not resolve");
      cur = next;
      if (end == std::string::npos) break;
      pos = end;
    }
    return cur;
  }

  void fill(int idx, const JValue& s) {
    static const std::set<std::string> kKeywords = {
        "type", "properties", "required", "additionalProperties", "enum", "const", "items",
        "prefixItems", "minItems", "maxItems", "anyOf", "oneOf", "$ref"};
    for (const std::string& k : s.keys)
      if (!kKeywords.count(k) && !annotation_key(k))
        throw std::invalid_argument("response_format: the JSON Schema keyword '" + k + "' is not supported");

    const bool has_any = s.get("anyOf") != nullptr, has_one = s.get("oneOf") != nullptr;
    if (has_any || has_one) {
      if (has_any && has_one) throw std::invalid_argument("response_format: anyOf and oneOf together are not supported");
      for (const char* k : {"type", "properties", "required", "additionalProperties", "enum", "const", "items",
                            "prefixItems", "minItems", "maxItems"})
        if (s.get(k))
          throw std::invalid_argument(std::string("response_format: '") + k + "' next to anyOf or oneOf is not supported");
      const JValue* list = s.get(has_any ? "anyOf" : "oneOf");
      if (!list->is_array() || list->items.empty())
        throw std::invalid_argument("response_format: anyOf and oneOf need a non-empty list of schemas");
      std::vector<int> alts;
      for (const JValue& a : list->items) alts.push_back(compile(&a));
      if (has_one) {
        for (size_t i = 0; i < alts.size(); ++i)
          for (size_t j = i + 1; j < alts.size(); ++j)
            if ((nodes_[alts[i]].types & nodes_[alts[j]].types) && !disjoint_literals(alts[i], alts[j]))
              throw std::invalid_argument("response_format: oneOf branches can match the same value; use anyOf");
      }
      uint8_t types = 0;
      for (int a : alts) types |= nodes_[a].types;
      nodes_[idx].kind = NK::kUnion;
      nodes_[idx].alts = alts;
      nodes_[idx].types = types;
      return;
    }

    if (s.get("enum") || s.get("const")) {
      fill_literal(idx, s);
      return;
    }

    std::vector<std::string> types;
    if (const JValue* t = s.get("type")) {
      if (t->is_string()) {
        types.push_back(t->text);
      } else if (t->is_array()) {
        for (const JValue& e : t->items) {
          if (!e.is_string()) throw std::invalid_argument("response_format: type must be a string or a list of strings");
          types.push_back(e.text);
        }
      } else {
        throw std::invalid_argument("response_format: type must be a string or a list of strings");
      }
      if (types.empty()) throw std::invalid_argument("response_format: type lists no types");
    } else {
      // Without a type, the object and array keywords name the kind they apply to.
      if (s.get("properties") || s.get("required") || s.get("additionalProperties")) types.push_back("object");
      if (s.get("items") || s.get("prefixItems") || s.get("minItems") || s.get("maxItems")) types.push_back("array");
    }
    const bool obj_kw = s.get("properties") || s.get("required") || s.get("additionalProperties");
    const bool arr_kw = s.get("items") || s.get("prefixItems") || s.get("minItems") || s.get("maxItems");
    const bool t_obj = std::find(types.begin(), types.end(), "object") != types.end();
    const bool t_arr = std::find(types.begin(), types.end(), "array") != types.end();
    if (obj_kw && !t_obj) throw std::invalid_argument("response_format: an object keyword next to a type that is not object");
    if (arr_kw && !t_arr) throw std::invalid_argument("response_format: an array keyword next to a type that is not array");
    if (types.empty()) {
      const int any_node = any();
      nodes_[idx] = nodes_[static_cast<size_t>(any_node)];
      return;
    }
    if (types.size() == 1) {
      fill_type(idx, types[0], s);
      return;
    }
    std::vector<int> alts;
    for (const std::string& t : types) {
      const int c = add();
      fill_type(c, t, s);
      alts.push_back(c);
    }
    nodes_[idx].kind = NK::kUnion;
    nodes_[idx].alts = alts;
    uint8_t bits = 0;
    for (int a : alts) bits |= nodes_[a].types;
    nodes_[idx].types = bits;
  }

  // Two literal branches can only overlap when they share a text.
  bool disjoint_literals(int a, int b) const {
    if (nodes_[a].kind != NK::kLiteral || nodes_[b].kind != NK::kLiteral) return false;
    for (const std::string& x : nodes_[a].literals)
      for (const std::string& y : nodes_[b].literals)
        if (x == y) return false;
    return true;
  }

  void fill_type(int idx, const std::string& type, const JValue& s) {
    if (type == "object") {
      fill_object(idx, s);
    } else if (type == "array") {
      fill_array(idx, s);
    } else if (type == "string") {
      nodes_[idx].kind = NK::kString;
      nodes_[idx].types = kTString;
    } else if (type == "integer" || type == "number") {
      nodes_[idx].kind = type == "integer" ? NK::kInteger : NK::kNumber;
      nodes_[idx].types = kTNumber;
    } else if (type == "boolean") {
      nodes_[idx].kind = NK::kLiteral;
      nodes_[idx].literals = {"true", "false"};
      nodes_[idx].types = kTBool;
    } else if (type == "null") {
      nodes_[idx].kind = NK::kLiteral;
      nodes_[idx].literals = {"null"};
      nodes_[idx].types = kTNull;
    } else {
      throw std::invalid_argument("response_format: type '" + type + "' is not supported");
    }
  }

  void fill_object(int idx, const JValue& s) {
    const JValue* props = s.get("properties");
    const JValue* addl = s.get("additionalProperties");
    const JValue* req = s.get("required");
    if (addl && addl->type == JValue::Type::kObject)
      throw std::invalid_argument("response_format: additionalProperties as a schema is not supported");
    const bool addl_false = addl && addl->type == JValue::Type::kBool && !addl->boolean;
    if (props) {
      if (!props->is_object()) throw std::invalid_argument("response_format: properties must be an object");
      if (addl && !addl_false)
        throw std::invalid_argument("response_format: additionalProperties other than false is not supported; the grammar writes the declared properties only");
      if (props->keys.size() > kMaxProps) throw std::invalid_argument("response_format: more than 64 properties");
      std::vector<std::string> keys;
      std::vector<int> nodes_of;
      for (size_t i = 0; i < props->keys.size(); ++i) {
        keys.push_back(escape_key(props->keys[i]));
        nodes_of.push_back(compile(&props->values[i]));
      }
      uint64_t required = 0;
      if (req) {
        if (!req->is_array()) throw std::invalid_argument("response_format: required must be a list of names");
        for (const JValue& r : req->items) {
          if (!r.is_string()) throw std::invalid_argument("response_format: required must be a list of names");
          const auto it = std::find(props->keys.begin(), props->keys.end(), r.text);
          if (it == props->keys.end())
            throw std::invalid_argument("response_format: required '" + r.text + "' is not a declared property");
          required |= 1ull << (it - props->keys.begin());
        }
      }
      uint32_t last = 0;
      for (size_t i = 0; i < keys.size(); ++i)
        if (required >> i & 1) last = static_cast<uint32_t>(i) + 1;
      Node& nd = nodes_[idx];
      nd.kind = NK::kObject;
      nd.open = false;
      nd.keys = std::move(keys);
      nd.props = std::move(nodes_of);
      nd.required = required;
      nd.last_required1 = last;
      nd.types = kTObject;
      return;
    }
    if (req && !(req->is_array() && req->items.empty()))
      throw std::invalid_argument("response_format: required names properties, and none are declared");
    Node& nd = nodes_[idx];
    nd.kind = NK::kObject;
    nd.types = kTObject;
    if (addl_false) {
      nd.open = false;
      nd.keys.clear();
      nd.props.clear();
      nd.required = 0;
      nd.last_required1 = 0;
      return;
    }
    // additionalProperties absent or true, with no properties declared: any object
    const int value_node = any();
    nodes_[idx].open = true;
    nodes_[idx].value = value_node;
  }

  void fill_array(int idx, const JValue& s) {
    const JValue* pre = s.get("prefixItems");
    const JValue* items = s.get("items");
    std::vector<int> prefix;
    if (pre) {
      if (!pre->is_array()) throw std::invalid_argument("response_format: prefixItems must be a list of schemas");
      for (const JValue& e : pre->items) prefix.push_back(compile(&e));
    }
    int items_node = -1;
    if (!items) {
      items_node = any();
    } else if (items->type == JValue::Type::kBool) {
      items_node = items->boolean ? any() : -1;
    } else if (items->type == JValue::Type::kObject) {
      items_node = compile(items);
    } else {
      throw std::invalid_argument("response_format: the list form of items is not supported; use prefixItems");
    }
    bool has_min = false, has_max = false;
    const uint32_t min = count_keyword(s, "minItems", 0, has_min);
    const uint32_t max_raw = count_keyword(s, "maxItems", 0, has_max);
    if (has_max && max_raw < min) throw std::invalid_argument("response_format: maxItems is below minItems");
    Node& nd = nodes_[idx];
    nd.kind = NK::kArray;
    nd.types = kTArray;
    nd.prefix = std::move(prefix);
    nd.items = items_node;
    nd.min_items = min;
    nd.max_items = has_max ? static_cast<int64_t>(max_raw) : -1;
  }

  void fill_literal(int idx, const JValue& s) {
    const JValue* en = s.get("enum");
    const JValue* cn = s.get("const");
    if (en && cn) throw std::invalid_argument("response_format: enum and const together are not supported");
    std::vector<const JValue*> values;
    if (en) {
      if (!en->is_array()) throw std::invalid_argument("response_format: enum must be a list");
      for (const JValue& e : en->items) values.push_back(&e);
    } else {
      values.push_back(cn);
    }
    std::vector<std::string> types;
    if (const JValue* t = s.get("type")) {
      if (t->is_string()) types.push_back(t->text);
      else if (t->is_array())
        for (const JValue& e : t->items)
          if (e.is_string()) types.push_back(e.text);
    }
    std::vector<std::string> texts;
    uint8_t bits = 0;
    for (const JValue* v : values) {
      if (!lx::text::json_is_finite(*v)) throw std::invalid_argument("response_format: enum and const cannot hold NaN or infinity");
      if (!types.empty() && !type_allows(types, *v)) continue;
      std::string text;
      compact(*v, text);
      if (std::find(texts.begin(), texts.end(), text) == texts.end()) texts.push_back(text);
      bits |= value_bit(*v);
    }
    if (texts.empty()) throw std::invalid_argument("response_format: no enum or const value has the declared type");
    if (texts.size() > kMaxLiterals) throw std::invalid_argument("response_format: more than 64 enum values");
    Node& nd = nodes_[idx];
    nd.kind = NK::kLiteral;
    nd.literals = std::move(texts);
    nd.types = bits;
  }

  const JValue& doc_;
  std::vector<Node>& nodes_;
  std::map<const JValue*, int> memo_;
  int any_ = -1;
};

// Unions must not refer back to themselves without a value in between (that would never end).
void check_unions(const std::vector<Node>& nodes) {
  std::vector<uint8_t> color(nodes.size(), 0);
  struct Visit {
    const std::vector<Node>& nodes;
    std::vector<uint8_t>& color;
    void operator()(int i) {
      if (nodes[static_cast<size_t>(i)].kind != NK::kUnion) return;
      if (color[static_cast<size_t>(i)] == 1) throw std::invalid_argument("response_format: anyOf or oneOf refers back to itself without a value in between");
      if (color[static_cast<size_t>(i)] == 2) return;
      color[static_cast<size_t>(i)] = 1;
      for (int a : nodes[static_cast<size_t>(i)].alts) (*this)(a);
      color[static_cast<size_t>(i)] = 2;
    }
  } visit{nodes, color};
  for (size_t i = 0; i < nodes.size(); ++i) visit(static_cast<int>(i));
}

// What a frame state's own transitions decided, cached per state (and per stack below, for the rest).
constexpr size_t kMaxBelowEntries = 4096;

}  // namespace jx

// The compiled schema, its node graph and its mask caches. Entries are keyed by the top frame's state; the
// stack-dependent results by (entry, stack below it). Both fill on the engine thread only.
class JsonProgram {
 public:
  std::vector<jx::Node> nodes;
  int root = -1;
  bool thinking = false;
  std::vector<jx::Entry> entries;
  std::unordered_map<std::string, int> entry_ids;
  std::unordered_map<std::string, std::vector<uint32_t>> below_cache;
};

std::shared_ptr<JsonProgram> compile_json(const JsonFormat& format) {
  auto prog = std::make_shared<JsonProgram>();
  prog->thinking = format.thinking;
  lx::text::JValue doc;
  if (format.object) {
    doc.type = lx::text::JValue::Type::kObject;
    doc.keys = {"type"};
    doc.values = {lx::text::JValue::string("object")};
  } else if (!lx::text::json_loads(format.schema, doc)) {
    throw std::invalid_argument("response_format: the schema is not valid JSON");
  }
  jx::Compiler compiler(doc, prog->nodes);
  prog->root = compiler.compile(&doc);
  jx::check_unions(prog->nodes);
  return prog;
}

// ---- the vocabulary -----------------------------------------------------------------------------

VocabIndex::VocabIndex(const std::vector<std::string>& token_bytes)
    : vocab_(static_cast<int32_t>(token_bytes.size())), words_((token_bytes.size() + 31) / 32), bytes_(token_bytes) {
  nodes_.emplace_back();  // the root
  for (size_t id = 0; id < bytes_.size(); ++id) {
    if (bytes_[id].empty()) continue;
    int32_t node = 0;
    for (char ch : bytes_[id]) {
      const uint8_t b = static_cast<uint8_t>(ch);
      int32_t child = -1;
      for (const auto& [cb, ci] : nodes_[static_cast<size_t>(node)].children)
        if (cb == b) child = ci;
      if (child < 0) {
        child = static_cast<int32_t>(nodes_.size());
        Node fresh;
        fresh.parent = node;
        fresh.byte = b;
        nodes_.push_back(std::move(fresh));
        auto& kids = nodes_[static_cast<size_t>(node)].children;
        kids.insert(std::lower_bound(kids.begin(), kids.end(), std::make_pair(b, int32_t{0}),
                                     [](const auto& x, const auto& y) { return x.first < y.first; }),
                    std::make_pair(b, child));
      }
      node = child;
    }
    nodes_[static_cast<size_t>(node)].tokens.push_back(static_cast<int32_t>(id));
  }
}

// ---- the request's grammar -----------------------------------------------------------------------

struct JsonGrammar::Impl {
  std::shared_ptr<JsonProgram> program;
  const VocabIndex* vocab = nullptr;
  std::vector<jx::Stack> configs;  // the stacks the output could be in; empty stacks are complete replies
  bool cached = false;
  std::vector<uint32_t> cache;
};

namespace {

// The entry for a top frame's state, computed on first use.
int entry_for(JsonProgram& p, const VocabIndex& v, const jx::Machine& m, const jx::Frame& top) {
  std::string key;
  jx::put(key, top);
  const auto found = p.entry_ids.find(key);
  if (found != p.entry_ids.end()) return found->second;
  jx::Entry e;
  e.accept.assign(v.words(), 0);
  jx::local_dfs(v, m, 0, std::vector<jx::Stack>{jx::Stack{top}}, e);
  p.entries.push_back(std::move(e));
  const int id = static_cast<int>(p.entries.size()) - 1;
  p.entry_ids.emplace(std::move(key), id);
  return id;
}

}  // namespace

JsonGrammar::JsonGrammar(std::shared_ptr<JsonProgram> program, const VocabIndex& vocab)
    : impl_(std::make_unique<Impl>()) {
  impl_->program = std::move(program);
  impl_->vocab = &vocab;
  jx::Stack start;
  jx::Frame wrap;
  wrap.node = static_cast<uint16_t>(impl_->program->root);
  wrap.kind = jx::kWrap;
  wrap.phase = jx::kWait;
  start.push_back(wrap);
  if (impl_->program->thinking) {
    jx::Frame think;
    think.kind = jx::kThink;
    start.push_back(think);
  }
  impl_->configs.push_back(std::move(start));
}

JsonGrammar::~JsonGrammar() = default;

bool JsonGrammar::accept(std::string_view bytes) {
  const jx::Machine m(impl_->program->nodes);
  std::vector<jx::Stack> cur = impl_->configs;
  for (unsigned char c : bytes) {
    cur = jx::advance_full(m, cur, c);
    if (cur.empty() || cur.size() > jx::kMaxConfigs) return false;
  }
  impl_->configs = std::move(cur);
  impl_->cached = false;
  return true;
}

bool JsonGrammar::may_end() const {
  const jx::Machine m(impl_->program->nodes);
  return jx::may_end_configs(m, impl_->configs);
}

const std::vector<uint32_t>& JsonGrammar::mask(const std::vector<int32_t>& stops) {
  Impl& im = *impl_;
  if (im.cached) return im.cache;
  JsonProgram& p = *im.program;
  const VocabIndex& v = *im.vocab;
  const jx::Machine m(p.nodes);
  im.cache.assign(v.words(), 0);
  for (const jx::Stack& cfg : im.configs) {
    if (cfg.empty()) continue;  // a complete reply consumes nothing
    const int ei = entry_for(p, v, m, cfg.back());
    const jx::Entry& e = p.entries[static_cast<size_t>(ei)];
    jx::or_into(im.cache, e.accept);
    if (e.frontier.empty()) continue;
    std::string key(reinterpret_cast<const char*>(&ei), sizeof ei);
    for (size_t i = 0; i + 1 < cfg.size(); ++i) jx::put(key, cfg[i]);
    auto it = p.below_cache.find(key);
    if (it == p.below_cache.end()) {
      if (p.below_cache.size() >= jx::kMaxBelowEntries) p.below_cache.clear();
      it = p.below_cache.emplace(key, jx::frontier_bits(v, m, e.frontier, cfg)).first;
    }
    jx::or_into(im.cache, it->second);
  }
  // Stops are allowed exactly when the output is complete.
  const bool may = jx::may_end_configs(m, im.configs);
  for (int32_t id : stops) {
    if (id < 0 || id >= v.vocab()) continue;
    uint32_t& word = im.cache[static_cast<size_t>(id) / 32];
    const uint32_t bit = 1u << (static_cast<size_t>(id) % 32);
    if (may) word |= bit;
    else word &= ~bit;
  }
  im.cached = true;
  return im.cache;
}

bool JsonGrammar::finished(const std::vector<int32_t>& stops) {
  if (!may_end()) return false;
  const std::vector<uint32_t>& bits = mask(stops);
  size_t allowed = 0;
  for (uint32_t w : bits) allowed += static_cast<size_t>(__builtin_popcount(w));
  std::set<int32_t> counted;
  size_t stop_allowed = 0;
  for (int32_t id : stops) {
    if (id < 0 || id >= impl_->vocab->vocab()) continue;
    const bool set = (bits[static_cast<size_t>(id) / 32] >> (static_cast<size_t>(id) % 32)) & 1u;
    if (set && counted.insert(id).second) stop_allowed += 1;
  }
  return allowed == stop_allowed;
}

}  // namespace lx::runtime
