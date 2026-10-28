// The Metis tokenizer: tokenizer.json read as tokenizers reads it (ByteLevel pre-tokenizer and
// decoder, BPE model, added tokens split out first), with decode and streaming decode as
// tokenizers implements them: the bytes of all tokens, concatenated, decoded as UTF-8 with
// replacement.
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "text/bpe.h"
#include "text/json.h"
#include "text/pretokenize.h"
#include "text/tokenizer.h"
#include "text/utf8.h"

namespace lx::text {
namespace {

std::string read_file(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) throw std::runtime_error("cannot read " + path);
  std::ostringstream ss;
  ss << in.rdbuf();
  return ss.str();
}

JValue json_loads_file(const std::string& path) {
  JValue value;
  if (!json_loads(read_file(path), value)) throw std::runtime_error("not valid JSON: " + path);
  return value;
}

// GPT-2's bytes_to_unicode: every byte gets a printable character, so tokens are strings.
struct ByteChars {
  uint32_t byte_to_char[256];
  uint32_t char_to_byte[512];  // indexed by code point below 512; the others are not byte chars

  ByteChars() {
    std::vector<uint32_t> bs;
    for (uint32_t b = 33; b < 127; ++b) bs.push_back(b);
    for (uint32_t b = 161; b < 173; ++b) bs.push_back(b);
    for (uint32_t b = 174; b < 256; ++b) bs.push_back(b);
    std::vector<uint32_t> cs = bs;
    uint32_t extra = 0;
    for (uint32_t b = 0; b < 256; ++b) {
      bool present = false;
      for (uint32_t x : bs) present = present || x == b;
      if (!present) {
        bs.push_back(b);
        cs.push_back(256 + extra);
        ++extra;
      }
    }
    for (size_t k = 0; k < bs.size(); ++k) {
      byte_to_char[bs[k]] = cs[k];
    }
    for (auto& v : char_to_byte) v = 0xFFFFFFFFu;
    for (uint32_t b = 0; b < 256; ++b) char_to_byte[byte_to_char[b]] = b;
  }
};

const ByteChars& byte_chars() {
  static const ByteChars table;
  return table;
}

std::string bytes_to_token(std::string_view token) {
  // The token's characters, each mapped back to its byte. A character outside the byte table keeps
  // its own UTF-8 (tokenizers does the same).
  std::string bytes;
  size_t i = 0;
  while (i < token.size()) {
    size_t len = 0;
    const uint32_t cp = utf8_decode(token, i, len);
    if (cp < 512 && byte_chars().char_to_byte[cp] != 0xFFFFFFFFu) {
      bytes.push_back(static_cast<char>(byte_chars().char_to_byte[cp]));
    } else {
      bytes.append(token.substr(i, len));
    }
    i += len;
  }
  return bytes;
}

// The byte-level string of a token given as bytes (used to look up merge halves).
std::string token_to_bytelevel(std::string_view bytes) {
  std::string out;
  for (const char c : bytes) {
    utf8_append(out, byte_chars().byte_to_char[static_cast<uint8_t>(c)]);
  }
  return out;
}

const JValue& require(const JValue* v, const char* what) {
  if (v == nullptr) throw std::runtime_error(std::string("tokenizer.json: missing ") + what);
  return *v;
}

// Only the configurations the Metis files use. Anything else is refused, not silently wrong.
void check_supported(const JValue& tj) {
  const JValue* normalizer = tj.get("normalizer");
  if (normalizer != nullptr && !normalizer->is_null()) {
    throw std::runtime_error("tokenizer.json: normalizers are not implemented");
  }
  const JValue* pre = tj.get("pre_tokenizer");
  if (pre == nullptr || pre->get("type") == nullptr || pre->get("type")->text != "ByteLevel" ||
      pre->get("use_regex") == nullptr || !pre->get("use_regex")->boolean ||
      pre->get("add_prefix_space")->boolean) {
    throw std::runtime_error("tokenizer.json: only the GPT-2 ByteLevel pre-tokenizer is implemented");
  }
  const JValue* decoder = tj.get("decoder");
  if (decoder == nullptr || decoder->get("type") == nullptr || decoder->get("type")->text != "ByteLevel") {
    throw std::runtime_error("tokenizer.json: only the ByteLevel decoder is implemented");
  }
  const JValue& model = require(tj.get("model"), "model");
  if (model.get("type") == nullptr || model.get("type")->text != "BPE") {
    throw std::runtime_error("tokenizer.json: only the BPE model is implemented");
  }
  for (const char* field : {"continuing_subword_prefix", "end_of_word_suffix", "dropout"}) {
    const JValue* v = model.get(field);
    if (v != nullptr && !v->is_null()) {
      throw std::runtime_error(std::string("tokenizer.json: BPE option ") + field + " is not implemented");
    }
  }
  const JValue* added = tj.get("added_tokens");
  if (added != nullptr && added->is_array()) {
    for (const JValue& a : added->items) {
      for (const char* flag : {"normalized", "lstrip", "rstrip", "single_word"}) {
        const JValue* v = a.get(flag);
        if (v != nullptr && v->boolean) {
          throw std::runtime_error(std::string("tokenizer.json: added token option ") + flag + " is not implemented");
        }
      }
    }
  }
}

}  // namespace

class BpeTokenizer final : public Tokenizer {
 public:
  struct Added {
    std::string content;  // the text, as written
    int32_t id;
    bool special;
  };

  class BpeStream final : public Stream {
   public:
    BpeStream(const BpeTokenizer& tok, bool skip_special) : tok_(tok), skip_special_(skip_special) {}

    std::string push(int32_t id) override {
      if (!tok_.append_token_bytes(id, skip_special_, pending_)) return {};
      const size_t tail = utf8_incomplete_tail(pending_);
      std::string out = utf8_lossy(std::string_view(pending_).substr(0, pending_.size() - tail));
      pending_.erase(0, pending_.size() - tail);
      return out;
    }

    std::string flush() override {
      std::string out = utf8_lossy(pending_);
      pending_.clear();
      return out;
    }

    std::string finish() override {
      std::string out = utf8_lossy_final(pending_);
      pending_.clear();
      return out;
    }

   private:
    const BpeTokenizer& tok_;
    bool skip_special_;
    std::string pending_;  // bytes not yet emitted: at most a few bytes of an unfinished character
  };

  void init(const JValue& tj, const std::string& model_dir);

  std::vector<int32_t> encode(std::string_view text, bool parse_special) const override {
    std::vector<int32_t> ids;
    // Invalid bytes become U+FFFD first, as a str built with errors="replace" would have them.
    const bool valid = utf8_valid(text);
    const std::string lossy = valid ? std::string() : utf8_lossy(text);
    const std::string_view s = valid ? text : std::string_view(lossy);
    if (!parse_special || added_.empty()) {
      encode_plain(s, ids);
      return ids;
    }
    // Leftmost, then longest: an added token starts with '<', which never sits inside a UTF-8 sequence.
    size_t segment = 0;
    size_t i = 0;
    while (i < s.size()) {
      const Added* best = nullptr;
      if (s[i] == '<') {
        for (const Added& a : added_) {
          if (s.substr(i, a.content.size()) == a.content && (best == nullptr || a.content.size() > best->content.size())) {
            best = &a;
          }
        }
      }
      if (best == nullptr) {
        ++i;
        continue;
      }
      encode_plain(s.substr(segment, i - segment), ids);
      ids.push_back(best->id);
      i += best->content.size();
      segment = i;
    }
    encode_plain(s.substr(segment), ids);
    return ids;
  }

  std::string decode(const std::vector<int32_t>& ids, bool skip_special) const override {
    std::string bytes;
    for (const int32_t id : ids) {
      append_token_bytes(id, skip_special, bytes);
    }
    return utf8_lossy(bytes);
  }

  std::unique_ptr<Stream> stream(bool skip_special) const override {
    return std::make_unique<BpeStream>(*this, skip_special);
  }

  int vocab_size() const override { return static_cast<int>(id_bytes_.size()); }

  std::string token_bytes(int32_t id) const override {
    if (id < 0 || static_cast<size_t>(id) >= id_bytes_.size()) return std::string();
    return id_bytes_[static_cast<size_t>(id)];
  }

  int32_t token_id(std::string_view token) const override {
    const auto it = vocab_.find(std::string(token));
    return it == vocab_.end() ? -1 : it->second;
  }

  bool is_special(int32_t id) const override {
    return id >= 0 && static_cast<size_t>(id) < special_.size() && special_[id];
  }

  std::vector<int32_t> stop_ids() const override { return stops_; }

  // Appends the bytes of one token. False for an id outside the vocabulary (or a skipped special).
  bool append_token_bytes(int32_t id, bool skip_special, std::string& out) const {
    if (id < 0 || static_cast<size_t>(id) >= id_bytes_.size()) return false;
    if (skip_special && special_[id]) return false;
    out += id_bytes_[id];
    return true;
  }

 private:
  void encode_plain(std::string_view s, std::vector<int32_t>& ids) const {
    if (s.empty()) return;
    std::vector<std::pair<uint32_t, uint32_t>> pieces;
    pretokenize(s, pieces);
    for (const auto& [begin, end] : pieces) {
      bpe_.encode(s.substr(begin, end - begin), ids);
    }
  }

  std::unordered_map<std::string, int32_t> vocab_;  // byte-level token text -> id
  std::vector<std::string> id_bytes_;               // the bytes each id decodes to
  std::vector<bool> special_;                       // per id: an added special token
  std::vector<Added> added_;
  std::vector<int32_t> stops_;
  BytePair bpe_;
};

void BpeTokenizer::init(const JValue& tj, const std::string& model_dir) {
  check_supported(tj);
  const JValue& model = require(tj.get("model"), "model");
  const JValue& vocab = require(model.get("vocab"), "model.vocab");
  if (!vocab.is_object()) throw std::runtime_error("tokenizer.json: model.vocab is not an object");

  int32_t max_id = -1;
  for (size_t k = 0; k < vocab.keys.size(); ++k) {
    const JValue& id = vocab.values[k];
    if (id.type != JValue::Type::kInt) throw std::runtime_error("tokenizer.json: vocab id is not an integer");
    const auto v = static_cast<int32_t>(std::stol(id.text));
    vocab_.emplace(vocab.keys[k], v);
    if (v > max_id) max_id = v;
  }
  id_bytes_.assign(static_cast<size_t>(max_id + 1), std::string());
  special_.assign(static_cast<size_t>(max_id + 1), false);
  for (const auto& [text, id] : vocab_) {
    id_bytes_[id] = bytes_to_token(text);
  }

  for (int b = 0; b < 256; ++b) {
    const std::string one = token_to_bytelevel(std::string(1, static_cast<char>(b)));
    const auto it = vocab_.find(one);
    if (it == vocab_.end()) throw std::runtime_error("tokenizer.json: byte token missing for byte " + std::to_string(b));
    bpe_.set_byte_token(static_cast<uint8_t>(b), it->second);
  }

  // merges: [left, right] pairs (or "left right" strings in older files); rank is the list index.
  const JValue& merges = require(model.get("merges"), "model.merges");
  uint32_t rank = 0;
  for (const JValue& m : merges.items) {
    std::string left, right;
    if (m.is_array() && m.items.size() == 2) {
      left = m.items[0].text;
      right = m.items[1].text;
    } else if (m.is_string()) {
      const size_t sp = m.text.find(' ');
      if (sp == std::string::npos) throw std::runtime_error("tokenizer.json: bad merge");
      left = m.text.substr(0, sp);
      right = m.text.substr(sp + 1);
    } else {
      throw std::runtime_error("tokenizer.json: bad merge");
    }
    const auto l = vocab_.find(left);
    const auto r = vocab_.find(right);
    const auto merged = vocab_.find(left + right);
    if (l == vocab_.end() || r == vocab_.end() || merged == vocab_.end()) {
      throw std::runtime_error("tokenizer.json: merge tokens missing from the vocabulary");
    }
    bpe_.add_merge(l->second, r->second, rank++, merged->second);
  }

  if (const JValue* added = tj.get("added_tokens"); added != nullptr && added->is_array()) {
    for (const JValue& a : added->items) {
      const JValue& content = require(a.get("content"), "added token content");
      const auto id = static_cast<int32_t>(std::stol(require(a.get("id"), "added token id").text));
      const JValue* special = a.get("special");
      const bool is_special_token = special != nullptr && special->boolean;
      added_.push_back(Added{content.text, id, is_special_token});
      if (id >= 0 && static_cast<size_t>(id) < id_bytes_.size()) {
        id_bytes_[id] = content.text;  // an added token decodes to its text
        special_[id] = is_special_token;
      }
    }
  }

  // Stop ids: generation_config's eos ids, then tokenizer_config's eos token.
  auto add_stop = [&](int32_t id) {
    for (int32_t s : stops_) {
      if (s == id) return;
    }
    stops_.push_back(id);
  };
  try {
    const JValue gen = json_loads_file(model_dir + "/generation_config.json");
    if (const JValue* eos = gen.get("eos_token_id")) {
      if (eos->type == JValue::Type::kInt) {
        add_stop(static_cast<int32_t>(std::stol(eos->text)));
      } else if (eos->is_array()) {
        for (const JValue& e : eos->items) add_stop(static_cast<int32_t>(std::stol(e.text)));
      }
    }
  } catch (const std::runtime_error&) {
    // no generation_config.json: the tokenizer_config eos below is the only stop
  }
  try {
    const JValue cfg = json_loads_file(model_dir + "/tokenizer_config.json");
    if (const JValue* eos = cfg.get("eos_token"); eos != nullptr && eos->is_string()) {
      const int32_t id = token_id(eos->text);
      if (id >= 0) add_stop(id);
    }
  } catch (const std::runtime_error&) {
    // no tokenizer_config.json
  }
}

std::unique_ptr<Tokenizer> Tokenizer::load(const std::string& model_dir) {
  const JValue tj = json_loads_file(model_dir + "/tokenizer.json");
  auto tok = std::make_unique<BpeTokenizer>();
  tok->init(tj, model_dir);
  return tok;
}

}  // namespace lx::text
