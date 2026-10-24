// The ByteLevel pre-tokenizer of the Metis tokenizer.json: the GPT-2 pattern
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// matched by hand over the code points of valid UTF-8, instead of through a regex engine.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

namespace lx::text {

// Appends the [begin, end) byte ranges of the pieces of s, in order, covering all of s.
void pretokenize(std::string_view s, std::vector<std::pair<uint32_t, uint32_t>>& pieces);

}  // namespace lx::text
