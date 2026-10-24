#include <doctest/doctest.h>

#include <string>

#include "core/digest.h"

TEST_CASE("SHA-256 matches the standard vectors") {
  CHECK(lx::sha256_hex("") == "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(lx::sha256_hex("abc") == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK(lx::sha256_hex("abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq") ==
        "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1");
}

TEST_CASE("SHA-256 is the same however the input is split") {
  const std::string text(1000000, 'a');
  CHECK(lx::sha256_hex(text) == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  lx::Sha256 s;
  size_t at = 0, step = 1;
  while (at < text.size()) {
    const size_t n = std::min(step, text.size() - at);
    s.update(text.data() + at, n);
    at += n;
    step = step * 3 + 1;  // odd sizes cross the 64-byte block boundary every which way
  }
  CHECK(s.hex_digest() == "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
}

TEST_CASE("SHA-1 matches the standard vectors") {
  lx::Sha1 empty;
  CHECK(empty.hex_digest() == "da39a3ee5e6b4b0d3255bfef95601890afd80709");
  lx::Sha1 abc;
  abc.update("abc", 3);
  CHECK(abc.hex_digest() == "a9993e364706816aba3e25717850c26c9cd0d89d");
}

TEST_CASE("a git blob id is the SHA-1 of the header and the content") {
  lx::GitBlobId id(6);
  id.update("hello\n", 6);
  CHECK(id.hex_digest() == "ce013625030ba8dba906f756967f9e9ca394464a");
}
