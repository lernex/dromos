// Release and update policy: version order, the release URLs and checksum listing, the daily cache and the notice it
// prints. Host-side only (no network, no GPU): the transfer itself is exercised against a release host (docs/RELEASE.md).
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>

#include "core/update.h"
#include "doctest.h"

namespace {

namespace fs = std::filesystem;

// A scratch file that no other test uses, removed when the test ends.
struct Scratch {
  fs::path path;
  explicit Scratch(const std::string& name)
      : path(fs::temp_directory_path() / ("dromos-update-test-" + name + "-" +
                                          std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {}
  ~Scratch() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
};

}  // namespace

TEST_CASE("versions are X.Y.Z with an optional pre-release or build tail") {
  using lx::update::valid;
  CHECK(valid("0.1.0"));
  CHECK(valid("10.20.30"));
  CHECK(valid("0.1.0-rc1"));
  CHECK(valid("0.0.0-dev+4f3a2b1"));
  CHECK(valid("0.1.0-dev.3+g4f3a2b1"));
  CHECK_FALSE(valid(""));
  CHECK_FALSE(valid("0.1"));
  CHECK_FALSE(valid("v0.1.0"));
  CHECK_FALSE(valid("0.1.0-"));
  CHECK_FALSE(valid("0.1.0 rc1"));
  CHECK_FALSE(valid("a.b.c"));
}

TEST_CASE("versions compare by their numbers, then a release beats its pre-releases") {
  using lx::update::compare;
  CHECK(compare("0.1.10", "0.1.9") > 0);
  CHECK(compare("0.2.0", "0.1.99") > 0);
  CHECK(compare("1.0.0", "0.9.9") > 0);
  CHECK(compare("0.1.0", "0.1.0") == 0);
  CHECK(compare("0.0.0-dev+4f3a2b1", "0.0.0-dev+9999999") == 0);
  CHECK(compare("0.1.0", "0.1.0-rc1") > 0);
  CHECK(compare("0.1.0-rc1", "0.1.0-rc2") < 0);
  CHECK(compare("0.1.0-dev.3+g4f3a2b1", "0.1.0") < 0);
  CHECK(compare("0.1.1", "0.1.0-dev.3+g4f3a2b1") > 0);
  CHECK(compare("not-a-version", "0.1.0") < 0);
  CHECK(compare("0.1.0", "not-a-version") > 0);
}

TEST_CASE("a release page's URL names its version") {
  using lx::update::version_from_url;
  CHECK(version_from_url("https://github.com/dromos/dromos/releases/tag/v0.1.1") == "0.1.1");
  CHECK(version_from_url("http://127.0.0.1:8000/tag/v0.1.1?expanded=true") == "0.1.1");
  CHECK(version_from_url("http://127.0.0.1:8000/tag/v0.2.0-rc1#top") == "0.2.0-rc1");
  CHECK(version_from_url("https://github.com/dromos/dromos/releases/latest") == "");
  CHECK(version_from_url("https://github.com/dromos/dromos/releases/download/v0.1.1/x.tar.gz") == "");
}

TEST_CASE("the asset is named for its flavor") {
  CHECK(lx::update::asset_name("linux-x86_64-cpu") == "dromos-linux-x86_64-cpu.tar.gz");
  CHECK(lx::update::asset_name("darwin-arm64") == "dromos-darwin-arm64.tar.gz");
}

TEST_CASE("SHA256SUMS gives the digest of the named asset, in either sha256sum form") {
  const std::string hash_a(64, 'a');
  const std::string hash_b = std::string(60, 'B') + "dead";
  const std::string sums = hash_a + "  dromos-linux-x86_64-cpu.tar.gz\n" + hash_b + " *dromos-darwin-arm64.tar.gz\r\n";
  CHECK(lx::update::checksum_for(sums, "dromos-linux-x86_64-cpu.tar.gz") == hash_a);
  CHECK(lx::update::checksum_for(sums, "dromos-darwin-arm64.tar.gz") == std::string(60, 'b') + "dead");
  CHECK(lx::update::checksum_for(sums, "dromos-linux-aarch64-cpu.tar.gz").empty());
  CHECK(lx::update::checksum_for(std::string("abc  dromos-darwin-arm64.tar.gz\n"), "dromos-darwin-arm64.tar.gz").empty());
}

TEST_CASE("an installed asset hashes to the SHA-256 of its bytes") {
  Scratch scratch("sha");
  fs::create_directories(scratch.path);
  const fs::path file = scratch.path / "abc.bin";
  {
    std::ofstream out(file, std::ios::binary);
    out << "abc";
  }
  CHECK(lx::update::sha256_file(file) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  CHECK_THROWS(lx::update::sha256_file(scratch.path / "missing.bin"));
}

TEST_CASE("the daily cache is stale a day after the last check") {
  using lx::update::Cache;
  using lx::update::stale;
  const long long now = 1800000000;
  CHECK(stale(Cache{"0.1.1", 0}, now));
  CHECK_FALSE(stale(Cache{"0.1.1", now - 100}, now));
  CHECK_FALSE(stale(Cache{"0.1.1", now - 86399}, now));
  CHECK(stale(Cache{"0.1.1", now - 86400}, now));
}

TEST_CASE("the cache round-trips, and a garbled one reads as empty") {
  Scratch scratch("cache");
  const fs::path file = scratch.path / "update-check";
  lx::update::write_cache(file, lx::update::Cache{"0.1.1", 1800000000});
  const lx::update::Cache back = lx::update::read_cache(file);
  CHECK(back.latest == "0.1.1");
  CHECK(back.checked == 1800000000);

  // A failed check keeps the version it knew and records the attempt
  lx::update::write_cache(file, lx::update::Cache{back.latest, 1800090000});
  CHECK(lx::update::read_cache(file).checked == 1800090000);

  {
    std::ofstream out(file);
    out << "latest=not-a-version\nchecked=12x\nnoise\n";
  }
  const lx::update::Cache garbled = lx::update::read_cache(file);
  CHECK(garbled.latest.empty());
  CHECK(garbled.checked == 12);

  CHECK(lx::update::read_cache(scratch.path / "never-written").latest.empty());
}

TEST_CASE("the notice names a newer release and the command of its channel, and says nothing otherwise") {
  using lx::update::Cache;
  using lx::update::notice;
  const Cache newer{"0.1.1", 1800000000};
  const std::string tarball = notice("0.1.0", newer, "tarball");
  CHECK(tarball.find("dromos 0.1.1 is out") != std::string::npos);
  CHECK(tarball.find("you have 0.1.0") != std::string::npos);
  CHECK(tarball.find("`dromos update`") != std::string::npos);
  CHECK(notice("0.1.0", newer, "homebrew").find("`brew upgrade dromos`") != std::string::npos);
  CHECK(notice("0.1.0", newer, "pip").find("`pip install -U dromos`") != std::string::npos);
  CHECK(notice("0.1.0", newer, "docker").find("`docker pull ghcr.io/lernex/dromos`") != std::string::npos);

  CHECK(notice("0.1.1", newer, "tarball").empty());
  CHECK(notice("0.1.2", newer, "tarball").empty());
  CHECK(notice("0.1.0", Cache{}, "tarball").empty());
  CHECK(notice("0.1.0", newer, "").empty());
  CHECK(notice("0.1.0", newer, "unknown").empty());
  CHECK(notice("0.1.0-dev+abc", Cache{"0.1.0", 1}, "tarball").size() > 0);
}

TEST_CASE("each channel has one update command") {
  CHECK(lx::update::update_command("tarball") == "dromos update");
  CHECK(lx::update::update_command("homebrew") == "brew upgrade dromos");
  CHECK(lx::update::update_command("pip") == "pip install -U dromos");
  CHECK(lx::update::update_command("docker") == "docker pull ghcr.io/lernex/dromos");
  CHECK(lx::update::update_command("").empty());
}
