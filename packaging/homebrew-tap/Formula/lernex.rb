# Homebrew formula for Lernex's inference engine on Apple silicon. It belongs in the lernex/homebrew-tap repository
# as Formula/lernex.rb, so `brew install lernex/tap/lernex` finds it. The asset is the darwin-arm64 tarball of
# docs/RELEASE.md: bin/lernex, lib/liblernex_metal.dylib, share/lernex/channel and LICENSE.
# packaging/homebrew-tap/bump-formula.sh moves version, url and sha256 to a release.
class Lernex < Formula
  desc "Inference engine for Lernex's Metis models, on the Metal GPU of an Apple silicon Mac"
  homepage "https://github.com/lernex/lernex-engine"
  version "0.1.0"
  url "https://github.com/lernex/lernex-engine/releases/download/v0.1.0/lernex-darwin-arm64.tar.gz"
  sha256 "0000000000000000000000000000000000000000000000000000000000000000"

  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    bin.install "bin/lernex"
    lib.install Dir["lib/*.dylib"]
    # The tuning tables and the channel file the tarball ships. The channel names this install path, so
    # `lernex update` and the daily notice point at brew upgrade lernex.
    (share/"lernex").install Dir["share/lernex/*"] if Dir.exist?("share/lernex")
    (share/"lernex/channel").atomic_write "homebrew\n"
    prefix.install "LICENSE" if File.exist?("LICENSE")
  end

  def caveats
    <<~EOS
      Models are downloaded on first use into ~/.cache/lernex/models.
        lernex run praxis      chat in the terminal
        lernex serve praxis    the OpenAI API on this Mac
    EOS
  end

  test do
    ENV["LERNEX_NO_UPDATE_CHECK"] = "1"
    assert_match "lernex #{version}", shell_output("#{bin}/lernex --version")
    doctor = shell_output("#{bin}/lernex doctor")
    assert_match "Metal", doctor
    assert_equal "homebrew\n", (share/"lernex/channel").read
  end
end
