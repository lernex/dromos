# Homebrew formula for Dromos, Lernex's inference engine on Apple silicon. It belongs in the lernex/homebrew-tap repository
# as Formula/dromos.rb, so `brew install lernex/tap/dromos` finds it. The asset is the darwin-arm64 tarball of
# docs/RELEASE.md: bin/dromos, lib/libdromos_metal.dylib, share/dromos/channel, LICENSE and NOTICE (Homebrew keeps those in the
# prefix itself), and the third-party licenses in share/dromos/licenses.
# packaging/homebrew-tap/bump-formula.sh moves version, url and sha256 to a release.
class Dromos < Formula
  desc "Metis inference engine on the Metal GPU of an Apple silicon Mac"
  homepage "https://github.com/lernex/dromos"
  url "https://github.com/lernex/dromos/releases/download/v0.1.0/dromos-darwin-arm64.tar.gz"
  version "0.1.0"
  sha256 "cdd136284cdea38b7a8773ef5a5dbfa7f6064e0744829ad7fc081ff7138b7662"
  license "Apache-2.0"

  depends_on arch: :arm64
  depends_on macos: :sonoma

  def install
    bin.install "bin/dromos"
    lib.install Dir["lib/*.dylib"]
    # The tuning tables and the channel file the tarball ships. The channel names this install path, so
    # `dromos update` and the daily notice point at brew upgrade dromos.
    pkgshare.install Dir["share/dromos/*"] if Dir.exist?("share/dromos")
    (pkgshare/"channel").atomic_write "homebrew\n"
    prefix.install "LICENSE" if File.exist?("LICENSE")
  end

  def caveats
    <<~EOS
      Models are downloaded on first use into ~/.cache/dromos/models.
        dromos run praxis      chat in the terminal
        dromos serve praxis    the OpenAI API on this Mac
    EOS
  end

  test do
    ENV["DROMOS_NO_UPDATE_CHECK"] = "1"
    assert_match "dromos #{version}", shell_output("#{bin}/dromos --version")
    doctor = shell_output("#{bin}/dromos doctor")
    assert_match "Metal", doctor
    assert_equal "homebrew\n", (pkgshare/"channel").read
  end
end
