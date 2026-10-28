#!/bin/sh
# Points Formula/lernex.rb at a release: its version, the url of lernex-darwin-arm64.tar.gz and its sha256.
#
#   packaging/homebrew-tap/bump-formula.sh 0.1.0
#   packaging/homebrew-tap/bump-formula.sh v0.1.0 path/to/Formula/lernex.rb
#   packaging/homebrew-tap/bump-formula.sh 0.1.0 Formula/lernex.rb --tarball PATH --url file://PATH
#
# Without --tarball the asset and the release's SHA256SUMS are downloaded, and the tarball must match its line
# there. LERNEX_RELEASES points at another release host, as install.sh and `lernex update` do (docs/RELEASE.md).
set -eu
usage() { echo "usage: $0 VERSION [FORMULA] [--tarball PATH] [--url URL]" >&2; exit 2; }
[ $# -ge 1 ] || usage
version=${1#v}
shift
formula=packaging/homebrew-tap/Formula/lernex.rb
if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then formula=$1; shift; fi
tarball=
url=
while [ $# -gt 0 ]; do
  case $1 in
    --tarball) tarball=$2; shift 2 ;;
    --url) url=$2; shift 2 ;;
    *) usage ;;
  esac
done
releases=${LERNEX_RELEASES:-https://github.com/lernex/lernex-engine/releases}
asset=lernex-darwin-arm64.tar.gz
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
want=
if [ -z "$tarball" ]; then
  tarball=$work/$asset
  curl -fsSL -o "$tarball" "$releases/download/v$version/$asset"
  curl -fsSL -o "$work/SHA256SUMS" "$releases/download/v$version/SHA256SUMS"
  want=$(awk -v a="$asset" '{ f = $2; sub(/^\*/, "", f); if (f == a) print $1 }' "$work/SHA256SUMS")
  [ -n "$want" ] || { echo "$asset is not in the release's SHA256SUMS" >&2; exit 1; }
fi
sha=$(shasum -a 256 "$tarball" | awk '{ print $1 }')
if [ -n "$want" ] && [ "$sha" != "$want" ]; then
  echo "checksum mismatch: $asset is $sha, SHA256SUMS says $want" >&2
  exit 1
fi
[ -n "$url" ] || url=$releases/download/v$version/$asset
perl -0pi -e "s|^  version \".*\"\$|  version \"$version\"|m; s|^  url \".*\"\$|  url \"$url\"|m; s|^  sha256 \".*\"\$|  sha256 \"$sha\"|m" "$formula"
if command -v ruby >/dev/null 2>&1; then ruby -c "$formula" >/dev/null; fi
echo "$formula: version $version, sha256 $sha"
