#!/usr/bin/env bash
# End to end: `lernex update` against a local release host (tests/release/serve.py), in a temporary HOME.
#
#   tests/release/update_e2e.sh TARBALL_OLD TARBALL_NEW
#
# TARBALL_OLD is a release tarball of the version the test starts on (unpacked as install.sh leaves it: versions/X,
# current, and ~/.local/bin/lernex), TARBALL_NEW the one the release host offers as latest. Both come from
# scripts/release/build.sh for the same flavor. Nothing here touches the real ~/.lernex, ~/.cache or ~/.local.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "usage: tests/release/update_e2e.sh TARBALL_OLD TARBALL_NEW" >&2
  exit 2
fi
old=$(cd "$(dirname "$1")" && pwd)/$(basename "$1")
new=$(cd "$(dirname "$2")" && pwd)/$(basename "$2")
here=$(cd "$(dirname "$0")/../.." && pwd)
flavor=$(basename "$old" .tar.gz)
flavor=${flavor#lernex-}
tmp=$(mktemp -d "${TMPDIR:-/tmp}/lernex-update-e2e.XXXXXX")
server_pid=""
cleanup() {
  [ -n "$server_pid" ] && kill "$server_pid" 2>/dev/null || true
  rm -rf "$tmp"
}
trap cleanup EXIT

fail() { echo "FAIL: $*" >&2; exit 1; }
version_in() { mkdir -p "$tmp/probe-$$"; tar -xzf "$1" -C "$tmp/probe-$$" && "$tmp/probe-$$/bin/lernex" --version | awk '{print $2}'; rm -rf "$tmp/probe-$$"; }

old_version=$(version_in "$old")
new_version=$(version_in "$new")
echo "flavor $flavor: old $old_version, new $new_version"

# The release host: one directory per release with its asset and SHA256SUMS, and LATEST naming the new one.
releases="$tmp/releases"
for v in "$old_version" "$new_version"; do
  mkdir -p "$releases/v$v"
  cp "$([ "$v" = "$old_version" ] && echo "$old" || echo "$new")" "$releases/v$v/lernex-$flavor.tar.gz"
  (cd "$releases/v$v" && "$here/scripts/release/sums.sh" . >/dev/null)
done
echo "$new_version" > "$releases/LATEST"

port=$((20000 + RANDOM % 20000))
python3 "$here/tests/release/serve.py" "$releases" "$port" 2>"$tmp/serve.log" &
server_pid=$!
for _ in $(seq 1 100); do
  grep -q " on http" "$tmp/serve.log" 2>/dev/null && break
  sleep 0.1
done

# The HOME a tarball install leaves behind.
home="$tmp/home"
mkdir -p "$home/.lernex/versions" "$home/.local/bin" "$home/.cache/lernex"
mkdir -p "$home/.lernex/versions/$old_version" && tar -xzf "$old" -C "$home/.lernex/versions/$old_version"
ln -s "$home/.lernex/versions/$old_version" "$home/.lernex/current"
ln -s "$home/.lernex/current/bin/lernex" "$home/.local/bin/lernex"
export HOME="$home" LERNEX_RELEASES="http://127.0.0.1:$port"
lernex="$home/.local/bin/lernex"
on_version() { "$lernex" --version | awk '{print $2}'; }

echo; echo "== version on the old install"; "$lernex" --version
[ "$(on_version)" = "$old_version" ] || fail "old install does not report $old_version"

echo; echo "== update --check reports the new release"
"$lernex" update --check
[ "$(on_version)" = "$old_version" ] || fail "--check changed the install"

echo; echo "== update installs it"
"$lernex" update
[ "$(on_version)" = "$new_version" ] || fail "not on $new_version after update"
[ "$(readlink "$home/.lernex/current")" = "$home/.lernex/versions/$new_version" ] || fail "current does not point at $new_version"
[ "$(readlink "$home/.lernex/previous")" = "$home/.lernex/versions/$old_version" ] || fail "previous does not point at $old_version"
[ -z "$(ls -A "$home/.lernex" | grep -E '^staging-|\.tmp' || true)" ] || fail "staging left behind"

echo; echo "== update again: already the latest"
"$lernex" update

echo; echo "== the daily notice, from the cache, on the old version"
"$home/.lernex/versions/$old_version/bin/lernex" --version >/dev/null
printf 'latest=%s\nchecked=%s\n' "$new_version" "$(date +%s)" > "$home/.cache/lernex/update-check"
notice=$("$home/.lernex/versions/$old_version/bin/lernex" run /nonexistent-model-folder 2>&1 || true)
head -1 <<<"$notice"
grep -q "lernex $new_version is out" <<<"$notice" || fail "no notice for $new_version"
quiet=$(LERNEX_NO_UPDATE_CHECK=1 "$home/.lernex/versions/$old_version/bin/lernex" run /nonexistent-model-folder 2>&1 || true)
if grep -q "is out" <<<"$quiet"; then fail "LERNEX_NO_UPDATE_CHECK=1 still printed the notice"; fi
echo "(with LERNEX_NO_UPDATE_CHECK=1 the same run prints no notice)"

echo; echo "== rollback returns to the old version"
"$lernex" update --rollback
[ "$(on_version)" = "$old_version" ] || fail "rollback did not return to $old_version"
[ "$(readlink "$home/.lernex/previous")" = "$home/.lernex/versions/$new_version" ] || fail "previous after rollback"

echo; echo "== a checksum mismatch installs nothing"
rm -rf "$home/.lernex/versions/$new_version"
printf '%064d  lernex-%s.tar.gz\n' 0 "$flavor" > "$releases/v$new_version/SHA256SUMS"
set +e
"$lernex" update; rc=$?
set -e
[ "$rc" -ne 0 ] || fail "a mismatch exited 0"
[ "$(on_version)" = "$old_version" ] || fail "a mismatch changed the install"
[ ! -e "$home/.lernex/versions/$new_version" ] || fail "a mismatch left $new_version installed"
[ -z "$(ls -A "$home/.lernex" | grep -E '^staging-' || true)" ] || fail "a mismatch left staging behind"

echo; echo "PASS: $flavor $old_version -> $new_version"
