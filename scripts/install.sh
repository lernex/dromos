#!/bin/sh
# Installs Lernex, the inference engine for Praxis and Logos, from its release:
#
#   curl -fsSL https://github.com/lernex/lernex-engine/releases/latest/download/install.sh | sh
#
# It picks the build this machine can run (CUDA when an NVIDIA driver is new enough, else CPU; Metal on a
# Mac), checks it against SHA256SUMS, unpacks it to ~/.lernex/versions/X.Y.Z, points ~/.lernex/current at it
# and links ~/.local/bin/lernex. Run it again to upgrade. Nothing here needs root.
#
# Environment:
#   LERNEX_VERSION=X.Y.Z              install this release (default: the latest)
#   LERNEX_FLAVOR=cpu|cuda12|cuda13   install this build (default: the best one for this machine)
#   LERNEX_RELEASES=URL               the release host, laid out like GitHub's (default: GitHub)
#
# Needs curl or wget, tar, gzip, sha256sum or shasum, awk and sed.

set -eu

# What each CUDA build needs from the machine. The driver is compared by its branch (580 for 580.65.06) and the
# first GPU by its compute capability (sm_80 is 80). cuda13 is preferred when the release ships it.
CUDA13_MIN_DRIVER=580
CUDA13_MIN_SM=80
CUDA12_MIN_DRIVER=525
CUDA12_MIN_SM=80

DEFAULT_RELEASES=https://github.com/lernex/lernex-engine/releases
MACOS_MIN_MAJOR=14

say() { printf 'lernex: %s\n' "$*"; }
warn() { printf 'lernex: %s\n' "$*" >&2; }
die() { printf 'lernex: %s\n' "$*" >&2; exit 1; }
have() { command -v "$1" >/dev/null 2>&1; }

# A path with the home folder written as ~, for messages.
short() {
  case "$1" in
    "$HOME"/*) printf '~%s' "${1#"$HOME"}" ;;
    *) printf '%s' "$1" ;;
  esac
}

usage() {
  cat <<'EOF'
usage: install.sh [--help]

Installs Lernex into ~/.lernex and links ~/.local/bin/lernex. Run it again to upgrade.

  LERNEX_VERSION=X.Y.Z              a release other than the latest
  LERNEX_FLAVOR=cpu|cuda12|cuda13   a build other than the one this machine would get
  LERNEX_RELEASES=URL               another release host, laid out like GitHub's
EOF
}

need_tools() {
  have tar || die "tar is needed to unpack Lernex"
  have gzip || die "gzip is needed to unpack Lernex"
  if ! have curl && ! have wget; then die "curl or wget is needed to download Lernex"; fi
  if ! have sha256sum && ! have shasum; then die "sha256sum or shasum is needed to check the download"; fi
  [ -n "${HOME:-}" ] || die "HOME is not set"
}

# Sets os, platform (linux-x86_64, linux-aarch64 or darwin-arm64) and label, or stops with the reason.
detect_platform() {
  os=$(uname -s)
  machine=$(uname -m)
  case "$os" in
    Linux)
      case "$machine" in
        x86_64|amd64) arch=x86_64 ;;
        aarch64|arm64) arch=aarch64 ;;
        *) die "there is no Lernex build for $machine on Linux (x86_64 and aarch64 are supported)" ;;
      esac
      if [ -r /proc/version ] && grep -qi microsoft /proc/version && ! grep -qi wsl2 /proc/version; then
        die "WSL1 is not supported: the GPU needs a real Linux kernel. Convert the distro with: wsl --set-version <distro> 2"
      fi
      if ls /lib/ld-musl-*.so.1 /usr/lib/ld-musl-*.so.1 >/dev/null 2>&1; then
        die "this is a musl system (Alpine, for example), and Lernex is built for glibc 2.28 or newer"
      fi
      platform=linux-$arch
      label="Linux $arch"
      ;;
    Darwin)
      case "$machine" in
        arm64) ;;
        x86_64)
          if [ "$(sysctl -n hw.optional.arm64 2>/dev/null || echo 0)" = 1 ]; then
            die "this terminal runs under Rosetta, so it is x86_64. Open a native terminal (or start one with: arch -arm64 zsh) and run the installer again"
          fi
          die "Intel Macs are not supported: Lernex runs on Apple silicon (M1 or later)"
          ;;
        *) die "there is no Lernex build for $machine on macOS" ;;
      esac
      major=$(sw_vers -productVersion | cut -d. -f1)
      if [ "$major" -lt "$MACOS_MIN_MAJOR" ] 2>/dev/null; then
        die "Lernex needs macOS $MACOS_MIN_MAJOR or newer, and this is $(sw_vers -productVersion)"
      fi
      platform=darwin-arm64
      label="macOS on Apple silicon"
      ;;
    *)
      die "there is no Lernex build for $os. Linux (x86_64, aarch64) and macOS (Apple silicon) are supported; on Windows, use WSL2"
      ;;
  esac
}

# Sets driver, gpu and sm from NVIDIA's tools: the driver branch, the first GPU's name and its compute capability
# as sm (12.0 is 120). Empty when a value cannot be read.
probe_nvidia() {
  driver=""; gpu=""; cc=""; sm=""
  if have nvidia-smi; then
    driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null </dev/null | sed -n 1p | tr -d ' \r')
    gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null </dev/null | sed -n 1p | tr -d '\r')
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null </dev/null | sed -n 1p | tr -d ' \r')
  fi
  if [ -z "$driver" ] && [ -r /proc/driver/nvidia/version ]; then
    driver=$(awk 'NR == 1 { for (i = 1; i <= NF; i++) if ($i ~ /^[0-9]+[.][0-9]+([.][0-9]+)?$/) { print $i; exit } }' /proc/driver/nvidia/version)
  fi
  case "$driver" in [0-9]*.[0-9]*) ;; *) driver="" ;; esac
  case "$cc" in [0-9]*.[0-9]*) sm=$(printf '%s' "$cc" | awk -F. '{ print $1 * 10 + $2 }') ;; *) cc="" ;; esac
  [ -n "$gpu" ] || gpu="NVIDIA GPU"
}

floor_driver() { case "$1" in cuda13) echo "$CUDA13_MIN_DRIVER" ;; *) echo "$CUDA12_MIN_DRIVER" ;; esac; }
floor_sm() { case "$1" in cuda13) echo "$CUDA13_MIN_SM" ;; *) echo "$CUDA12_MIN_SM" ;; esac; }

# The asset name for a flavor on this platform.
asset_for() {
  case "$platform" in
    darwin-arm64) echo "lernex-darwin-arm64.tar.gz" ;;
    *) echo "lernex-$platform-$1.tar.gz" ;;
  esac
}

# Sets flavor, asset and why (one line, for the user). Reads the SHA256SUMS in $1.
choose_build() {
  sums=$1
  case "$platform" in
    darwin-arm64)
      if [ -n "${LERNEX_FLAVOR:-}" ] && [ "$LERNEX_FLAVOR" != metal ]; then die "LERNEX_FLAVOR does not apply on macOS: the only build is metal"; fi
      flavor=metal; why="" ;;
    linux-aarch64)
      if [ -n "${LERNEX_FLAVOR:-}" ] && [ "$LERNEX_FLAVOR" != cpu ]; then die "LERNEX_FLAVOR does not apply on aarch64 Linux: the only build is cpu"; fi
      flavor=cpu; why="" ;;
    linux-x86_64)
      probe_nvidia
      if [ -n "${LERNEX_FLAVOR:-}" ]; then
        case "$LERNEX_FLAVOR" in
          cpu|cuda12|cuda13) flavor=$LERNEX_FLAVOR; why="LERNEX_FLAVOR asks for it" ;;
          *) die "LERNEX_FLAVOR must be cpu, cuda12 or cuda13, not $LERNEX_FLAVOR" ;;
        esac
      else
        flavor=""
        if [ -z "$driver" ]; then
          flavor=cpu; why="no NVIDIA driver found, so no CUDA"
        elif [ -z "$sm" ]; then
          flavor=cpu
          if have nvidia-smi; then
            why="the NVIDIA driver $driver is here, but nvidia-smi cannot read the GPU's compute capability"
          else
            why="the NVIDIA driver $driver is here, but nvidia-smi is not on PATH to check the GPU"
          fi
        else
          for candidate in cuda13 cuda12; do
            [ "${driver%%.*}" -ge "$(floor_driver "$candidate")" ] || continue
            [ "$sm" -ge "$(floor_sm "$candidate")" ] || continue
            if [ -n "$(listed_sha "$sums" "$(asset_for "$candidate")")" ]; then
              flavor=$candidate; why="$gpu, driver $driver"
              break
            fi
          done
          if [ -z "$flavor" ]; then
            flavor=cpu
            if [ "${driver%%.*}" -lt "$CUDA12_MIN_DRIVER" ]; then
              why="driver $driver is older than $CUDA12_MIN_DRIVER, the oldest CUDA build needs"
            elif [ "$sm" -lt "$CUDA12_MIN_SM" ]; then
              why="$gpu is sm_$sm, and the CUDA builds need sm_$CUDA12_MIN_SM or newer"
            else
              why="this release has no CUDA build for driver $driver"
            fi
          fi
        fi
      fi
      ;;
  esac
  asset=$(asset_for "$flavor")
}

# The SHA-256 of a file, as a bare hex string.
sha256_of() {
  if have sha256sum; then sha256sum "$1" | awk '{ print $1 }'; else shasum -a 256 "$1" | awk '{ print $1 }'; fi
}

# The hash SHA256SUMS lists for an asset (empty when it does not list it). Accepts the binary-mode "*" too.
listed_sha() {
  awk -v want="$2" '{ name = $2; sub(/^\*/, "", name); if (name == want) { print $1; exit } }' "$1"
}

# Downloads URL to FILE, quietly unless stderr is a terminal (then with a progress bar).
fetch() {
  if have curl; then
    if [ -t 2 ]; then
      curl -fL --retry 3 --connect-timeout 20 --progress-bar -o "$2" "$1"
    else
      curl -fsL --retry 3 --connect-timeout 20 -o "$2" "$1" </dev/null
    fi
  else
    wget -q -T 20 -O "$2" "$1" </dev/null
  fi
}

# mv that replaces a symlink in place (rename(2)) instead of moving into the folder it points to.
replace_link() {
  if [ "$os" = Darwin ]; then mv -fh "$1" "$2"; else mv -fT "$1" "$2"; fi
}

# The folder the current symlink points to, absolute, or empty.
current_dir() {
  target=$(readlink "$HOME/.lernex/current" 2>/dev/null || true)
  [ -n "$target" ] || return 0
  case "$target" in
    /*) printf '%s' "$target" ;;
    *) printf '%s/.lernex/%s' "$HOME" "$target" ;;
  esac
}

# Refuses before anything is downloaded when ~/.local/bin/lernex is a real file the installer must not replace.
check_bin_free() {
  if [ -e "$HOME/.local/bin/lernex" ] && [ ! -L "$HOME/.local/bin/lernex" ]; then
    die "$(short "$HOME/.local/bin/lernex") is a file, not a link from the installer. Move it aside and run the installer again"
  fi
}

link_bin() {
  bin_dir="$HOME/.local/bin"
  target="$HOME/.lernex/current/bin/lernex"
  mkdir -p "$bin_dir"
  if [ "$(readlink "$bin_dir/lernex" 2>/dev/null || true)" = "$target" ]; then return 0; fi
  ln -s "$target" "$bin_dir/.lernex-link.$$"
  replace_link "$bin_dir/.lernex-link.$$" "$bin_dir/lernex"
  say "linked $(short "$bin_dir/lernex")"
}

# Adds ~/.local/bin to the shell's startup file once, unless PATH already has it.
path_setup() {
  case ":$PATH:" in
    *":$HOME/.local/bin:"*) return 0 ;;
  esac
  marker="# Added by the Lernex installer"
  # The literal text for the startup file: $HOME and $PATH are for the shell that reads it, not this script.
  # shellcheck disable=SC2016
  line='export PATH="$HOME/.local/bin:$PATH"'
  case "$(basename "${SHELL:-sh}")" in
    zsh) rc="${ZDOTDIR:-$HOME}/.zshrc" ;;
    bash)
      if [ "$os" = Darwin ]; then rc="$HOME/.bash_profile"; else rc="$HOME/.bashrc"; fi
      ;;
    fish)
      rc="$HOME/.config/fish/config.fish"
      # shellcheck disable=SC2016
      line='contains -- "$HOME/.local/bin" $PATH; or set -gx PATH "$HOME/.local/bin" $PATH'
      ;;
    *) rc="$HOME/.profile" ;;
  esac
  if [ -f "$rc" ] && grep -qF "$marker" "$rc"; then
    say "$(short "$rc") already adds ~/.local/bin to PATH: open a new terminal to use lernex"
    return 0
  fi
  mkdir -p "$(dirname "$rc")"
  printf '\n%s\n%s\n' "$marker" "$line" >> "$rc"
  say "added ~/.local/bin to PATH in $(short "$rc"): open a new terminal to use lernex"
}

# Models download through the system's libcurl, which the engine opens at run time.
check_libcurl() {
  [ "$os" = Darwin ] && return 0
  for dir in /usr/lib /usr/lib64 /lib /lib64 /usr/lib/x86_64-linux-gnu /usr/lib/aarch64-linux-gnu /usr/local/lib; do
    [ -e "$dir/libcurl.so.4" ] && return 0
  done
  if have ldconfig && ldconfig -p 2>/dev/null | grep -q 'libcurl\.so\.4'; then return 0; fi
  if [ -x /sbin/ldconfig ] && /sbin/ldconfig -p 2>/dev/null | grep -q 'libcurl\.so\.4'; then return 0; fi
  warn "libcurl is missing, so lernex cannot download models yet. Install it: sudo apt-get install -y curl (Debian, Ubuntu), sudo dnf install -y libcurl (Fedora), or sudo pacman -S curl (Arch)"
}

# The version a tree reports, from its own bin/lernex (no backend loads for --version).
tree_version() {
  "$1/bin/lernex" --version </dev/null 2>/dev/null | awk 'NR == 1 && $1 == "lernex" { print $2; exit }'
}

cleanup() {
  rm -rf "$tmp"
  if [ -n "$stage" ]; then rm -rf "$stage"; fi
  if [ -n "$replaced" ]; then rm -rf "$replaced"; fi
}

main() {
  case "${1:-}" in
    "") ;;
    -h|--help) usage; exit 0 ;;
    *) die "unknown argument $1 (try --help)" ;;
  esac
  need_tools
  detect_platform
  check_bin_free

  releases=${LERNEX_RELEASES:-$DEFAULT_RELEASES}
  releases=${releases%/}
  if [ -n "${LERNEX_VERSION:-}" ]; then
    version=${LERNEX_VERSION#v}
    base="$releases/download/v$version"
  else
    version=""
    base="$releases/latest/download"
  fi

  versions="$HOME/.lernex/versions"
  tmp=$(mktemp -d "${TMPDIR:-/tmp}/lernex-install.XXXXXX") || die "cannot make a temporary folder"
  stage=""; replaced=""
  trap cleanup EXIT
  trap 'exit 130' INT
  trap 'exit 143' TERM

  if ! fetch "$base/SHA256SUMS" "$tmp/SHA256SUMS"; then
    if [ -n "$version" ]; then
      die "there is no release $version at $releases (check LERNEX_VERSION)"
    fi
    die "cannot download the release from $base (check your connection, or LERNEX_RELEASES)"
  fi

  choose_build "$tmp/SHA256SUMS"
  expected=$(listed_sha "$tmp/SHA256SUMS" "$asset")
  if [ -z "$expected" ]; then
    die "this release has no $asset, so there is no $flavor build for this machine"
  fi
  if [ -n "$why" ]; then say "$label: $flavor build, $why"; else say "$label: $flavor build"; fi
  case "$flavor" in
    cuda12|cuda13)
      if [ -n "${LERNEX_FLAVOR:-}" ] && { [ -z "$driver" ] || [ "${driver%%.*}" -lt "$(floor_driver "$flavor")" ]; }; then
        warn "$flavor needs an NVIDIA driver of $(floor_driver "$flavor") or newer, and this machine has ${driver:-none}: the engine may not start"
      fi
      ;;
  esac

  # Already current: the installed tree was made from this asset, byte for byte.
  current=$(current_dir)
  if [ -n "$current" ] && [ -f "$current/.install" ] && [ "$(cat "$current/.install")" = "$asset $expected" ]; then
    say "lernex $(basename "$current") is already installed and current"
    link_bin
    path_setup
    check_libcurl
    say "done. Try: lernex doctor"
    exit 0
  fi

  say "downloading $asset"
  fetch "$base/$asset" "$tmp/$asset" || die "cannot download $base/$asset"
  actual=$(sha256_of "$tmp/$asset")
  if [ "$actual" != "$expected" ]; then
    die "$asset does not match SHA256SUMS (expected $expected, got $actual). Nothing was installed; try again"
  fi

  mkdir -p "$versions"
  stage=$(mktemp -d "$versions/.incoming.XXXXXX") || die "cannot make a folder in $(short "$versions")"
  tar -xzf "$tmp/$asset" -C "$stage" || die "cannot unpack $asset"
  [ -x "$stage/bin/lernex" ] || die "$asset has no bin/lernex, so it is not a Lernex build"
  ver=$(tree_version "$stage")
  case "$ver" in
    [0-9]*.[0-9]*.[0-9]*) ;;
    *) die "the lernex in $asset does not report a version (it printed: $("$stage/bin/lernex" --version </dev/null 2>&1 | head -1))" ;;
  esac
  printf '%s %s\n' "$asset" "$expected" > "$stage/.install"

  dest="$versions/$ver"
  if [ -e "$dest" ]; then
    replaced="$versions/.replaced.$$"
    mv "$dest" "$replaced"
  fi
  mv "$stage" "$dest"
  stage=""

  # Switch, then tidy. The previous version stays for one rollback; everything older goes.
  previous=""
  if [ -n "$current" ]; then previous=$(basename "$current"); fi
  ln -s "versions/$ver" "$HOME/.lernex/.current-link.$$"
  replace_link "$HOME/.lernex/.current-link.$$" "$HOME/.lernex/current"
  for old in "$versions"/*; do
    [ -d "$old" ] || continue
    name=$(basename "$old")
    case "$name" in
      "$ver"|"$previous") ;;
      *) rm -rf "$old" ;;
    esac
  done
  if [ -n "$previous" ] && [ "$previous" != "$ver" ]; then
    say "installed lernex $ver in $(short "$dest") (was $previous)"
  else
    say "installed lernex $ver in $(short "$dest")"
  fi

  link_bin
  path_setup
  check_libcurl
  say "done. Try: lernex doctor"
}

main "$@"
