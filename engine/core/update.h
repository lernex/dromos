// Releases: `lernex update` (install the latest release of this install's flavor, verified, and switch to it) and the
// daily notice that `run` and `serve` print. The release layout, the install tree and the channels are in
// docs/RELEASE.md. HTTP goes through core/http (libcurl, opened at run time).
#pragma once

#include <filesystem>
#include <string>

namespace lx::update {

// "X.Y.Z", then an optional "-pre" and "+build" tail (rc1, dev.3, gabc1234). Only the numbers and the pre-release
// tail order versions; a release is newer than its own pre-releases.
bool valid(const std::string& version);
// <0, 0 or >0 as a is older than, the same as, or newer than b. An invalid version sorts oldest.
int compare(const std::string& a, const std::string& b);

// The version a release page's URL names: .../releases/tag/vX.Y.Z gives X.Y.Z, and anything else gives "".
std::string version_from_url(const std::string& url);
// lernex-FLAVOR.tar.gz, the asset a flavor is released as.
std::string asset_name(const std::string& flavor);
// The SHA-256 (lower-case hex) that a sha256sum-format listing gives for `file`, or "" when it lists none.
std::string checksum_for(const std::string& sums, const std::string& file);
// The SHA-256 (lower-case hex) of a file's contents. Throws std::runtime_error when it cannot be read.
std::string sha256_file(const std::filesystem::path& file);

// Where releases are downloaded from: LERNEX_RELEASES, else the GitHub releases page of lernex-engine.
std::string releases_url();
// The newest release, from the release host's /latest redirect (a pre-release is never the latest). Throws
// std::runtime_error when the host names none. `timeout_seconds` bounds the request.
std::string latest_version(const std::string& releases, long timeout_seconds);

// The command a channel updates an install with, for the user to run: "lernex update" for tarball, and the
// channel's own command for the others. "" for a channel this build does not know.
std::string update_command(const std::string& channel);

// The daily check's cache (~/.cache/lernex/update-check): the newest release the last check saw, and when the last
// check ran. Both are written together, so a failed check still counts as the day's check.
struct Cache {
  std::string latest;     // a version, or "" when no check has succeeded
  long long checked = 0;  // unix seconds of the last attempt, or 0
};
std::filesystem::path cache_path();
Cache read_cache(const std::filesystem::path& file);  // a missing or garbled file reads as an empty cache
void write_cache(const std::filesystem::path& file, const Cache& cache);  // atomic, and best effort
// True when the last check is a day old or more, at unix time `now`.
bool stale(const Cache& cache, long long now);
// The one line the notice prints, or "" when nothing newer than `current` is known or the channel has no command.
std::string notice(const std::string& current, const Cache& cache, const std::string& channel);

// `lernex update`: with check_only, reports whether a newer release exists; with rollback, switches back to the
// version the last update replaced. Otherwise downloads the latest release of this flavor, checks it against
// SHA256SUMS, unpacks it under ~/.lernex/versions, switches ~/.lernex/current to it, and keeps the version it
// replaced as ~/.lernex/previous. Returns the exit status. Throws std::runtime_error with a plain message.
int run_update(bool check_only, bool rollback);

// Called by `run` and `serve` before anything else. Prints the notice from the cache, if it names a newer release,
// and when the cache is a day old refreshes it on a detached thread with a short timeout. It never waits on the
// network. Off with LERNEX_NO_UPDATE_CHECK=1, and for a source build (which has no channel).
void start_daily_check();

}  // namespace lx::update
