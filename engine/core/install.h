// Where the running lernex lives. An install is a tree (docs/RELEASE.md): bin/lernex beside share/lernex and lib/.
// Everything is found from the real path of the executable, so a symlink (~/.local/bin/lernex) leads to the same tree.
#pragma once

#include <string>

namespace lx::install {

// The running executable (symlinks resolved), or "".
std::string executable_path();
// The directory of the running executable, with symlinks resolved, or "" when it cannot be found.
std::string executable_dir();

// <executable dir>/../share/lernex: the tree's shared data (tuning tables, the channel file). The directory is
// returned whether or not it exists; "" when the executable cannot be found.
std::string share_dir();

// How this install was made: "tarball", "homebrew", "pip" or "docker", from share/lernex/channel. "" for a source
// build, which has no such file. A tree that names no known channel is "" too.
std::string channel();

}  // namespace lx::install
