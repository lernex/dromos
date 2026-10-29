// What differs between Linux, macOS and Windows beyond a call's spelling (that is posix.h): where the executable and
// the user's folders are, running another program, the terminal, and the links that name the current install.
//
// The install's own names live here and in scripts/install.sh / install.ps1, so renaming the engine is one sweep:
//
//   Linux, macOS   ~/.dromos (versions, current, previous)      ~/.cache/dromos (models, tune, update-check)
//   Windows        %LOCALAPPDATA%\Programs\Dromos               %LOCALAPPDATA%\Dromos
//
// Windows keeps the product's capitalised name for its folders, as per-user programs there do.
#pragma once

#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

namespace lx::platform {

// The running executable, with symlinks (and on Windows junctions) resolved, or "" when it cannot be found.
std::filesystem::path executable_path();

// The user's home folder: HOME, and on Windows USERPROFILE before it. Throws std::runtime_error when there is none.
std::filesystem::path home();
// Where release installs live: versions/X.Y.Z, and the current and previous links.
std::filesystem::path install_root();
// The per-user cache: models/, tune/ and update-check (DROMOS_MODELS moves only the models).
std::filesystem::path cache_root();
// A path as messages show it: under the home folder as ~/..., and on Windows under %LOCALAPPDATA% as that.
std::string display(const std::filesystem::path& path);
// "~", "~/rest" (and "~\rest" on Windows) under the home folder; any other path as it is.
std::filesystem::path expand_home(const std::string& path);

// Runs a program and waits for it. args[0] is the program, looked up on PATH when it names no directory. With
// `output`, the program's standard output (and its standard error too, with merge_stderr) is handed over as it
// arrives; without it, both go where this process's go. Returns the exit status, or -1 when the program did not
// start or did not run to an exit.
using OutputSink = std::function<void(const char* data, std::size_t size)>;
int run_process(const std::vector<std::string>& args, const OutputSink& output = {}, bool merge_stderr = false);

// The terminal. console_init() sets up a Windows console for the engine's output (UTF-8 in and out, and VT
// sequences for colour and cursor movement), and puts it back as it was at exit; elsewhere it does nothing.
void console_init();
bool is_terminal(std::FILE* stream);
// The width of the terminal `stream` writes to, in columns, or 0 when it is not a terminal.
int terminal_columns(std::FILE* stream);
// One line of standard input, without its line break, in UTF-8 (a Windows console is read as UTF-16, so any
// character typed arrives intact). 1 for a line, 0 at the end of the input, -1 when Ctrl-C ended the read.
int read_line(std::string& line);

// The install's switchable names (current, previous): a symlink on Linux and macOS, a directory junction on
// Windows (which needs no privilege). set_link points `link` at `target`, replacing what it pointed at in one step;
// link_target reads one back, absolute, or "" when `link` is not a link.
void set_link(const std::filesystem::path& link, const std::filesystem::path& target);
std::filesystem::path link_target(const std::filesystem::path& link);
// Removes the link itself, never what it points at. Nothing happens when it does not exist.
void remove_link(const std::filesystem::path& link);

// This process's id.
long process_id();

}  // namespace lx::platform
