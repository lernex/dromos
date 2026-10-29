// dromos pull MODEL: downloads a model into the cache, resumably and verified.
// dromos list: the models dromos knows, and which of them are installed.
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#include "cli/common.h"
#include "core/hub.h"

namespace cli = lx::cli;
namespace hub = lx::hub;

namespace {

std::string short_path(const std::filesystem::path& p) {
  const char* home = std::getenv("HOME");
  const std::string s = p.string();
  if (home && s.rfind(home, 0) == 0) return "~" + s.substr(std::string(home).size());
  return s;
}

}  // namespace

int cmd_pull(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: dromos pull MODEL   (praxis, logos, praxis:fp8, logos:nvfp4, or owner/name)\n");
    return 2;
  }
  const std::string name = argv[1];
  if (cli::looks_like_path(name)) throw std::runtime_error(name + " is a folder; there is nothing to pull");
  const hub::ModelRef ref = hub::resolve(name);
  const std::filesystem::path dir = hub::install_dir(ref.repo);
  if (hub::installed(dir)) {
    std::printf("%s is already installed at %s\n", name.c_str(), short_path(dir).c_str());
    return 0;
  }
  const std::string token = hub::hf_token();
  const hub::RemoteModel remote = hub::describe(ref.repo, token);
  std::printf("%s  %s, %zu files\n", ref.repo.c_str(), cli::human_bytes(remote.total_bytes()).c_str(), remote.files.size());
  const bool live = cli::stderr_is_terminal();
  std::string printed_file;
  hub::pull(remote, dir, token, [&](const hub::Progress& p) {
    const double fraction = p.model_total ? static_cast<double>(p.model_done) / static_cast<double>(p.model_total) : 1.0;
    if (live) {
      std::fprintf(stderr, "\r  %5.1f%%  %s / %s  %-40.40s", 100.0 * fraction, cli::human_bytes(p.model_done).c_str(),
                   cli::human_bytes(p.model_total).c_str(), p.file.c_str());
      std::fflush(stderr);
    } else if (p.done >= p.total && p.file != printed_file) {
      // a log or pipe cannot redraw a line: one line per finished file instead
      std::fprintf(stderr, "  %5.1f%%  %s / %s  %s\n", 100.0 * fraction, cli::human_bytes(p.model_done).c_str(),
                   cli::human_bytes(p.model_total).c_str(), p.file.c_str());
      printed_file = p.file;
    }
  });
  if (live) std::fprintf(stderr, "\r%100s\r", "");
  std::printf("Installed %s at %s\n", name.c_str(), short_path(dir).c_str());
  std::printf("Run it with: dromos run %s\n", name.c_str());
  return 0;
}

int cmd_list(int argc, char** argv) {
  (void)argc;
  (void)argv;
  std::printf("Models\n");
  for (const hub::Alias& a : hub::aliases()) {
    for (const char* format : {"bf16", "fp8", "nvfp4"}) {
      const std::string repo = hub::alias_repo(a, format);
      const std::filesystem::path dir = hub::install_dir(repo);
      const bool ok = hub::installed(dir);
      const std::string label = a.name + (std::string(format) == "bf16" ? "" : std::string(":") + format);
      std::printf("  %-14s %-28s %s\n", label.c_str(), repo.c_str(),
                  ok ? ("installed  " + short_path(dir)).c_str() : "not installed");
    }
  }
  return 0;
}
