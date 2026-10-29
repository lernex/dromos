// dromos update [--check|--rollback]: the latest release of this install's flavor, checked and switched to, or the
// report of whether one exists (docs/RELEASE.md). `dromos run` and `dromos serve` print the daily notice themselves.
#include <cstdio>
#include <stdexcept>
#include <string>

#include "core/update.h"

int cmd_update(int argc, char** argv) {
  bool check_only = false, rollback = false;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--check") check_only = true;
    else if (a == "--rollback") rollback = true;
    else throw std::runtime_error("unknown option " + a + " (dromos update [--check|--rollback])");
  }
  if (check_only && rollback) throw std::runtime_error("--check and --rollback do not go together");
  return lx::update::run_update(check_only, rollback);
}
