// dromos tune MODEL_DIR [--out PATH] [--log]
//
// Times every variant of every tunable kernel on the shapes the model's plan runs, on this GPU, and
// writes the winners to the cache (or to --out). Run it under the GPU lock: scripts/remote.sh SLOT x.
#include <cstdio>
#include <cstring>
#include <exception>
#include <map>
#include <stdexcept>
#include <string>

#include "core/config.h"
#include "runtime/plan.h"
#include "runtime/tuner.h"

int cmd_tune(int argc, char** argv) {
  using namespace lx::runtime;
  std::string dir, out;
  TuneOptions options;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--out" && i + 1 < argc) {
        out = argv[++i];
      } else if (arg == "--log") {
        options.log = true;
      } else if (!arg.empty() && arg[0] != '-' && dir.empty()) {
        dir = arg;
      } else {
        throw std::runtime_error("unknown argument " + arg);
      }
    }
    if (dir.empty()) throw std::runtime_error("usage: dromos tune MODEL_DIR [--out PATH] [--log]");

    const DeviceInfo device = query_device(0);
    const lx::MetisConfig config = lx::load_config(dir);
    const CheckpointInfo checkpoint = inspect_checkpoint(dir);
    const Plan plan = make_plan(device, config, checkpoint, PlanRequest{});
    const std::vector<lx::TuneProblem> problems = model_problems(config, plan);
    std::printf("tuning %s on %s: %zu problems\n", dir.c_str(), device.slug().c_str(), problems.size());

    const TuneReport report = tune(problems, options);
    std::map<std::string, std::map<std::string, int>> wins;  // kernel -> variant name -> entries it won
    for (const TuneEntry& e : report.entries) {
      const std::string kernel = e.key.substr(0, e.key.find('@'));
      ++wins[kernel][e.variant_name];
    }
    for (const auto& [kernel, by_variant] : wins) {
      std::printf("%-14s", kernel.c_str());
      for (const auto& [variant, count] : by_variant) std::printf("  %s x%d", variant.c_str(), count);
      std::printf("\n");
    }
    std::printf("%d kernels, %d problems, %d variants timed, %d skipped (not supported here), %d rejected\n",
                report.kernels, report.problems, report.timed, report.skipped, report.rejected);
    std::printf("%.1f s, %zu table entries\n", report.seconds, report.entries.size());

    const std::string path = out.empty() ? tune_cache_path(device) : out;
    save_table(report, device, path);
    std::printf("table %s\n", path.c_str());
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
