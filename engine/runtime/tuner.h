// The autotuner. For the Metis shapes a plan runs, it times every variant of every registered tunable
// kernel and keeps the winner per (kernel, M bucket, N, K). A variant whose output disagrees with
// variant 0 is never chosen, whatever its speed. The winners go to a table on disk, one per GPU,
// driver and engine version, and are loaded at startup into lx_tune_set.
#pragma once

#include <string>
#include <vector>

#include "core/config.h"
#include "cuda/tunable.h"
#include "runtime/plan.h"

namespace lx::runtime {

// Tables from another engine version are ignored: a kernel's variants change with the kernels.
inline constexpr const char* kEngineVersion = "0.2";

struct TuneOptions {
  int warmup = 2;          // launches before timing
  int repeats = 7;         // timed samples; the median is kept
  double sample_ms = 2.0;  // each sample runs the variant long enough to last about this many ms
  bool log = false;        // print each problem's timings as they are measured
};

struct TuneEntry {
  std::string key;          // tune_key of the kernel and the problem; kernels call lx_tune_get with it
  int M = 0, N = 0, K = 0;  // M is the bucket
  int variant = 0;
  std::string variant_name;
  double us = 0;            // the winner's time per launch
  double us_reference = 0;  // variant 0's
};

struct TuneReport {
  std::vector<TuneEntry> entries;
  int kernels = 0, problems = 0, timed = 0, skipped = 0, rejected = 0;
  double seconds = 0;
  std::string gpu;  // DeviceInfo::slug()
};

// The problems a plan's model runs: every projection at each M bucket (1 to 4096, powers of two), the
// routed expert stacks, and the hyper-connection row shape (D, S) the stream kernels use.
std::vector<TuneProblem> model_problems(const MetisConfig& config, const Plan& plan);

// Times every variant of every registered tunable kernel that has a runner, on each problem, on the
// current device. The caller holds the GPU (scripts/remote.sh SLOT x).
TuneReport tune(const std::vector<TuneProblem>& problems, const TuneOptions& options);

// The cache file for a device: ~/.cache/lernex/tune/<slug>_<driver>_<engine>.json.
std::string tune_cache_path(const DeviceInfo& device);
// The table shipped for a GPU: <executable>/../share/lernex/tuning/<slug>_<engine>.json in an install, else the
// same name in the source tree (tuning/). The path is returned whether or not the file exists.
std::string shipped_table_path(const DeviceInfo& device);

void save_table(const TuneReport& report, const DeviceInfo& device, const std::string& path);
// Applies a table to the kernels (lx_tune_set). Returns the entries applied, or -1 when the file does
// not exist or is for another GPU or engine version (`why` says which).
int load_table(const std::string& path, const DeviceInfo& device, std::string* why = nullptr);
// The table this device should use: the cache when it has one, else the shipped table. Returns the
// path it loaded, or an empty string when there is none.
std::string load_tuning(const DeviceInfo& device);

}  // namespace lx::runtime
