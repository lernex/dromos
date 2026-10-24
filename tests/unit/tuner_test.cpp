// The autotuner, on a kernel registered for the test: a correct variant, a variant that is wrong, and
// one this device does not run. The tuner must reject the wrong one and skip the other, whatever their
// speed. Needs the GPU: the test binary runs on the server.
#include <cuda_runtime.h>

#include <filesystem>
#include <string>
#include <vector>

#include "cuda/tunable.h"
#include "doctest.h"
#include "runtime/plan.h"
#include "runtime/tuner.h"

namespace {

// out[m, n] = scale * (i % 7 - 3): variant 0 is the reference, variant 1 is off by 10% and so must be
// rejected, and variant 2 is not supported here and must be skipped. The values are made on the host,
// so the test needs no device code of its own.
int run_test_scale(int variant, const lx::TuneProblem& p, float* out, void* stream) {
  if (variant == 2) return static_cast<int>(cudaErrorNotSupported);
  std::vector<float> host(static_cast<size_t>(p.M) * p.N);
  const float scale = variant == 1 ? 2.2f : 2.0f;
  for (size_t i = 0; i < host.size(); ++i) host[i] = scale * (static_cast<float>(i % 7) - 3.0f);
  cudaError_t err = cudaMemcpyAsync(out, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice,
                                    static_cast<cudaStream_t>(stream));
  if (err == cudaSuccess) err = cudaStreamSynchronize(static_cast<cudaStream_t>(stream));
  return static_cast<int>(err);
}

const lx::TunableRegistration reg_test_scale("test_scale", {"exact", "off_by_ten_percent", "unsupported"},
                                             &run_test_scale);

}  // namespace

TEST_CASE("tune keys: the kernel's name, with format, math and experts when they differ") {
  lx::TuneProblem p;
  CHECK(lx::tune_key("mhc_read", p) == "mhc_read");
  p.weight = 2;
  p.math = 2;
  CHECK(lx::tune_key("gemm_w", p) == "gemm_w@w2m2");
  p.experts = 128;
  CHECK(lx::tune_key("moe", p) == "moe@w2m2@e128");
}

TEST_CASE("the tuner never picks a variant that disagrees with variant 0, or one the device does not run") {
  std::vector<lx::TuneProblem> problems;
  for (int m : {1, 8, 64}) {
    lx::TuneProblem p;
    p.M = m;
    p.N = 96;
    p.K = 16;
    problems.push_back(p);
  }
  lx::runtime::TuneOptions options;
  options.warmup = 1;
  options.repeats = 3;
  options.sample_ms = 0.5;
  const lx::runtime::TuneReport report = lx::runtime::tune(problems, options);
  CHECK(report.rejected >= 3);  // variant 1, once per problem
  CHECK(report.skipped >= 3);   // variant 2, once per problem
  int test_entries = 0;
  for (const auto& e : report.entries) {
    if (e.key != "test_scale") continue;
    ++test_entries;
    CHECK(e.variant == 0);
    CHECK(e.variant_name == "exact");
    CHECK(e.us > 0);
  }
  CHECK(test_entries == 3);
}

TEST_CASE("a table saved and loaded sets the variants the kernels read") {
  const lx::runtime::DeviceInfo d = lx::runtime::query_device(0);
  lx::runtime::TuneReport report;
  report.gpu = d.slug();
  lx::runtime::TuneEntry e;
  e.key = "tuner_test_key";
  e.M = 4;
  e.N = 32;
  e.K = 8;
  e.variant = 1;
  e.variant_name = "cluster8";
  e.us = 3.5;
  e.us_reference = 4.0;
  report.entries.push_back(e);
  const std::string path = (std::filesystem::temp_directory_path() / "lx_tuner_test.json").string();
  lx::runtime::save_table(report, d, path);

  lx_tune_clear();
  std::string why;
  CHECK(lx::runtime::load_table(path, d, &why) == 1);
  CHECK(lx_tune_get("tuner_test_key", 3, 32, 8) == 1);  // M buckets to 4
  CHECK(lx_tune_get("tuner_test_key", 8, 32, 8) == -1);

  // a table for another GPU is refused, and says why
  lx::runtime::DeviceInfo other = d;
  other.name = "Some Other GPU";
  CHECK(lx::runtime::load_table(path, other, &why) == -1);
  CHECK(why.find("table is for") != std::string::npos);
  std::filesystem::remove(path);
  lx_tune_clear();
}
