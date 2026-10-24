// The autotuner through the backend ABI alone: the table keys the backend writes, the kernels it registers and
// their variants, and what the tuner reports for them. Needs the GPU: the test binary runs on the server.
#include <filesystem>
#include <string>
#include <vector>

#include "cuda/kernels.h"
#include "cuda/tunable.h"
#include "doctest.h"
#include "runtime/plan.h"
#include "runtime/tuner.h"

namespace {

// A problem in the given format, math and expert count, with the default shape.
lx_tune_problem problem(int weight, int math, int experts) { return lx_tune_problem{1, 1, 1, experts, weight, math}; }

// The table key the backend writes for a problem (lx_tune_key).
std::string key_of(const char* kernel, const lx_tune_problem& p) {
  char buf[256] = {};
  lx_tune_key(kernel, &p, buf, sizeof buf);
  return buf;
}

// The names of the kernels the backend registered (lx_tune_kernels).
std::vector<std::string> registered_kernels() {
  const char* names[256];
  const int total = lx_tune_kernels(names, 256);
  std::vector<std::string> out;
  for (int k = 0; k < total && k < 256; ++k) out.emplace_back(names[k]);
  return out;
}

}  // namespace

TEST_CASE("tune keys: the kernel's name, with format, math and experts when they differ") {
  CHECK(key_of("mhc_read", problem(0, 0, 1)) == "mhc_read");
  CHECK(key_of("gemm_w", problem(2, 2, 1)) == "gemm_w@w2m2");
  CHECK(key_of("moe", problem(2, 2, 128)) == "moe@w2m2@e128");
}

TEST_CASE("an unknown kernel runs nothing and has no variants") {
  float out[4] = {};
  const lx_tune_problem p = problem(0, 0, 1);
  CHECK(lx_tune_runnable("no_such_kernel") == 0);
  CHECK(lx_tune_variants("no_such_kernel") == 0);
  CHECK(lx_tune_variant_name("no_such_kernel", 0) == nullptr);
  CHECK(lx_tune_run("no_such_kernel", 0, &p, out, nullptr) == LX_ERR_NOT_SUPPORTED);
}

TEST_CASE("the backend registers memory_fuse as a runnable kernel with its two variants") {
  bool found = false;
  for (const std::string& name : registered_kernels()) found = found || name == "memory_fuse";
  REQUIRE(found);
  CHECK(lx_tune_runnable("memory_fuse") == 1);
  CHECK(lx_tune_variants("memory_fuse") == 2);
  CHECK(std::string(lx_tune_variant_name("memory_fuse", 0)) == "block");
  CHECK(std::string(lx_tune_variant_name("memory_fuse", 1)) == "cluster8");
}

TEST_CASE("the tuner times the runnable kernels, and each entry names a variant the backend registered") {
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

  const std::vector<std::string> kernels = registered_kernels();
  int runnable = 0;
  for (const std::string& name : kernels) runnable += lx_tune_runnable(name.c_str());
  CHECK(report.kernels == runnable);
  for (const auto& e : report.entries) {
    // an entry's key is its kernel's name, with the format, math and experts appended when they differ
    const std::string* kernel = nullptr;
    for (const std::string& name : kernels)
      if (e.key == name || e.key.rfind(name + "@", 0) == 0) kernel = &name;
    REQUIRE(kernel != nullptr);
    CHECK(e.variant >= 0);
    CHECK(e.variant < lx_tune_variants(kernel->c_str()));
    const char* vname = lx_tune_variant_name(kernel->c_str(), e.variant);
    REQUIRE(vname != nullptr);
    CHECK(e.variant_name == vname);
    CHECK(e.us > 0);
  }
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
