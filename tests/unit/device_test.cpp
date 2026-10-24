// Device query and the architecture override. Needs the GPU: the test binary runs on the server.
#include <cuda_runtime.h>

#include <cmath>
#include <string>

#include "cuda/tunable.h"
#include "doctest.h"
#include "runtime/plan.h"

using lx::runtime::DeviceInfo;
using lx::runtime::query_device;

TEST_CASE("the override lowers the architecture the kernels see, and never raises it") {
  const int real = lx_arch();
  CHECK(real >= 80);
  CHECK(lx_set_arch_override(80) == 80);
  CHECK(lx_arch() == 80);
  CHECK(lx_set_arch_override(89) == 89);
  // a higher override is ignored: the card is what it is
  CHECK(lx_set_arch_override(200) == real);
  CHECK(lx_set_arch_override(0) == real);
}

TEST_CASE("the device report matches the card") {
  const DeviceInfo d = query_device(0);
  CHECK(d.real_cc() >= 80);
  CHECK(d.sms > 0);
  CHECK(d.total_bytes > 0);
  CHECK(d.free_bytes <= d.total_bytes);
  CHECK(d.bf16_mma);
  CHECK(d.int8_mma);
  CHECK(d.driver > 0);
  CHECK(d.runtime > 0);
  CHECK(d.slug().find("sm" + std::to_string(d.real_cc())) != std::string::npos);
  CHECK(d.slug().find(' ') == std::string::npos);
  CHECK(d.built);
  if (d.real_cc() >= 89) CHECK(d.fp8_mma);
  if (d.real_cc() >= 90) CHECK(d.clusters);
  if (d.real_cc() < 90) CHECK_FALSE(d.clusters);
}

TEST_CASE("the override changes the device the plan sees, not the card") {
  const DeviceInfo real = query_device(0);
  lx_set_arch_override(80);
  const DeviceInfo as80 = query_device(0);
  lx_set_arch_override(0);
  CHECK(as80.cc() == 80);
  CHECK(as80.real_cc() == real.real_cc());
  CHECK(as80.name == real.name);
  CHECK_FALSE(as80.fp8_mma);
  CHECK_FALSE(as80.clusters);
  CHECK(as80.slug().find("_as80") != std::string::npos);
}

TEST_CASE("the bandwidth of the card is its memory clock times bus width") {
  const DeviceInfo d = query_device(0);
  CHECK(d.bandwidth_gbs > 0);
  // The RTX 5070 Ti, the development card: 256-bit GDDR7 at 28 Gbps, 896 GB/s.
  if (d.name.find("5070 Ti") != std::string::npos) CHECK(std::fabs(d.bandwidth_gbs - 896.0) < 1.0);
}
