// lernex doctor [MODEL_DIR] [--weights bf16|int8|fp8|nvfp4] [--context N] [--sequences N]
//
// Prints what this GPU can do, which kernels this build has, which tuning table is in use, and, with a
// model folder, the plan the engine would run it under, in plain language.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "core/config.h"
#include "runtime/plan.h"
#include "runtime/tuner.h"

namespace {

const char* weight_flag(const std::string& name) {
  if (name == "bf16") return "0";
  if (name == "int8") return "1";
  if (name == "fp8") return "2";
  if (name == "nvfp4") return "3";
  return nullptr;
}

std::string archs_text(const std::vector<int>& archs) {
  std::string out;
  for (int cc : archs) out += (out.empty() ? "" : " ") + std::string("sm_") + std::to_string(cc);
  return out;
}

void print_device(const lx::runtime::DeviceInfo& d) {
  using lx::runtime::DeviceInfo;
  std::printf("GPU        %s (device %d)\n", d.name.c_str(), d.ordinal);
  if (d.cc() != d.real_cc())
    std::printf("arch       sm_%d as the kernels see it (the card is sm_%d; LX_ARCH_OVERRIDE lowers it)\n", d.cc(),
                d.real_cc());
  else
    std::printf("arch       sm_%d\n", d.cc());
  std::printf("SMs        %d, L2 %.0f MB\n", d.sms, static_cast<double>(d.l2_bytes) / 1e6);
  std::printf("memory     %.2f GB total, %.2f GB free now\n", static_cast<double>(d.total_bytes) / 1e9,
              static_cast<double>(d.free_bytes) / 1e9);
  std::printf("bandwidth  %.0f GB/s (memory clock x bus width)\n", d.bandwidth_gbs);
  std::string math = "BF16";
  if (d.int8_mma) math += ", INT8";
  if (d.fp8_mma) math += ", FP8";
  if (d.fp4_mma) math += ", FP4";
  std::printf("math       tensor cores: %s\n", math.c_str());
  std::printf("clusters   %s\n", d.clusters ? "yes (sm_90 and up: the cluster paths run)"
                                            : "no (the one-block paths run)");
  std::printf("CUDA       driver %s, runtime %s\n", d.driver_text.c_str(), d.runtime_text.c_str());
  std::printf("build      kernels for %s\n", archs_text(lx::runtime::built_archs()).c_str());
  if (!d.built)
    std::printf("           this card (sm_%d) has no kernels in this build: rebuild with -DLX_ARCHS=all\n", d.real_cc());
}

}  // namespace

int cmd_doctor(int argc, char** argv) {
  using namespace lx::runtime;
  std::string dir;
  PlanRequest req;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      if (arg == "--weights" && i + 1 < argc) {
        const char* code = weight_flag(argv[++i]);
        if (!code) throw std::runtime_error("--weights takes bf16, int8, fp8 or nvfp4");
        req.weights = std::atoi(code);
      } else if (arg == "--context" && i + 1 < argc) {
        req.context = std::stoi(argv[++i]);
      } else if (arg == "--sequences" && i + 1 < argc) {
        req.sequences = std::stoi(argv[++i]);
      } else if (!arg.empty() && arg[0] != '-' && dir.empty()) {
        dir = arg;
      } else {
        throw std::runtime_error("unknown argument " + arg);
      }
    }

    const DeviceInfo device = query_device(0);
    print_device(device);
    const std::string table = load_tuning(device);
    std::printf("tuning     %s\n", table.empty() ? "none for this GPU; the kernels use their own heuristics "
                                                    "(lernex tune MODEL_DIR makes one)"
                                              : table.c_str());

    if (!dir.empty()) {
      const lx::MetisConfig config = lx::load_config(dir);
      const CheckpointInfo checkpoint = inspect_checkpoint(dir);
      std::printf("\nmodel      %s: %d layers (%zu attention), hidden %d, %d routed experts, %d passes\n",
                  config.family.c_str(), config.num_hidden_layers, config.attention_layer_indices.size(),
                  config.hidden_size, config.n_routed_experts, config.num_passes);
      std::printf("checkpoint %s\n", checkpoint.description.c_str());
      const Plan plan = make_plan(device, config, checkpoint, req);
      std::printf("\nplan\n");
      for (const std::string& note : plan.notes) std::printf("  %s\n", note.c_str());
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
