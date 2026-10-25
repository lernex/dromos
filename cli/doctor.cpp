// lernex doctor [MODEL_DIR] [--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x] [--context N] [--sequences N]
//
// Prints what this GPU or CPU can do, which kernels this build has, which tuning table is in use, and, with a
// model folder, the plan the engine would run it under, in plain language: the format the file stores,
// every format with whether it loads from this file, fits this GPU, and how fast one stream could decode
// in it, the one chosen and why, whether the experts are offloaded, whether decode reads BF16X copies,
// and the plan's notes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <stdexcept>
#include <string>

#include "cli/common.h"
#include "core/config.h"
#include "dispatch/dispatch.h"
#include "runtime/plan.h"
#include "runtime/tuner.h"

namespace {

std::string archs_text(const std::vector<int>& archs) {
  std::string out;
  for (int cc : archs) out += (out.empty() ? "" : " ") + std::string("sm_") + std::to_string(cc);
  return out;
}

std::string names_text(const std::vector<std::string>& names) {
  std::string out;
  for (const std::string& name : names) out += (out.empty() ? "" : ", ") + name;
  return out.empty() ? "none" : out;
}

// "A", "A and B", "A, B and C".
std::string and_list(const std::vector<std::string>& items) {
  std::string out;
  for (std::size_t i = 0; i < items.size(); ++i)
    out += (i == 0 ? "" : i + 1 == items.size() ? " and " : ", ") + items[i];
  return out;
}

std::string bytes_text(std::size_t bytes) {
  char buf[32];
  if (bytes >= (std::size_t{1} << 20))
    std::snprintf(buf, sizeof buf, "%.4g MB", static_cast<double>(bytes) / (1 << 20));
  else
    std::snprintf(buf, sizeof buf, "%zu KB", bytes >> 10);
  return buf;
}

// A processor: what the kernels run on it, how wide its memory is, and which maths it multiplies itself.
void print_cpu(const lx::runtime::DeviceInfo& d) {
  std::printf("CPU        %s (device %d)\n", d.name.c_str(), d.ordinal);
  if (d.cc() != d.real_cc())
    std::printf("ISA        level %d: %s (LX_ARCH_OVERRIDE lowers it from %d)\n", d.cc(), d.isa.c_str(), d.real_cc());
  else
    std::printf("ISA        level %d: %s\n", d.cc(), d.isa.c_str());
  std::printf("threads    %d%s\n", d.sms,
              std::getenv("LX_THREADS") ? " (LX_THREADS)" : ", one per physical core (LX_THREADS sets them)");
  std::printf("memory     %.2f GB total, %.2f GB free now\n", static_cast<double>(d.total_bytes) / 1e9,
              static_cast<double>(d.free_bytes) / 1e9);
  if (d.bandwidth_gbs > 0)
    std::printf("bandwidth  %.0f GB/s, measured reading memory on %d threads\n", d.bandwidth_gbs, d.sms);
  std::vector<std::string> caches;
  if (d.l1_bytes) caches.push_back("L1d " + bytes_text(d.l1_bytes));
  if (d.l2_bytes) caches.push_back("L2 " + bytes_text(d.l2_bytes) + (d.l3_bytes ? " per core" : ""));
  if (d.l3_bytes) caches.push_back("L3 " + bytes_text(d.l3_bytes) + " shared");
  if (!caches.empty()) std::printf("caches     %s\n", and_list(caches).c_str());
  // The maths the kernels multiply in: a format's dot instruction where the processor has one, else the format
  // widened (BF16 to f32, INT8 to 16-bit products). FP8 and FP4 products are exact on every processor.
  const bool bf16_native = d.native_math & (1 << LX_MATH_BF16), int8_native = d.native_math & (1 << LX_MATH_INT8);
  std::printf("math       %s\n", bf16_native ? "BF16 in the processor's bf16 dot instruction (batches of more than 8 rows)"
                                             : "BF16 widened to f32 and multiplied with FMA (no bf16 dot instruction here)");
  std::printf("           %s\n", int8_native ? "INT8 in the processor's int8 dot instructions (VNNI or SDOT)"
                                               : "INT8 with 16-bit multiplies (no int8 dot instruction here)");
  std::printf("           FP8 and FP4 products are exact, summed in double\n");
}

void print_gpu(const lx::runtime::DeviceInfo& d) {
  const lx::dispatch::Info& backend = lx::dispatch::info();
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
  std::printf("%-10s driver %s, runtime %s\n", backend.backend.empty() ? "backend" : backend.backend.c_str(),
              d.driver_text.c_str(), d.runtime_text.c_str());
}

void print_device(const lx::runtime::DeviceInfo& d) {
  if (d.cpu) print_cpu(d);
  else print_gpu(d);
  const lx::dispatch::Info& backend = lx::dispatch::info();
  if (lx_rt_backend() == LX_BACKEND_CUDA)
    std::printf("build      kernels for %s\n", archs_text(lx::runtime::built_archs()).c_str());
  if (!d.built)
    std::printf("           this card (sm_%d) has no kernels in this build: rebuild with -DLX_ARCHS=all\n", d.real_cc());
  if (backend.backend.empty()) {
    std::printf("backend    none: %s\n", backend.error.c_str());
    std::printf("missing    not checked, no backend loaded\n");
  } else {
    const std::size_t slash = backend.path.rfind('/');
    const std::string file = slash == std::string::npos ? backend.path : backend.path.substr(slash + 1);
    std::printf("backend    %s, %s (%s)\n", backend.backend.c_str(), file.c_str(), backend.path.c_str());
    std::printf("missing    %s\n", names_text(backend.missing).c_str());
  }
}

// The plan for a model folder, as lernex doctor prints it.
void print_plan(const lx::runtime::Plan& plan) {
  using namespace lx::runtime;
  std::printf("\nformats    %-6s %-4s %10s %12s %-14s %-18s %s\n", "", "fits", "weights", "decode est.", "maths",
              "with --fast-math", "loads");
  for (const FormatOption& o : plan.options) {
    char ceiling[48] = "-";
    if (o.decode_ceiling_tps > 0)
      std::snprintf(ceiling, sizeof ceiling, "%.0f tok/s%s", o.decode_ceiling_tps, o.bf16x_bytes ? "*" : "");
    std::printf("           %-6s %-4s %7.2f GB %12s %-14s %-18s %s%s\n", format_name(o.format).c_str(),
                o.fits ? "yes" : "no", static_cast<double>(o.weight_bytes) / 1e9, ceiling, o.maths.c_str(),
                o.fast_maths.c_str(), o.loadable ? "" : "no: ", o.load.c_str());
  }
  for (const FormatOption& o : plan.options)
    if (o.bf16x_bytes)
      std::printf("           * decoding from the BF16X copies of the BF16 weights (%.2f GB more)\n",
                  static_cast<double>(o.bf16x_bytes) / 1e9);
  std::printf("           fits: resident with room for %d sequences of %d tokens; decode est.: one stream, the "
              "measured share of the card's bandwidth over the bytes a token reads\n",
              kUsefulSlots, kUsefulContext);

  std::printf("\nstored     %s\n", format_name(plan.stored).c_str());
  std::printf("chosen     %s, %s\n", format_name(plan.weights).c_str(), plan.reason.c_str());
  if (plan.offload.enabled) {
    std::printf("offload    yes: %.0f%% of the %s experts cached in %.2f GB of VRAM, all of them in %s\n",
                100.0 * plan.offload.resident_fraction, format_name(plan.recipe.experts).c_str(),
                static_cast<double>(plan.offload.expert_cache_bytes) / 1e9,
                plan.offload.disk_expert_bytes ? "pinned RAM and on NVMe" : "pinned RAM");
  } else {
    std::printf("offload    no: the model is resident (%d sequence%s of %d tokens)\n", plan.max_slots,
                plan.max_slots == 1 ? "" : "s", plan.kv_capacity);
  }
  if (!plan.bf16x_reason.empty())
    std::printf("BF16X      %s %s\n", plan.bf16x_decode ? "on:" : "off:", plan.bf16x_reason.c_str());
  else
    std::printf("BF16X      off: decode copies are for BF16 weights, and these are %s\n",
                format_name(plan.weights).c_str());
  if (plan.decode_ceiling_tps > 0)
    std::printf("decode     about %.0f tokens per second for one stream (the measured share of the card's bandwidth "
                "over the bytes a token reads)\n",
                plan.decode_ceiling_tps);
  else
    std::printf("decode     set by the experts that miss the VRAM cache, which only a run measures\n");
  std::printf("\nplan\n");
  for (const std::string& note : plan.notes) std::printf("  %s\n", note.c_str());
}

}  // namespace

int cmd_doctor(int argc, char** argv) {
  using namespace lx::runtime;
  std::string dir;
  PlanRequest req;
  try {
    for (int i = 1; i < argc; ++i) {
      const std::string arg = argv[i];
      lx::cli::FormatFlags format;
      format.weights = req.weights;
      format.bf16x = req.bf16x;
      if (lx::cli::parse_format_flag(argc, argv, i, format)) {
        req.weights = format.weights;
        req.bf16x = format.bf16x;
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
    if (device.cpu && table.empty())
      std::printf("tuning     none needed: each CPU product has one order, so there is nothing to choose\n");
    else
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
      if (const char* v = std::getenv("LX_VRAM_BUDGET_GB")) req.vram_budget = static_cast<size_t>(std::atof(v) * 1073741824.0);
      if (const char* v = std::getenv("LX_RAM_BUDGET_GB")) req.ram_budget = static_cast<size_t>(std::atof(v) * 1073741824.0);
      print_plan(make_plan(device, config, checkpoint, req));
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
