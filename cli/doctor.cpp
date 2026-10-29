// dromos doctor [MODEL] [--weights bf16|int8|fp8|nvfp4|bf16x] [--no-bf16x] [--context N] [--sequences N] [--modelled]
//
// --modelled counts the roofline of a folder in a format it cannot be loaded in yet: the loader's recipe for that
// format applied to the folder's config, with nothing read from the folder (NVFP4 needs a GPTQ folder to load).
//
// Prints what this GPU or CPU can do, which kernels this build has, which tuning table is in use, and, with a
// model folder, the plan the engine would run it under, in plain language: the format the file stores,
// every format with whether it loads from this file, fits this GPU, and how fast one stream could decode
// in it, the one chosen and why, whether the experts are offloaded, whether decode reads BF16X copies,
// and the plan's notes.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <exception>
#include <stdexcept>
#include <set>
#include <string>
#include <vector>

#include "cli/common.h"
#include "cli/roofline.h"
#include "core/config.h"
#include "dispatch/dispatch.h"
#include "model/traffic.h"
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

// An Apple GPU: its family (the Metal backend's arch is 1000 + the family), its cores, the unified memory macOS lets
// it hold, and whether the GEMMs and attention run on its neural accelerators. None of CUDA's terms apply here.
void print_apple_gpu(const lx::runtime::DeviceInfo& d) {
  const int family = d.cc() - 1000, real_family = d.real_cc() - 1000;
  std::printf("GPU        %s (device %d)\n", d.name.c_str(), d.ordinal);
  if (family != real_family)
    std::printf("family     Apple %d as the kernels see it (the GPU is Apple %d; LX_ARCH_OVERRIDE lowers it)\n", family,
                real_family);
  else
    std::printf("family     Apple %d: Metal compiles the kernels for this GPU at run time\n", family);
  std::printf("cores      %d GPU cores\n", d.sms);
  std::printf("memory     %.2f GB of unified memory the GPU can use, %.2f GB free now\n",
              static_cast<double>(d.total_bytes) / 1e9, static_cast<double>(d.free_bytes) / 1e9);
  if (d.bandwidth_gbs > 0)
    std::printf("bandwidth  %.0f GB/s (memory transfer rate x bus width)\n", d.bandwidth_gbs);
  else
    std::printf("bandwidth  not known for this chip\n");
  // The neural accelerators are Apple family 10 (the M5) and later; Device::nax() in the Metal backend decides.
  const char* nax = std::getenv("LX_METAL_NAX");
  if (real_family < 10)
    std::printf("math       simdgroup matrices (this GPU has no neural accelerators)\n");
  else if (nax && nax[0] == '0')
    std::printf("math       simdgroup matrices (LX_METAL_NAX=0 keeps the GEMMs off the neural accelerators)\n");
  else
    std::printf("math       neural accelerators for the GEMMs and attention (LX_METAL_NAX=0 turns them off)\n");
  std::string metal = d.runtime_text;  // "4.0" is Metal 4, "3.2" stays
  if (metal.size() > 2 && metal.compare(metal.size() - 2, 2, ".0") == 0) metal.resize(metal.size() - 2);
  std::printf("Metal      macOS %s, Metal %s\n", d.driver_text.c_str(), metal.c_str());
}

void print_gpu(const lx::runtime::DeviceInfo& d) {
  if (lx_rt_backend() == LX_BACKEND_METAL) return print_apple_gpu(d);
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
    const std::size_t slash = backend.path.find_last_of("/\\");
    const std::string file = slash == std::string::npos ? backend.path : backend.path.substr(slash + 1);
    std::printf("backend    %s, %s (%s)\n", backend.backend.c_str(), file.c_str(), backend.path.c_str());
    // CUDA-only fast paths (lx::dispatch::optional) have a fallback everywhere else, so their absence is no gap.
    std::vector<std::string> missing;
    for (const std::string& name : backend.missing)
      if (!lx::dispatch::optional(name)) missing.push_back(name);
    std::printf("missing    %s\n", names_text(missing).c_str());
  }
}

// The plan for a model folder, as dromos doctor prints it.
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
  // One footnote for every starred format: the size once when the copies weigh the same in each, else per format.
  std::string copies;
  bool same = true;
  std::size_t first = 0;
  for (const FormatOption& o : plan.options) {
    if (!o.bf16x_bytes) continue;
    if (first && o.bf16x_bytes != first) same = false;
    if (!first) first = o.bf16x_bytes;
    char part[48];
    std::snprintf(part, sizeof part, "%s%.2f GB to %s", copies.empty() ? "" : ", ",
                  static_cast<double>(o.bf16x_bytes) / 1e9, format_name(o.format).c_str());
    copies += part;
  }
  if (first && same)
    std::printf("           * decoding from the BF16X copies of the BF16 tensors (%.2f GB more)\n",
                static_cast<double>(first) / 1e9);
  else if (first)
    std::printf("           * decoding from the BF16X copies of the BF16 tensors, which add %s\n", copies.c_str());
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
    std::printf("BF16X      off: no BF16 tensors to copy in %s\n",
                format_name(plan.weights).c_str());
  if (plan.decode_ceiling_tps > 0)
    std::printf("decode     about %.0f tokens per second for one stream (the measured share of the card's bandwidth "
                "over the bytes a token reads)\n",
                plan.decode_ceiling_tps);
  else
    std::printf("decode     set by the experts that miss the VRAM cache, which only a run measures\n");
  if (plan.offload.enabled)
    std::printf("speculate  off: the experts are offloaded, which the verify path does not yet support\n");
  else
    std::printf("speculate  on for one stream: prompt-lookup drafts are checked in one pass, output unchanged "
                "(--no-spec turns it off)\n");
  std::printf("\nplan\n");
  for (const std::string& note : plan.notes) std::printf("  %s\n", note.c_str());
}

// Tokens per second: whole numbers from ten up, one decimal below.
std::string tps_text(double tps) {
  char buf[32];
  if (tps >= 10) std::snprintf(buf, sizeof buf, "%.0f", tps);
  else std::snprintf(buf, sizeof buf, "%.1f", tps);
  return buf;
}

// The roofline of a model folder in `target`: the bytes one decode step moves, against the card's bandwidth and tensor
// throughput, as ceilings (bandwidth / bytes), with the binding limit named. With a plan's formats (`options`), every
// loadable format gets a row too. `dropped`: the BF16X copy units the plan left out, so the count matches the model's.
void print_roofline(const lx::MetisConfig& config, lx_wtype target, bool bf16x_decode,
                    const lx::runtime::DeviceInfo& device, const std::vector<lx::runtime::FormatOption>* options,
                    const std::set<std::string>* dropped = nullptr) {
  using namespace lx::cli;
  using lx::DecodeTraffic;
  using lx::config_decode_traffic;
  const int contexts[2] = {512, lx::runtime::kUsefulContext};
  const Bandwidth bw = measure_bandwidth(device.bandwidth_gbs);
  const ComputePeak peak = device.cpu ? ComputePeak{} : compute_peak(device.cc());
  const double tflops = peak.known ? peak.bf16_tflops : 0.0;
  const std::string spec = bw.spec_gbs > 0 ? std::to_string(static_cast<long>(bw.spec_gbs + 0.5)) + " GB/s" : "n/a";
  const std::string measured = bw.measured_gbs > 0 ? std::to_string(static_cast<long>(bw.measured_gbs + 0.5)) + " GB/s" : "n/a";

  std::printf("\nroofline   a decode step's memory traffic against the card (one stream unless said; passes included)\n");
  std::printf("bandwidth  %s spec, %s measured%s%s\n", spec.c_str(), measured.c_str(),
              bw.measured ? " (1 GiB read, best of 5)" : "", bw.note.empty() ? "" : (" (" + bw.note + ")").c_str());
  if (peak.known)
    std::printf("compute    %.1f TFLOPS BF16 tensor, dense with FP32 accumulate (%s, sm_%d; %s)\n", peak.bf16_tflops,
                peak.card.c_str(), device.cc(), peak.source.c_str());
  else
    std::printf("compute    no tensor peak in the table for this device: the compute ceiling is not shown\n");

  const DecodeTraffic at = config_decode_traffic(config, target, bf16x_decode, contexts[0], dropped);
  const DecodeTraffic at_long = config_decode_traffic(config, target, bf16x_decode, contexts[1], dropped);
  const std::string title = "components of " + lx::runtime::format_name(target) + (bf16x_decode ? " (BF16X decode copies)" : "") +
                            " at context " + std::to_string(contexts[0]) + ":";
  print_components(at, title.c_str());
  const ByteSplit parts = split(at);
  std::printf("  reads: %.1f MB of weights and scales, %.1f MB of state, cache and per-stream rows (at context %d)\n",
              parts.weights / 1e6, parts.per_stream / 1e6, contexts[0]);

  // One stream: the ceilings at each context, from the card's peak and from the measured rate.
  std::printf("\none stream   bytes per token      ceiling at spec     ceiling at measured\n");
  for (int c = 0; c < 2; ++c) {
    const DecodeTraffic& t = c == 0 ? at : at_long;
    std::printf("  context %5d  %9.1f MB  %14s tok/s  %14s tok/s\n", contexts[c], t.bytes_per_token() / 1e6,
                bw.spec_gbs > 0 ? tps_text(ceiling_tps(t, 1, bw.spec_gbs, 0)).c_str() : "n/a",
                bw.measured_gbs > 0 ? tps_text(ceiling_tps(t, 1, bw.measured_gbs, 0)).c_str() : "n/a");
  }
  std::printf("  compute ceiling, any stream count: %s tok/s (%.2f GMAC per token)\n",
              tflops > 0 ? tps_text(ceiling_tps(at, 1, 0, tflops)).c_str() : "n/a", at.macs_per_token() / 1e9);

  // Batched: the routed experts a step reads are the union of the streams' choices, the dense weights are read once.
  std::printf("\nbatched    streams  experts/layer-pass   step MB at 512   experts MB (share)   tok/s spec   tok/s measured   "
              "step MB at %d   tok/s spec   tok/s measured   binds\n", contexts[1]);
  for (int n : {8, 32, 64}) {
    const double step = at.step_bytes(n), step_long = at_long.step_bytes(n);
    const double bw_spec = bw.spec_gbs > 0 ? ceiling_tps(at, n, bw.spec_gbs, 0) : 0;
    const double bw_meas = bw.measured_gbs > 0 ? ceiling_tps(at, n, bw.measured_gbs, 0) : 0;
    const double bw_spec_long = bw.spec_gbs > 0 ? ceiling_tps(at_long, n, bw.spec_gbs, 0) : 0;
    const double bw_meas_long = bw.measured_gbs > 0 ? ceiling_tps(at_long, n, bw.measured_gbs, 0) : 0;
    const double compute = tflops > 0 ? ceiling_tps(at, n, 0, tflops) : 0;
    const bool compute_binds = compute > 0 && bw_meas > 0 && compute < bw_meas;
    const double experts = expert_bytes(at, n);
    std::printf("           %7d  %18.1f  %15.1f  %8.1f (%4.1f%%)  %11s  %15s   %15.1f  %11s  %15s   %s\n", n,
                at.expected_experts(n), step / 1e6, experts / 1e6, 100.0 * experts / step,
                bw_spec > 0 ? tps_text(bw_spec).c_str() : "n/a",
                bw_meas > 0 ? tps_text(bw_meas).c_str() : "n/a", step_long / 1e6,
                bw_spec_long > 0 ? tps_text(bw_spec_long).c_str() : "n/a",
                bw_meas_long > 0 ? tps_text(bw_meas_long).c_str() : "n/a",
                compute_binds ? "compute" : "bandwidth");
  }
  std::printf("           the union is E (1 - (1 - k / E)^n) experts per layer-pass for uniform routing; real routing is\n"
              "           more concentrated, so the union is an upper bound. Ceilings count the weights once per step.\n");

  if (!options) return;
  // Every format the model loads in, at one stream and context 512.
  std::printf("\nformats    %-6s %14s %16s %16s   %s\n", "", "bytes per token", "ceiling at spec", "ceiling at measured",
              "decode copies");
  for (const lx::runtime::FormatOption& o : *options) {
    if (!o.loadable) continue;
    const bool copies = o.format != LX_W_BF16X && lx::runtime::bf16x_available();
    const DecodeTraffic t = config_decode_traffic(config, o.format, copies, 512);
    std::printf("           %-6s %11.1f MB %13s tok/s %13s tok/s   %s\n", lx::runtime::format_name(o.format).c_str(),
                t.bytes_per_token() / 1e6,
                bw.spec_gbs > 0 ? tps_text(ceiling_tps(t, 1, bw.spec_gbs, 0)).c_str() : "n/a",
                bw.measured_gbs > 0 ? tps_text(ceiling_tps(t, 1, bw.measured_gbs, 0)).c_str() : "n/a",
                copies ? "BF16X" : "as held");
  }
}

}  // namespace

int cmd_doctor(int argc, char** argv) {
  using namespace lx::runtime;
  std::string dir;
  bool modelled = false;
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
      } else if (arg == "--modelled") {
        modelled = true;
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
                                                      "(dromos tune MODEL_DIR makes one)"
                                                : table.c_str());

    if (modelled && dir.empty()) throw std::runtime_error("--modelled needs a model folder");
    if (modelled && req.weights < 0) throw std::runtime_error("--modelled needs --weights: the format to count");
    if (!dir.empty()) {
      // MODEL is a folder, or an alias or repository installed in the cache (praxis, logos:fp8), as for bench.
      if (!std::filesystem::is_directory(dir)) dir = lx::cli::find_model(dir, false).dir.string();
      const lx::MetisConfig config = lx::load_config(dir);
      std::printf("\nmodel      %s: %d layers (%zu attention), hidden %d, %d routed experts, %d passes\n",
                  config.family.c_str(), config.num_hidden_layers, config.attention_layer_indices.size(),
                  config.hidden_size, config.n_routed_experts, config.num_passes);
      if (modelled) {
        // The format the folder is not stored in: the loader's recipe for it, applied to this config. Nothing is read, so
        // a folder converted or stored in that format reads the same bytes per token.
        const lx_wtype target = static_cast<lx_wtype>(req.weights);
        std::printf("checkpoint not read: counted as the loader's %s recipe holds this config (--modelled)\n",
                    format_name(target).c_str());
        print_roofline(config, target, req.bf16x && bf16x_available(), device, nullptr);
      } else {
        const CheckpointInfo checkpoint = inspect_checkpoint(dir);
        std::printf("checkpoint %s\n", checkpoint.description.c_str());
        if (const char* v = std::getenv("LX_VRAM_BUDGET_GB")) req.vram_budget = static_cast<size_t>(std::atof(v) * 1073741824.0);
        if (const char* v = std::getenv("LX_RAM_BUDGET_GB")) req.ram_budget = static_cast<size_t>(std::atof(v) * 1073741824.0);
        const Plan plan = make_plan(device, config, checkpoint, req);
        print_plan(plan);
        print_roofline(config, plan.weights, plan.bf16x_decode, device, &plan.options, &plan.bf16x_dropped);
      }
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "error: %s\n", e.what());
    return 1;
  }
  return 0;
}
