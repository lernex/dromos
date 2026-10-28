// The roofline's numbers and their printing (roofline.h). The traffic is model/traffic.h's; this file holds the peaks
// and the arithmetic against them.
#include "cli/roofline.h"

#include <algorithm>
#include <cstdio>
#include <map>
#include <vector>

#include "cuda/kernels.h"

namespace lx::cli {
namespace {

// The probe reads 1 GiB, far past any L2, the same size the owner's roofline is stated for.
constexpr int64_t kProbeBytes = int64_t{1} << 30;
constexpr int kProbeReps = 5;

// Dense tensor-core peaks by compute capability. BF16 is the dense rate with FP32 accumulate: the decode products
// multiply in BF16 under the default maths (plan.cpp, native_math), so that is the rate that binds. FP8 is the dense
// FP8 rate with FP32 accumulate, for a run with --fast-math on FP8 weights. Sources:
//   sm_80   NVIDIA A100 Tensor Core GPU datasheet (A100 SXM): BF16 312 TFLOPS dense; no FP8 tensor cores.
//   sm_86   NVIDIA GA102 GPU Architecture whitepaper (GeForce RTX 3090): FP16 tensor, FP32 accumulate, 71 TFLOPS dense.
//   sm_89   NVIDIA Ada GPU Architecture whitepaper (GeForce RTX 4090): FP16 tensor, FP32 accumulate, 165.2 TFLOPS dense;
//           FP8 tensor, FP32 accumulate, 330.3 TFLOPS dense.
//   sm_90   NVIDIA H100 Tensor Core GPU datasheet (SXM): BF16 989.4 TFLOPS dense; FP8 1978.9 TFLOPS dense.
//   sm_100  NVIDIA Blackwell datasheet (B200): BF16 2.25 PFLOPS dense; FP8 4.5 PFLOPS dense.
//   sm_120  GeForce RTX 5070 Ti (GB203). NVIDIA's product page gives 1406 AI TOPS, an FP4 figure that is sparse; halved
//           for dense, 703 TFLOPS FP4, whose FP8 is half that, 351.5 with FP16 accumulate and 175.8 with FP32 (the
//           GeForce half rate). BF16 with FP32 accumulate is half of its FP16 rate, 87.9. The third-party tables
//           (hmc-tech's RTX 5070 Ti full specs) list 87.88 and 175.8 under those labels.
struct PeakRow {
  int cc;
  const char* card;
  double bf16_tflops, fp8_tflops;
  const char* source;
};
constexpr PeakRow kPeaks[] = {
    {80, "A100 SXM", 312.0, 0.0, "NVIDIA A100 Tensor Core GPU datasheet"},
    {86, "GeForce RTX 3090", 71.0, 0.0, "NVIDIA GA102 GPU Architecture whitepaper"},
    {89, "GeForce RTX 4090", 165.2, 330.3, "NVIDIA Ada GPU Architecture whitepaper"},
    {90, "H100 SXM", 989.4, 1978.9, "NVIDIA H100 Tensor Core GPU datasheet"},
    {100, "B200", 2250.0, 4500.0, "NVIDIA Blackwell datasheet (B200)"},
    {120, "GeForce RTX 5070 Ti", 87.9, 175.8,
     "NVIDIA RTX 5070 Ti product page (1406 AI TOPS, sparse FP4), halved to dense; hmc-tech spec tables"},
};

}  // namespace

Bandwidth measure_bandwidth(double fallback_gbs) {
  Bandwidth b;
  double read = 0, spec = 0;
  const int status = lx_bandwidth_probe(kProbeBytes, kProbeReps, &read, &spec, nullptr);
  if (status == 0) {
    b.measured = true;
    b.measured_gbs = read;  // lx_bandwidth_probe reports GB/s already
    b.spec_gbs = spec;
    return b;
  }
  if (status == LX_ERR_NOT_SUPPORTED) {
    b.measured_gbs = fallback_gbs;
    b.note = "this backend has no probe: its own reading";
  } else {
    b.measured_gbs = fallback_gbs;
    b.note = "the probe failed (CUDA error " + std::to_string(status) + "): the clock's peak stands in";
  }
  return b;
}

ComputePeak compute_peak(int cc) {
  ComputePeak p;
  for (const PeakRow& row : kPeaks) {
    if (row.cc != cc) continue;
    p.known = true;
    p.bf16_tflops = row.bf16_tflops;
    p.fp8_tflops = row.fp8_tflops;
    p.card = row.card;
    p.source = row.source;
  }
  return p;
}

double ceiling_tps(const DecodeTraffic& traffic, int streams, double gbs, double tflops) {
  double best = -1;
  const double bytes = traffic.step_bytes(streams);
  if (gbs > 0 && bytes > 0) best = streams * gbs * 1e9 / bytes;
  const double macs = traffic.macs_per_token();
  if (tflops > 0 && macs > 0) {
    const double compute = tflops * 1e12 / (2.0 * macs);
    best = best < 0 ? compute : std::min(best, compute);
  }
  return best < 0 ? 0 : best;
}

ByteSplit split(const DecodeTraffic& traffic) {
  ByteSplit s;
  for (const TrafficItem& it : traffic.items) {
    const double bytes = traffic.item_bytes(it, 1);
    (it.scope == Scope::PerStream ? s.per_stream : s.weights) += bytes;
  }
  return s;
}

void print_components(const DecodeTraffic& traffic, const char* title) {
  const double total = traffic.bytes_per_token();
  std::printf("%s\n", title);

  // The groups, in the order their first component appears, then every component under its group.
  struct Row {
    std::string name, group;
    double count = 0, bytes = 0;
  };
  std::vector<Row> rows;
  std::map<std::string, std::size_t> row_of;
  std::map<std::string, double> group_bytes;
  std::vector<std::string> groups;
  for (const TrafficItem& it : traffic.items) {
    const double bytes = traffic.item_bytes(it, 1);
    auto found = row_of.find(it.name);
    if (found == row_of.end()) {
      found = row_of.emplace(it.name, rows.size()).first;
      rows.push_back(Row{it.name, it.group});
    }
    rows[found->second].count += it.count;
    rows[found->second].bytes += bytes;
    if (!group_bytes.count(it.group)) groups.push_back(it.group);
    group_bytes[it.group] += bytes;
  }

  std::printf("  %-40s %12s %11s %8s\n", "group", "", "MB/token", "share");
  for (const std::string& g : groups)
    std::printf("  %-40s %12s %11.1f %7.1f%%\n", g.c_str(), "", group_bytes[g] / 1e6,
                total > 0 ? 100.0 * group_bytes[g] / total : 0.0);
  std::printf("  %-40s %12s %11.1f %7.1f%%\n", "total", "", total / 1e6, 100.0);

  std::printf("\n  %-40s %12s %11s %8s\n", "component", "instances", "MB/token", "share");
  for (const std::string& g : groups) {
    std::printf("  %s\n", g.c_str());
    for (const Row& r : rows) {
      if (r.group != g) continue;
      std::printf("    %-38s %12.0f %11.2f %7.1f%%\n", r.name.c_str(), r.count, r.bytes / 1e6,
                  total > 0 ? 100.0 * r.bytes / total : 0.0);
    }
  }

  // Each layer's total over its passes, three to a line. Attention layers are the ones with an attention component.
  std::map<int, std::pair<double, bool>> layers;  // layer -> (bytes, attention)
  for (const TrafficItem& it : traffic.items) {
    if (it.layer < 0) continue;
    auto& entry = layers[it.layer];
    entry.first += traffic.item_bytes(it, 1);
    if (it.name.rfind("attention", 0) == 0) entry.second = true;
  }
  if (!layers.empty()) {
    std::printf("\n  layers, MB per token over their passes (A attention, M Mamba-2)\n   ");
    int column = 0;
    for (const auto& [layer, entry] : layers) {
      std::printf(" %2d %s %8.2f", layer, entry.second ? "A" : "M", entry.first / 1e6);
      if (++column % 4 == 0) std::printf("\n   ");
    }
    if (column % 4 != 0) std::printf("\n");
  }
}

}  // namespace lx::cli
