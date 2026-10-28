// The roofline, as lernex doctor and lernex bench print it. A decode step's memory traffic (model/traffic.h) against
// the card's memory bandwidth and tensor throughput. The bandwidth ceiling is bandwidth / bytes per step: the tokens
// per second the memory system allows, and every measured speed reads as a share of it.
#pragma once

#include <string>

#include "model/traffic.h"

namespace lx::cli {

// The memory bandwidth the roofline uses: the device's peak (memory clock x bus width x 2, the spec) and the rate a
// read-only kernel sustains over 1 GiB (lx_bandwidth_probe, the best of five runs). A backend without the probe reports
// its own reading as the measured rate and no spec.
struct Bandwidth {
  bool measured = false;     // the probe ran
  double spec_gbs = 0;      // 0 when the backend does not say
  double measured_gbs = 0;  // 0 when nothing was measured
  std::string note;         // why a figure is missing, or ""
};
Bandwidth measure_bandwidth(double fallback_gbs);

// The dense tensor-core peak of an architecture, in TFLOPS, with the card it describes and where the figure comes from
// (roofline.cpp). bf16 is the dense BF16 rate with FP32 accumulate, the rate the decode products run at under the default
// maths; fp8 the dense FP8 rate with FP32 accumulate, 0 where the architecture has no FP8 tensor cores.
struct ComputePeak {
  bool known = false;
  double bf16_tflops = 0;
  double fp8_tflops = 0;
  std::string card;
  std::string source;
};
ComputePeak compute_peak(int cc);

// Tokens per second a set of `streams` decoding together can make at most: the smaller of the bandwidth bound,
// streams x gbs / the step's bytes, and the compute bound, tflops / (2 x the MACs per token), which does not depend on
// the stream count. A bound of 0 (no figure) is not applied.
double ceiling_tps(const DecodeTraffic& traffic, int streams, double gbs, double tflops);

// The components of one decode token at one stream: the groups, then every component with its count per token, its
// bytes per token and its share, then each layer's total. `title` heads the block.
// Bytes per step at `streams` streams (one by default: per token), by component.
void print_components(const DecodeTraffic& traffic, const char* title, int streams = 1);

// The bytes a step of `streams` streams reads for the routed experts: the expert items of step_bytes, each over the
// union of the streams' choices.
double expert_bytes(const DecodeTraffic& traffic, int streams);

// The bytes a token of the stream reads, split by what they are: weights (shared, expert) and state, cache and the
// streams' own rows.
struct ByteSplit {
  double weights = 0, per_stream = 0;
};
ByteSplit split(const DecodeTraffic& traffic);

}  // namespace lx::cli
