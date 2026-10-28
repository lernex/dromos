// What one decode step moves through memory, by component: the bytes the roofline divides the card's bandwidth by.
//
// The count is the decode path the model runs by default (run() in metis.cpp, kernel by kernel, graphs captured): every
// product is read from the bytes it is held in (or its BF16X decode copy), every routed expert a stream chose, every
// stream's state and cache. Nothing is counted as L2-resident between passes: a pass is about a gigabyte of weights,
// far over a 48 MB L2, so every pass reads its weights again. Activations (the residual streams, the scratch rows) are
// left out: they are a few hundred kilobytes per step and stay in L2 between the kernels of a step.
//
// Two providers feed one traversal (traffic.cpp), so the number cannot drift between them: MetisModel::decode_traffic
// reads the matrices the model loaded (their held formats, their decode copies), and config_decode_traffic reads the
// same rules the loader applies (projection_format, the BF16X copy rule) for a folder nothing has been loaded from,
// which is what lernex doctor prints. `lernex bench` prints both and checks that they agree.
#pragma once

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "core/config.h"
#include "cuda/kernels.h"

namespace lx {

// A projection as decode reads it: the format it is held in, whether decode reads its BF16X copy instead, and its
// shape (E experts of [N, K]; E = 1 for a matrix). Names are the checkpoint's ("layers.4.mixer.qkv", "depth_memory.query",
// "embedding" for the tied head), as in core/checkpoint.cpp.
struct MatrixDesc {
  lx_wtype format = LX_W_BF16;
  bool copy = false;  // decode reads the BF16X copy (the copy's bytes), not the held BF16 rows
  int64_t E = 1, N = 0, K = 0;
};
// The descriptor of the projection `name` of shape E x [N, K]. The traversal asks with the shape its config gives; a
// provider that holds the matrix checks it against the one it has.
using MatrixLookup = std::function<MatrixDesc(const std::string& name, int64_t E, int64_t N, int64_t K)>;

enum class Scope {
  Shared,     // read once per step, whatever the number of streams: weights, plain tensors, the head
  PerStream,  // read once per stream: the state and the cache, a stream's own rows
  Expert,     // one routed expert, read by every stream that chose it: the union of their choices
};

// One instance of a component, in one layer (or the model's). Totals are count x the per-instance bytes.
struct TrafficItem {
  std::string name;   // "Mamba in_proj": the component, the same across layers
  std::string group;  // "mixer", "moe", "depth memory", "connections", "state and cache", "head"
  int layer = -1;     // the checkpoint layer it belongs to, -1 for the model's own
  Scope scope = Scope::Shared;
  double count = 0;   // instances per step: a layer's passes, the retrievals, the entries read
  double data = 0;    // bytes per instance: the values as decode reads them
  double scales = 0;  // bytes per instance: quantization scales (INT8/FP8 row scales, NVFP4 block scales and globals)
  double per_position = 0;  // bytes per instance per context position (the attention cache read)
  double macs = 0;    // multiply-accumulates per instance (per expert pick for Expert items), for the compute ceiling
};

struct DecodeTraffic {
  int context = 0;  // the cache positions each stream reads
  int top_k = 0;    // routed experts per token per layer-pass
  int experts = 0;  // routed experts per layer
  std::vector<TrafficItem> items;

  // Bytes one step of `streams` decoding tokens moves: shared items once, stream items per stream, and each expert item
  // over the expected union of the streams' choices, E * (1 - (1 - k / E)^n) experts per layer-pass.
  double step_bytes(int streams) const;
  // The expected distinct experts a layer-pass reads for `streams` streams, uniform routing (k when streams is 1).
  double expected_experts(int streams) const;
  // What one item adds to a step of `streams` streams (its scope's rule above).
  double item_bytes(const TrafficItem& item, int streams) const;
  // The step's bytes per token, at one stream: step_bytes(1).
  double bytes_per_token() const { return step_bytes(1); }
  // Multiply-accumulates one token takes (every product it multiplies: an expert counts its top_k picks, the depth
  // memory's reads count every stream's row).
  double macs_per_token() const;
};

// The traversal over a config and a provider of its matrices.
DecodeTraffic count_decode_traffic(const MetisConfig& config, const MatrixLookup& matrices, int context);

// The same for the folder as the loader would hold it in `target`, with decode's BF16X copies when `bf16x_decode`
// (the plan's rule: copies of BF16 matrices, every routed expert and the wide dense ones). Nothing is loaded.
DecodeTraffic config_decode_traffic(const MetisConfig& config, lx_wtype target, bool bf16x_decode, int context);

// The bytes one matrix of [N, K] takes in decode, split into data and scales. Exposed for the model's own accounting.
struct MatrixBytes {
  double data = 0;
  double scales = 0;
};
MatrixBytes matrix_bytes(lx_wtype format, bool copy, int64_t N, int64_t K);

}  // namespace lx
