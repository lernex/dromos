// What this GPU can do, and how to run a model on it. The user picks a model and, if they want,
// a weight format; everything else (which instructions multiply, how many sequences fit, how long
// they can be, which kernel variant runs each product) is decided here, from the hardware.
#pragma once

#include <stddef.h>
#include <stdint.h>

#include <set>
#include <string>
#include <vector>

#include "core/config.h"
#include "cuda/kernels.h"
#include "model/metis.h"

namespace lx::runtime {

struct DeviceInfo {
  int ordinal = 0;
  std::string name;
  int cc_major = 0, cc_minor = 0;  // as the kernels see it (after LX_ARCH_OVERRIDE)
  int real_cc_major = 0, real_cc_minor = 0;
  int sms = 0;
  size_t total_bytes = 0, free_bytes = 0, l2_bytes = 0;
  size_t l1_bytes = 0, l3_bytes = 0;  // where the backend reports them (a CPU's)
  double bandwidth_gbs = 0;  // peak, from the memory clock and bus width, or measured (bandwidth_measured)
  bool bandwidth_measured = false;
  bool cpu = false;          // the CPU backend: host memory is the device's
  std::string isa;           // the instruction set the kernels run, in words (a CPU's)
  int native_math = 0;       // bit (1 << m): the hardware multiplies lx_math m with an instruction of its own
  bool bf16_mma = false, int8_mma = false, fp8_mma = false, fp4_mma = false, clusters = false;
  bool built = false;  // this binary has kernels for the architecture the engine runs as
  int driver = 0, runtime = 0;  // CUDA versions as integers: 13020 for 13.2
  std::string driver_text, runtime_text;

  int cc() const noexcept { return cc_major * 10 + cc_minor; }
  int real_cc() const noexcept { return real_cc_major * 10 + real_cc_minor; }
  // A short, filesystem-safe identity for tuning tables: "NVIDIA_GeForce_RTX_5070_Ti_sm120".
  std::string slug() const;
};

// Queries device `ordinal`. LX_ARCH_OVERRIDE=80|86|89|90|100|120 (or lx_set_arch_override) makes the
// engine behave as that architecture: only lowering is allowed, so every fallback runs and is tested
// on one card.
DeviceInfo query_device(int ordinal = 0);

// Whether this binary has kernels for compute capability `cc` (80, 86, 89, 90, 100, 120, ...): the
// architectures listed in LX_ARCHS when it was built.
bool kernels_built_for(int cc);
std::vector<int> built_archs();

// The formats a checkpoint folder holds, read from its safetensors headers without loading it.
struct CheckpointInfo {
  lx_wtype stored = LX_W_BF16;  // the format of the large projections in the file (ModelWeights::stored_format)
  size_t bytes = 0;             // the file's tensor bytes
  std::string description;      // plain language: "BF16 checkpoint, 6.2 GB on disk"
};
CheckpointInfo inspect_checkpoint(const std::string& model_dir);

// A format fits a GPU usefully when it leaves room for kUsefulSlots sequences (or as many as were asked for, when
// fewer) of kUsefulContext tokens.
constexpr int kUsefulSlots = 4;
constexpr int kUsefulContext = 4096;

// The user's wishes; anything left unset is decided from the checkpoint and the hardware.
struct PlanRequest {
  // An lx_wtype the user named (--weights), or -1 for the format the file stores. Nothing narrower than the
  // file is ever chosen for the user, and nothing wider is planned at all.
  int weights = -1;
  // A model decodes from lossless BF16X copies of its BF16 tensors when they fit: every BF16 weight of a BF16 model,
  // and the wide ones a quantized recipe keeps (--no-bf16x turns them off, for measuring plain BF16 decode).
  bool bf16x = true;
  // --fast-math: the quantized tensor-core maths in place of BF16 (native_math): faster prefill, less accurate.
  bool fast_math = false;
  // The engine builds its model speculative (EngineOptions::speculate): the plan reserves one speculative history, shared by
  // the stream that verifies. Off (--no-spec), the plan gives that memory back to the KV cache.
  bool speculative = true;
  int context = 0;    // context each sequence is sized for, 0 for automatic
  int sequences = 0;  // concurrent sequences wanted, 0 for automatic
  double memory_fraction = 0.92;  // of the free memory at startup; the rest is left for others
  // The GPU memory this process may use, 0 for the card (LX_VRAM_BUDGET_GB), and the pinned RAM offloaded
  // experts may take, 0 for all of them (LX_RAM_BUDGET_GB; the rest are read from the checkpoint on NVMe).
  // A model that does not fit the GPU, or the budget, runs with its experts offloaded (runtime/offload_plan.h).
  size_t vram_budget = 0;
  size_t ram_budget = 0;
};

// The format of each kind of tensor in a target format: dense projections, routed experts, and latent_up. The
// loader's recipe (core/checkpoint.h, projection_format) keeps some projections of each kind wider (BF16);
// ModelFormats takes these three as the formats the model accepts besides BF16.
struct WeightRecipe {
  lx_wtype dense = LX_W_BF16;    // attention, Mamba in_proj, latent_down, shared expert, depth memory, the head
  lx_wtype experts = LX_W_BF16;  // routed expert stacks
  lx_wtype latent_up = LX_W_BF16;
};
WeightRecipe recipe_for(lx_wtype target);

// The math a product on weights held in `format` multiplies in: BF16 by default, for every format and device (the
// KL in plan.cpp). With `fast` (--fast-math), the quantized tensor-core maths the weights can run: INT8 IMMA for INT8,
// FP8 on sm_89 and up for FP8, FP4 for NVFP4 experts on the sm_120 family (FP8 for NVFP4's dense products).
lx_math native_math(lx_wtype format, const DeviceInfo& device, bool fast = false);
// The model's formats for a target on a device: the recipe's kinds, native_math for the dense kind (prefill) and
// the experts, and the BF16X decode copies when asked for (every target but the lossless one, which is its own
// encoding).
lx::ModelFormats model_formats(lx_wtype target, const DeviceInfo& device, bool bf16x_decode, bool fast = false);
// Whether a target can be loaded from a folder stored as `stored`: the file's own format; INT8, FP8 or BF16X
// narrower than or as wide as the file (round-to-nearest, or the lossless encoding); NVFP4 only from a stored
// NVFP4 folder (the GPTQ checkpoint: round-to-nearest fails its KL). A format wider than the file holds the same
// values in more bytes and is refused. `how` gets a plain description of the load, or why it is refused.
bool can_load(lx_wtype target, lx_wtype stored, const MetisConfig& config, std::string* how = nullptr);
// "BF16", "INT8", "FP8", "NVFP4", "BF16X"; and the --weights value for each ("bf16", ... "bf16x").
std::string format_name(lx_wtype format);
std::string weights_flag(lx_wtype format);
// The lx_wtype a --weights value names. Throws std::runtime_error, listing the values, for anything else.
int weights_from_flag(const std::string& name);

// One weight format for this model on this device, as lernex doctor lists them.
struct FormatOption {
  lx_wtype format = LX_W_BF16;
  bool loadable = false;
  std::string load;        // "as stored", "round-to-nearest from BF16", or why it cannot load
  bool fits = false;       // resident, with room for the useful sequences and context
  size_t weight_bytes = 0; // one copy of every weight, resident
  size_t bf16x_bytes = 0;  // the BF16X decode copies the model adds when it keeps them
  double decode_ceiling_tps = 0;  // resident, one stream; BF16 with its BF16X copies when this build has them
  std::string maths;       // the maths the format's products multiply in by default
  std::string fast_maths;  // with --fast-math, or "-" when that changes nothing
};

struct Plan {
  lx_wtype stored = LX_W_BF16;    // the file's format
  lx_wtype weights = LX_W_BF16;   // the target format (the headline: "BF16", "FP8", ...)
  WeightRecipe recipe;            // what each kind of tensor is held in
  bool convert_at_load = false;   // the file is in another format (round-to-nearest or the lossless encoding at load)
  bool bf16x_decode = false;      // decode reads lossless BF16X copies of the BF16 weights (budgeted in weight_bytes)
  lx_math prefill_math = LX_MATH_BF16;  // dense products, prefill and batched steps
  lx_math batch_math = LX_MATH_BF16;    // the same, for the rows of a batched decode step
  lx_math expert_math = LX_MATH_BF16;   // routed experts
  int max_slots = 1;          // concurrent sequences
  int kv_capacity = 0;        // cache positions per sequence
  int max_step_tokens = 0;    // tokens in one step (decode rows and prefill rows together)
  int prefill_chunk = 0;      // prompt tokens a sequence takes per step while it is being prefilled
  size_t weight_bytes = 0, state_bytes = 0, kv_bytes = 0, scratch_bytes = 0, budget_bytes = 0;
  size_t bf16x_bytes = 0;          // of weight_bytes, the BF16X decode copies that are made
  std::set<std::string> bf16x_dropped;  // copy units the room beside the sequences did not hold (metis.h, bf16x_copy_made)
  size_t prefix_device_bytes = 0;  // prefix snapshots kept in device memory (what the model leaves over)
  size_t prefix_host_bytes = 0;    // and in pinned host memory, when the device share is full
  double decode_ceiling_tps = 0;   // bandwidth / bytes read per token: the most tokens per second one stream can make
  std::string reason;              // why this format: "the format the file stores", "as asked: ..."
  std::string bf16x_reason;        // why the BF16X decode copies are on or off, in one line
  std::vector<FormatOption> options;  // every format, for lernex doctor
  std::vector<std::string> notes;  // plain-language lines `lernex` prints (why this format, etc.)

  // Running larger than the GPU (make_offload_plan in runtime/offload_plan.h): every routed expert is
  // in pinned RAM (or, beyond the RAM budget, on NVMe), and VRAM keeps the ones in use in an LRU cache
  // the decode step manages on the GPU. weight_bytes then counts the dense weights and the cache.
  struct Offload {
    bool enabled = false;
    size_t expert_cache_bytes = 0;  // VRAM for expert slots
    size_t ram_expert_bytes = 0;    // pinned RAM holding experts
    size_t disk_expert_bytes = 0;   // experts the CPU lane reads from the checkpoint on NVMe
    double resident_fraction = 0;   // share of the experts the cache holds at once
    double cpu_fraction = 0;        // share of a step's misses the CPU computes from RAM
    int max_cpu_pairs = 0;          // at most this many pairs per step on the CPU
    int prefetch = 0;               // next-layer experts copied ahead per token, 0 for none
    size_t vram_budget = 0, ram_budget = 0;  // the request's budgets, which the runtime holds itself to
  } offload;

  // The formats MetisModel runs in under this plan.
  lx::ModelFormats formats() const;
};

// The plan. The format is the file's (or the one --weights names); a model that does not fit the GPU usefully
// in it runs with its routed experts offloaded before anything smaller is considered, and never in a narrower
// format than asked for. Throws std::runtime_error, saying what is missing and which narrower format would fit,
// when the model does not run on this device in that format at all.
Plan make_plan(const DeviceInfo& device, const MetisConfig& config, const CheckpointInfo& checkpoint,
               const PlanRequest& request);

// Exact byte accounting, shared by the plan and its tests. Each weight is counted once, in the format the loader
// holds it in (core/checkpoint.h, projection_format), with the 256-byte alignment of a device allocation.
struct WeightBytes {
  size_t plain = 0;        // bf16 and f32 tensors kept as they are (norms, biases, embeddings, routers), the
                           // bf16 gather table included
  size_t projections = 0;  // every projection but the head and the routed experts
  size_t head = 0;         // the tied head's own copy: 0 when it reads the bf16 gather table
  size_t experts = 0;      // the routed expert stacks: data, scales, NVFP4 per-expert globals
  size_t derived = 0;      // what the model computes at load: the hyper-connections' per-pass f32 parameters
  size_t bf16x = 0;        // the lossless BF16X decode copies, when the plan keeps them
  // What ModelWeights::device_bytes reports for the folder (the plain tensors the engine never reads aside).
  size_t loaded() const noexcept { return plain + projections + head + experts; }
  size_t total() const noexcept { return loaded() + derived + bf16x; }
};
// With `dropped` (copy units the plan leaves out), only the copies not named there count.
WeightBytes weight_bytes(const MetisConfig& config, lx_wtype target, bool bf16x_decode = false,
                         const std::set<std::string>* dropped = nullptr);
// Bytes of one projection of `rows` x `cols` held in `format` (an expert stack is rows = E * N): its data and
// scales, and for an NVFP4 stack of `stack_experts` experts their per-expert globals. BF16X is the encoding's
// fixed rows (lx_bf16x_row_bytes; the rare overflow records are not counted), 0 without a backend that has it.
size_t projection_bytes(lx_wtype format, int64_t rows, int64_t cols, int64_t stack_experts = 0);
// What one stream reads per decoded token in a target format at `context` positions, and the most tokens per second
// bandwidth allows. The count is the traffic model's (model/traffic.h), the one `lernex bench` checks its roofline against.
double decode_bytes_per_token(const MetisConfig& config, lx_wtype target, bool bf16x_decode, int context,
                             const std::set<std::string>* dropped = nullptr);
double decode_ceiling_tps(const DeviceInfo& device, const MetisConfig& config, lx_wtype target, bool bf16x_decode,
                          int context, const std::set<std::string>* dropped = nullptr);
// Whether the loaded backend has the BF16X kernels (the CUDA one does).
bool bf16x_available();
// Persistent state: the Mamba-2 conv and SSM state of one slot, and the attention KV cache of one slot
// at `capacity` positions. Both count every pass.
size_t state_bytes_per_slot(const MetisConfig& config);
size_t kv_bytes_per_position(const MetisConfig& config);
// The per-step scratch the model allocates for `tokens` tokens and `slots` slots (metis.cpp's constructor):
// the step's activations, the GEMM workspace for every dense math the step can use, the grouped expert
// workspace (laid out for the ordered sum), the decode experts' workspace, and the bf16 scratch BF16X-held
// weights expand into for prefill.
size_t scratch_bytes(const MetisConfig& config, int tokens, int slots, const lx::ModelFormats& formats = {});

}  // namespace lx::runtime
