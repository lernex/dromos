// Offloading: running a model larger than the GPU from every memory the machine has at once.
//
// Metis is mostly routed experts (Logos: 9.1B of 10.2B parameters) and a token touches 4 of 192 per
// layer per pass, so the plan keeps the dense weights and the experts the routing uses most in VRAM
// and every expert in pinned host memory. A decode step decides on the GPU where each pair runs
// (lx_offload_plan): a hit reads VRAM; a miss is read across PCIe by the expert kernel and written
// into a VRAM slot as it is read (the next use is a hit), or computed by the CPU straight from RAM
// while the GPU works on the rest (only the latent rows and the result cross the bus). Experts that
// do not fit in RAM stay in the checkpoint on NVMe and the CPU lane reads them. Prefill steps, which
// touch most experts of a layer, stage the misses into VRAM in batches.
//
// Where an expert lives never changes the result: every tier holds the same bytes, and the GPU path
// computes them with the same kernel. Only the CPU lane rounds differently (float summation order).
//
// Decode steps stay one CUDA graph launch: the plan, the expert kernel and the join are kernels, the
// cache tables live in device memory, and the CPU lane is a mailbox in mapped host memory served by
// a spinning worker thread (the GPU never waits for the host to launch anything).
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "cuda/kernels.h"

namespace lx::offload {

// What the machine can move, measured (probe_machine). Bandwidths in GB/s (1e9 bytes per second).
struct MachineProfile {
  double h2d_gbs = 0;         // pinned host to device, copy engine, 64 MB copies
  double h2d_small_gbs = 0;   // the same, 4 MB copies (one BF16 expert is 4.7 MB)
  double d2h_gbs = 0;
  double zero_copy_gbs = 0;   // a kernel reading pinned host memory directly
  double ram_read_gbs = 0;    // the CPU reading RAM, every thread
  double ram_read_1t_gbs = 0; // one thread
  double nvme_gbs = 0;        // O_DIRECT reads of the checkpoint, 4 MB requests, 8 in flight
  double cpu_expert_gbs = 0;  // the CPU expert kernel (decode, one token), bytes of weights per second
  double cpu_expert_us = 0;   // one expert, one token, at the probe's format
  int cpu_threads = 0;
  std::vector<std::string> lines;  // every measurement, as `lernex offload probe` prints it
};

struct ProbeOptions {
  std::filesystem::path nvme_file;  // a large file on the drive to read (the checkpoint)
  lx_wtype expert_type = LX_W_BF16;
  int L = 1024, I = 768;
  bool quick = false;
};
MachineProfile probe_machine(const ProbeOptions& options);

// The shape of the routed experts: n_layers stacks of n_experts experts, each gate_up [2I, L] and
// down [L, I] in `type` (scales f32 [2I] and [L] for INT8 / FP8).
struct ExpertShape {
  int n_layers = 0, n_experts = 0, top_k = 0, L = 0, I = 0;
  lx_wtype type = LX_W_BF16;
  int64_t gate_up_bytes() const { return 2ll * I * L * elem(); }
  int64_t down_bytes() const { return static_cast<int64_t>(L) * I * elem(); }
  int64_t gate_up_scale_bytes() const { return scaled() ? 2ll * I * 4 : 0; }
  int64_t down_scale_bytes() const { return scaled() ? static_cast<int64_t>(L) * 4 : 0; }
  int64_t expert_bytes() const { return gate_up_bytes() + down_bytes() + gate_up_scale_bytes() + down_scale_bytes(); }
  int elem() const { return type == LX_W_BF16 ? 2 : 1; }
  bool scaled() const { return type == LX_W_INT8 || type == LX_W_FP8; }
};

// Per (layer, expert) how often the routing chose it, over both passes of a calibration set.
struct RoutingStats {
  int n_layers = 0, n_experts = 0;
  std::vector<double> frequency;  // [n_layers * n_experts], sums to top_k * passes per layer per token
  static RoutingStats load(const std::filesystem::path& path);  // one line per layer, n_experts counts
  bool empty() const { return frequency.empty(); }
};

// The VRAM slots of each layer and what they hold at load (finalize uses it; pure, so it is tested
// on its own). capacity slots in all. With routing statistics, the share of slots given by
// 1 - dynamic_fraction is pinned to the hottest experts over all layers, and each layer gets dynamic
// slots in proportion to the routing mass its pinned set leaves uncovered, filled with its next
// hottest experts; the experts left out start cold (the CPU lane takes their misses first). Without
// statistics every layer gets an equal share holding its first experts.
struct SlotLayout {
  std::vector<int> slots, pinned;          // per layer
  std::vector<std::vector<int>> contents;  // per layer, experts in slot order, the pinned ones first
  std::vector<uint8_t> cpu_first;          // [n_layers * n_experts]
};
SlotLayout layout_slots(int n_layers, int n_experts, int capacity, const RoutingStats& stats, double dynamic_fraction);

// How to split the experts across the tiers. Zero / negative fields are filled by plan_placement.
struct OffloadOptions {
  int64_t vram_cache_bytes = 0;   // VRAM for expert slots
  int64_t ram_bytes = -1;         // host RAM for experts, -1 for all of them
  double dynamic_fraction = 1.0;  // share of each layer's slots under LRU (the rest pinned by frequency)
  double cpu_fraction = 0;        // share of a step's misses the CPU takes
  int max_cpu_pairs = 0;          // per step; 0 disables the CPU lane (unless experts are on disk)
  bool fill = true;               // write GPU misses into VRAM slots
  int prefill_batch = 0;          // experts staged per prefill batch, 0 for the layer's dynamic slots
};

// Where one expert's bytes come from at load: host pointers into the mapped checkpoint (or a
// converted copy), plus for the NVMe tier the file and byte offsets of the same bytes.
struct ExpertSource {
  const void* gate_up = nullptr;
  const void* gate_up_scale = nullptr;
  const void* down = nullptr;
  const void* down_scale = nullptr;
};
struct DiskLocation {
  std::filesystem::path file;
  int64_t gate_up = -1, gate_up_scale = -1, down = -1, down_scale = -1;  // byte offsets, -1 when absent
};

struct OffloadCounters {
  uint64_t hits = 0, gpu_misses = 0, cpu_misses = 0, fills = 0, prefetches = 0, prefetch_hits = 0, calls = 0;
  uint64_t prefill_pairs = 0, prefill_staged = 0, disk_reads = 0;
  double cpu_busy_us = 0;   // time the CPU lane spent computing
  uint64_t cpu_jobs = 0;
};

class ExpertOffload {
 public:
  ExpertOffload(const ExpertShape& shape, const OffloadOptions& options);
  ~ExpertOffload();
  ExpertOffload(const ExpertOffload&) = delete;
  ExpertOffload& operator=(const ExpertOffload&) = delete;

  const ExpertShape& shape() const { return shape_; }
  // Which experts get a host RAM slot (the hottest when RAM is short); call before add_expert.
  void plan_host(const RoutingStats& stats);
  // Copies one expert into its tier (host RAM, or only remembers the disk location).
  void add_expert(int layer, int expert, const ExpertSource& source, const DiskLocation& disk);
  // Allocates `vram_cache_bytes` of VRAM slots and fills them with the experts the stats say are
  // hottest (the first ones when there are no stats), builds the device tables and starts the CPU
  // worker. The policy fields of options() may be changed until then.
  void finalize(const RoutingStats& stats, int64_t vram_cache_bytes);
  OffloadOptions& options() { return options_; }

  // One MoE layer's routed experts for a decode step (T <= 8): plan, experts, join, on `stream`.
  // Capturable. ids [T, top_k], weights [T, top_k], latent bf16 [T, L], out f32 [T, L].
  void decode(int layer, const int32_t* ids, const float* weights, const void* latent, float* out, int T,
              void* stream);
  // The same for a larger step: synchronizes on `stream`, stages missing experts into VRAM in batches
  // and runs the grouped kernel. Not capturable.
  void prefill(int layer, const int32_t* ids, const float* weights, const void* latent, float* out, int T,
               void* stream);
  // Starts copying the predicted experts of `layer` into VRAM on `stream` (a side stream the caller
  // joins before the layer's decode). predicted [n] in device memory. Capturable.
  void prefetch(int layer, const int32_t* predicted, int n, void* stream);

  OffloadCounters counters();  // synchronizes the device
  void reset_counters();
  std::string describe() const;  // the placement, in lines
  int64_t vram_bytes() const { return vram_bytes_; }
  int64_t host_bytes() const { return host_bytes_; }
  int vram_slots() const { return n_slots_; }
  bool error();  // a join timed out (the CPU worker stopped)

 private:
  struct Device;
  void cpu_worker();
  void serve_job();
  void load_disk_expert(int layer, int expert, int cache_slot);

  ExpertShape shape_;
  OffloadOptions options_;
  int n_slots_ = 0;
  int64_t vram_bytes_ = 0, host_bytes_ = 0;

  // host tier: one pinned, mapped region; slot s of each array at base + s * the per-expert size
  void* host_region_ = nullptr;
  size_t host_region_bytes_ = 0;
  void* host_region_dev_ = nullptr;
  char *h_gu_ = nullptr, *h_gus_ = nullptr, *h_dn_ = nullptr, *h_dns_ = nullptr;
  int host_slots_ = 0;
  std::vector<int32_t> host_slot_;  // [n_layers * n_experts]
  std::vector<DiskLocation> disk_;  // [n_layers * n_experts]
  std::vector<uint8_t> loaded_;     // add_expert seen

  // disk tier: the CPU lane's RAM cache of experts read from NVMe
  int disk_cache_slots_ = 0;
  char *d_gu_ = nullptr, *d_gus_ = nullptr, *d_dn_ = nullptr, *d_dns_ = nullptr;  // host memory
  std::vector<int32_t> disk_cache_owner_;       // slot -> layer * E + expert, -1
  std::vector<uint64_t> disk_cache_stamp_;
  std::map<int32_t, int> disk_cache_index_;     // layer * E + expert -> slot
  uint64_t disk_clock_ = 0;

  // VRAM tier and the device-side tables
  std::unique_ptr<Device> dev_;

  // the CPU lane
  using ExpertsFn = int (*)(const void*, const int32_t*, const float*, const lx_weight*, const lx_weight*, float*,
                            void*, int, int, int, int, void*);
  ExpertsFn cpu_experts_ = nullptr;
  void* cpu_lib_ = nullptr;
  std::thread worker_;
  std::atomic<bool> stop_{false};
  lx_offload_mailbox* mailbox_ = nullptr;  // host address
  uint16_t* cpu_latent_ = nullptr;         // host address
  float* cpu_result_ = nullptr;            // host address
  void* mapped_block_ = nullptr;
  std::vector<char> cpu_workspace_;
  std::atomic<double> cpu_busy_us_{0};
  std::atomic<uint64_t> cpu_jobs_{0}, disk_reads_{0};
  uint64_t prefill_pairs_ = 0, prefill_staged_ = 0;
};

// The plan's share: how much VRAM goes to expert slots, and the CPU lane policy, from the budget, the
// model and the measured machine. The CPU lane is planned only when the cache holds less than
// kCpuLaneBelow of the experts (see plan_placement).
constexpr double kCpuLaneBelow = 0.5;
struct Placement {
  int64_t vram_cache_bytes = 0;
  int64_t ram_bytes = -1;
  double cpu_fraction = 0;
  int max_cpu_pairs = 0;
  std::vector<std::string> notes;
};
Placement plan_placement(const ExpertShape& shape, int64_t vram_free_for_experts, int64_t ram_available,
                         const MachineProfile* machine);

// The VRAM budget from LX_VRAM_BUDGET_GB (GiB), or the card's total memory when unset. It bounds what
// this process holds on the GPU, its CUDA context included; other processes only limit it through
// the memory they leave free.
int64_t vram_budget_bytes();
// What the driver charges this process on the GPU (NVML), or -1 when NVML cannot say.
int64_t process_vram_bytes();
// The host RAM budget for experts from LX_RAM_BUDGET_GB (GiB), or -1 for no limit.
int64_t ram_budget_bytes();

}  // namespace lx::offload
