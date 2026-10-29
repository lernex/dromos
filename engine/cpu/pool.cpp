// The thread pool behind every parallel kernel. One job at a time: the caller publishes it, works on it
// itself, and returns when every chunk is done.
//
// Decode runs a few hundred parallel ops per token, each a few microseconds long, so the cost of a job's
// start and end matters as much as its throughput. Workers spin on the job generation for a while
// before sleeping (LX_SPIN_US, default 400 us), so back-to-back ops hand over without a syscall, and an
// idle engine still leaves the machine alone. Chunks are claimed dynamically, which balances the
// performance and efficiency cores of a hybrid CPU without knowing which is which.
//
// Claims are one fetch-add on a cursor that carries the job's generation next to the chunk index (no
// retry loop when fourteen threads claim at once). The claim says which job the chunk belongs to, and
// jobs alternate between two descriptors, so a worker that wakes up late runs a chunk with the right
// job's function: the descriptor of job g cannot be reused by job g + 2 before job g's last claimed
// chunk is done.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu/cpu.h"
#include "cpu/dromos_cpu.h"
#include "platform/posix.h"

#if defined(__linux__)
#include <dirent.h>
#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <set>
#include <utility>
#elif defined(__APPLE__)
#include <pthread.h>
#include <sys/qos.h>
#include <sys/sysctl.h>
#elif defined(_WIN32)
#include <windows.h>
#include <intrin.h>
#endif

namespace lx::cpu {
namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ volatile("isb" ::: "memory");
#elif defined(_M_X64)
  _mm_pause();
#elif defined(_M_ARM64)
  __isb(_ARM64_BARRIER_SY);
#endif
}

#if defined(_WIN32)
// Each physical core of processor group 0 as one logical processor (its first), from the system's own topology.
std::vector<int> windows_cores() {
  std::vector<int> out;
  DWORD len = 0;
  GetLogicalProcessorInformationEx(RelationProcessorCore, nullptr, &len);
  if (len == 0) return out;
  std::vector<char> buf(len);
  auto* info = reinterpret_cast<SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data());
  if (!GetLogicalProcessorInformationEx(RelationProcessorCore, info, &len)) return out;
  for (DWORD at = 0; at < len;) {
    const auto* e = reinterpret_cast<const SYSTEM_LOGICAL_PROCESSOR_INFORMATION_EX*>(buf.data() + at);
    const GROUP_AFFINITY& g = e->Processor.GroupMask[0];
    if (g.Group == 0 && g.Mask) {
      unsigned long first = 0;
      _BitScanForward64(&first, static_cast<unsigned long long>(g.Mask));
      out.push_back(static_cast<int>(first));
    }
    at += e->Size;
  }
  return out;
}

// Windows 11 runs a thread it judges to be background work slowly, on efficiency cores at a low clock (EcoQoS). The
// kernels are the opposite of background work, so their threads opt out, as the macOS pool asks for
// QOS_CLASS_USER_INITIATED. Systems that predate the setting ignore it.
void full_speed_thread() {
  THREAD_POWER_THROTTLING_STATE state{};
  state.Version = THREAD_POWER_THROTTLING_CURRENT_VERSION;
  state.ControlMask = THREAD_POWER_THROTTLING_EXECUTION_SPEED;
  state.StateMask = 0;
  SetThreadInformation(GetCurrentThread(), ThreadPowerThrottling, &state, sizeof state);
}

void full_speed_process() {
  PROCESS_POWER_THROTTLING_STATE state{};
  state.Version = PROCESS_POWER_THROTTLING_CURRENT_VERSION;
  state.ControlMask = PROCESS_POWER_THROTTLING_EXECUTION_SPEED;
  state.StateMask = 0;
  SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &state, sizeof state);
}
#endif

// Logical CPUs this process may run on, one per physical core (the first sibling), on Linux.
std::vector<int> core_cpus() {
  std::vector<int> out;
#if defined(__linux__)
  cpu_set_t mask;
  CPU_ZERO(&mask);
  if (sched_getaffinity(0, sizeof(mask), &mask) != 0) return out;
  std::set<std::pair<int, int>> seen;
  for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
    if (!CPU_ISSET(cpu, &mask)) continue;
    char path[128];
    int core = cpu, pkg = 0;
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/core_id", cpu);
    if (FILE* f = fopen(path, "r")) {
      if (fscanf(f, "%d", &core) != 1) core = cpu;
      fclose(f);
    }
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
    if (FILE* f = fopen(path, "r")) {
      if (fscanf(f, "%d", &pkg) != 1) pkg = 0;
      fclose(f);
    }
    if (seen.insert({pkg, core}).second) out.push_back(cpu);
  }
#elif defined(_WIN32)
  out = windows_cores();
#endif
  return out;
}

int physical_cores() {
#if defined(__linux__) || defined(_WIN32)
  const std::vector<int> cpus = core_cpus();
  if (!cpus.empty()) return static_cast<int>(cpus.size());
#elif defined(__APPLE__)
  int n = 0;
  size_t len = sizeof(n);
  if (sysctlbyname("hw.physicalcpu", &n, &len, nullptr, 0) == 0 && n > 0) return n;
#endif
  const unsigned hc = std::thread::hardware_concurrency();
  return hc ? static_cast<int>(hc) : 1;
}

int env_int(const char* name, int fallback) {
  const char* v = getenv(name);
  return v && *v ? atoi(v) : fallback;
}

thread_local bool t_in_job = false;

class Pool {
 public:
  Pool() {
    n_ = env_int("LX_THREADS", physical_cores());
    if (n_ < 1) n_ = 1;
    spin_ns_ = static_cast<int64_t>(env_int("LX_SPIN_US", 400)) * 1000;
    const bool pin = env_int("LX_PIN", 0) != 0;
#if defined(_WIN32)
    full_speed_process();
    full_speed_thread();
#endif
    const std::vector<int> cpus = pin ? core_cpus() : std::vector<int>{};
    for (int i = 1; i < n_; ++i) {
      const int cpu = pin && i < static_cast<int>(cpus.size()) ? cpus[i] : -1;
      workers_.emplace_back([this, cpu] { worker(cpu); });
    }
  }

  int size() const { return n_; }

  void run(int64_t n, int64_t grain, RangeFn fn, void* ctx) {
    const int64_t chunks = (n + grain - 1) / grain;
    if (t_in_job || n_ == 1 || chunks <= 1) {
      fn(ctx, 0, n);
      return;
    }
    std::lock_guard<std::mutex> lock(submit_);
    const uint64_t gen = generation_.load(std::memory_order_relaxed) + 1;
    Job& job = jobs_[gen & 1];
    job.fn = fn;
    job.ctx = ctx;
    job.n = n;
    job.grain = grain;
    job.done.store(0, std::memory_order_relaxed);
    job.tag.store((gen << 32) | static_cast<uint64_t>(chunks), std::memory_order_release);
    cursor_.store(gen << 32, std::memory_order_release);
    generation_.store(gen, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) generation_.notify_all();
    t_in_job = true;
    work();
    t_in_job = false;
    while (job.done.load(std::memory_order_acquire) < chunks) cpu_relax();
  }

 private:
  // Claims and runs chunks of the current job until a claim comes back past its end.
  void work() {
    for (;;) {
      const uint64_t c = cursor_.fetch_add(1, std::memory_order_acquire);
      Job& job = jobs_[(c >> 32) & 1];
      // The descriptor's tag names its job and chunk count in one atomic word. A claim from a job that
      // is over (its descriptor reused, or every chunk handed out) touches nothing else; a claim in
      // range holds its job open, so the rest of the descriptor stays put while the chunk runs.
      const uint64_t tag = job.tag.load(std::memory_order_acquire);
      const int64_t i = static_cast<int64_t>(c & 0xffffffffu);
      if ((tag >> 32) != (c >> 32) || i >= static_cast<int64_t>(tag & 0xffffffffu)) return;
      const int64_t b = i * job.grain, e = b + job.grain < job.n ? b + job.grain : job.n;
      job.fn(job.ctx, b, e);
      job.done.fetch_add(1, std::memory_order_release);
    }
  }

  void worker(int cpu) {
#if defined(__linux__)
    if (cpu >= 0) {
      cpu_set_t set;
      CPU_ZERO(&set);
      CPU_SET(cpu, &set);
      pthread_setaffinity_np(pthread_self(), sizeof(set), &set);
    }
#elif defined(__APPLE__)
    (void)cpu;
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INITIATED, 0);
#elif defined(_WIN32)
    if (cpu >= 0) {
      GROUP_AFFINITY affinity{};
      affinity.Group = 0;
      affinity.Mask = static_cast<KAFFINITY>(1) << cpu;
      SetThreadGroupAffinity(GetCurrentThread(), &affinity, nullptr);
    }
    full_speed_thread();
#else
    (void)cpu;
#endif
    t_in_job = true;  // a kernel that calls parallel_for from a worker runs it inline
    uint64_t seen = 0;
    for (;;) {
      uint64_t gen = generation_.load(std::memory_order_acquire);
      if (gen == seen) {
        const auto start = std::chrono::steady_clock::now();
        int polls = 0;
        while ((gen = generation_.load(std::memory_order_acquire)) == seen) {
          cpu_relax();
          if (++polls == 256) {
            polls = 0;
            if (std::chrono::steady_clock::now() - start > std::chrono::nanoseconds(spin_ns_)) {
              sleepers_.fetch_add(1, std::memory_order_seq_cst);
              if (generation_.load(std::memory_order_seq_cst) == seen) generation_.wait(seen, std::memory_order_seq_cst);
              sleepers_.fetch_sub(1, std::memory_order_seq_cst);
            }
          }
        }
      }
      seen = gen;
      work();
    }
  }

  int n_ = 1;
  int64_t spin_ns_ = 0;
  std::vector<std::thread> workers_;
  std::mutex submit_;
  // The jobs, alternating by generation. Written by the caller before the cursor and the generation
  // are published (release), read by a worker after its claim (acquire).
  struct alignas(64) Job {
    std::atomic<uint64_t> tag{0};  // generation << 32 | chunks
    RangeFn fn = nullptr;
    void* ctx = nullptr;
    int64_t n = 0, grain = 1;
    alignas(64) std::atomic<int64_t> done{0};
  };
  Job jobs_[2];
  alignas(64) std::atomic<uint64_t> generation_{0};
  alignas(64) std::atomic<uint64_t> cursor_{0};
  alignas(64) std::atomic<int> sleepers_{0};
};

Pool& pool() {
  // Never destroyed: workers sleeping at exit are harmless, joining them from a static destructor is not.
  static Pool* p = new Pool;
  return *p;
}

}  // namespace

int threads() { return pool().size(); }

void run_parallel(int64_t n, int64_t grain, RangeFn fn, void* ctx) { pool().run(n, grain, fn, ctx); }

// ---- Profile --------------------------------------------------------------------------------------------

namespace {
struct ProfEntry {
  int64_t calls = 0, ns = 0;
};
struct Profile {
  std::mutex mu;
  std::vector<std::pair<const char*, ProfEntry>> ops;
  const char* level = "";
  int n_threads = 0;
  ~Profile() {
    if (ops.empty()) return;
    std::sort(ops.begin(), ops.end(), [](const auto& a, const auto& b) { return a.second.ns > b.second.ns; });
    int64_t total = 0;
    for (const auto& [name, e] : ops) total += e.ns;
    fprintf(stderr, "dromos_cpu profile (%s, %d threads): %.1f ms in ops\n", level, n_threads, total / 1e6);
    for (const auto& [name, e] : ops)
      fprintf(stderr, "  %-22s %8lld calls %10.2f ms %6.1f%% %9.2f us/call\n", name, static_cast<long long>(e.calls),
              e.ns / 1e6, 100.0 * e.ns / total, e.ns / 1e3 / e.calls);
  }
};
Profile& profile() {
  static Profile p;
  return p;
}
}  // namespace

bool g_profile = [] {
  const char* v = getenv("LX_CPU_PROFILE");
  if (v && *v && *v != '0') profile();  // constructed first, so it is destroyed (and prints) last
  return v && *v && *v != '0';
}();

int64_t ProfScope::now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

void ProfScope::record(const char* name, int64_t ns) {
  Profile& p = profile();
  std::lock_guard<std::mutex> g(p.mu);
  for (auto& [n, e] : p.ops) {
    if (n == name || strcmp(n, name) == 0) {
      ++e.calls;
      e.ns += ns;
      return;
    }
  }
  if (p.ops.empty()) {
    p.level = kern().name;
    p.n_threads = threads();
  }
  p.ops.push_back({name, ProfEntry{1, ns}});
}

// ---- Scratch --------------------------------------------------------------------------------------------

namespace {
constexpr int kSlots = 8;
struct Scratch {
  void* p[kSlots] = {};
  size_t cap[kSlots] = {};
  ~Scratch() {
    for (int i = 0; i < kSlots; ++i) lx_aligned_free(p[i]);
  }
};
thread_local Scratch t_scratch;
}  // namespace

void* scratch(size_t bytes, int slot) {
  Scratch& s = t_scratch;
  if (bytes < 64) bytes = 64;
  if (s.cap[slot] < bytes) {
    lx_aligned_free(s.p[slot]);
    size_t want = 4096;
    while (want < bytes) want *= 2;
    void* p = lx_aligned_alloc(64, want);
    s.p[slot] = p;
    s.cap[slot] = p ? want : 0;
  }
  return s.p[slot];
}

}  // namespace lx::cpu

extern "C" int lx_cpu_threads(void) { return lx::cpu::threads(); }
