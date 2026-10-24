// The thread pool behind every parallel kernel. One job at a time: the caller publishes it, works on it
// itself, and returns when every chunk is done.
//
// Decode runs a few hundred parallel ops per token, each a few microseconds long, so the cost of a job's
// start and end matters as much as its throughput. Workers spin on the job generation for a while
// before sleeping (LX_SPIN_US, default 400 us), so back-to-back ops hand over without a syscall, and an
// idle engine still leaves the machine alone. Chunks are claimed dynamically, which balances the
// performance and efficiency cores of a hybrid CPU without knowing which is which.
//
// Claims are a compare-and-swap on a cursor that carries the job's generation next to the chunk index,
// so a worker that wakes up late can never claim a chunk of a newer job with the old job's function.
#include <stdlib.h>
#include <string.h>

#include <chrono>
#include <mutex>
#include <thread>
#include <vector>

#include "cpu/cpu.h"
#include "cpu/lernex_cpu.h"

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
#endif

namespace lx::cpu {
namespace {

inline void cpu_relax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  __asm__ volatile("isb" ::: "memory");
#endif
}

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
#endif
  return out;
}

int physical_cores() {
#if defined(__linux__)
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
    fn_.store(fn, std::memory_order_relaxed);
    ctx_.store(ctx, std::memory_order_relaxed);
    n_items_.store(n, std::memory_order_relaxed);
    grain_.store(grain, std::memory_order_relaxed);
    chunks_.store(chunks, std::memory_order_relaxed);
    done_.store(0, std::memory_order_relaxed);
    const uint64_t gen = generation_.load(std::memory_order_relaxed) + 1;
    cursor_.store(gen << 32, std::memory_order_relaxed);
    generation_.store(gen, std::memory_order_seq_cst);
    if (sleepers_.load(std::memory_order_seq_cst) > 0) generation_.notify_all();
    t_in_job = true;
    work(gen);
    t_in_job = false;
    while (done_.load(std::memory_order_acquire) < chunks) cpu_relax();
  }

 private:
  // Claims and runs chunks of job `gen` until none are left.
  void work(uint64_t gen) {
    // A worker that woke late may read a newer job's fields here; its claims then fail on the cursor's
    // generation, so it never runs them.
    const RangeFn fn = fn_.load(std::memory_order_relaxed);
    void* const ctx = ctx_.load(std::memory_order_relaxed);
    const int64_t n = n_items_.load(std::memory_order_relaxed), grain = grain_.load(std::memory_order_relaxed),
                  chunks = chunks_.load(std::memory_order_relaxed);
    uint64_t c = cursor_.load(std::memory_order_relaxed);
    while ((c >> 32) == gen && static_cast<int64_t>(c & 0xffffffffu) < chunks) {
      if (!cursor_.compare_exchange_weak(c, c + 1, std::memory_order_acq_rel, std::memory_order_relaxed)) continue;
      const int64_t i = static_cast<int64_t>(c & 0xffffffffu);
      const int64_t b = i * grain, e = b + grain < n ? b + grain : n;
      fn(ctx, b, e);
      done_.fetch_add(1, std::memory_order_release);
      c = cursor_.load(std::memory_order_relaxed);
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
      work(gen);
    }
  }

  int n_ = 1;
  int64_t spin_ns_ = 0;
  std::vector<std::thread> workers_;
  std::mutex submit_;
  // The job. Written by the caller before the generation is published, read by workers after.
  std::atomic<RangeFn> fn_{nullptr};
  std::atomic<void*> ctx_{nullptr};
  std::atomic<int64_t> n_items_{0}, grain_{1}, chunks_{0};
  alignas(64) std::atomic<uint64_t> generation_{0};
  alignas(64) std::atomic<uint64_t> cursor_{0};
  alignas(64) std::atomic<int64_t> done_{0};
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

// ---- Scratch --------------------------------------------------------------------------------------------

namespace {
constexpr int kSlots = 8;
struct Scratch {
  void* p[kSlots] = {};
  size_t cap[kSlots] = {};
  ~Scratch() {
    for (int i = 0; i < kSlots; ++i) free(p[i]);
  }
};
thread_local Scratch t_scratch;
}  // namespace

void* scratch(size_t bytes, int slot) {
  Scratch& s = t_scratch;
  if (bytes < 64) bytes = 64;
  if (s.cap[slot] < bytes) {
    free(s.p[slot]);
    size_t want = 4096;
    while (want < bytes) want *= 2;
    void* p = nullptr;
    if (posix_memalign(&p, 64, want) != 0) p = nullptr;
    s.p[slot] = p;
    s.cap[slot] = p ? want : 0;
  }
  return s.p[slot];
}

}  // namespace lx::cpu

extern "C" int lx_cpu_threads(void) { return lx::cpu::threads(); }
