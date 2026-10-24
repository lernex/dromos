// probe_machine: what the machine can move, measured the way the offload runtime moves it.
#include <dlfcn.h>
#include <fcntl.h>
#include <immintrin.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

#include "runtime/offload.h"

namespace lx::offload {
namespace {

using Clock = std::chrono::steady_clock;

void check(int status, const char* what) {
  if (status != 0) throw std::runtime_error(std::string(what) + " failed with error " + std::to_string(status));
}

double since(Clock::time_point t) { return std::chrono::duration<double>(Clock::now() - t).count(); }

std::string format(const char* f, ...) {
  char buf[512];
  va_list args;
  va_start(args, f);
  std::vsnprintf(buf, sizeof(buf), f, args);
  va_end(args);
  return buf;
}

// Copies `total` bytes host to device (or back) in `chunk`-sized async copies; GB/s, best of three.
double copy_rate(void* host, void* dev, int64_t chunk, int64_t total, bool h2d, void* stream) {
  double best = 0;
  for (int rep = 0; rep < 3; ++rep) {
    const auto t0 = Clock::now();
    for (int64_t off = 0; off < total; off += chunk) {
      const int64_t o = off % (256ll << 20);
      check(h2d ? lx_rt_copy(static_cast<char*>(dev) + o, static_cast<char*>(host) + o, chunk, LX_COPY_H2D, stream)
                : lx_rt_copy(static_cast<char*>(host) + o, static_cast<char*>(dev) + o, chunk, LX_COPY_D2H, stream),
            "copy");
    }
    check(lx_rt_stream_sync(stream), "sync");
    best = std::max(best, total / since(t0) / 1e9);
  }
  return best;
}

__attribute__((target("avx2"))) uint64_t sum_avx2(const char* p, size_t bytes) {
  __m256i acc = _mm256_setzero_si256(), acc2 = _mm256_setzero_si256();
  for (size_t i = 0; i + 64 <= bytes; i += 64) {
    acc = _mm256_xor_si256(acc, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i)));
    acc2 = _mm256_xor_si256(acc2, _mm256_load_si256(reinterpret_cast<const __m256i*>(p + i + 32)));
  }
  acc = _mm256_xor_si256(acc, acc2);
  return static_cast<uint64_t>(_mm256_extract_epi64(acc, 0) ^ _mm256_extract_epi64(acc, 3));
}

double ram_rate(const char* buf, size_t bytes, int threads) {
  double best = 0;
  std::atomic<uint64_t> sink{0};
  for (int rep = 0; rep < 3; ++rep) {
    std::vector<std::thread> pool;
    const size_t slice = bytes / threads / 64 * 64;
    const auto t0 = Clock::now();
    for (int t = 0; t < threads; ++t)
      pool.emplace_back([&, t] { sink += sum_avx2(buf + t * slice, slice); });
    for (auto& th : pool) th.join();
    best = std::max(best, slice * threads / since(t0) / 1e9);
  }
  return best;
}

double nvme_rate(const std::filesystem::path& file, int threads, int64_t request, int64_t total, bool sequential) {
  const int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) return 0;
  const int64_t size = lseek(fd, 0, SEEK_END);
  if (size < request * 4) {
    close(fd);
    return 0;
  }
  std::atomic<int64_t> next{0};
  std::vector<std::thread> pool;
  const auto t0 = Clock::now();
  for (int t = 0; t < threads; ++t)
    pool.emplace_back([&, t] {
      void* buf = std::aligned_alloc(4096, static_cast<size_t>(request));
      std::mt19937_64 rng(1234 + t);
      for (;;) {
        const int64_t i = next.fetch_add(1);
        if (i * request >= total) break;
        const int64_t blocks = (size - request) / 4096;
        const int64_t off = sequential ? (i * request) % (size - request) / 4096 * 4096
                                       : static_cast<int64_t>(rng() % static_cast<uint64_t>(blocks)) * 4096;
        if (pread(fd, buf, static_cast<size_t>(request), off) <= 0) break;
      }
      std::free(buf);
    });
  for (auto& th : pool) th.join();
  close(fd);
  return total / since(t0) / 1e9;
}

// cuFile (GPUDirect Storage) reads into device memory, through dlopen so the engine does not depend on
// it. Without the nvidia-fs driver cuFile runs in compatibility mode (POSIX reads and a bounce buffer).
struct CuFileDescr {
  int type;
  union {
    int fd;
    void* handle;
  } handle;
  const void* fs_ops;
};
double cufile_rate(const std::filesystem::path& file, int64_t request, int64_t total, std::string& note) {
  void* lib = dlopen("libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
  if (!lib) lib = dlopen("/usr/local/cuda/lib64/libcufile.so.0", RTLD_NOW | RTLD_LOCAL);
  if (!lib) {
    note = "libcufile not found";
    return 0;
  }
  struct Err {
    int err;
    int cu_err;
  };
  auto drv_open = reinterpret_cast<Err (*)()>(dlsym(lib, "cuFileDriverOpen"));
  auto reg = reinterpret_cast<Err (*)(void**, CuFileDescr*)>(dlsym(lib, "cuFileHandleRegister"));
  auto dereg = reinterpret_cast<void (*)(void*)>(dlsym(lib, "cuFileHandleDeregister"));
  auto buf_reg = reinterpret_cast<Err (*)(const void*, size_t, int)>(dlsym(lib, "cuFileBufRegister"));
  auto read = reinterpret_cast<ssize_t (*)(void*, void*, size_t, off_t, off_t)>(dlsym(lib, "cuFileRead"));
  if (!drv_open || !reg || !read || !buf_reg) {
    note = "libcufile lacks the read API";
    return 0;
  }
  if (Err e = drv_open(); e.err != 0) {
    note = format("cuFileDriverOpen error %d", e.err);
    return 0;
  }
  const int fd = open(file.c_str(), O_RDONLY | O_DIRECT);
  if (fd < 0) return 0;
  CuFileDescr d{};
  d.type = 1;  // CU_FILE_HANDLE_TYPE_OPAQUE_FD
  d.handle.fd = fd;
  void* fh = nullptr;
  if (Err e = reg(&fh, &d); e.err != 0) {
    note = format("cuFileHandleRegister error %d", e.err);
    close(fd);
    return 0;
  }
  void* dev = nullptr;
  check(lx_rt_alloc(&dev, request), "alloc");
  buf_reg(dev, static_cast<size_t>(request), 0);
  const int64_t size = lseek(fd, 0, SEEK_END);
  std::mt19937_64 rng(7);
  const auto t0 = Clock::now();
  int64_t done = 0;
  while (done < total) {
    const int64_t off = static_cast<int64_t>(rng() % static_cast<uint64_t>((size - request) / 4096)) * 4096;
    const ssize_t r = read(fh, dev, static_cast<size_t>(request), off, 0);
    if (r <= 0) {
      note = format("cuFileRead returned %zd", r);
      break;
    }
    done += r;
  }
  const double rate = done / since(t0) / 1e9;
  if (dereg) dereg(fh);
  close(fd);
  lx_rt_free(dev);
  return rate;
}

}  // namespace

MachineProfile probe_machine(const ProbeOptions& o) {
  MachineProfile p;
  auto line = [&](const std::string& s) { p.lines.push_back(s); };
  void* stream = nullptr;
  check(lx_rt_stream_create(&stream), "stream");
  const int64_t big = 256ll << 20;
  void *host = nullptr, *dev = nullptr;
  check(lx_rt_host_alloc(&host, big), "pinned alloc");
  check(lx_rt_alloc(&dev, big), "alloc");
  std::memset(host, 1, static_cast<size_t>(big));

  // the link idles at a low PCIe generation and ramps under load: warm it up first
  copy_rate(host, dev, 64ll << 20, 2ll << 30, true, stream);
  line("PCIe, copy engine (pinned):");
  for (int64_t chunk : {64ll << 10, 256ll << 10, 1ll << 20, 4ll << 20, 16ll << 20, 64ll << 20}) {
    const double h2d = copy_rate(host, dev, chunk, 2ll << 30, true, stream);
    const double d2h = copy_rate(host, dev, chunk, 1ll << 30, false, stream);
    if (chunk == (4ll << 20)) p.h2d_small_gbs = h2d;
    if (chunk == (64ll << 20)) {
      p.h2d_gbs = h2d;
      p.d2h_gbs = d2h;
    }
    line(format("  %6lld KB copies: H2D %5.1f GB/s, D2H %5.1f GB/s", (long long)(chunk >> 10), h2d, d2h));
  }

  // zero-copy: a kernel reading mapped host memory
  void *mh = nullptr, *md = nullptr;
  const int64_t zbytes = 1ll << 30;
  check(lx_rt_host_alloc_mapped(&mh, &md, zbytes), "mapped alloc");
  std::memset(mh, 3, static_cast<size_t>(zbytes));
  unsigned long long* sink = nullptr;
  check(lx_rt_alloc(reinterpret_cast<void**>(&sink), 8), "alloc");
  line("PCIe, zero-copy kernel reads of pinned host memory (1 GB):");
  for (int blocks : {16, 35, 70, 140, 280, 560}) {
    double best = 0;
    for (int rep = 0; rep < 3; ++rep) {
      const auto t0 = Clock::now();
      check(lx_offload_read_probe(md, zbytes, sink, blocks, stream), "probe");
      check(lx_rt_stream_sync(stream), "sync");
      best = std::max(best, zbytes / since(t0) / 1e9);
    }
    p.zero_copy_gbs = std::max(p.zero_copy_gbs, best);
    line(format("  %4d blocks of 256 threads: %5.1f GB/s", blocks, best));
  }
  // the same for VRAM, for scale
  {
    double best = 0;
    for (int rep = 0; rep < 5; ++rep) {
      const auto t0 = Clock::now();
      check(lx_offload_read_probe(dev, big, sink, 560, stream), "probe");
      check(lx_rt_stream_sync(stream), "sync");
      best = std::max(best, big / since(t0) / 1e9);
    }
    line(format("  (VRAM, the same kernel: %.0f GB/s)", best));
  }

  // the expert kernel itself, with every expert in VRAM, and with every expert in host memory
  {
    ExpertShape s;
    s.n_layers = 1;
    s.n_experts = 64;
    s.top_k = 4;
    s.L = o.L;
    s.I = o.I;
    s.type = o.expert_type;
    const int64_t eb = s.expert_bytes();
    void *vd = nullptr, *hh = nullptr, *hd = nullptr;
    check(lx_rt_alloc(&vd, eb * s.n_experts), "alloc");
    check(lx_rt_host_alloc_mapped(&hh, &hd, eb * s.n_experts), "mapped alloc");
    // weights near zero so no format overflows: zeros, plus unit scales
    std::memset(hh, 0, static_cast<size_t>(eb * s.n_experts));
    if (s.scaled()) {
      float* sc = reinterpret_cast<float*>(static_cast<char*>(hh) + s.n_experts * (s.gate_up_bytes() + s.down_bytes()));
      for (int64_t i = 0; i < s.n_experts * (2ll * s.I + s.L); ++i) sc[i] = 1.0f;
    }
    check(lx_rt_copy(vd, hh, eb * s.n_experts, LX_COPY_H2D, stream), "copy");
    auto ref_at = [&](char* base, int e) {
      char* gu = base;
      char* dn = gu + s.n_experts * s.gate_up_bytes();
      char* gus = dn + s.n_experts * s.down_bytes();
      char* dns = gus + s.n_experts * s.gate_up_scale_bytes();
      return lx_expert_ref{gu + e * s.gate_up_bytes(), s.scaled() ? gus + e * s.gate_up_scale_bytes() : nullptr,
                           dn + e * s.down_bytes(), s.scaled() ? dns + e * s.down_scale_bytes() : nullptr};
    };
    lx_expert_ref* refs = nullptr;
    float* w = nullptr;
    float* out = nullptr;
    void *lat = nullptr, *ws = nullptr;
    check(lx_rt_alloc(reinterpret_cast<void**>(&refs), 4 * sizeof(lx_expert_ref) * 16), "alloc");
    check(lx_rt_alloc(reinterpret_cast<void**>(&w), 64), "alloc");
    check(lx_rt_alloc(reinterpret_cast<void**>(&out), s.L * 4), "alloc");
    check(lx_rt_alloc(&lat, s.L * 2), "alloc");
    check(lx_rt_alloc(&ws, lx_offload_workspace_bytes(1, s.L, s.I, 4)), "alloc");
    for (int where = 0; where < 2; ++where) {
      std::vector<lx_expert_ref> h(4 * 16);
      for (int i = 0; i < 4 * 16; ++i)
        h[i] = ref_at(static_cast<char*>(where ? hd : vd), (i * 7) % s.n_experts);
      check(lx_rt_copy(refs, h.data(), sizeof(lx_expert_ref) * h.size(), LX_COPY_H2D, stream), "copy");
      check(lx_rt_stream_sync(stream), "sync");
      const int calls = 16;
      double best = 1e9;
      for (int rep = 0; rep < 3; ++rep) {
        const auto t0 = Clock::now();
        for (int c = 0; c < calls; ++c)
          check(lx_offload_experts(lat, w, refs + 4 * c, nullptr, s.type, out, ws, 1, s.L, s.I, 4, stream), "experts");
        check(lx_rt_stream_sync(stream), "sync");
        best = std::min(best, since(t0) / (calls * 4));
      }
      line(format("expert kernel, one token, 4 experts per call, every expert %s: %.1f us per expert (%.1f GB/s)",
                  where ? "in pinned host memory" : "in VRAM", best * 1e6, eb / best / 1e9));
      if (where) p.zero_copy_gbs = std::max(p.zero_copy_gbs, eb / best / 1e9);
    }
    lx_rt_free(refs);
    lx_rt_free(w);
    lx_rt_free(out);
    lx_rt_free(lat);
    lx_rt_free(ws);
    lx_rt_free(vd);
    lx_rt_host_free(hh);
  }

  // CPU: RAM bandwidth, then the CPU expert kernel (decode, one token)
  {
    const size_t bytes = 2ull << 30;
    char* buf = static_cast<char*>(std::aligned_alloc(4096, bytes));
    std::memset(buf, 5, bytes);
    p.ram_read_1t_gbs = ram_rate(buf, bytes, 1);
    line("CPU reading RAM (2 GB, AVX2 loads):");
    line(format("  %2d thread:  %5.1f GB/s", 1, p.ram_read_1t_gbs));
    for (int t : {4, 6, 8, 12, 14, 20}) {
      const double r = ram_rate(buf, bytes, t);
      p.ram_read_gbs = std::max(p.ram_read_gbs, r);
      line(format("  %2d threads: %5.1f GB/s", t, r));
    }
    std::free(buf);
  }
  {
    std::string path;
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&lx_offload_plan), &info) && info.dli_fname) {
      path = info.dli_fname;
      path = path.substr(0, path.rfind('/')) + "/liblernex_cpu.so";
    }
    if (const char* env = std::getenv("LX_CPU_KERNELS")) path = env;
    void* lib = dlopen(path.c_str(), RTLD_NOW | RTLD_LOCAL);
    using Fn = int (*)(const void*, const int32_t*, const float*, const lx_weight*, const lx_weight*, float*, void*, int,
                       int, int, int, void*);
    auto fn = lib ? reinterpret_cast<Fn>(dlsym(lib, "lx_moe_experts_decode")) : nullptr;
    auto threads = lib ? reinterpret_cast<int (*)()>(dlsym(lib, "lx_cpu_threads")) : nullptr;
    if (fn) {
      ExpertShape s;
      s.L = o.L;
      s.I = o.I;
      s.type = o.expert_type;
      const int E = 96;  // larger than the last-level cache, so every call streams from RAM
      void* hh = nullptr;
      check(lx_rt_host_alloc(&hh, s.expert_bytes() * E), "pinned alloc");
      std::memset(hh, 0, static_cast<size_t>(s.expert_bytes() * E));
      char* gu = static_cast<char*>(hh);
      char* dn = gu + E * s.gate_up_bytes();
      float* gus = reinterpret_cast<float*>(dn + E * s.down_bytes());
      float* dns = gus + E * 2 * s.I;
      if (s.scaled())
        for (int64_t i = 0; i < E * (2ll * s.I + s.L); ++i) gus[i] = 1.0f;
      const lx_weight wg{s.type, gu, s.scaled() ? gus : nullptr, 1.0f};
      const lx_weight wd{s.type, dn, s.scaled() ? dns : nullptr, 1.0f};
      std::vector<uint16_t> lat(s.L, 0x3f80);
      std::vector<float> out(s.L), ws(8 * s.I);
      p.cpu_threads = threads ? threads() : 0;
      line(format("CPU expert kernel (%d threads), one token:", p.cpu_threads));
      for (int k : {1, 2, 4}) {
        double best = 1e9;
        for (int rep = 0; rep < 3; ++rep) {
          const int calls = 48;
          const auto t0 = Clock::now();
          for (int c = 0; c < calls; ++c) {
            int32_t ids[4];
            float wts[4] = {0.25f, 0.25f, 0.25f, 0.25f};
            for (int j = 0; j < k; ++j) ids[j] = (c * 4 + j) % E;
            check(fn(lat.data(), ids, wts, &wg, &wd, out.data(), ws.data(), 1, s.L, s.I, k, nullptr), "cpu experts");
          }
          best = std::min(best, since(t0) / calls);
        }
        const double per = best / k;
        if (k == 4) {
          p.cpu_expert_us = per * 1e6;
          p.cpu_expert_gbs = s.expert_bytes() / per / 1e9;
        }
        line(format("  %d experts per call: %.1f us per call, %.1f us per expert (%.1f GB/s of weights)", k, best * 1e6,
                    per * 1e6, s.expert_bytes() / per / 1e9));
      }
      // the CPU lane and the GPU's zero-copy reads at the same time (they share RAM bandwidth)
      std::atomic<bool> stop{false};
      std::atomic<int64_t> gpu_bytes{0};
      std::thread gpu([&] {
        while (!stop.load()) {
          if (lx_offload_read_probe(md, 256ll << 20, sink, 280, stream) != 0) break;
          lx_rt_stream_sync(stream);
          gpu_bytes += 256ll << 20;
        }
      });
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
      const int64_t g0 = gpu_bytes.load();
      const auto t0 = Clock::now();
      int calls = 0;
      while (since(t0) < 1.0) {
        int32_t ids[4] = {(calls * 4) % E, (calls * 4 + 1) % E, (calls * 4 + 2) % E, (calls * 4 + 3) % E};
        float wts[4] = {0.25f, 0.25f, 0.25f, 0.25f};
        check(fn(lat.data(), ids, wts, &wg, &wd, out.data(), ws.data(), 1, s.L, s.I, 4, nullptr), "cpu experts");
        ++calls;
      }
      const double el = since(t0);
      const double g = (gpu_bytes.load() - g0) / el / 1e9;
      stop.store(true);
      gpu.join();
      const double c = calls * 4.0 * s.expert_bytes() / el / 1e9;
      line(format("  together with GPU zero-copy reads: CPU experts %.1f GB/s + GPU %.1f GB/s = %.1f GB/s", c, g, c + g));
      lx_rt_host_free(hh);
    } else {
      line("CPU expert kernel: liblernex_cpu.so not found");
    }
  }

  if (!o.nvme_file.empty()) {
    line("NVMe, O_DIRECT reads of " + o.nvme_file.string() + ":");
    for (int threads : {1, 4, 8}) {
      const double r = nvme_rate(o.nvme_file, threads, 4ll << 20, 2ll << 30, false);
      p.nvme_gbs = std::max(p.nvme_gbs, r);
      line(format("  4 MB random reads, %d in flight: %.2f GB/s", threads, r));
    }
    line(format("  4 MB sequential reads, 4 in flight: %.2f GB/s", nvme_rate(o.nvme_file, 4, 4ll << 20, 2ll << 30, true)));
    std::string note;
    const double cf = cufile_rate(o.nvme_file, 4ll << 20, 1ll << 30, note);
    line(format("  cuFile to VRAM, 4 MB random reads: %.2f GB/s%s", cf, note.empty() ? "" : (" (" + note + ")").c_str()));
  }

  lx_rt_free(sink);
  lx_rt_host_free(mh);
  lx_rt_free(dev);
  lx_rt_host_free(host);
  lx_rt_stream_destroy(stream);
  return p;
}

}  // namespace lx::offload
