// dromos weights MODEL_DIR bf16|int8|fp8|nvfp4|bf16x [--reference DIR] [--bytes-against DIR] [--samples N]
//
// The acceptance check of the weight formats. Loads MODEL_DIR in the target format and reports the device
// memory the load held. Sampled projections are read back as bf16 through lx_dequantize_weight, expert by
// expert. With --reference (a BF16 folder), each read value is compared with the reference's bf16 value:
// the largest absolute error, the relative RMS error, and the values whose bits differ. With
// --bytes-against (a folder that is in the target's format), the sampled bytes of the two loads are
// compared, so a conversion can be checked against a published folder bit for bit.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "core/checkpoint.h"
#include "core/config.h"
#include "cuda/kernels.h"

namespace {

using Clock = std::chrono::steady_clock;

void check(int status, const std::string& what) {
  if (status != 0) throw std::runtime_error(what + ": " + lx_rt_error_string(status));
}

void usage() {
  std::fprintf(stderr,
               "usage: dromos weights MODEL_DIR bf16|int8|fp8|nvfp4|bf16x [--reference DIR] "
               "[--bytes-against DIR] [--samples N]\n");
}

lx_wtype parse_target(const std::string& s) {
  if (s == "bf16") return LX_W_BF16;
  if (s == "int8") return LX_W_INT8;
  if (s == "fp8") return LX_W_FP8;
  if (s == "nvfp4") return LX_W_NVFP4;
  if (s == "bf16x") return LX_W_BF16X;
  throw std::runtime_error("unknown target '" + s + "' (bf16, int8, fp8, nvfp4, bf16x)");
}

const char* format_name(lx_wtype t) {
  switch (t) {
    case LX_W_BF16: return "bf16";
    case LX_W_INT8: return "int8";
    case LX_W_FP8: return "fp8";
    case LX_W_NVFP4: return "nvfp4";
    case LX_W_BF16X: return "bf16x";
  }
  return "?";
}

std::string mib(size_t bytes) {
  char buf[32];
  std::snprintf(buf, sizeof buf, "%.1f MiB", static_cast<double>(bytes) / 1048576.0);
  return buf;
}

float bf16_value(uint16_t bits) {
  const uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, sizeof f);
  return f;
}

// The experts read for a projection: the first and the last of a stack, the matrix itself otherwise.
std::vector<int64_t> experts_to_read(const lx::ModelWeights::Shape& s) {
  if (!s.stack || s.E == 1) return {0};
  return {0, s.E - 1};
}

// Evenly spaced projection names, always including the tied head.
std::vector<std::string> sample_names(const lx::ModelWeights& m, int count) {
  const std::vector<std::string> all = m.projection_names();
  if (static_cast<size_t>(count) >= all.size()) return all;
  std::vector<std::string> picked;
  for (int i = 0; i < count; ++i) picked.push_back(all[static_cast<size_t>(i) * (all.size() - 1) / (count - 1)]);
  if (std::find(all.begin(), all.end(), "lm_head") != all.end() &&
      std::find(picked.begin(), picked.end(), "lm_head") == picked.end())
    picked.push_back("lm_head");
  return picked;
}

// One expert of a projection (or the matrix), dequantized to bf16 on the device, as bf16 bits on the host.
std::vector<uint16_t> bf16_on_host(const lx::ModelWeights& m, const std::string& name, int64_t e) {
  const lx::ModelWeights::Shape s = m.projection_shape(name);
  const lx_weight w = m.projection_expert(name, e);
  const size_t n = static_cast<size_t>(s.N * s.K);
  void* dst = nullptr;
  const int64_t bytes = static_cast<int64_t>(n * sizeof(uint16_t));
  check(lx_rt_alloc(&dst, bytes), "allocating the readback of " + name);
  std::vector<uint16_t> host(n);
  try {
    check(lx_dequantize_weight(&w, dst, s.N, static_cast<int>(s.K), nullptr), "dequantizing " + name);
    check(lx_rt_copy(host.data(), dst, bytes, LX_COPY_D2H, nullptr), "reading " + name);
    check(lx_rt_stream_sync(nullptr), "reading " + name);
  } catch (...) {
    lx_rt_free(dst);
    throw;
  }
  lx_rt_free(dst);
  return host;
}

struct Error {
  double max_abs = 0, sum_sq_err = 0, sum_sq_ref = 0;
  int64_t values = 0, mismatched = 0;

  void add(const std::vector<uint16_t>& got, const std::vector<uint16_t>& ref) {
    for (size_t i = 0; i < got.size(); ++i) {
      if (got[i] != ref[i]) ++mismatched;
      const double r = bf16_value(ref[i]), d = static_cast<double>(bf16_value(got[i])) - r;
      max_abs = std::max(max_abs, std::fabs(d));
      sum_sq_err += d * d;
      sum_sq_ref += r * r;
    }
    values += static_cast<int64_t>(got.size());
  }

  double rel_rms() const { return sum_sq_ref > 0 ? std::sqrt(sum_sq_err / sum_sq_ref) : 0.0; }
};

// Bytes of one expert's data and scales on both loads, compared. Returns the bytes that differ.
size_t differing_bytes(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
  if (a_bytes != b_bytes) return std::max(a_bytes, b_bytes);
  if (a_bytes == 0) return 0;
  std::vector<unsigned char> ha(a_bytes), hb(b_bytes);
  check(lx_rt_copy(ha.data(), a, static_cast<int64_t>(a_bytes), LX_COPY_D2H, nullptr), "reading the bytes");
  check(lx_rt_stream_sync(nullptr), "reading the bytes");
  check(lx_rt_copy(hb.data(), b, static_cast<int64_t>(b_bytes), LX_COPY_D2H, nullptr), "reading the bytes");
  check(lx_rt_stream_sync(nullptr), "reading the bytes");
  size_t diff = 0;
  for (size_t i = 0; i < a_bytes; ++i) diff += ha[i] != hb[i];
  return diff;
}

}  // namespace

int cmd_weights(int argc, char** argv) {
  if (argc < 3) {
    usage();
    return 2;
  }
  const std::string dir = argv[1];
  const lx_wtype target = parse_target(argv[2]);
  std::string reference, against;
  int samples = 24;
  for (int i = 3; i < argc; ++i) {
    const std::string a = argv[i];
    if (a == "--reference" && i + 1 < argc) {
      reference = argv[++i];
    } else if (a == "--bytes-against" && i + 1 < argc) {
      against = argv[++i];
    } else if (a == "--samples" && i + 1 < argc) {
      samples = std::stoi(argv[++i]);
    } else {
      usage();
      return 2;
    }
  }
  if (samples < 2) throw std::runtime_error("--samples takes at least 2");

  const lx::MetisConfig config = lx::load_config(dir);
  std::printf("target   %s as %s (the folder holds %s)\n", dir.c_str(), format_name(target),
              format_name(lx::ModelWeights::stored_format(dir)));

  // The reference is read first, so the target is the only large load on the card at a time.
  std::vector<std::string> names;
  std::map<std::string, std::map<int64_t, std::vector<uint16_t>>> ref;
  if (!reference.empty()) {
    lx::LoadOptions bf16;
    bf16.target = LX_W_BF16;
    lx::ModelWeights r = lx::ModelWeights::load(reference, config, bf16);
    std::printf("reference %s as bf16: %s resident\n", reference.c_str(), mib(r.device_bytes()).c_str());
    names = sample_names(r, samples);
    for (const std::string& name : names)
      for (int64_t e : experts_to_read(r.projection_shape(name))) ref[name][e] = bf16_on_host(r, name, e);
  }

  lx::LoadOptions options;
  options.target = target;
  const auto t0 = Clock::now();
  lx::ModelWeights model = lx::ModelWeights::load(dir, config, options);
  const double seconds = std::chrono::duration<double>(Clock::now() - t0).count();
  std::printf("load     %.1f s: %s resident, %s peak during the load\n", seconds,
              mib(model.device_bytes()).c_str(), mib(model.peak_device_bytes()).c_str());
  if (names.empty()) names = sample_names(model, samples);

  if (!ref.empty()) {
    std::printf("\n%-34s %-6s %-16s %12s %12s %10s\n", "projection", "format", "shape", "max |err|", "rel RMS",
                "bits differ");
    Error total;
    double worst_rms = 0, worst_abs = 0;
    for (const std::string& name : names) {
      const lx::ModelWeights::Shape s = model.projection_shape(name);
      Error err;
      for (int64_t e : experts_to_read(s)) err.add(bf16_on_host(model, name, e), ref.at(name).at(e));
      total.values += err.values;
      total.mismatched += err.mismatched;
      worst_abs = std::max(worst_abs, err.max_abs);
      worst_rms = std::max(worst_rms, err.rel_rms());
      char shape[40];
      if (s.stack) {
        std::snprintf(shape, sizeof shape, "%lld x %lld x %lld", static_cast<long long>(s.E),
                      static_cast<long long>(s.N), static_cast<long long>(s.K));
      } else {
        std::snprintf(shape, sizeof shape, "%lld x %lld", static_cast<long long>(s.N), static_cast<long long>(s.K));
      }
      std::printf("%-34s %-6s %-16s %12.3e %12.3e %10lld\n", name.c_str(), format_name(model.projection(name)->type),
                  shape, err.max_abs, err.rel_rms(), static_cast<long long>(err.mismatched));
    }
    std::printf("\n%lld values over %zu projections: worst max |err| %.3e, worst rel RMS %.3e, %lld bits differ\n",
                static_cast<long long>(total.values), names.size(), worst_abs, worst_rms,
                static_cast<long long>(total.mismatched));
  }

  if (!against.empty()) {
    lx::LoadOptions as_is;
    as_is.target = lx::ModelWeights::stored_format(against);
    lx::ModelWeights other = lx::ModelWeights::load(against, config, as_is);
    std::printf("\nagainst  %s holds %s (%s resident)\n", against.c_str(), format_name(as_is.target),
                mib(other.device_bytes()).c_str());
    int identical = 0;
    for (const std::string& name : names) {
      if (model.projection(name)->type != other.projection(name)->type) {
        std::printf("  %-34s format differs (%s here, %s there)\n", name.c_str(),
                    format_name(model.projection(name)->type), format_name(other.projection(name)->type));
        continue;
      }
      size_t differing = 0;
      for (int64_t e : experts_to_read(model.projection_shape(name))) {
        const lx_weight a = model.projection_expert(name, e), b = other.projection_expert(name, e);
        const auto ab = model.expert_bytes(name), bb = other.expert_bytes(name);
        differing += differing_bytes(a.data, ab.first, b.data, bb.first);
        if (a.scale) differing += differing_bytes(a.scale, ab.second, b.scale, bb.second);
      }
      if (differing == 0) ++identical;
      std::printf("  %-34s %s\n", name.c_str(), differing == 0 ? "identical bytes" : "BYTES DIFFER");
      if (differing) std::printf("    %zu bytes differ\n", differing);
    }
    std::printf("%d of %zu sampled projections are byte for byte the folder's own\n", identical, names.size());
  }
  return 0;
}
