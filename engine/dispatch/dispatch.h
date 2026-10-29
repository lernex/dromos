// The backend dispatcher, as host code sees it. The lx_* functions of engine/cuda/kernels.h are stubs that
// forward to one backend shared library, opened at run time, so the lernex executable links no GPU backend.
// This header says which library was opened and what it does not export.
#pragma once

#include <algorithm>
#include <string>
#include <vector>

namespace lx::dispatch {

// What the loader found. Filled in on the first use of info().
struct Info {
  // The name the library gives itself: "CUDA", "HIP", "Metal" or "CPU". "Unknown" for an id the loader does
  // not know, and "" when nothing loaded.
  std::string backend;
  std::string path;                  // the library loaded, as a full path, or ""
  std::vector<std::string> missing;  // lx_ symbols the loaded backend does not export
  std::string error;                 // why nothing loaded, or ""
};

// The loaded backend. The first call loads it: LX_KERNELS, then LX_BACKEND, then a probe of the installed
// libraries. Every later call returns the same object.
const Info& info();

// Whether the loaded backend exports symbol.
inline bool has(const char* symbol) {
  const std::vector<std::string>& missing = info().missing;
  return std::find(missing.begin(), missing.end(), symbol) == missing.end();
}

// Whether a backend may leave symbol out without that being a gap: host code checks has() before it calls one of
// these, and falls back to a path every backend exports. They are CUDA's alone: the persistent decode program
// (lx_program_*, opt-in with LX_PROGRAM, else the step runs kernel by kernel) and the skinny decode GEMM
// (lx_gemm_skinny, else lx_gemm). doctor does not list them as missing.
inline bool optional(const std::string& symbol) {
  return symbol.rfind("lx_program_", 0) == 0 || symbol == "lx_gemm_skinny";
}

// The last message about a missing symbol ("backend CUDA lacks lx_x"), or "" when there was none.
std::string last_message();

}  // namespace lx::dispatch
