// The dispatcher's internals, shared by dispatch.cpp (the loader) and the generated table
// (generated/lx_dispatch_gen.cpp, from tools/gen_dispatch.py). Host code does not include this; it
// includes dispatch/dispatch.h.
#pragma once

#include <string>
#include <vector>

namespace lx::dispatch::detail {

// Loads the backend on the first call and does nothing after. Safe to call from several threads, and it
// never calls an lx_ stub.
void ensure();

// Points every lx_ slot at the symbol `lib` exports, or at its missing function when it exports none. The
// names of the missing symbols are appended to `missing`. Defined by the generated table.
void bind(void* lib, std::vector<std::string>& missing);

}  // namespace lx::dispatch::detail

namespace lx::dispatch {

// Called by a missing function: records "backend <name> lacks <symbol>" and prints it to stderr, once per
// symbol. Never aborts.
void report_missing(const char* symbol);

}  // namespace lx::dispatch
