// Embedding, the residual streams and the depth memory (kernels/streams.metal, kernels/memory.metal).
#include <algorithm>
#include <initializer_list>

#include "cuda/tunable.h"
#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kMaxStreams = 8;
constexpr int kMaxEntries = 8;
constexpr int kDecodeTokens = 8;  // at most this many tokens take the spread-out decode forms

NSUInteger row_threads(int dim, int cap = 256) {
  return static_cast<NSUInteger>(std::min(cap, std::max(32, (dim + 31) / 32 * 32)));
}

// Whether rows of D floats can take the float4 kernels: D a multiple of 4, the f32 buffers 16-byte
// aligned and the bf16 ones 8-byte aligned.
bool vec4(int D, std::initializer_list<const void*> f32, std::initializer_list<const void*> bf16 = {}) {
  if (D % 4 != 0) return false;
  for (const void* p : f32)
    if (reinterpret_cast<uintptr_t>(p) & 15) return false;
  for (const void* p : bf16)
    if (reinterpret_cast<uintptr_t>(p) & 7) return false;
  return true;
}
constexpr int kReadKeep = 4;  // kernels/streams.metal mhc_read_vec

// Variants the tests and the tuner may pick (tunable.h); -1 in the table means the heuristic.
const lx::TunableRegistration reg_mhc("mhc_read", {"row", "split"});
const lx::TunableRegistration reg_fuse("memory_fuse", {"token", "stream"});

}  // namespace

extern "C" {

int lx_embed_streams(const void* table, const int32_t* ids, const void* stream_emb, float* streams, int T, int S, int D,
                     void* stream) {
  static Kernel k("embed_streams");
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  // ids pick the table's rows, and its height is not passed: an unknown extent.
  EmbedArgs a{addr(stream, table, kUnknownExtent), addr(stream, ids, size_t(T) * 4),
              addr(stream, stream_emb, size_t(S) * D * 2), addr_w(stream, streams, size_t(T) * S * D * 4), S, D};
  return k.launch(stream, a, size1(T), size1(row_threads(D)));
}

int lx_stream_mean(const float* streams, float* mean, int T, int S, int D, void* stream) {
  static Kernel k("stream_mean");
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  static Kernel kv("stream_mean_vec");
  StreamMeanArgs a{addr(stream, streams, size_t(T) * S * D * 4), addr_w(stream, mean, size_t(T) * D * 4), T, S, D};
  if (vec4(D, {streams, mean})) return kv.launch(stream, a, size1(ceil_div(size_t(T) * (D / 4), 256)), size1(256));
  return k.launch(stream, a, size2(ceil_div(D, 256), std::min(T, 65535)), size1(256));
}

int lx_mhc_read(const float* streams, const float* matrix, const float* read, const void* norm_w, float eps, float* mixed,
                void* h, int T, int S, int D, void* stream) {
  static Kernel row("mhc_read_row"), split("mhc_read_split"), vec("mhc_read_vec");
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (S > kMaxStreams) return kInvalidValue;
  int variant = lx_tune_get("mhc_read", T, D, S);
  if (variant < 0) variant = T <= kDecodeTokens && lx_arch() >= 1000 ? 1 : 0;
  // The row form keeps the source row in threadgroup memory; very wide rows take the split form.
  if (D * 4 > 28 * 1024) variant = 1;
  MhcReadArgs a{addr(stream, streams, size_t(T) * S * D * 4), addr(stream, matrix, size_t(S) * S * 4),
                addr(stream, read, size_t(S) * 4), addr(stream, norm_w, size_t(D) * 2),
                addr_w(stream, mixed, size_t(T) * S * D * 4), addr_w(stream, h, size_t(T) * D * 2), eps, S, D, 1};
  const bool v4 = vec4(D, {streams, mixed}, {norm_w, h});
  if (v4 && variant == 0 && D / 4 <= kReadKeep * 1024) {
    // Two chunks of four per thread, so a token's loads are all in flight at once.
    return vec.launch(stream, a, size1(T), size1(row_threads(ceil_div(D / 4, 2), 1024)));
  }
  if (variant == 0) return row.launch(stream, a, size1(T), size1(row_threads(D, 1024)), size_t(D) * 4);
  a.split = std::max(1, std::min(8, D / 256));
  if (v4) return vec.launch(stream, a, size1(size_t(T) * a.split), size1(256));
  return split.launch(stream, a, size1(size_t(T) * a.split), size1(256));
}

int lx_mhc_write(const float* mixed, const float* write, const float* update, float* streams, int T, int S, int D,
                 void* stream) {
  static Kernel k("mhc_write");
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  static Kernel kv("mhc_write_vec");
  MhcWriteArgs a{addr(stream, mixed, size_t(T) * S * D * 4), addr(stream, write, size_t(S) * 4),
                 addr(stream, update, size_t(T) * D * 4), addr_w(stream, streams, size_t(T) * S * D * 4), T, S, D};
  if (vec4(D, {mixed, update, streams})) return kv.launch(stream, a, size1(ceil_div(size_t(T) * (D / 4), 256)), size1(256));
  return k.launch(stream, a, size2(ceil_div(D, 256), std::min(T, 65535)), size1(256));
}

// The mixer's write and the MoE's read as two launches (Metal has no fused kernel for them yet).
int lx_mhc_write_read(float* mixed, const float* write, const float* update, float* streams, const float* matrix,
                      const float* read, const void* norm_w, float eps, void* h, int T, int S, int D, void* stream) {
  const int status = lx_mhc_write(mixed, write, update, streams, T, S, D, stream);
  if (status != kOk) return status;
  return lx_mhc_read(streams, matrix, read, norm_w, eps, mixed, h, T, S, D, stream);
}

int lx_memory_attend(const float* q, const float* bank_k, const float* bank_v, int n_entries, int max_entries, void* out,
                     int T, int S, int M, void* stream) {
  static Kernel k("memory_attend");
  if (n_entries < 1 || n_entries > max_entries || n_entries > kMaxEntries) return kInvalidValue;
  if (T <= 0 || S <= 0 || M <= 0) return kOk;
  const int rows = T * S;
  const size_t bank = size_t(T) * max_entries * M * 4;
  MemoryAttendArgs a{addr(stream, q, size_t(rows) * M * 4), addr(stream, bank_k, bank), addr(stream, bank_v, bank),
                     addr_w(stream, out, size_t(rows) * M * 2), n_entries, max_entries, rows, S, M};
  constexpr int kRowsPerGroup = 4;  // one simdgroup per row
  return k.launch(stream, a, size1(ceil_div(rows, kRowsPerGroup)), size1(32 * kRowsPerGroup));
}

int lx_memory_fuse(float* streams, const float* projected, const void* gate, const void* gate_bias, float* summary,
                   int T, int S, int D, void* stream) {
  static Kernel k("memory_fuse"), kv("memory_fuse_vec");
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  int variant = lx_tune_get("memory_fuse", T, D, S);
  if (variant < 0) variant = T <= kDecodeTokens && lx_arch() >= 1000 ? 1 : 0;
  const size_t rows = size_t(T) * S * D * 4;
  MemoryFuseArgs a{addr_w(stream, streams, rows), addr(stream, projected, rows), addr(stream, gate, size_t(S) * D * 2),
                   addr(stream, gate_bias, size_t(S) * 2), addr_w(stream, summary, size_t(T) * D * 4), S, D,
                   variant == 1 ? S : 1};
  const NSUInteger groups = variant == 1 ? size_t(T) * S : size_t(T);
  if (vec4(D, {streams, projected, summary}, {gate}))
    return kv.launch(stream, a, size1(groups), size1(row_threads(D / 4, variant == 1 ? 256 : 1024)));
  return k.launch(stream, a, size1(groups), size1(row_threads(D, variant == 1 ? 256 : 1024)));
}

int lx_continuation_finish(const float* logit, const float* state, const float* difference, float limit, float* out,
                           int T, int D, void* stream) {
  static Kernel k("continuation_finish");
  if (T <= 0 || D <= 0) return kOk;
  ContinuationArgs a{addr(stream, logit, size_t(T) * 4), addr(stream, state, size_t(T) * D * 4),
                     addr(stream, difference, size_t(T) * D * 4), addr_w(stream, out, size_t(T) * 4), limit, D};
  return k.launch(stream, a, size1(T), size1(row_threads(D)));
}

}  // extern "C"
