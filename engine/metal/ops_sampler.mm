// Sampling on Metal (kernels/sampler.metal): lx_sample, the penalty counts, and constrained decoding's
// token masks.
#include "metal/backend.h"
#include "metal/kernels/args.h"

using namespace lx::metal;

namespace {

constexpr int kThreads = 1024;
constexpr int kMaxTopN = 20;

}  // namespace

static_assert(sizeof(lx_sampling) == sizeof(SamplingRow), "SamplingRow mirrors lx_sampling");

extern "C" {

int lx_sample(const float* logits, int V, const lx_sampling* params, int n, int32_t* seen, int32_t* out_counts,
              float* work, int32_t* tokens, float* token_logprob, int top_n, int32_t* top_tokens, float* top_logprobs,
              void* stream) {
  static Kernel k("sample");
  if (n <= 0) return kOk;
  if (top_n < 0 || top_n > kMaxTopN || V <= 0 || V > (1 << 24)) return kInvalidValue;
  if (!logits || !params || !seen || !out_counts || !work || !tokens) return kInvalidValue;
  if (top_n > 0 && (!top_tokens || !top_logprobs)) return kInvalidValue;
  // Rows of the logits and of the counts are picked by the parameters on the device.
  SampleArgs a{};
  a.logits = addr(stream, logits, kUnknownExtent);
  a.params = addr(stream, params, size_t(n) * sizeof(lx_sampling));
  a.seen = addr_w(stream, seen, kUnknownExtent);
  a.out_counts = addr_w(stream, out_counts, kUnknownExtent);
  a.work = addr_w(stream, work, size_t(n) * V * 4);
  a.tokens = addr_w(stream, tokens, size_t(n) * 4);
  a.token_logprob = addr_w(stream, token_logprob, size_t(n) * 4);
  a.top_tokens = top_n > 0 ? addr_w(stream, top_tokens, size_t(n) * top_n * 4) : 0;
  a.top_logprobs = top_n > 0 ? addr_w(stream, top_logprobs, size_t(n) * top_n * 4) : 0;
  a.V = V;
  a.top_n = top_n;
  return k.launch(stream, a, size1(NSUInteger(n)), size1(kThreads));
}

int lx_sampler_reset(int32_t* seen, int32_t* out_counts, int slot, int V, void* stream) {
  if (!seen || !out_counts || slot < 0 || V <= 0) return kInvalidValue;
  const size_t offset = size_t(slot) * V;
  if (int e = lx_rt_memset(seen + offset, 0, int64_t(V) * 4, stream)) return e;
  return lx_rt_memset(out_counts + offset, 0, int64_t(V) * 4, stream);
}

int lx_sampler_observe(int32_t* seen, int32_t* out_counts, int slot, int V, const int32_t* tokens, int n, int is_output,
                       void* stream) {
  static Kernel k("sampler_observe");
  if (n <= 0) return kOk;
  if (!seen || !out_counts || !tokens || slot < 0 || V <= 0) return kInvalidValue;
  const size_t offset = size_t(slot) * V;
  ObserveArgs a{addr_w(stream, seen + offset, size_t(V) * 4), addr(stream, tokens, size_t(n) * 4), n};
  if (int e = k.launch(stream, a, size1(ceil_div(n, 256)), size1(256))) return e;
  if (!is_output) return kOk;
  ObserveArgs b{addr_w(stream, out_counts + offset, size_t(V) * 4), addr(stream, tokens, size_t(n) * 4), n};
  return k.launch(stream, b, size1(ceil_div(n, 256)), size1(256));
}

int lx_mask_logits(float* logits, int V, const int32_t* rows, const uint32_t* bits, int n, void* stream) {
  static Kernel k("mask_logits");
  if (n <= 0) return kOk;
  if (!logits || !rows || !bits || V <= 0) return kInvalidValue;
  const int words = (V + 31) / 32;
  MaskLogitsArgs a{addr_w(stream, logits, kUnknownExtent), addr(stream, rows, size_t(n) * 4),
                   addr(stream, bits, size_t(n) * words * 4), V, words};
  return k.launch(stream, a, size2(ceil_div(V, 256), NSUInteger(n)), size2(256, 1));
}

}  // extern "C"
