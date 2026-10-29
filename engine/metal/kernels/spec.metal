// Speculative decoding's kernels (kernels.h, "Speculative decoding"; host side in ops_spec.mm).

// ---- Batch-invariant f32 GEMV ---------------------------------------------------------------
// One simdgroup per output column n, four columns per threadgroup. Lane l takes k = l, l + 32, ... in
// order with one sum per input row, then the simdgroup adds its lanes in a fixed tree. No step of that
// order depends on M, so a row's bits never depend on the other rows of the step.
constant constexpr int kSpecGemvRows = 8;

kernel void spec_gemv_f32(constant SpecGemvArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                          uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const int n = int(tg) * 4 + int(sg);
  if (n >= a.N) return;
  device const float* wr = a.w + long(n) * a.K;
  float acc[kSpecGemvRows];
  for (int m = 0; m < kSpecGemvRows; ++m) acc[m] = 0.0f;
  for (int k = lane; k < a.K; k += 32) {
    const float wv = wr[k];
    for (int m = 0; m < kSpecGemvRows; ++m)
      if (m < a.M) acc[m] = fma(wv, a.x[long(m) * a.ld_x + k], acc[m]);
  }
  for (int m = 0; m < kSpecGemvRows; ++m) {
    if (m >= a.M) break;
    float sum = simd_sum(acc[m]);
    if (lane == 0) {
      if (a.bias) sum += a.bias[n];
      a.y[long(m) * a.ld_y + n] = sum;
    }
  }
}

// ---- Causal conv with pending positions ------------------------------------------------------
// The window of new token t reads the combined sequence [base state (W - 1 rows, zeros without state),
// pending rows 0 .. fold + keep - 1, new rows]: the inputs a plain step would have seen at that position,
// in the same order, so the output is conv1d_out's float for float.
constant constexpr int kSpecMaxPending = 8;  // LX_SPEC_MAX_PENDING

kernel void spec_conv_out(constant SpecConvArgs& a [[buffer(0)]], uint2 tg [[threadgroup_position_in_grid]],
                          uint2 tid [[thread_position_in_threadgroup]], uint2 tpg [[threads_per_threadgroup]]) {
  const int s = tg.y;
  const int c = tg.x * tpg.x + tid.x;
  if (c >= a.C) return;
  const int W = a.W, C = a.C;
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bool has = a.has_state[s] != 0;
  const long slot = a.slot_ids[s];
  const long hidx = a.hist_ids ? a.hist_ids[s] : 0;  // the history is indexed by history, not slot
  const device float* st = a.state + slot * (W - 1) * C;
  const device float* hs = a.hist + hidx * kSpecMaxPending * C;
  const int pending = a.fold[s] + a.keep[s];
  float wk[8];
  for (int k = 0; k < 8; ++k) wk[k] = k < W ? bf(a.w[c * W + k]) : 0.0f;
  const float b = bf(a.bias[c]);
  for (int t = tid.y; t < len; t += tpg.y) {
    float acc = b;
    for (int k = 0; k < 8; ++k) {
      if (k < W) {
        // Position relative to the first new token: negative positions are pending rows, then the state.
        const int p = t - (W - 1) + k;
        float v = 0.0f;
        if (p >= 0) v = a.x[long(start + p) * a.ld_x + c];
        else if (pending + p >= 0) v = hs[long(pending + p) * C + c];
        else if (has) v = st[(pending + p + W - 1) * C + c];
        acc = fma(wk[k], v, acc);
      }
    }
    a.out[long(start + t) * C + c] = silu(acc);
  }
}

// One thread per (channel, sequence). With fold > 0 the base state becomes the last W - 1 rows of
// [base state, pending rows 0 .. fold - 1] and the kept rows move to the front; then the new rows are
// appended after the kept ones. Every value is read before its row is written.
kernel void spec_conv_state(constant SpecConvArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int s = gid.y;
  const int c = gid.x;
  if (c >= a.C) return;
  const int W = a.W, C = a.C;
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const bool has = a.has_state[s] != 0;
  const long slot = a.slot_ids[s];
  const long hidx = a.hist_ids ? a.hist_ids[s] : 0;  // the history is indexed by history, not slot
  device float* st = a.state + slot * (W - 1) * C;
  device float* hs = a.hist + hidx * kSpecMaxPending * C;
  const int f = a.fold[s], kp = a.keep[s];
  if (f > 0) {
    float nv[7];
    for (int j = 0; j < 7; ++j) {
      if (j < W - 1) {
        const int m = f + j;  // row of [base state, pending rows]
        nv[j] = m >= W - 1 ? hs[long(m - (W - 1)) * C + c] : (has ? st[m * C + c] : 0.0f);
      }
    }
    float kv[kSpecMaxPending];
    for (int j = 0; j < kSpecMaxPending; ++j)
      if (j < kp) kv[j] = hs[long(f + j) * C + c];
    for (int j = 0; j < 7; ++j)
      if (j < W - 1) st[j * C + c] = nv[j];
    for (int j = 0; j < kSpecMaxPending; ++j)
      if (j < kp) hs[long(j) * C + c] = kv[j];
  }
  for (int t = 0; t < len; ++t) hs[long(kp + t) * C + c] = a.x[long(start + t) * a.ld_x + c];
}

// ---- Selective scan with pending positions ---------------------------------------------------
// ssm_scan's layout, staging and per-token math (mamba.metal; read that one first): one threadgroup per
// (head, sequence), four lanes per state row. Before the new tokens, the threadgroup replays the pending
// positions from the slot's history (the fold positions, then the base state write, then the kept ones),
// and as each chunk of new tokens is staged it copies their x, B and dt_raw into the history after the
// kept rows. A history row of one head is x [round4(P)] | B [round4(N)] | dt_raw | pad, kSpecMaxPending
// rows per (slot, head). Threadgroup memory: the token staging, then the replayed rows.
inline int spec_round4(int v) { return (v + 3) & ~3; }

template <int NI>
kernel void spec_ssm_scan(constant SpecScanArgs& a [[buffer(0)]], threadgroup float* stage [[threadgroup(0)]],
                          uint2 tg [[threadgroup_position_in_grid]], uint2 tid2 [[thread_position_in_threadgroup]],
                          uint2 tpg [[threads_per_threadgroup]]) {
  const uint tid = tid2.x, threads = tpg.x;
  threadgroup float dts[kScanChunk], decays[kScanChunk];
  constexpr int BW = 16 * NI;
  const int P = a.P, N = a.N;
  const int BO = spec_round4(P), CO = BO + BW, DO = CO + BW;
  const int SW = DO + 4;
  const int HB = spec_round4(P), HD = HB + spec_round4(N), HS = HD + 4;
  threadgroup float* pend = stage + kScanChunk * SW;
  const int q = tid & 3, p = tid >> 2;
  const bool valid = p < P;
  const int h = tg.x, s = tg.y;
  const int g = h / (a.H / a.G);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const long slot = a.slot_ids[s];
  const long hidx = a.hist_ids ? a.hist_ids[s] : 0;  // the history is indexed by history, not slot
  device float* st = a.state + (slot * a.H + h) * P * N;
  device float* hs = a.hist + (hidx * a.H + h) * kSpecMaxPending * HS;
  const bool has = a.has_state[s] != 0;
  const int fold = a.fold[s], keep = a.keep[s];
  const float negA = -exp(a.A_log[h]);
  const float Dh = a.D[h], dt_bias = a.dt_bias[h];

  for (int i = tid; i < kScanChunk * SW; i += threads) stage[i] = 0.0f;
  for (int i = tid; i < (fold + keep) * HS; i += threads) pend[i] = hs[i];
  threadgroup_barrier(mem_flags::mem_threadgroup);

  float S[NI][4];
  for (int i = 0; i < NI; ++i)
    for (int e = 0; e < 4; ++e) {
      const int n = 4 * (q + 4 * i) + e;
      S[i][e] = (has && valid && n < N) ? st[p * N + n] : 0.0f;
    }

  // One pending position's state update: the operations of ssm_scan's token loop, without y.
  for (int r = 0; r < fold + keep; ++r) {
    if (r == fold && fold > 0 && valid) {
      for (int i = 0; i < NI; ++i)
        for (int e = 0; e < 4; ++e) {
          const int n = 4 * (q + 4 * i) + e;
          if (n < N) st[p * N + n] = S[i][e];
        }
    }
    const threadgroup float* row = pend + r * HS;
    const float dt = softplus(row[HD] + dt_bias);
    const float decay = exp(dt * negA);
    const float xp = valid ? row[p] : 0.0f;
    const float dtx = dt * xp;
    for (int i = 0; i < NI; ++i) {
      const int c0 = 4 * (q + 4 * i);
      for (int e = 0; e < 4; ++e) {
        const float bv = c0 + e < N ? row[HB + c0 + e] : 0.0f;
        S[i][e] = fma(decay, S[i][e], dtx * bv);
      }
    }
  }
  if (fold > 0 && keep == 0 && valid) {
    for (int i = 0; i < NI; ++i)
      for (int e = 0; e < 4; ++e) {
        const int n = 4 * (q + 4 * i) + e;
        if (n < N) st[p * N + n] = S[i][e];
      }
  }
  // The old history is in pend, so the kept rows can move to the front; the new rows follow them.
  if (fold > 0)
    for (int i = tid; i < keep * HS; i += threads) hs[i] = pend[fold * HS + i];

  for (int c0 = 0; c0 < len; c0 += kScanChunk) {
    const int count = min(kScanChunk, len - c0);
    for (int tok = 0; tok < count; ++tok) {
      const long t = start + c0 + tok;
      threadgroup float* d = stage + tok * SW;
      const device float* xr = a.x + t * a.ld_x + h * P;
      const device float* br = a.B + t * a.ld_bc + g * N;
      const device float* cr = a.C + t * a.ld_bc + g * N;
      device float* hrow = hs + (keep + c0 + tok) * HS;
      for (int j = tid; j < P; j += threads) {
        d[j] = xr[j];
        hrow[j] = xr[j];
      }
      for (int j = tid; j < N; j += threads) {
        d[BO + j] = br[j];
        d[CO + j] = cr[j];
        hrow[HB + j] = br[j];
      }
      if (tid == 0) {
        d[DO] = a.dt_raw[t * a.ld_dt + h];
        hrow[HD] = d[DO];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (int(tid) < count) {
      const float dt = softplus(stage[tid * SW + DO] + dt_bias);
      dts[tid] = dt;
      decays[tid] = exp(dt * negA);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int tok = 0; tok < count; ++tok) {
      const threadgroup float* row = stage + tok * SW;
      const float xp = valid ? row[p] : 0.0f;
      const float dtx = dts[tok] * xp;
      const float decay = decays[tok];
      float acc = 0.0f;
      for (int i = 0; i < NI; ++i) {
        const int c = 4 * (q + 4 * i);
        const float4 bb = *(const threadgroup float4*)(row + BO + c);
        const float4 cc = *(const threadgroup float4*)(row + CO + c);
        S[i][0] = fma(decay, S[i][0], dtx * bb.x);
        S[i][1] = fma(decay, S[i][1], dtx * bb.y);
        S[i][2] = fma(decay, S[i][2], dtx * bb.z);
        S[i][3] = fma(decay, S[i][3], dtx * bb.w);
        acc = fma(S[i][0], cc.x, acc);
        acc = fma(S[i][1], cc.y, acc);
        acc = fma(S[i][2], cc.z, acc);
        acc = fma(S[i][3], cc.w, acc);
      }
      acc += simd_shuffle_xor(acc, 1);
      acc += simd_shuffle_xor(acc, 2);
      if (valid && q == 0) a.y[(long(start + c0 + tok) * a.H + h) * P + p] = acc + Dh * xp;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
template [[host_name("spec_ssm_scan_1")]] kernel void spec_ssm_scan<1>(constant SpecScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("spec_ssm_scan_2")]] kernel void spec_ssm_scan<2>(constant SpecScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("spec_ssm_scan_4")]] kernel void spec_ssm_scan<4>(constant SpecScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("spec_ssm_scan_8")]] kernel void spec_ssm_scan<8>(constant SpecScanArgs&, threadgroup float*, uint2, uint2, uint2);
template [[host_name("spec_ssm_scan_16")]] kernel void spec_ssm_scan<16>(constant SpecScanArgs&, threadgroup float*, uint2, uint2, uint2);

// ---- Sampling ---------------------------------------------------------------------------------
// The filtering is the sampler's (kept_key in sampler.metal), so a row's kept set is the one lx_sample
// computes. Philox domains: each kind of draw has its own counter space, apart from lx_sample's (0x4C58).
constant constexpr uint kDomainAccept = 0x53504131u;    // "SPA1": the acceptance uniforms
constant constexpr uint kDomainResidual = 0x53505231u;  // "SPR1": residual and bonus draws

// One threadgroup per row of a verify step: sampler.metal's penalties (step 1) on the row, with the counts the step's
// drafts before the row add (tokens[1 .. r]). The expressions are sampler.metal's, as penalize_kernel's are
// sample_kernel's.
kernel void spec_penalize(constant SpecPenalizeArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  const int V = a.V;
  device float* l = a.logits + long(r) * V;
  for (int e = tid; e < V; e += threads) {
    int extra = 0;
    for (uint t = 1; t <= r; ++t) extra += a.tokens[t] == e;
    const int sc = a.seen[e] + extra;
    const int co = a.out_counts[e] + extra;
    float x = l[e];
    if (a.repetition_penalty != 1.0f && sc > 0) x = x > 0.0f ? x / a.repetition_penalty : x * a.repetition_penalty;
    if (co > 0) x -= a.frequency_penalty * float(co) + a.presence_penalty;
    l[e] = x;
  }
}

// One threadgroup per row: the row's filtered distribution, in place of its scaled logits.
kernel void spec_probs(constant SpecProbsArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup KeepScratch sh;
  threadgroup int redi[32];
  const SpecFilter f = a.filters[r];
  const int V = a.V;
  device const float* l = a.logits + long(r) * V;
  device float* out = a.probs + long(r) * V;
  if (!(f.temperature > 0.0f)) {
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += threads)
      if (l[e] > best) {
        best = l[e];
        best_i = e;
      }
    float bv;
    int bi;
    group_argmax(best, best_i, sh.red, redi, tid, threads, bv, bi);
    for (int e = tid; e < V; e += threads) out[e] = e == bi ? 1.0f : 0.0f;
    return;
  }
  const float inv_t = 1.0f / f.temperature;
  float m = -INFINITY;
  for (int e = tid; e < V; e += threads) {
    const float s = l[e] * inv_t;
    out[e] = s;
    m = max(m, s);
  }
  threadgroup_barrier(mem_flags::mem_device);  // kept_key reads every thread's scaled logits
  const float top = group_max(m, sh.red, tid, threads);
  const uint keep_key = kept_key(out, V, f.top_k, f.top_p, f.min_p, top, sh, tid, threads);
  float z = 0.0f;
  for (int e = tid; e < V; e += threads)
    if (float_key(out[e]) >= keep_key) z += exp(out[e] - top);
  const float inv_z = 1.0f / group_sum(z, sh.red, tid, threads);
  for (int e = tid; e < V; e += threads) {
    const float v = out[e];
    out[e] = float_key(v) >= keep_key ? exp(v - top) * inv_z : 0.0f;
  }
}

// v = m 2^e for a positive finite float, m an integer below 2^24.
inline uint float_mantissa(float v, thread int& e) {
  const uint u = as_type<uint>(v);
  const int biased = int((u >> 23) & 0xffu);
  if (biased == 0) {
    e = -149;
    return u & 0x7fffffu;
  }
  e = biased - 150;
  return (u & 0x7fffffu) | 0x800000u;
}

// The acceptance test u q < p for u = (bits + 0.5) / 2^53 (bits below 2^53), decided exactly in integers,
// since the GPU has no doubles: (2 bits + 1) q < p 2^54.
inline bool accept_draft(ulong bits, float p, float q) {
  if (!(p > 0.0f) || isnan(q)) return false;
  if (!(q > 0.0f)) return true;
  int eq, ep;
  const ulong mq = float_mantissa(q, eq), mp = float_mantissa(p, ep);
  const ulong lhs_a = 2ul * bits + 1ul;  // 54 bits
  const ulong lo = lhs_a * mq, hi = mulhi(lhs_a, mq);  // the 78-bit product
  const int k = ep + 54 - eq;  // is the product below mp 2^k?
  if (k >= 0) {
    if (k >= 104) return true;  // mp 2^k >= 2^104 is past any product
    ulong rhi = 0, rlo = 0;
    if (k >= 64) {
      rhi = mp << (k - 64);
    } else if (k == 0) {
      rlo = mp;
    } else {
      rlo = mp << k;
      rhi = mp >> (64 - k);
    }
    return hi < rhi || (hi == rhi && lo < rlo);
  }
  const int shift = -k;  // is the product times 2^shift below mp (< 2^24)?
  if (shift >= 24 || hi != 0 || lo >= (1ul << 24)) return false;
  return (lo << shift) < mp;
}

// One threadgroup per sequence. Thread 0 walks the drafts while they hold, then the threadgroup draws the
// token that ends the step: from the residual at the first rejection, or from the last row when every
// draft holds. The draws are Gumbel-max over log(mass), the draw lx_sample makes over the kept logits.
kernel void spec_verify(constant SpecVerifyArgs& a [[buffer(0)]], uint s [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  threadgroup int redi[32];
  threadgroup int s_j;  // thread 0's count of drafts that held
  const int V = a.V;
  const int start = a.cu[s], rows = a.cu[s + 1] - start;
  const int k = rows - 1;
  const SpecRng r = a.rng[s];
  int j = 0;
  if (tid == 0) {
    for (; j < k; ++j) {
      const int d = a.tokens[start + j + 1];
      const long row = long(start + j) * V;
      const float pd = a.p[row + d];
      const float qd = a.q ? a.q[row + d] : 1.0f;
      uint c[4] = {0u, uint(r.position + ulong(j)), uint((r.position + ulong(j)) >> 32), kDomainAccept};
      philox4x32(c, uint(r.seed), uint(r.seed >> 32));
      const ulong bits = ((ulong(c[0]) << 21) ^ ulong(c[1] >> 11)) & ((1ul << 53) - 1ul);
      if (!accept_draft(bits, pd, qd)) break;
    }
    s_j = j;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  j = s_j;
  const bool residual = j < k;
  const int d = residual ? a.tokens[start + j + 1] : -1;
  device const float* pr = a.p + long(start + j) * V;
  device const float* qr = a.q ? a.q + long(start + j) * V : nullptr;
  const ulong pos = r.position + ulong(j);
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  for (int e = tid; e < V; e += threads) {
    float m = pr[e];
    if (residual) m = qr ? max(m - qr[e], 0.0f) : (e == d ? 0.0f : m);
    if (m > 0.0f) {
      const float v = log(m) - log(-log(philox_uniform(r.seed, pos, uint(e), kDomainResidual)));
      if (v > best) {
        best = v;
        best_i = e;
      }
    }
  }
  float bv;
  int bi;
  group_argmax(best, best_i, red, redi, tid, threads, bv, bi);
  if (bv == -INFINITY) {
    // No residual mass left (p equals q up to rounding): the row's own distribution.
    best = -INFINITY;
    best_i = 0x7fffffff;
    for (int e = tid; e < V; e += threads)
      if (pr[e] > 0.0f) {
        const float v = log(pr[e]) - log(-log(philox_uniform(r.seed, pos, uint(e), kDomainResidual)));
        if (v > best) {
          best = v;
          best_i = e;
        }
      }
    group_argmax(best, best_i, red, redi, tid, threads, bv, bi);
  }
  if (tid == 0) {
    a.accepted[s] = j;
    a.next[s] = bi;
  }
}

// One threadgroup per row: argmax with ties to the lower index.
kernel void spec_argmax(constant SpecGreedyArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  threadgroup int redi[32];
  device const float* l = a.logits + long(row) * a.V;
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  for (int e = tid; e < a.V; e += threads)
    if (l[e] > best) {
      best = l[e];
      best_i = e;
    }
  float bv;
  int bi;
  group_argmax(best, best_i, red, redi, tid, threads, bv, bi);
  if (tid == 0) a.argmax[row] = bi;
}

// One thread per sequence: drafts hold while each equals the argmax of the row before it.
kernel void spec_greedy_accept(constant SpecGreedyArgs& a [[buffer(0)]], uint s [[thread_position_in_grid]]) {
  if (int(s) >= a.n_seq) return;
  const int start = a.cu[s], k = a.cu[s + 1] - start - 1;
  int j = 0;
  while (j < k && a.argmax[start + j] == a.tokens[start + j + 1]) ++j;
  a.accepted[s] = j;
  a.next[s] = a.argmax[start + j];
}

// One threadgroup per draw: Gumbel-max over log(probs[rows[i], :]).
kernel void spec_sample(constant SpecSampleArgs& a [[buffer(0)]], uint i [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  threadgroup int redi[32];
  const SpecRng r = a.rng[i];
  device const float* pr = a.probs + long(a.rows ? a.rows[i] : int(i)) * a.V;
  float best = -INFINITY;
  int best_i = 0x7fffffff;
  for (int e = tid; e < a.V; e += threads)
    if (pr[e] > 0.0f) {
      const float v = log(pr[e]) - log(-log(philox_uniform(r.seed, r.position, uint(e), a.domain)));
      if (v > best) {
        best = v;
        best_i = e;
      }
    }
  float bv;
  int bi;
  group_argmax(best, best_i, red, redi, tid, threads, bv, bi);
  if (tid == 0) a.tokens[i] = bi;
}

// ---- Block drafter: per-head norm and RoPE, block attention ------------------------------------

// a * b exactly, as an unevaluated pair (product, error).
inline float2 two_prod(float a, float b) {
  const float p = a * b;
  return float2(p, fma(a, b, -p));
}

// One simdgroup per (row, head), in place: v = x[r, h * Dh ..], with norm weights
// v = v * rsqrt(mean(v^2) + eps) * w, then the pairs (i, i + Dh / 2) rotate by (pos0 + r) theta^(-2i / Dh).
// Positions run to 2^17 and more, where a float angle would be off by more than its own rounding, so the
// angle and its reduction to about [-pi, pi] are in float pairs (the frequencies come from the host's
// doubles as hi + lo), and only the reduced angle goes to sin and cos.
kernel void spec_head_norm_rope(constant SpecRopeArgs& a [[buffer(0)]], uint gid [[thread_position_in_grid]],
                                uint lane [[thread_index_in_simdgroup]]) {
  const int warp = int(gid >> 5);
  if (warp >= a.rows * a.heads) return;  // whole simdgroups leave together
  const int r = warp / a.heads, h = warp % a.heads, Dh = a.Dh, half_dim = Dh / 2;
  device float* v = a.x + long(r) * a.ld_x + long(h) * Dh;
  float av[4], bv[4], ss = 0.0f;
  for (int i = lane, n = 0; i < half_dim; i += 32, ++n) {
    av[n] = v[i];
    bv[n] = v[i + half_dim];
    ss += av[n] * av[n] + bv[n] * bv[n];
  }
  ss = simd_sum(ss);
  const float scale = a.w ? rsqrt(ss / float(Dh) + a.eps) : 1.0f;
  const float pos = float(a.pos0 + r);  // exact: the host keeps positions below 2^24
  constexpr float kTwoPiHi = 6.28318548202514648f, kTwoPiLo = -1.74845553146951215e-7f, kInvTwoPi = 0.159154943091895336f;
  for (int i = lane, n = 0; i < half_dim; i += 32, ++n) {
    float x1 = av[n], x2 = bv[n];
    if (a.w) {
      x1 = x1 * scale * bf(a.w[i]);
      x2 = x2 * scale * bf(a.w[i + half_dim]);
    }
    const float2 angle = two_prod(pos, a.inv_hi[i]);
    const float angle_lo = fma(pos, a.inv_lo[i], angle.y);
    const float k = rint(angle.x * kInvTwoPi);
    const float2 turns = two_prod(k, kTwoPiHi);
    const float reduced = (angle.x - turns.x) + (angle_lo - turns.y - k * kTwoPiLo);
    float c;
    const float s = sincos(reduced, c);
    v[i] = x1 * c - x2 * s;
    v[i + half_dim] = x2 * c + x1 * s;
  }
}

// One threadgroup per (head h, query row b). Keys: n_ctx context rows, then the block's B rows, all visible
// to every query. Every thread scores whole keys (float4 loads); the weighted sum splits the keys over
// threads / Dh groups of Dh threads and adds the groups' partial sums in a fixed order. Scores, softmax and
// sums in f32. The scores go through threadgroup memory in chunks of a.chunk keys with a running max (one
// chunk whenever they fit, which is then the plain two-pass softmax), so any context length runs within
// Metal's 32 KB. Threadgroup memory: query [Dh], partial sums [threads], scores [chunk].
kernel void spec_block_attention(constant SpecBlockAttnArgs& a [[buffer(0)]], threadgroup float* sm [[threadgroup(0)]],
                                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                                 uint2 tpg [[threads_per_threadgroup]]) {
  threadgroup float red[32];
  const uint threads = tpg.x;
  const int h = tg.x, b = tg.y, Dh = a.Dh, kv = h / (a.H / a.KVH), n = a.n_ctx + a.B;
  threadgroup float* qs = sm;
  threadgroup float* part = sm + Dh;
  threadgroup float* sc = part + threads;
  for (int d = tid; d < Dh; d += threads) qs[d] = a.q[long(b) * a.ld_q + h * Dh + d];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int groups = int(threads) / Dh, g = int(tid) / Dh, d = int(tid) % Dh;
  float m_run = -INFINITY, z_run = 0.0f, acc = 0.0f;
  for (int j0 = 0; j0 < n; j0 += a.chunk) {
    const int count = min(a.chunk, n - j0);
    for (int j = tid; j < count; j += threads) {
      const int key = j0 + j;
      const device float4* kr = (const device float4*)(key < a.n_ctx ? a.k_ctx + (long(key) * a.KVH + kv) * Dh
                                                                      : a.k_blk + long(key - a.n_ctx) * a.ld_blk + kv * Dh);
      float s = 0.0f;
      for (int d4 = 0; d4 < Dh / 4; ++d4) {
        const float4 k = kr[d4];
        s += qs[4 * d4] * k.x + qs[4 * d4 + 1] * k.y + qs[4 * d4 + 2] * k.z + qs[4 * d4 + 3] * k.w;
      }
      sc[j] = s * a.scale;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float m = -INFINITY;
    for (int j = tid; j < count; j += threads) m = max(m, sc[j]);
    const float m_new = max(m_run, group_max(m, red, tid, threads));
    float z = 0.0f;
    for (int j = tid; j < count; j += threads) {
      const float e = exp(sc[j] - m_new);
      sc[j] = e;
      z += e;
    }
    z = group_sum(z, red, tid, threads);  // its barriers also publish the exponentials
    const float f = exp(m_run - m_new);   // 0 for the first chunk
    z_run = z_run * f + z;
    float part_acc = 0.0f;
    if (g < groups) {
      for (int j = g; j < count; j += groups) {
        const int key = j0 + j;
        const device float* vr = key < a.n_ctx ? a.v_ctx + (long(key) * a.KVH + kv) * Dh
                                               : a.v_blk + long(key - a.n_ctx) * a.ld_blk + kv * Dh;
        part_acc += sc[j] * vr[d];
      }
    }
    acc = acc * f + part_acc;
    m_run = m_new;
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the next chunk overwrites the scores
  }
  part[tid] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (int(tid) < Dh) {
    float sum = 0.0f;
    for (int i = 0; i < groups; ++i) sum += part[i * Dh + tid];
    a.out[long(b) * a.ld_out + h * Dh + tid] = sum / z_run;
  }
}

// lx_spec_logprobs: one threadgroup per verify row, on its raw logits (before the penalties). The steps are sample's
// step 7 (sampler.metal), so a verify row reports the plain step's numbers at its position.
kernel void spec_logprobs(constant SpecLogprobsArgs& a [[buffer(0)]], uint r [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup SampleShared sh;
  const int V = a.V;
  const int token = a.tokens[r];
  device const float* l = a.logits + long(r) * V;
  float raw_max = -INFINITY;
  for (int e = tid; e < V; e += threads) raw_max = max(raw_max, l[e]);
  raw_max = group_max(raw_max, sh.keep.red, tid, threads);
  float raw_sum = 0.0f;
  for (int e = tid; e < V; e += threads) raw_sum += exp(l[e] - raw_max);
  const float lse = raw_max + log(group_sum(raw_sum, sh.keep.red, tid, threads));
  if (a.want_token && tid == 0) a.token_lp[r] = l[token] - lse;
  for (int j = 0; j < a.top_n; ++j) {
    float best = -INFINITY;
    int best_i = 0x7fffffff;
    for (int e = tid; e < V; e += threads) {
      bool taken = false;
      for (int c = 0; c < j; ++c) taken |= sh.chosen[c] == e;
      if (!taken && l[e] > best) {
        best = l[e];
        best_i = e;
      }
    }
    float bv;
    int bi;
    group_argmax(best, best_i, sh.keep.red, sh.redi, tid, threads, bv, bi);
    if (tid == 0) {
      sh.chosen[j] = bi;
      a.top_tokens[long(r) * a.top_n + j] = bi;
      a.top_logprobs[long(r) * a.top_n + j] = l[bi] - lse;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}
