// Mamba-2 mixer kernels: the causal conv that carries its state between steps, the selective scan
// (the SSD recurrence) with f32 state per slot, and the gated RMSNorm that follows the scan.
#include <cooperative_groups.h>
#include <cstdlib>

#include "cuda/common.cuh"

namespace {

// ---- Causal conv ----------------------------------------------------------------------------

constexpr int kConvMaxW = 8;  // widest conv the kernel takes: W - 1 <= 7 old rows per channel
constexpr int kConvCh = 32;   // channels per block: one warp's lanes

// One block owns 32 channels of one sequence and does both halves of the update. The outputs read the
// old state, and the update rewrites it, so the old rows of the block's channels are staged in shared
// memory first; the outputs and the new state then read the staged copy, and the state writes go to
// columns no other block touches. Each output depends on the W inputs ending at its own token, never on
// another output, so the tokens of a sequence are spread over the block's eight token lanes (threadIdx.y).
// Every product sums in the same order as the two kernels it replaces, so the bits do not change.
__global__ void conv1d_kernel(const float* __restrict__ x, int ld_x, const __nv_bfloat16* __restrict__ w,
                              const __nv_bfloat16* __restrict__ bias, float* __restrict__ state,
                              const int32_t* __restrict__ slot_ids, const int32_t* __restrict__ has_state,
                              const int32_t* __restrict__ cu, float* __restrict__ out, int C, int W) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ float old[kConvMaxW - 1][kConvCh];  // the old state rows of this block's channels (0 when none)
  const int tx = threadIdx.x, ty = threadIdx.y;
  const int s = blockIdx.y;
  const int ch0 = blockIdx.x * kConvCh;
  const int c = ch0 + tx;
  const bool live = c < C;  // no early return: every thread reaches the barrier
  const int start = cu[s], len = cu[s + 1] - start;
  const bool has = has_state[s] != 0;
  float* st = state + static_cast<int64_t>(slot_ids[s]) * (W - 1) * C;

  for (int i = ty * kConvCh + tx; i < (W - 1) * kConvCh; i += 8 * kConvCh) {
    const int r = i / kConvCh, cc = i % kConvCh, ch = ch0 + cc;
    old[r][cc] = (has && ch < C) ? st[static_cast<int64_t>(r) * C + ch] : 0.0f;
  }
  __syncthreads();

  if (live) {
    // Unrolled to a fixed bound so the taps stay in registers.
    float wk[kConvMaxW];
#pragma unroll
    for (int k = 0; k < kConvMaxW; ++k) wk[k] = k < W ? __bfloat162float(w[c * W + k]) : 0.0f;
    const float b = __bfloat162float(bias[c]);

    for (int t = ty; t < len; t += 8) {
      float acc = b;
#pragma unroll
      for (int k = 0; k < kConvMaxW; ++k) {
        if (k < W) {
          // Position relative to the sequence start; negative positions are the carried state.
          const int p = t - (W - 1) + k;
          float v = 0.0f;
          if (p >= 0) v = x[static_cast<int64_t>(start + p) * ld_x + c];
          else v = old[p + W - 1][tx];
          acc = fmaf(wk[k], v, acc);
        }
      }
      out[static_cast<int64_t>(start + t) * C + c] = lx::silu(acc);
    }

    // The new state is the last W - 1 inputs of [old state rows, x rows]. Row j takes combined input
    // len + j; a row below W - 1 is an old row, and every old row is in the staged copy.
    if (ty == 0) {
      for (int j = 0; j < W - 1; ++j) {
        const int m = len + j;
        float v = 0.0f;
        if (m >= W - 1) v = x[static_cast<int64_t>(start + m - (W - 1)) * ld_x + c];
        else v = old[m][tx];
        st[static_cast<int64_t>(j) * C + c] = v;
      }
    }
  }
}

// ---- Selective scan -------------------------------------------------------------------------

struct ScanArgs {
  const float* x;
  int ld_x;
  const float* dt_raw;
  int ld_dt;
  const float* B;
  const float* C;
  int ld_bc;
  const float* A_log;
  const float* D;
  const float* dt_bias;
  float* state;
  const int32_t* slot_ids;
  const int32_t* has_state;
  const int32_t* cu;
  float* y;
  int H, P, G, N;
};

// Tokens staged per chunk: one block barrier pair per chunk, and the next chunk's copies run behind
// this chunk's math.
constexpr int kChunk = 8;
// Floats per staged token at the widest shape (P <= 64, N <= 256): x[64] | B[256] | C[256] | dt | pad.
constexpr int kRowMax = 64 + 2 * 256 + 4;

__host__ __device__ constexpr int round4(int v) { return (v + 3) & ~3; }

__device__ __forceinline__ void cp_async_f32(float* smem, const float* gmem) {
  const uint32_t dst = static_cast<uint32_t>(__cvta_generic_to_shared(smem));
  asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n" ::"r"(dst), "l"(gmem) : "memory");
}
__device__ __forceinline__ void cp_async_commit() { asm volatile("cp.async.commit_group;\n" ::: "memory"); }
__device__ __forceinline__ void cp_async_wait_all() { asm volatile("cp.async.wait_group 0;\n" ::: "memory"); }

// One block per (head, sequence). Four lanes own each state row p. Lane q holds the chunks of four
// columns at 4 * (q + 4 i) for i < NI in registers for the whole sequence, so a row needs NI * 4
// state values per lane. The row's y is summed over its four lanes with two shuffles.
//
// Tokens go through shared memory in chunks of kChunk. B and C rows are padded to 16 * NI columns
// with zeros, which covers every lane's chunks without guards: the padding is zeroed once and copies
// never write it. The copies for chunk c + 1 are issued before chunk c is computed, so the memory
// latency is paid behind a whole chunk of math. The per-chunk barriers are the only block-wide
// synchronization.
template <int NI>
__global__ void __launch_bounds__(256) ssm_scan_kernel(const ScanArgs a) {
  lx::pdl_trigger();
  lx::pdl_wait();
  constexpr int BW = 16 * NI;  // B and C row width, padded so every lane's four-column load is in range
  __shared__ __align__(16) float stage[2][kChunk * kRowMax];
  __shared__ float dts[2][kChunk], decays[2][kChunk];
  float* const sm = &stage[0][0];

  const int P = a.P, N = a.N;
  const int BO = round4(P), CO = BO + BW, DO = CO + BW;
  const int SW = DO + 4;  // stride of one staged token, a multiple of 4
  const int bd = blockDim.x;
  const int tid = threadIdx.x, q = tid & 3, p = tid >> 2;
  // Lanes past the last row run the shuffles and read finite staging, but store nothing.
  const bool valid = p < P;
  const int h = blockIdx.x, s = blockIdx.y;
  const int g = h / (a.H / a.G);
  const int start = a.cu[s], len = a.cu[s + 1] - start;
  const int nchunks = (len + kChunk - 1) / kChunk;
  float* st = a.state + (static_cast<int64_t>(a.slot_ids[s]) * a.H + h) * P * N;
  const bool has = a.has_state[s] != 0;
  const float negA = -expf(a.A_log[h]);
  const float Dh = a.D[h], dt_bias = a.dt_bias[h];

  // Zero the staging once, padding included. The barrier keeps these stores from landing on a copy.
  for (int i = tid; i < 2 * kChunk * kRowMax; i += bd) sm[i] = 0.0f;
  __syncthreads();

  float S[NI][4];
#pragma unroll
  for (int i = 0; i < NI; ++i)
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      const int n = 4 * (q + 4 * i) + e;
      S[i][e] = (has && valid && n < N) ? st[p * N + n] : 0.0f;
    }

  // Copies chunk c's tokens into buffer buf, one 4-byte copy per element.
  auto issue = [&](int c, int buf) {
    const int base = c * kChunk;
    const int count = min(kChunk, len - base);
    float* dst = sm + buf * kChunk * kRowMax;
    for (int tok = 0; tok < count; ++tok) {
      const int t = start + base + tok;
      float* d = dst + tok * SW;
      const float* xr = a.x + static_cast<int64_t>(t) * a.ld_x + h * P;
      const float* br = a.B + static_cast<int64_t>(t) * a.ld_bc + g * N;
      const float* cr = a.C + static_cast<int64_t>(t) * a.ld_bc + g * N;
      for (int j = tid; j < P; j += bd) cp_async_f32(d + j, xr + j);
      for (int j = tid; j < N; j += bd) {
        cp_async_f32(d + BO + j, br + j);
        cp_async_f32(d + CO + j, cr + j);
      }
      if (tid == 0) cp_async_f32(d + DO, a.dt_raw + static_cast<int64_t>(t) * a.ld_dt + h);
    }
  };

  if (nchunks > 0) {
    issue(0, 0);
    cp_async_commit();
    cp_async_wait_all();
    __syncthreads();
  }

  for (int c = 0; c < nchunks; ++c) {
    if (c + 1 < nchunks) issue(c + 1, (c + 1) & 1);
    cp_async_commit();
    const float* sb = sm + (c & 1) * kChunk * kRowMax;
    const int count = min(kChunk, len - c * kChunk);
    // dt and the decay depend on the token and head only, not on the row, so they are formed once
    // per chunk rather than by every row.
    if (tid < count) {
      const float dt = lx::softplus(sb[tid * SW + DO] + dt_bias);
      dts[c & 1][tid] = dt;
      decays[c & 1][tid] = expf(dt * negA);
    }
    __syncthreads();
    for (int tok = 0; tok < count; ++tok) {
      const float* row = sb + tok * SW;
      const float xp = row[p];
      const float dtx = dts[c & 1][tok] * xp;
      const float decay = decays[c & 1][tok];
      float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
      for (int i = 0; i < NI; ++i) {
        const int c0 = 4 * (q + 4 * i);  // this lane's four columns within B and within C
        const float4 bb = *reinterpret_cast<const float4*>(row + BO + c0);
        const float4 cc = *reinterpret_cast<const float4*>(row + CO + c0);
        const float bv[4] = {bb.x, bb.y, bb.z, bb.w};
        const float cv[4] = {cc.x, cc.y, cc.z, cc.w};
#pragma unroll
        for (int e = 0; e < 4; ++e) {
          // Columns past N have zero B and C, so their state stays zero and adds nothing to y.
          S[i][e] = fmaf(decay, S[i][e], dtx * bv[e]);
          acc[e] = fmaf(S[i][e], cv[e], acc[e]);
        }
      }
      float y = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
      for (int o = 1; o < 4; o <<= 1) y += __shfl_xor_sync(0xffffffffu, y, o);
      if (valid && q == 0) a.y[(static_cast<int64_t>(start + c * kChunk + tok) * a.H + h) * P + p] = y + Dh * xp;
    }
    cp_async_wait_all();
    __syncthreads();
  }

  if (valid) {
#pragma unroll
    for (int i = 0; i < NI; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) {
        const int n = 4 * (q + 4 * i) + e;
        if (n < N) st[p * N + n] = S[i][e];
      }
  }
}

template <int NI>
int launch_scan(const ScanArgs& args, int n_seq, int threads, cudaStream_t s) {
  lx::launch_pdl(ssm_scan_kernel<NI>, dim3(args.H, n_seq), threads, 0, s, args);
  return lx::launch_status();
}

// ---- Gated RMSNorm --------------------------------------------------------------------------

// One block per (token, group): v = y * silu(z), normalized by the RMS over the group's K channels.
__global__ void gated_rmsnorm_kernel(const float* __restrict__ y, const float* __restrict__ z, int ld_z,
                                     const __nv_bfloat16* __restrict__ w, __nv_bfloat16* __restrict__ out,
                                     int I, int groups, float eps) {
  lx::pdl_trigger();
  lx::pdl_wait();
  __shared__ float scratch[32];
  const int K = I / groups;
  const int t = blockIdx.x / groups, g = blockIdx.x % groups;
  const int base = g * K;
  const float* yr = y + static_cast<int64_t>(t) * I + base;
  const float* zr = z + static_cast<int64_t>(t) * ld_z + base;
  float sum = 0.0f;
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    const float v = yr[i] * lx::silu(zr[i]);
    sum += v * v;
  }
  const float scale = rsqrtf(lx::block_sum(sum, scratch) / K + eps);
  // The gate is recomputed rather than held across the reduction: K is only a few hundred.
  for (int i = threadIdx.x; i < K; i += blockDim.x) {
    const float v = yr[i] * lx::silu(zr[i]);
    out[static_cast<int64_t>(t) * I + base + i] = lx::from_f32<__nv_bfloat16>(v * scale * __bfloat162float(w[base + i]));
  }
}

// ---- Fused decode step ----------------------------------------------------------------------

// One token per sequence, the conv, the scan and the gated norm in one kernel. Block (h, s) is head h of
// sequence s: it convolves its x channels and its group's B and C channels (every head of a group reads
// the same B and C, and each head's taps are the same arithmetic as lx_conv1d_varlen's), scans its own
// [P, N] state (the same lanes and summation order as ssm_scan_kernel, so the bits are the same), and
// writes its y. Only a group's last block to arrive finishes it: that block has every head's y, so it
// computes the group's gated norm (the same reduction as gated_rmsnorm_kernel, over the same 256
// threads), and it moves the group's B and C conv state on, since every head has read the old one by
// then. The arrival counter is the only cross-block ordering; the sums run in a fixed order, so a row's
// bits never depend on which block arrives last or on the other rows.
constexpr int kDecodeThreads = 256;  // four lanes per state row, a row per channel of the head (P <= 64)
constexpr int kDecodeMaxP = 64;
constexpr int kDecodeMaxN = 256;

struct DecodeArgs {
  const float* zxbcdt;
  int ld;
  const __nv_bfloat16* conv_w;
  const __nv_bfloat16* conv_b;
  float* conv_state;
  float* ssm_state;
  const float* A_log;
  const float* D;
  const float* dt_bias;
  const int32_t* slot_ids;
  const int32_t* has_state;
  const __nv_bfloat16* norm_w;
  float eps;
  float* y;
  __nv_bfloat16* out;
  int32_t* sync;
  int H, P, G, N, W;
};

// One chunk of a block's new state: four columns from one thread, as one float4 when the row is whole float4s.
__device__ __forceinline__ void store_state_chunk(float* st, int N, bool vec, int n0, const float* v) {
  if (vec) {
    if (n0 < N) __stcg(reinterpret_cast<float4*>(st + n0), make_float4(v[0], v[1], v[2], v[3]));
  } else {
#pragma unroll
    for (int e = 0; e < 4; ++e)
      if (n0 + e < N) __stcg(st + n0 + e, v[e]);
  }
}

// One body for both launches. kCluster = false: the head blocks of a group count arrivals on a global counter and the
// last one to arrive finishes the group. kCluster = true (sm_90 and up): the group's head blocks are one cluster, and
// they share y through distributed shared memory, so there is no counter, no fence and no read-back from L2.
template <int NI, bool kCluster>
__device__ __forceinline__ void mamba_decode_body(const DecodeArgs& a) {
  lx::pdl_trigger();
  __shared__ float xs[kDecodeMaxP];                  // the head's x conv outputs
  __shared__ __align__(16) float Bs[kDecodeMaxN];    // the group's B conv outputs, zero past N
  __shared__ __align__(16) float Cs[kDecodeMaxN];    // the group's C conv outputs, zero past N
  __shared__ float nst[7][kDecodeMaxP];             // the head's x channels' new conv state rows
  __shared__ float red[32];
  __shared__ int last;
  __shared__ float ysh[kDecodeMaxP];  // the head's y, read by the cluster's blocks (kCluster only)

  const int tid = threadIdx.x, q = tid & 3, p = tid >> 2;
  const int h = blockIdx.x, s = blockIdx.y;
  const int H = a.H, P = a.P, G = a.G, N = a.N, W = a.W;
  const int I = H * P, GN = G * N, conv_dim = I + 2 * GN;
  const int g = h / (H / G);
  const int slot = a.slot_ids[s];
  const bool has = a.has_state[s] != 0;
  const bool valid = p < P;
  const bool vec = (N & 3) == 0;  // whole float4 chunks: a chunk past N is wholly past N
  const float* zrow = a.zxbcdt + static_cast<int64_t>(s) * a.ld;
  float* cst = a.conv_state + static_cast<int64_t>(slot) * (W - 1) * conv_dim;
  float* st = a.ssm_state + ((static_cast<int64_t>(slot) * H + h) * P + (valid ? p : 0)) * N;
  // The head's weights come first too, with the state, so none of their loads waits behind the conv.
  const float dt_bias = a.dt_bias[h];
  const float A_log = a.A_log[h];
  const float Dh = a.D[h];

  // The state comes first, so its loads are in flight behind the conv. Lane q holds the chunks of four
  // columns at 4 * (q + 4 i) for i < NI, as in ssm_scan_kernel.
  float S[NI][4];
  const bool load = valid && has;
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int n0 = 4 * (q + 4 * i);
    if (vec) {
      const float4 v = (load && n0 < N) ? __ldcg(reinterpret_cast<const float4*>(st + n0)) : make_float4(0, 0, 0, 0);
      S[i][0] = v.x;
      S[i][1] = v.y;
      S[i][2] = v.z;
      S[i][3] = v.w;
    } else {
#pragma unroll
      for (int e = 0; e < 4; ++e) S[i][e] = (load && n0 + e < N) ? __ldcg(st + n0 + e) : 0.0f;
    }
  }

  // The conv's weights and the norm's weights are static (never written by any kernel), and their addresses come
  // from the head's indices, so the first two conv iterations' taps and the norm's first two elements load with the
  // state, before the wait. Two sets of registers, not more, so two blocks still fit an SM (see the kernel's use).
  const int pK = I / G, pbase = g * pK, prank = h - g * (H / G);
  float cw[2][8], cb[2], nw[2];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int j = tid + r * kDecodeThreads;
    const bool ok = j < P + 2 * N;
    const int c = j < P ? h * P + j : (j < P + N ? I + g * N + (j - P) : I + GN + g * N + (j - P - N));
    cb[r] = ok ? __bfloat162float(a.conv_b[c]) : 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) cw[r][k] = (ok && k < W) ? __bfloat162float(a.conv_w[c * W + k]) : 0.0f;
    const int ni = kCluster ? prank * P + tid : tid + r * kDecodeThreads;
    const bool nok = kCluster ? (r == 0 && tid < P) : ni < pK;
    nw[r] = nok ? __bfloat162float(a.norm_w[pbase + ni]) : 0.0f;
  }

  // The state and the weights are not written by the predecessor kernel (only by an earlier step's scan), so their
  // loads above went out before the dependency wait. Everything that reads zxbcdt waits for it.
  lx::pdl_wait();
  const float dt_raw = zrow[I + conv_dim + h];

  // The conv of the head's x channels [h P, h P + P), then the group's B and C channels, in conv order
  // (x | B | C). Each channel's taps are read once. The loop only loads and writes shared memory, so the
  // loads of its next iterations can go out early; the x channels' new state rows wait in shared memory
  // for the barrier, and B and C wait for the last block of the group.
#pragma unroll
  for (int r = 0; r < 3; ++r) {
    const int j = tid + r * kDecodeThreads;
    if (j < P + 2 * N) {
      const int c = j < P ? h * P + j : (j < P + N ? I + g * N + (j - P) : I + GN + g * N + (j - P - N));
      const float xn = zrow[I + c];
      float old[8];
#pragma unroll
      for (int k = 0; k < 8; ++k) old[k] = (k < W - 1 && has) ? __ldcg(cst + static_cast<int64_t>(k) * conv_dim + c) : 0.0f;
      float acc = r < 2 ? cb[r < 2 ? r : 0] : __bfloat162float(a.conv_b[c]);
#pragma unroll
      for (int k = 0; k < 8; ++k)
        if (k < W) {
          const float wk = r < 2 ? cw[r < 2 ? r : 0][k] : __bfloat162float(a.conv_w[c * W + k]);
          acc = fmaf(wk, k < W - 1 ? old[k] : xn, acc);
        }
      const float out = lx::silu(acc);
      if (j < P) {
        xs[j] = out;
        if (W >= 2) {
#pragma unroll
          for (int k = 0; k < 7; ++k)
            if (k < W - 1) nst[k][j] = k < W - 2 ? old[k + 1] : xn;
        }
      } else if (j < P + N) {
        Bs[j - P] = out;
      } else {
        Cs[j - P - N] = out;
      }
    }
  }
  // Padding past N is zero, as in the scan: a chunk's lanes read whole float4s.
  for (int i = N + tid; i < 16 * NI; i += kDecodeThreads) {
    Bs[i] = 0.0f;
    Cs[i] = 0.0f;
  }
  __syncthreads();
  // The x channels' new state, each row written by one thread (nothing else touches these channels).
  if (W >= 2)
    for (int i = tid; i < (W - 1) * P; i += kDecodeThreads)
      cst[static_cast<int64_t>(i / P) * conv_dim + h * P + i % P] = nst[i / P][i % P];

  const float dt = lx::softplus(dt_raw + dt_bias);
  const float negA = -expf(A_log);
  const float decay = expf(dt * negA);
  const float xp = valid ? xs[p] : 0.0f;
  const float dtx = dt * xp;
  float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
  for (int i = 0; i < NI; ++i) {
    const int c0 = 4 * (q + 4 * i);
    const float4 bb = *reinterpret_cast<const float4*>(Bs + c0);
    const float4 cc = *reinterpret_cast<const float4*>(Cs + c0);
    const float bv[4] = {bb.x, bb.y, bb.z, bb.w};
    const float cv[4] = {cc.x, cc.y, cc.z, cc.w};
#pragma unroll
    for (int e = 0; e < 4; ++e) {
      S[i][e] = fmaf(decay, S[i][e], dtx * bv[e]);
      acc[e] = fmaf(S[i][e], cv[e], acc[e]);
    }
    // This chunk's new state goes out as soon as it is computed, so its stores drain while the later chunks' state
    // loads are still in flight. The stores are the same values in the same places, whatever order they go out in.
    // (The cluster's last chunk goes out behind its barrier arrival instead, below.)
    if (valid && !(kCluster && i == NI - 1)) store_state_chunk(st, N, vec, 4 * (q + 4 * i), S[i]);
  }
  float y = (acc[0] + acc[1]) + (acc[2] + acc[3]);
#pragma unroll
  for (int o = 1; o < 4; o <<= 1) y += __shfl_xor_sync(0xffffffffu, y, o);
  const float yo = y + Dh * xp;
  if (valid && q == 0) {
    a.y[static_cast<int64_t>(s) * I + h * P + p] = yo;
    if (kCluster) ysh[p] = yo;
  }

  // The group's B and C conv state, read now, behind the arrival: only the last block moves it on, and nothing writes
  // it before that block does, so every block may read it. The last block then has it in registers, a round trip
  // sooner. A block that is not last drops it. Two elements per thread cover 2N up to 2 * kDecodeThreads.
  float bc_xn[2] = {0.0f, 0.0f};
  float bc_old[2][8];
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int j = tid + r * kDecodeThreads;
    if (j < 2 * N) {
      const int c = I + (j < N ? g * N + j : GN + g * N + (j - N));
      bc_xn[r] = zrow[I + c];
#pragma unroll
      for (int k = 0; k < 8; ++k)
        bc_old[r][k] = (k < W - 1 && has) ? __ldcg(cst + static_cast<int64_t>(k) * conv_dim + c) : 0.0f;
    }
  }

  // The cluster path: the group's head blocks are one cluster, consecutive in x, so the rank is the head's place in its
  // group. Each block reads the group's B and C state before the barrier, and rank 0 moves it on after it, so no block
  // writes state another still reads. The norm sums the group's K channels in the order the last block did (thread t
  // takes channels t, t + 256, ...), reading each channel's y from the head block that owns it, so the bits are the same.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  if constexpr (kCluster) {
    const cooperative_groups::cluster_group cluster = cooperative_groups::this_cluster();
    const int hpg = H / G, rank = h - g * hpg;
    // The arrival orders this block's y and B and C reads before the barrier; the last chunk of state goes out behind it,
    // so the barrier waits only for the stores that were issued before the last chunk.
    __cluster_barrier_arrive();
    if (valid) store_state_chunk(st, N, vec, 4 * (q + 4 * (NI - 1)), S[NI - 1]);
    __cluster_barrier_wait();
    if (rank == 0 && W >= 2) {
#pragma unroll
      for (int r = 0; r < 2; ++r) {
        const int j = tid + r * kDecodeThreads;
        if (j < 2 * N) {
          const int c = I + (j < N ? g * N + j : GN + g * N + (j - N));
#pragma unroll
          for (int k = 0; k < 6; ++k)
            if (k < W - 2) cst[static_cast<int64_t>(k) * conv_dim + c] = bc_old[r][k + 1];
          cst[static_cast<int64_t>(W - 2) * conv_dim + c] = bc_xn[r];
        }
      }
    }
    const int K = hpg * P, base = g * K;
    const float* zr = zrow + base;
    float sum = 0.0f;
    for (int i = tid; i < K; i += kDecodeThreads) {
      const float v = cluster.map_shared_rank(ysh, i / P)[i % P] * lx::silu(zr[i]);
      sum += v * v;
    }
    const float scale = rsqrtf(lx::block_sum(sum, red) / K + a.eps);
    for (int j = tid; j < P; j += kDecodeThreads) {
      const int i = rank * P + j;
      const float v = ysh[j] * lx::silu(zr[i]);
      a.out[static_cast<int64_t>(s) * I + base + i] = lx::from_f32<__nv_bfloat16>(v * scale * nw[0]);
    }
    cluster.sync();  // no block leaves while another still reads its shared memory
    return;
  }
#endif

  // Arrival: the group's last block to finish takes over. Every thread's writes are fenced before the count.
  __threadfence();
  __syncthreads();
  if (tid == 0) last = atomicAdd(a.sync + static_cast<int64_t>(s) * G + g, 1) == H / G - 1;
  __syncthreads();
  if (!last) return;
  __threadfence();

  // The group's B and C conv state moves on: every head of the group has read the old state.
#pragma unroll
  for (int r = 0; r < 2; ++r) {
    const int j = tid + r * kDecodeThreads;
    if (j < 2 * N && W >= 2) {
      const int c = I + (j < N ? g * N + j : GN + g * N + (j - N));
#pragma unroll
      for (int k = 0; k < 6; ++k)
        if (k < W - 2) cst[static_cast<int64_t>(k) * conv_dim + c] = bc_old[r][k + 1];
      cst[static_cast<int64_t>(W - 2) * conv_dim + c] = bc_xn[r];
    }
  }

  // The group's gated norm over its K channels (the head blocks' y, read from L2 past their writes).
  const int K = I / G, base = g * K;
  const float* yr = a.y + static_cast<int64_t>(s) * I + base;
  const float* zr = zrow + base;
  float sum = 0.0f;
  for (int i = tid; i < K; i += kDecodeThreads) {
    const float v = __ldcg(yr + i) * lx::silu(zr[i]);
    sum += v * v;
  }
  const float scale = rsqrtf(lx::block_sum(sum, red) / K + a.eps);
  for (int i = tid; i < K; i += kDecodeThreads) {
    const float v = __ldcg(yr + i) * lx::silu(zr[i]);
    const float nwv = i == tid ? nw[0] : (i == tid + kDecodeThreads ? nw[1] : __bfloat162float(a.norm_w[base + i]));
    a.out[static_cast<int64_t>(s) * I + base + i] = lx::from_f32<__nv_bfloat16>(v * scale * nwv);
  }
  if (tid == 0) a.sync[static_cast<int64_t>(s) * G + g] = 0;  // the next step starts from zero counters
}

template <int NI>
__global__ void __launch_bounds__(kDecodeThreads) mamba_decode_kernel(const DecodeArgs a) {
  mamba_decode_body<NI, false>(a);
}

template <int NI>
__global__ void __launch_bounds__(kDecodeThreads) mamba_decode_cluster_kernel(const DecodeArgs a) {
  mamba_decode_body<NI, true>(a);
}

// Whether the group norm runs on clusters: the device launches them (sm_90 and up). LX_MAMBA_ATOMIC=1 takes the counter
// path on any device, so the suite can check both.
inline bool cluster_on() {
  static const bool forced_atomic = std::getenv("LX_MAMBA_ATOMIC") != nullptr;
  if (forced_atomic) return false;
  int dev = 0, v = 0;
  if (cudaGetDevice(&dev) != cudaSuccess || cudaDeviceGetAttribute(&v, cudaDevAttrClusterLaunch, dev) != cudaSuccess)
    return false;
  return v != 0;
}

// One cluster per group: its hpg head blocks, consecutive in x. A cluster is at most 8 blocks, the portable size.
template <int NI>
int launch_decode_cluster(const DecodeArgs& args, int n_seq, int H, int hpg, cudaStream_t s) {
  cudaLaunchConfig_t cfg = {};
  cfg.gridDim = dim3(H, n_seq);
  cfg.blockDim = dim3(kDecodeThreads);
  cfg.dynamicSmemBytes = 0;
  cfg.stream = s;
  cudaLaunchAttribute attrs[2] = {};
  int na = 0;
  attrs[na].id = cudaLaunchAttributeClusterDimension;
  attrs[na].val.clusterDim.x = hpg;
  attrs[na].val.clusterDim.y = 1;
  attrs[na].val.clusterDim.z = 1;
  ++na;
  if (lx::pdl_new_on()) {
    attrs[na].id = cudaLaunchAttributeProgrammaticStreamSerialization;
    attrs[na].val.programmaticStreamSerializationAllowed = 1;
    ++na;
  }
  cfg.attrs = attrs;
  cfg.numAttrs = na;
  cudaLaunchKernelEx(&cfg, mamba_decode_cluster_kernel<NI>, args);
  return lx::launch_status();
}

template <int NI>
int launch_decode(const DecodeArgs& args, int n_seq, int H, cudaStream_t s) {
  const int hpg = args.H / args.G;  // the head blocks of one group
  if (hpg <= 8 && cluster_on()) return launch_decode_cluster<NI>(args, n_seq, H, hpg, s);
  lx::launch_pdl(mamba_decode_kernel<NI>, dim3(H, n_seq), kDecodeThreads, 0, s, args);
  return lx::launch_status();
}

}  // namespace

extern "C" int lx_conv1d_varlen(const float* x, int ld_x, const void* w, const void* bias, float* conv_state,
                                const int32_t* slot_ids, const int32_t* has_state, const int32_t* cu_seqlens,
                                int n_seq, float* out, int C, int W, void* stream) {
  if (n_seq <= 0) return 0;
  if (C <= 0 || W < 1 || W > 8 || ld_x < C || n_seq > 65535) return static_cast<int>(cudaErrorInvalidValue);
  auto s = static_cast<cudaStream_t>(stream);
  lx::launch_pdl(conv1d_kernel, dim3((C + kConvCh - 1) / kConvCh, n_seq), dim3(kConvCh, 8), 0, s, x, ld_x, static_cast<const __nv_bfloat16*>(w), static_cast<const __nv_bfloat16*>(bias), conv_state, slot_ids,
      has_state, cu_seqlens, out, C, W);
  return lx::launch_status();
}

extern "C" int lx_ssm_scan_varlen(const float* x, int ld_x, const float* dt_raw, int ld_dt, const float* B,
                                  const float* C, int ld_bc, const float* A_log, const float* D,
                                  const float* dt_bias, float* ssm_state, const int32_t* slot_ids,
                                  const int32_t* has_state, const int32_t* cu_seqlens, int n_seq, float* y,
                                  int H, int P, int G, int N, void* stream) {
  if (n_seq <= 0) return 0;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || P > 64 || N < 1 || N > 256 || n_seq > 65535)
    return static_cast<int>(cudaErrorInvalidValue);
  if (ld_x < H * P || ld_dt < H || ld_bc < G * N) return static_cast<int>(cudaErrorInvalidValue);

  // Four lanes per row, rounded up to a whole warp; rows past P are idle.
  const int threads = (4 * P + 31) / 32 * 32;
  // N/4 column chunks, dealt to the four lanes of a row: NI chunks each (up to 16 for N = 256).
  const int ni = ((N + 3) / 4 + 3) / 4;
  const ScanArgs args{x,      ld_x,     dt_raw,  ld_dt,  B,        C,     ld_bc,   A_log,
                      D,      dt_bias,  ssm_state, slot_ids, has_state, cu_seqlens, y, H, P, G, N};
  auto s = static_cast<cudaStream_t>(stream);
  if (ni <= 1) return launch_scan<1>(args, n_seq, threads, s);
  if (ni <= 2) return launch_scan<2>(args, n_seq, threads, s);
  if (ni <= 4) return launch_scan<4>(args, n_seq, threads, s);
  if (ni <= 8) return launch_scan<8>(args, n_seq, threads, s);
  return launch_scan<16>(args, n_seq, threads, s);
}

extern "C" int lx_gated_rmsnorm(const float* y, const float* z, int ld_z, const void* w, void* out, int T, int I,
                                int groups, float eps, void* stream) {
  if (T <= 0) return 0;
  if (I <= 0 || groups <= 0 || I % groups != 0 || ld_z < I) return static_cast<int>(cudaErrorInvalidValue);
  const int K = I / groups;
  const int threads = K >= 256 ? 256 : ((K + 31) / 32) * 32;
  auto s = static_cast<cudaStream_t>(stream);
  lx::launch_pdl(gated_rmsnorm_kernel, T * groups, threads, 0, s, y, z, ld_z, static_cast<const __nv_bfloat16*>(w),
                                                      static_cast<__nv_bfloat16*>(out), I, groups, eps);
  return lx::launch_status();
}

extern "C" int lx_mamba_decode_step(const float* zxbcdt, int ld, const void* conv_w, const void* conv_b, float* conv_state,
                                    float* ssm_state, const float* A_log, const float* D, const float* dt_bias,
                                    const int32_t* slot_ids, const int32_t* has_state, const void* norm_w, float eps,
                                    float* y, void* out, int32_t* sync, int n_seq, int H, int P, int G, int N, int W,
                                    void* stream) {
  if (n_seq <= 0) return 0;
  if (H <= 0 || G <= 0 || H % G != 0 || P < 1 || P > kDecodeMaxP || N < 1 || N > kDecodeMaxN || W < 1 || W > 8 ||
      n_seq > 65535)
    return static_cast<int>(cudaErrorInvalidValue);
  const int I = H * P;
  if (ld < 2 * I + 2 * G * N + H) return static_cast<int>(cudaErrorInvalidValue);
  const DecodeArgs args{zxbcdt,
                        ld,
                        static_cast<const __nv_bfloat16*>(conv_w),
                        static_cast<const __nv_bfloat16*>(conv_b),
                        conv_state,
                        ssm_state,
                        A_log,
                        D,
                        dt_bias,
                        slot_ids,
                        has_state,
                        static_cast<const __nv_bfloat16*>(norm_w),
                        eps,
                        y,
                        static_cast<__nv_bfloat16*>(out),
                        sync,
                        H,
                        P,
                        G,
                        N,
                        W};
  auto s = static_cast<cudaStream_t>(stream);
  // The same chunk counts as lx_ssm_scan_varlen, so every shape runs the scan's lanes.
  const int ni = ((N + 3) / 4 + 3) / 4;
  if (ni <= 1) return launch_decode<1>(args, n_seq, H, s);
  if (ni <= 2) return launch_decode<2>(args, n_seq, H, s);
  if (ni <= 4) return launch_decode<4>(args, n_seq, H, s);
  if (ni <= 8) return launch_decode<8>(args, n_seq, H, s);
  return launch_decode<16>(args, n_seq, H, s);
}
