// The tile GEMM on simdgroup matrices, for prefill and batches: y = alpha * (x @ W^T + bias) + beta * y
// with W [N, K] in any weight format and x in any activation format (kernels.h lx_gemm_w, lx_gemm_bf16,
// lx_gemm_f32), and the two stages of the grouped experts (moe.metal). GPUs with neural accelerators run
// nax/gemm.metal instead, except for exact f32.
//
// A threadgroup of four simdgroups computes a 64 x 64 tile of y, each simdgroup a 32 x 32 quarter as
// 4 x 4 accumulators of 8 x 8. Per 32-wide step of K, every thread stages 16 values of one activation
// row and 16 of one weight row in threadgroup memory as bf16 (copied where they are bf16, else their
// exact decoded values: a quantized code times its block scale has at most 6 significant bits, which
// bf16 holds), and the simdgroups multiply bf16 into f32 accumulators. Row factors (the weight's row
// scale or global, the activation's per-token scale) are applied once in the epilogue. Products are
// exact, so only the f32 summation order differs from a float64 reference. The next step's values are
// loaded into registers while the simdgroups multiply. Exact f32 (lx_gemm_f32) has a kernel of its own
// at the end, gemm_f32x.

template <int AF> struct GemmStage {
  using S = bfloat;
  static constexpr constant int KS = kTileK + 8;  // padded row: 80 bytes
};

// One thread's share of a K step in registers, in the staged type.
template <int AF, int WF, typename S> struct GemmRegs;
template <int AF, int WF> struct GemmRegs<AF, WF, bfloat> {
  uint4 a[2], b[2];
  void load(const device uchar* arow, const device uchar* ablk, const device uchar* brow, const device uchar* bscl,
            int k, int K, bool xv, bool wv, bool a_valid, bool b_valid) thread {
    stage_act<AF, false>(arow, ablk, k, K, xv, a_valid, a);
    stage_w<WF, false>(brow, bscl, k, K, wv, b_valid, b);
  }
  void store(threadgroup bfloat* ad, threadgroup bfloat* bd) const thread {
    stage_store(ad, a);
    stage_store(bd, b);
  }
};

// MODE 0: a plain GEMM, grid (N / 64, M / 64).
// MODE 1: grouped gate_up. Grid (I / 32, tiles). The tile's 64 weight rows interleave gate row c and up
//         row I + c for 32 intermediate columns c, so a lane holds a column's gate and up side by side
//         and writes inter[pair, c] = bf16(silu(gate) * up). N is I; the stack has 2I rows per expert.
// MODE 2: grouped down. Grid (L / 64, tiles). Writes y[pair, l] = pair_w[pair] * (inter . down[l]), f32.
template <int AF, int WF, int MODE>
kernel void gemm(constant GemmArgs& a [[buffer(0)]], threadgroup uchar* smem [[threadgroup(0)]],
                 uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                 uint sgu [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  using S = typename GemmStage<AF>::S;
  constexpr int KS = GemmStage<AF>::KS;
  threadgroup S* As = (threadgroup S*)smem;
  threadgroup S* Bs = As + kTileM * KS;
  const int K = a.K, N = a.N;

  // The tile's rows: a block of M, or one expert's run of pairs.
  int m_base = 0, rows = 0, start = 0;
  const device uchar* wdata = a.w.data;
  const device uchar* wscale = a.w.scale;
  float wglobal = a.w.global;
  if (MODE == 0) {
    m_base = int(tg.y) * kTileM;
    rows = min(kTileM, a.M - m_base);
  } else {
    if (int(tg.y) >= a.tiles[0]) return;  // the whole threadgroup leaves together
    const int e = a.tiles[1 + 3 * tg.y];
    start = a.tiles[2 + 3 * tg.y];
    rows = a.tiles[3 + 3 * tg.y];
    expert_slice<WF>(a.w, e, a.stack_n, K, wdata, wscale, wglobal);
  }

  // This thread's load rows: activation row ar and weight row br, 16 values from column ac.
  const int ar = int(tid) >> 1, ac = (int(tid) & 1) * 16;
  const bool a_valid = ar < rows;
  int src = 0;
  if (MODE == 0) src = min(m_base + ar, a.M - 1);
  else {
    const int pair = a.order[start + min(ar, rows - 1)];
    src = MODE == 1 ? pair / a.top_k : pair;
  }
  const device uchar* arow = a.x + (long(src) * a.ld_x << act_bytes_shift<AF>());
  const device uchar* ablk = AF == A_FP4 ? a.x_block + long(src) * a.ld_xb : a.x_block;

  int wrow = 0;
  bool b_valid = false;
  if (MODE == 1) {
    const int c = int(tg.x) * (kTileN / 2) + (ar >> 1);
    b_valid = c < N;
    wrow = (ar & 1) ? N + min(c, N - 1) : min(c, N - 1);
  } else {
    const int n = int(tg.x) * kTileN + ar;
    b_valid = n < N;
    wrow = min(n, N - 1);
  }
  const device uchar* brow = wdata + row_offset<WF>(wrow, K);
  const device uchar* bscl = WF == W_NVFP4 ? wscale + long(wrow) * (K / 16) : wscale;
  const bool w_vec = WF == W_NVFP4 ? ((ulong(wdata) & 7ul) == 0 && (K / 2) % 8 == 0)
                                   : ((ulong(wdata) & 15ul) == 0 && (row_offset<WF>(1, K) & 15) == 0);

  const int sm = int(sgu) >> 1, sn = int(sgu) & 1;
  simdgroup_float8x8 acc[4][4];
  for (int i = 0; i < 4; ++i)
    for (int j = 0; j < 4; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

  const bool xv = a.x_vec != 0;
  const int steps = (K + kTileK - 1) / kTileK;
  GemmRegs<AF, WF, S> r;
  r.load(arow, ablk, brow, bscl, ac, K, kTileK <= K && xv, kTileK <= K && w_vec, a_valid, b_valid);
  r.store(As + ar * KS + ac, Bs + ar * KS + ac);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = 0; s < steps; ++s) {
    const bool more = s + 1 < steps;
    if (more) {
      const int k0 = (s + 1) * kTileK;
      const bool full = k0 + kTileK <= K;
      r.load(arow, ablk, brow, bscl, k0 + ac, K, full && xv, full && w_vec, a_valid, b_valid);
    }
    for (int kk = 0; kk < kTileK; kk += 8) {
      simdgroup_matrix<S, 8, 8> af[4], bt[4];
      for (int i = 0; i < 4; ++i) simdgroup_load(af[i], As + (sm * 32 + i * 8) * KS + kk, KS);
      for (int j = 0; j < 4; ++j) simdgroup_load(bt[j], Bs + (sn * 32 + j * 8) * KS + kk, KS, ulong2(0, 0), true);
      for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bt[j], acc[i][j]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (more) {
      r.store(As + ar * KS + ac, Bs + ar * KS + ac);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }

  // Epilogue: lane's elements of each 8 x 8 are (fm, fn) and (fm, fn + 1).
  const int qid = int(lane) >> 2;
  const int fm = (qid & 4) + ((int(lane) >> 1) & 3);
  const int fn = (qid & 2) * 2 + (int(lane) & 1) * 2;
  for (int i = 0; i < 4; ++i) {
    const int r = sm * 32 + i * 8 + fm;
    if (r >= rows) continue;
    int pair = 0, xr = 0;
    if (MODE == 0) xr = m_base + r;
    else {
      pair = a.order[start + r];
      xr = MODE == 1 ? pair / a.top_k : pair;
    }
    const float fa = (AF == A_INT8 || AF == A_E4M3 || AF == A_FP4) ? a.x_scale[xr] : 1.0f;
    for (int j = 0; j < 4; ++j) {
      const float2 e = float2(acc[i][j].thread_elements()[0], acc[i][j].thread_elements()[1]);
      const int c = sn * 32 + j * 8 + fn;
      if (MODE == 1) {
        const int col = int(tg.x) * (kTileN / 2) + (c >> 1);
        if (col >= N) continue;
        const float g = e.x * fa * row_factor<WF>(wscale, col, wglobal);
        const float u = e.y * fa * row_factor<WF>(wscale, N + col, wglobal);
        ((device ushort*)a.y)[long(pair) * N + col] = to_bf(silu(g) * u);
      } else {
        for (int h = 0; h < 2; ++h) {
          const int n = int(tg.x) * kTileN + c + h;
          if (n >= N) continue;
          float val = (h ? e.y : e.x) * fa * row_factor<WF>(wscale, n, wglobal);
          if (MODE == 2) {
            ((device float*)a.y)[long(pair) * N + n] = a.pair_w[pair] * val;
          } else {
            if (a.bias) val += load_val(a.bias, n, a.bias_bf16);
            val *= a.alpha;
            const long at = long(xr) * a.ld_y + n;
            if (a.beta != 0.0f) val += a.beta * load_val(a.y, at, a.y_bf16);
            store_val(a.y, at, val, a.y_bf16);
          }
        }
      }
    }
  }
}

#define LX_GEMM(AF, WF, MODE)                                                                                   \
  template [[host_name("gemm_" #AF #WF "_" #MODE)]] kernel void gemm<AF, WF, MODE>(constant GemmArgs&,          \
                                                                                    threadgroup uchar*, uint2,  \
                                                                                    uint, uint, uint);
// Plain: BF16 math on every format from bf16 or f32 activations, and the quantized maths.
LX_GEMM(0, 0, 0) LX_GEMM(0, 1, 0) LX_GEMM(0, 2, 0) LX_GEMM(0, 3, 0)
LX_GEMM(1, 0, 0) LX_GEMM(1, 1, 0) LX_GEMM(1, 2, 0) LX_GEMM(1, 3, 0)
LX_GEMM(2, 1, 0) LX_GEMM(3, 2, 0) LX_GEMM(3, 3, 0) LX_GEMM(4, 3, 0)
// Grouped experts, both stages.
LX_GEMM(0, 0, 1) LX_GEMM(0, 1, 1) LX_GEMM(0, 2, 1) LX_GEMM(0, 3, 1)
LX_GEMM(2, 1, 1) LX_GEMM(3, 2, 1) LX_GEMM(3, 3, 1) LX_GEMM(4, 3, 1)
LX_GEMM(0, 0, 2) LX_GEMM(0, 1, 2) LX_GEMM(0, 2, 2) LX_GEMM(0, 3, 2)
LX_GEMM(2, 1, 2) LX_GEMM(3, 2, 2) LX_GEMM(3, 3, 2) LX_GEMM(4, 3, 2)

// ---- Exact f32 (lx_gemm_f32) ------------------------------------------------------------------
// The routers and the depth head: f32 weights and a narrow output (a few hundred columns), where 64 x 64
// tiles would leave most of the GPU idle. A threadgroup of four simdgroups computes a 32 x 32 tile of y,
// each simdgroup a 16 x 16 quarter as 2 x 2 accumulators (with f32 operands, four times that many
// accumulators per simdgroup runs several times slower on the M5). Per 32-wide step of K every thread
// stages 8 values of one activation row and 8 of one weight row; the next step's are loaded into
// registers while the simdgroups multiply. Grid (N / 32, M / 32), 128 threads.
constant constexpr int kF32Tile = 32, kF32Step = 32, kF32Pitch = kF32Step + 4;

// 8 f32 values [k, k + 8) of a row, zero past K and for rows past the end.
inline void load8_f32(const device float* row, int k, int K, bool vec, bool valid, thread float4& lo, thread float4& hi) {
  if (valid && vec && k + 8 <= K) {
    lo = ((const device float4*)(row + k))[0];
    hi = ((const device float4*)(row + k))[1];
    return;
  }
  float v[8];
  for (int i = 0; i < 8; ++i) v[i] = valid && k + i < K ? row[k + i] : 0.0f;
  lo = float4(v[0], v[1], v[2], v[3]);
  hi = float4(v[4], v[5], v[6], v[7]);
}

kernel void gemm_f32x(constant GemmArgs& a [[buffer(0)]], threadgroup float* smem [[threadgroup(0)]],
                      uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]]) {
  const int K = a.K, N = a.N, M = a.M;
  const int m0 = int(tg.y) * kF32Tile, n0 = int(tg.x) * kF32Tile;
  threadgroup float* As = smem;
  threadgroup float* Bs = smem + kF32Tile * kF32Pitch;
  // Thread t stages row t / 4 of the activation and weight blocks, 8 values from column (t % 4) * 8.
  const int lr = int(tid) >> 2, lc = (int(tid) & 3) * 8;
  const int am = m0 + lr, bn = n0 + lr;
  const device float* xr = (const device float*)a.x + long(min(am, M - 1)) * a.ld_x;
  const device float* wr = (const device float*)a.w.data + long(min(bn, N - 1)) * K;
  const bool xv = a.x_vec != 0;
  const bool wv = (ulong(a.w.data) & 15ul) == 0 && (K & 3) == 0;
  const int sm = int(sg) >> 1, sn = int(sg) & 1;

  simdgroup_float8x8 acc[2][2];
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j) acc[i][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
  float4 a0, a1, b0, b1;
  load8_f32(xr, lc, K, xv, am < M, a0, a1);
  load8_f32(wr, lc, K, wv, bn < N, b0, b1);
  const int steps = (K + kF32Step - 1) / kF32Step;
  for (int s = 0; s < steps; ++s) {
    threadgroup_barrier(mem_flags::mem_threadgroup);  // the last step's reads of the staging are done
    ((threadgroup float4*)(As + lr * kF32Pitch + lc))[0] = a0;
    ((threadgroup float4*)(As + lr * kF32Pitch + lc))[1] = a1;
    ((threadgroup float4*)(Bs + lr * kF32Pitch + lc))[0] = b0;
    ((threadgroup float4*)(Bs + lr * kF32Pitch + lc))[1] = b1;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (s + 1 < steps) {
      const int k = (s + 1) * kF32Step + lc;
      load8_f32(xr, k, K, xv, am < M, a0, a1);
      load8_f32(wr, k, K, wv, bn < N, b0, b1);
    }
    for (int kk = 0; kk < kF32Step; kk += 8) {
      simdgroup_float8x8 af[2], bt[2];
      for (int i = 0; i < 2; ++i) simdgroup_load(af[i], As + (sm * 16 + i * 8) * kF32Pitch + kk, kF32Pitch);
      for (int j = 0; j < 2; ++j)
        simdgroup_load(bt[j], Bs + (sn * 16 + j * 8) * kF32Pitch + kk, kF32Pitch, ulong2(0, 0), true);
      for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j) simdgroup_multiply_accumulate(acc[i][j], af[i], bt[j], acc[i][j]);
    }
  }

  // The finished tile over the staging, then the epilogue: thread t takes row t / 4, 8 columns.
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
      simdgroup_store(acc[i][j], smem + (sm * 16 + i * 8) * kF32Tile + sn * 16 + j * 8, kF32Tile);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (am >= M) return;
  for (int c = lc; c < lc + 8; ++c) {
    const int n = n0 + c;
    if (n >= N) break;
    float val = smem[lr * kF32Tile + c];
    if (a.bias) val += load_val(a.bias, n, a.bias_bf16);
    val *= a.alpha;
    const long o = long(am) * a.ld_y + n;
    if (a.beta != 0.0f) val += a.beta * load_val(a.y, o, a.y_bf16);
    store_val(a.y, o, val, a.y_bf16);
  }
}
