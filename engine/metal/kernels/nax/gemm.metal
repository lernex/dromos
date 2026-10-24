// The tile GEMM on the neural accelerators of the GPU cores (Apple M5 and later), through the Metal 4
// tensor ops of MetalPerformancePrimitives: the same products, modes, grid and epilogue as gemm.metal
// (lx_gemm_w and both stages of the grouped experts), built as a second library where the GPU has them.
//
// A threadgroup of four simdgroups computes a 64 x 64 tile of y. Per 32-wide step of K every thread
// stages 16 values of one activation row and 16 of one weight row in threadgroup memory, and one
// matmul2d over the four simdgroups multiplies the 64 x 32 by 32 x 64 into a cooperative f32 tile.
// bf16 values and int8 codes are staged as they are stored; the other formats as their exact decoded
// values (as in gemm.metal: a quantized code times its block scale has at most 6 significant bits) in
// bf16, which represents every one of them. The accelerators multiply bf16 by bf16 or by int8 into
// f32, and int8 by int8 (INT8 math) exactly into int32. Row factors are applied once in the epilogue,
// which reads the finished tile back from threadgroup memory.
//
// Two stages of threadgroup memory: the next step's values are loaded into registers before the
// accelerators run on the current stage and stored into the other one after, so one barrier per step
// keeps the loads in flight under the multiply.
#include <metal_tensor>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>

using namespace mpp::tensor_ops;

// The staged element types. int8 codes stay int8: INT8 weights under BF16 math multiply bf16 by int8,
// and INT8 math multiplies int8 by int8 into int32 (exact). Every other value is staged as bf16.
template <int AF> struct NaxA { using T = bfloat; };
template <> struct NaxA<A_INT8> { using T = int8_t; };
template <int WF> struct NaxB { using T = bfloat; };
template <> struct NaxB<W_INT8> { using T = int8_t; };
template <int AF, int WF> struct NaxAcc { using T = float; };
template <> struct NaxAcc<A_INT8, W_INT8> { using T = int; };

// MODE 0: a plain GEMM, grid (N / 64, M / 64). MODE 1: grouped gate_up, grid (I / 32, tiles), weight rows
// interleaving gate and up. MODE 2: grouped down, grid (L / 64, tiles). As in gemm.metal.
template <int AF, int WF, int MODE>
kernel void gemm_nax(constant GemmArgs& a [[buffer(0)]], threadgroup uchar* smem [[threadgroup(0)]],
                     uint2 tg [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]]) {
  using TA = typename NaxA<AF>::T;
  using TB = typename NaxB<WF>::T;
  using Acc = typename NaxAcc<AF, WF>::T;
  // Per stage: the 64 x 32 activation tile, then the 64 x 32 weight tile.
  constexpr int kABytes = kTileM * kTileK * int(sizeof(TA));
  constexpr int kStageBytes = kABytes + kTileN * kTileK * int(sizeof(TB));
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

  // This thread's load rows: activation row ar and weight row br = ar, 16 values from column ac.
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
  const bool x_vec = a.x_vec != 0;

  constexpr auto desc = matmul2d_descriptor(kTileM, kTileN, kTileK, false, true, false,
                                            matmul2d_descriptor::mode::multiply_accumulate);
  matmul2d<desc, execution_simdgroups<4>> op;
  using TileA = tensor<threadgroup TA, extents<int32_t, kTileK, kTileM>, tensor_inline>;
  using TileB = tensor<threadgroup TB, extents<int32_t, kTileK, kTileN>, tensor_inline>;
  auto acc = op.template get_destination_cooperative_tensor<TileA, TileB, Acc>();
  for (uint16_t i = 0; i < acc.get_capacity(); ++i)
    if (acc.is_valid_element(i)) acc[i] = Acc(0);

  const int steps = (K + kTileK - 1) / kTileK;
  const int at = ar * kTileK + ac;  // this thread's element in either tile
  uint4 wa[2], wb[2];
  stage_act<AF, true>(arow, ablk, ac, K, kTileK <= K && x_vec, a_valid, wa);
  stage_w<WF, true>(brow, bscl, ac, K, kTileK <= K && w_vec, b_valid, wb);
  stage_store((threadgroup TA*)smem + at, wa);
  stage_store((threadgroup TB*)(smem + kABytes) + at, wb);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int s = 0; s < steps; ++s) {
    threadgroup uchar* cur = smem + (s & 1) * kStageBytes;
    threadgroup uchar* nxt = smem + ((s + 1) & 1) * kStageBytes;
    const bool more = s + 1 < steps;
    if (more) {
      const int k0 = (s + 1) * kTileK;
      const bool full = k0 + kTileK <= K;
      stage_act<AF, true>(arow, ablk, k0 + ac, K, full && x_vec, a_valid, wa);
      stage_w<WF, true>(brow, bscl, k0 + ac, K, full && w_vec, b_valid, wb);
    }
    TileA ta((threadgroup TA*)cur, extents<int32_t, kTileK, kTileM>());
    TileB tb((threadgroup TB*)(cur + kABytes), extents<int32_t, kTileK, kTileN>());
    op.run(ta, tb, acc);
    if (more) {
      stage_store((threadgroup TA*)nxt + at, wa);
      stage_store((threadgroup TB*)(nxt + kABytes) + at, wb);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  // The finished tile into threadgroup memory (over the stages, which every simdgroup is done with).
  threadgroup Acc* cs = (threadgroup Acc*)smem;
  tensor<threadgroup Acc, extents<int32_t, kTileN, kTileM>, tensor_inline> tc(cs, extents<int32_t, kTileN, kTileM>());
  acc.store(tc);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Epilogue: thread tid takes row r = tid / 2 and 32 columns from c0.
  const int r = int(tid) >> 1, c0 = (int(tid) & 1) * 32;
  if (r >= rows) return;
  int pair = 0, xr = 0;
  if (MODE == 0) xr = m_base + r;
  else {
    pair = a.order[start + r];
    xr = MODE == 1 ? pair / a.top_k : pair;
  }
  const float fa = (AF == A_INT8 || AF == A_E4M3 || AF == A_FP4) ? a.x_scale[xr] : 1.0f;
  const threadgroup Acc* crow = cs + r * kTileN;
  if (MODE == 1) {
    for (int c = c0; c < c0 + 32; c += 2) {
      const int col = int(tg.x) * (kTileN / 2) + (c >> 1);
      if (col >= N) break;
      const float g = float(crow[c]) * fa * row_factor<WF>(wscale, col, wglobal);
      const float u = float(crow[c + 1]) * fa * row_factor<WF>(wscale, N + col, wglobal);
      ((device ushort*)a.y)[long(pair) * N + col] = to_bf(silu(g) * u);
    }
    return;
  }
  for (int c = c0; c < c0 + 32; ++c) {
    const int n = int(tg.x) * kTileN + c;
    if (n >= N) break;
    float val = float(crow[c]) * fa * row_factor<WF>(wscale, n, wglobal);
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

#define LX_GEMM_NAX(AF, WF, MODE)                                                                    \
  template [[host_name("gemm_nax_" #AF #WF "_" #MODE)]] kernel void gemm_nax<AF, WF, MODE>(          \
      constant GemmArgs&, threadgroup uchar*, uint2, uint);
// Plain: BF16 math on every format from bf16 or f32 activations, and the quantized maths.
LX_GEMM_NAX(0, 0, 0) LX_GEMM_NAX(0, 1, 0) LX_GEMM_NAX(0, 2, 0) LX_GEMM_NAX(0, 3, 0)
LX_GEMM_NAX(1, 0, 0) LX_GEMM_NAX(1, 1, 0) LX_GEMM_NAX(1, 2, 0) LX_GEMM_NAX(1, 3, 0)
LX_GEMM_NAX(2, 1, 0) LX_GEMM_NAX(3, 2, 0) LX_GEMM_NAX(3, 3, 0) LX_GEMM_NAX(4, 3, 0)
// Grouped experts, both stages.
LX_GEMM_NAX(0, 0, 1) LX_GEMM_NAX(0, 1, 1) LX_GEMM_NAX(0, 2, 1) LX_GEMM_NAX(0, 3, 1)
LX_GEMM_NAX(2, 1, 1) LX_GEMM_NAX(3, 2, 1) LX_GEMM_NAX(3, 3, 1) LX_GEMM_NAX(4, 3, 1)
LX_GEMM_NAX(0, 0, 2) LX_GEMM_NAX(0, 1, 2) LX_GEMM_NAX(0, 2, 2) LX_GEMM_NAX(0, 3, 2)
LX_GEMM_NAX(2, 1, 2) LX_GEMM_NAX(3, 2, 2) LX_GEMM_NAX(3, 3, 2) LX_GEMM_NAX(4, 3, 2)
