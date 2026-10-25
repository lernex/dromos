// Quantization: bf16 weights into INT8, FP8 and NVFP4 (lx_quantize_weight, lx_quantize_w8), back to
// bf16 (lx_dequantize_weight), and activations per token (lx_quantize_act). Every quotient is an IEEE
// division and every rounding is to nearest even, so the bytes match the PyTorch references exactly.

// INT8 (Q = 127) and FP8 (Q = 448) per-row absmax: scale = amax / Q (1 for a zero row), q = w / scale
// rounded. One threadgroup per row.
template <int Q>
kernel void quantize_rows(constant QuantArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                          uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int cols = a.cols;
  const device ushort* w = a.w + long(row) * cols;
  float m = 0.0f;
  for (int c = tid; c < cols; c += threads) m = max(m, fabs(bf(w[c])));
  const float amax = group_max(m, scratch, tid, threads);
  const float scale = amax > 0.0f ? amax / float(Q) : 1.0f;
  if (tid == 0) ((device float*)a.scale)[row] = scale;
  device uchar* q = a.q + long(row) * cols;
  for (int c = tid; c < cols; c += threads) {
    const float v = bf(w[c]) / scale;
    if (Q == 127) q[c] = uchar(char(clamp(rint(v), -127.0f, 127.0f)));
    else q[c] = uchar(e4m3_encode(v));
  }
}
template [[host_name("quantize_rows_int8")]] kernel void quantize_rows<127>(constant QuantArgs&, uint, uint, uint);
template [[host_name("quantize_rows_fp8")]] kernel void quantize_rows<448>(constant QuantArgs&, uint, uint, uint);

// NVFP4, pass 1: the absmax of the whole matrix. Non-negative floats order as their bits, so an atomic
// max on the bits is a max on the values.
kernel void nvfp4_amax(constant QuantArgs& a [[buffer(0)]], uint gid [[thread_position_in_grid]],
                       uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]],
                       uint grid [[threads_per_grid]]) {
  threadgroup float scratch[32];
  const long n = a.rows * a.cols;
  float m = 0.0f;
  for (long i = gid; i < n; i += grid) m = max(m, fabs(bf(a.w[i])));
  m = group_max(m, scratch, tid, threads);
  if (tid == 0) atomic_fetch_max_explicit((device atomic_uint*)a.amax, as_type<uint>(m), memory_order_relaxed);
}

// NVFP4, pass 2, one thread per block of 16: the block's e4m3 scale amax_b / (6 * global), then each
// value's nearest e2m1 of w / (scale * global) (0 where the scale is 0), packed low nibble first.
kernel void nvfp4_quant(constant QuantArgs& a [[buffer(0)]], uint gid [[thread_position_in_grid]]) {
  const long blocks = a.rows * (a.cols / 16);
  if (long(gid) >= blocks) return;
  const device ushort* w = a.w + long(gid) * 16;
  float v[16];
  float bmax = 0.0f;
  for (int i = 0; i < 16; ++i) {
    v[i] = bf(w[i]);
    bmax = max(bmax, fabs(v[i]));
  }
  const float g = a.global;
  const uint sb = e4m3_encode(bmax / (6.0f * g));
  a.scale[gid] = uchar(sb);
  const float eff = e4m3(sb) * g;
  device uchar* q = a.q + long(gid) * 8;
  for (int i = 0; i < 8; ++i) {
    const uint lo = eff > 0.0f ? e2m1_encode(v[2 * i] / eff) : 0u;
    const uint hi = eff > 0.0f ? e2m1_encode(v[2 * i + 1] / eff) : 0u;
    q[i] = uchar(lo | (hi << 4));
  }
}

// out[r, c] = w[r, c] with its row factor, as bf16. Grid (columns, rows).
template <int F>
kernel void dequantize(constant DequantArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int c = gid.x;
  const long r = gid.y;
  if (c >= a.cols || r >= a.rows) return;
  const float v = element<F>(a.w.data, a.w.scale, r, c, a.cols) * row_factor<F>(a.w.scale, r, a.w.global);
  a.out[r * a.cols + c] = to_bf(v);
}
template [[host_name("dequantize_0")]] kernel void dequantize<0>(constant DequantArgs&, uint2);
template [[host_name("dequantize_1")]] kernel void dequantize<1>(constant DequantArgs&, uint2);
template [[host_name("dequantize_2")]] kernel void dequantize<2>(constant DequantArgs&, uint2);
template [[host_name("dequantize_3")]] kernel void dequantize<3>(constant DequantArgs&, uint2);

// Activations, one threadgroup per row (kernels.h lx_quantize_act). MATH 1: int8, 2: e4m3, 3: NVFP4.
template <int MATH>
kernel void quantize_act(constant QuantActArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const int K = a.K;
  const long xrow = long(row) * a.ld_x;
  float m = 0.0f;
  for (int c = tid; c < K; c += threads) m = max(m, fabs(load_val(a.x, xrow + c, a.x_bf16)));
  const float amax = group_max(m, scratch, tid, threads);
  device uchar* q = a.q + long(row) * a.ld_q;
  if (MATH == 1 || MATH == 2) {
    const float Q = MATH == 1 ? 127.0f : 448.0f;
    const float inv = amax > 0.0f ? Q / amax : 0.0f;
    if (tid == 0) a.scale[row] = amax > 0.0f ? amax / Q : 1.0f;
    for (int c = tid; c < K; c += threads) {
      const float v = load_val(a.x, xrow + c, a.x_bf16) * inv;
      q[c] = MATH == 1 ? uchar(char(clamp(rint(v), -127.0f, 127.0f))) : uchar(e4m3_encode(v));
    }
  } else {
    const float g = amax > 0.0f ? amax / 2688.0f : 1.0f;
    if (tid == 0) a.scale[row] = g;
    device uchar* bs = a.block_scale + long(row) * a.ld_bs;
    for (int b = tid; b < K / 16; b += threads) {
      float v[16];
      float bmax = 0.0f;
      for (int i = 0; i < 16; ++i) {
        v[i] = load_val(a.x, xrow + b * 16 + i, a.x_bf16);
        bmax = max(bmax, fabs(v[i]));
      }
      const uint sb = e4m3_encode(bmax / (6.0f * g));
      bs[b] = uchar(sb);
      const float s = e4m3(sb);
      const float r = s > 0.0f ? 1.0f / (s * g) : 0.0f;
      for (int i = 0; i < 8; ++i) {
        const uint lo = s > 0.0f ? e2m1_encode(v[2 * i] * r) : 0u;
        const uint hi = s > 0.0f ? e2m1_encode(v[2 * i + 1] * r) : 0u;
        q[b * 8 + i] = uchar(lo | (hi << 4));
      }
    }
  }
}
template [[host_name("quantize_act_1")]] kernel void quantize_act<1>(constant QuantActArgs&, uint, uint, uint);
template [[host_name("quantize_act_2")]] kernel void quantize_act<2>(constant QuantActArgs&, uint, uint, uint);
template [[host_name("quantize_act_3")]] kernel void quantize_act<3>(constant QuantActArgs&, uint, uint, uint);
