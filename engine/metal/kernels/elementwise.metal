// Norms and elementwise kernels, and the model's small glue kernels.

// One threadgroup per row, float32 throughout. Dtypes are template parameters: 0 f32, 1 bf16.
template <int XB, int YB>
kernel void rmsnorm(constant RmsnormArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                    uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  threadgroup float scratch[32];
  const long base = long(row) * a.dim;
  float sum = 0.0f;
  for (int i = tid; i < a.dim; i += threads) {
    const float v = load_val(a.x, base + i, XB);
    sum += v * v;
  }
  const float scale = rsqrt(group_sum(sum, scratch, tid, threads) / a.dim + a.eps);
  for (int i = tid; i < a.dim; i += threads)
    store_val(a.y, base + i, load_val(a.x, base + i, XB) * scale * bf(a.w[i]), YB);
}
template [[host_name("rmsnorm_00")]] kernel void rmsnorm<0, 0>(constant RmsnormArgs&, uint, uint, uint);
template [[host_name("rmsnorm_01")]] kernel void rmsnorm<0, 1>(constant RmsnormArgs&, uint, uint, uint);
template [[host_name("rmsnorm_10")]] kernel void rmsnorm<1, 0>(constant RmsnormArgs&, uint, uint, uint);
template [[host_name("rmsnorm_11")]] kernel void rmsnorm<1, 1>(constant RmsnormArgs&, uint, uint, uint);

// dst[r, c] = cast(src[r, c]); x covers columns, y and z rows (grid y * z may exceed one dimension).
template <int SB, int DB>
kernel void copy2d(constant Copy2dArgs& a [[buffer(0)]], uint3 gid [[thread_position_in_grid]],
                   uint3 grid [[threads_per_grid]]) {
  const int c = gid.x;
  if (c >= a.cols) return;
  for (long r = gid.y + long(gid.z) * grid.y; r < a.rows; r += long(grid.y) * grid.z)
    store_val(a.dst, r * a.ld_dst + c, load_val(a.src, r * a.ld_src + c, SB), DB);
}
template [[host_name("copy2d_00")]] kernel void copy2d<0, 0>(constant Copy2dArgs&, uint3, uint3);
template [[host_name("copy2d_01")]] kernel void copy2d<0, 1>(constant Copy2dArgs&, uint3, uint3);
template [[host_name("copy2d_10")]] kernel void copy2d<1, 0>(constant Copy2dArgs&, uint3, uint3);
template [[host_name("copy2d_11")]] kernel void copy2d<1, 1>(constant Copy2dArgs&, uint3, uint3);

template <int ACT>
kernel void activate(constant ActivateArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]],
                     uint grid [[threads_per_grid]]) {
  for (long j = i; j < a.n; j += grid) {
    const float v = a.y[j];
    if (ACT == 0) a.y[j] = precise::tanh(v);
    else if (ACT == 1) a.y[j] = v * sigmoid(v);
    else a.y[j] = sigmoid(v);
  }
}
template [[host_name("activate_0")]] kernel void activate<0>(constant ActivateArgs&, uint, uint);
template [[host_name("activate_1")]] kernel void activate<1>(constant ActivateArgs&, uint, uint);
template [[host_name("activate_2")]] kernel void activate<2>(constant ActivateArgs&, uint, uint);

template <int OB>
kernel void silu_mul(constant SiluMulArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]],
                     uint2 grid [[threads_per_grid]]) {
  const int c = gid.x;
  if (c >= a.cols) return;
  for (long r = gid.y; r < a.rows; r += grid.y) {
    const device float* xr = a.x + r * a.ld_x;
    const float g = xr[c];
    store_val(a.out, r * a.cols + c, g * sigmoid(g) * xr[a.cols + c], OB);
  }
}
template [[host_name("silu_mul_0")]] kernel void silu_mul<0>(constant SiluMulArgs&, uint2, uint2);
template [[host_name("silu_mul_1")]] kernel void silu_mul<1>(constant SiluMulArgs&, uint2, uint2);

kernel void add_row(constant AddRowArgs& a [[buffer(0)]], uint2 gid [[thread_position_in_grid]],
                    uint2 grid [[threads_per_grid]]) {
  const int c = gid.x;
  if (c >= a.cols) return;
  const float add = bf(a.row[c]);
  for (long r = gid.y; r < a.rows; r += grid.y) a.y[r * a.cols + c] += add;
}

kernel void axpby(constant AxpbyArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]],
                  uint grid [[threads_per_grid]]) {
  for (long j = i; j < a.n; j += grid) a.y[j] = a.a * a.y[j] + a.b * a.x[j];
}

kernel void sub_f32(constant SubArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]],
                    uint grid [[threads_per_grid]]) {
  for (long j = i; j < a.n; j += grid) a.out[j] = a.a[j] - a.b[j];
}

kernel void meta_scalars(constant MetaScalarsArgs& a [[buffer(0)]], uint t [[thread_position_in_grid]]) {
  if (int(t) >= a.T) return;
  device ushort* row = a.meta + long(t) * a.ld + a.col;
  row[0] = to_bf(a.mean_k);
  row[1] = to_bf(a.entropy[t] * a.inv_log_experts);
  row[2] = to_bf(a.confidence[t]);
  row[3] = to_bf(a.continuation[t]);
}

// out[i, :] = src[cu[i + 1] - 1, :], read on the device so a recorded step works for any lengths.
kernel void gather_last(constant GatherLastArgs& a [[buffer(0)]], uint i [[threadgroup_position_in_grid]],
                        uint tid [[thread_index_in_threadgroup]], uint threads [[threads_per_threadgroup]]) {
  const long row = a.cu[i + 1] - 1;
  for (int d = tid; d < a.D; d += threads) a.out[long(i) * a.D + d] = a.src[row * a.D + d];
}

// The model's fused row kernels (the glue ops in kernels.h); columns run over x, tokens over y (one group per token).
kernel void concat4_bf16(constant Concat4Args& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int j = int(gid.x), t = int(gid.y), W = 3 * p.D + p.R;
  if (j >= W || t >= p.T) return;
  float v;
  if (j < p.D) v = p.a[long(t) * p.D + j];
  else if (j < 2 * p.D) v = p.b[long(t) * p.D + (j - p.D)];
  else if (j < 3 * p.D) v = p.c[long(t) * p.D + (j - 2 * p.D)];
  else v = p.d[long(t) * p.R + (j - 3 * p.D)];
  p.out[long(t) * W + j] = to_bf(v);
}

kernel void pack_router(constant PackRouterArgs& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int j = int(gid.x), t = int(gid.y), W = p.Lat + p.R;
  if (j >= W || t >= p.T) return;
  if (j < p.Lat) p.latent[long(t) * p.Lat + j] = to_bf(p.router_in[long(t) * W + j]);
  else p.router_in[long(t) * W + j] = p.features[long(t) * p.R + (j - p.Lat)];
}

kernel void meta_entry(constant MetaEntryArgs& p [[buffer(0)]], uint2 gid [[thread_position_in_grid]]) {
  const int j = int(gid.x), t = int(gid.y), ld = p.R + 4;
  if (j >= ld || t >= p.T) return;
  device ushort* row = p.meta + long(t) * ld;
  if (j < p.R) row[j] = to_bf(p.summary[long(t) * p.R + j]);
  else if (j == p.R) row[j] = to_bf(p.mean_k);
  else if (j == p.R + 1) row[j] = to_bf(p.entropy[t] * p.inv_log_experts);
  else if (j == p.R + 2) row[j] = to_bf(p.confidence[t]);
  else row[j] = to_bf(p.continuation[t]);
}
