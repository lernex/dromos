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
