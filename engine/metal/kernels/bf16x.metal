// Lossless bf16 (LX_W_BF16X; layout in common.metal): the encoder and the decoder. The products that read
// the format are the GEMV and few-token expert kernels themselves (gemv.metal, moe.metal).
//
// One simdgroup per row, lane l holding values 4l .. 4l + 3 of each block of 128. The encoder runs in two
// passes around the host: bf16x_bases writes each block's base exponent and counts the row's patches,
// the host places the overflow records of rows with more patches than slots, then bf16x_rows writes the
// rest. Every byte of the encoding is written, so it is a function of the weights alone.
constant constexpr int kBf16xRowsPerGroup = 4;  // simdgroups (rows) per threadgroup

inline uint bf16x_exponent(uint u) { return (u >> 7) & 0xffu; }

kernel void bf16x_bases(constant Bf16xArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                        uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const long r = long(tg) * kBf16xRowsPerGroup + sg;
  if (r >= a.rows) return;
  const int K = a.K;
  const device ushort* src = a.w + r * K;
  device uchar* base = a.data ? a.data + r * bf16x_row_bytes(K) + bf16x_base_offset(K) : nullptr;
  uint total = 0;
  for (int j = 0; j < bf16x_blocks(K); ++j) {
    uint e[4], mx = 0;
    for (int t = 0; t < 4; ++t) {
      const int k = 128 * j + 4 * int(lane) + t;
      e[t] = k < K ? bf16x_exponent(src[k]) : 0u;
      mx = max(mx, e[t]);
    }
    const uint e0 = simd_max(mx);
    uint n = 0;
    for (int t = 0; t < 4; ++t) n += (128 * j + 4 * int(lane) + t < K && e0 - e[t] > 15u) ? 1u : 0u;
    total += simd_sum(n);
    if (base && lane == 0) base[j] = uchar(e0);
  }
  if (lane == 0) a.counts[r] = total;
}

kernel void bf16x_rows(constant Bf16xArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                       uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const long r = long(tg) * kBf16xRowsPerGroup + sg;
  if (r >= a.rows) return;
  const int K = a.K;
  const int P = bf16x_slots(K), B = bf16x_blocks(K), H = bf16x_header(K), bo = bf16x_base_offset(K);
  const long rb = bf16x_row_bytes(K);
  const device ushort* src = a.w + r * K;
  device uchar* row = a.data + r * rb;
  device uint* slots = (device uint*)row;
  const uint count = a.counts[r];

  // The row's patches go to its header slots, or to its overflow record after the rows.
  device uint* dst = slots;
  for (int s = lane; s < P; s += 32) slots[s] = kBf16xNoPatch;
  for (int b = bo + B + int(lane); b < H; b += 32) row[b] = 0;  // the header's padding
  if (count > uint(P)) {
    const long region = a.rows * rb;  // rows are whole multiples of 8 bytes
    device uint* rec = (device uint*)(a.data + region + a.offsets[r]);
    simdgroup_barrier(mem_flags::mem_device);
    if (lane == 0) {
      slots[0] = 0x80000000u | uint((region + a.offsets[r] - r * rb) / 8);
      rec[0] = count;
      rec[1 + count] = kBf16xNoPatch;
      if ((count & 1u) != 0) rec[2 + count] = 0;  // the record's padding to 8 bytes
    }
    dst = rec + 1;
  }
  simdgroup_barrier(mem_flags::mem_device);

  uint written = 0;
  for (int j = 0; j < B; ++j) {
    const uint e0 = row[bo + j];
    uint sm = 0, cn = 0, n = 0;
    uint pk[4];
    for (int t = 0; t < 4; ++t) {
      const int k = 128 * j + 4 * int(lane) + t;
      pk[t] = kBf16xNoPatch;
      if (k < K) {
        const uint u = src[k];
        const uint e = bf16x_exponent(u);
        sm |= (((u >> 8) & 0x80u) | (u & 0x7fu)) << (8 * t);
        if (e0 - e > 15u) {
          pk[t] = (uint(k) << 8) | e;
          ++n;
        } else {
          cn |= (e0 - e) << (8 * t);
        }
      }
    }
    device uchar* blk = row + H + long(j) * 192;
    ((device uint*)blk)[lane] = sm;
    // Lane 2g holds values 8g .. 8g + 3 (low nibbles), lane 2g + 1 values 8g + 4 .. 8g + 7 (high).
    const uint other = simd_shuffle_xor(cn, 1);
    if ((lane & 1) == 0) ((device uint*)(blk + 128))[lane >> 1] = cn | (other << 4);
    // Patches in k order: lanes in order, values in order within a lane.
    const uint before = simd_prefix_exclusive_sum(n);
    const uint block_total = simd_sum(n);
    uint idx = written + before;
    for (int t = 0; t < 4; ++t)
      if (pk[t] != kBf16xNoPatch) dst[idx++] = pk[t];
    written += block_total;
  }
}

kernel void bf16x_decode(constant Bf16xArgs& a [[buffer(0)]], uint tg [[threadgroup_position_in_grid]],
                         uint sg [[simdgroup_index_in_threadgroup]], uint lane [[thread_index_in_simdgroup]]) {
  const long r = long(tg) * kBf16xRowsPerGroup + sg;
  if (r >= a.rows) return;
  const int K = a.K;
  const device uchar* row = a.data + r * bf16x_row_bytes(K);
  const int H = bf16x_header(K), bo = bf16x_base_offset(K);
  Bf16xPatches pc;
  pc.start(row, K);
  for (int j = 0; j < bf16x_blocks(K); ++j) {
    const device uchar* blk = row + H + long(j) * 192;
    const uint k0 = 128u * uint(j) + 4u * lane;
    const uint sm = ((const device uint*)blk)[lane];
    const uint cd = ((const device uint*)(blk + 128))[lane >> 1] >> (4 * (lane & 1));
    uint e = uint(row[bo + j]) * 0x01010101u - (cd & 0x0f0f0f0fu);
    while (pc.pos() < k0 + 4) {
      e = bf16x_patch(e, pc.next, k0);
      pc.advance();
    }
    const uint2 b = bf16x_assemble(sm, e);
    const ushort v[4] = {ushort(b.x), ushort(b.x >> 16), ushort(b.y), ushort(b.y >> 16)};
    for (int t = 0; t < 4; ++t)
      if (int(k0) + t < K) a.out[r * K + k0 + t] = v[t];
  }
}
