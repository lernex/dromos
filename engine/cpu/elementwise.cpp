// Norms and elementwise ops (kernels.h, "Norms and elementwise"). Small calls (a decode step's rows)
// run on the calling thread: waking the pool costs more than the work. Large ones split by rows.
#include "cpu/cpu.h"

namespace lx::cpu {
namespace {

// Elements per parallel chunk: below this a row range runs on one thread.
constexpr int64_t kChunkElems = 32 * 1024;

int64_t row_grain(int64_t rows, int64_t cols) {
  const int64_t per = cols > 0 ? (kChunkElems + cols - 1) / cols : rows;
  return grain_for(rows, per < 1 ? 1 : per);
}

// sum of squares of n f32 values, eight lanes then a fixed reduction.
inline float sum_squares(const float* x, int64_t n) {
  float acc[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  int64_t i = 0;
  for (; i + 8 <= n; i += 8)
    for (int j = 0; j < 8; ++j) acc[j] += x[i + j] * x[i + j];
  float s = ((acc[0] + acc[4]) + (acc[1] + acc[5])) + ((acc[2] + acc[6]) + (acc[3] + acc[7]));
  for (; i < n; ++i) s += x[i] * x[i];
  return s;
}

// A row of x (f32 or bf16) as f32: the row itself, or converted into tmp.
inline const float* row_f32(const void* x, int dtype, int64_t offset, int64_t n, float* tmp) {
  if (dtype == LX_F32) return static_cast<const float*>(x) + offset;
  const uint16_t* p = static_cast<const uint16_t*>(x) + offset;
  for (int64_t i = 0; i < n; ++i) tmp[i] = bf16_to_f32(p[i]);
  return tmp;
}

inline void store_row(void* y, int dtype, int64_t offset, const float* v, int64_t n) {
  if (dtype == LX_F32) {
    memcpy(static_cast<float*>(y) + offset, v, sizeof(float) * n);
  } else {
    uint16_t* p = static_cast<uint16_t*>(y) + offset;
    for (int64_t i = 0; i < n; ++i) p[i] = f32_to_bf16(v[i]);
  }
}

}  // namespace
}  // namespace lx::cpu

using namespace lx::cpu;

extern "C" {

int lx_rmsnorm(const void* x, lx_dtype x_dtype, const void* w, void* y, lx_dtype y_dtype, int rows, int dim, float eps,
               void* stream) {
  LX_PROFILE("rmsnorm");
  (void)stream;
  if (rows <= 0) return kOk;
  if (dim <= 0 || !x || !w || !y || !valid_dtype(x_dtype) || !valid_dtype(y_dtype)) return fail(kInvalid);
  const uint16_t* wb = static_cast<const uint16_t*>(w);
  parallel_for(rows, row_grain(rows, dim), [&](int64_t r0, int64_t r1) {
    float* tmp = static_cast<float*>(scratch(sizeof(float) * 2 * dim));
    float* out = tmp + dim;
    for (int64_t r = r0; r < r1; ++r) {
      const float* v = row_f32(x, x_dtype, r * dim, dim, tmp);
      const float scale = 1.0f / std::sqrt(sum_squares(v, dim) / dim + eps);
      for (int i = 0; i < dim; ++i) out[i] = v[i] * scale * bf16_to_f32(wb[i]);
      store_row(y, y_dtype, r * dim, out, dim);
    }
  });
  return kOk;
}

int lx_copy2d(const void* src, lx_dtype src_dtype, int ld_src, void* dst, lx_dtype dst_dtype, int ld_dst, int rows,
              int cols, void* stream) {
  LX_PROFILE("copy2d");
  (void)stream;
  if (rows <= 0 || cols <= 0) return kOk;
  if (!valid_dtype(src_dtype) || !valid_dtype(dst_dtype) || !src || !dst) return fail(kInvalid);
  // A window read out of a wider buffer needs strides that clear it. One row needs no stride.
  if (rows > 1 && (ld_src < cols || ld_dst < cols)) return fail(kInvalid);
  parallel_for(rows, row_grain(rows, cols), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      const int64_t s = r * ld_src, d = r * ld_dst;
      if (src_dtype == dst_dtype) {
        const size_t es = src_dtype == LX_F32 ? 4 : 2;
        memcpy(static_cast<char*>(dst) + d * es, static_cast<const char*>(src) + s * es, es * cols);
      } else if (src_dtype == LX_F32) {
        const float* a = static_cast<const float*>(src) + s;
        uint16_t* b = static_cast<uint16_t*>(dst) + d;
        for (int c = 0; c < cols; ++c) b[c] = f32_to_bf16(a[c]);
      } else {
        const uint16_t* a = static_cast<const uint16_t*>(src) + s;
        float* b = static_cast<float*>(dst) + d;
        for (int c = 0; c < cols; ++c) b[c] = bf16_to_f32(a[c]);
      }
    }
  });
  return kOk;
}

int lx_activate_f32(float* y, int64_t n, int act, void* stream) {
  LX_PROFILE("activate_f32");
  (void)stream;
  if (n <= 0) return kOk;
  if (act < 0 || act > 2 || !y) return fail(kInvalid);
  const Kern& k = kern();
  parallel_for(n, grain_for(n, kChunkElems), [&](int64_t b, int64_t e) {
    if (act == 0) {
      for (int64_t i = b; i < e; ++i) y[i] = std::tanh(y[i]);
      return;
    }
    // silu and sigmoid through the vector exp: e = exp(-v), in blocks.
    float t[256];
    for (int64_t i = b; i < e; i += 256) {
      const int64_t m = e - i < 256 ? e - i : 256;
      for (int64_t j = 0; j < m; ++j) t[j] = -y[i + j];
      k.exp_f32(t, t, m);
      if (act == 1) {
        for (int64_t j = 0; j < m; ++j) y[i + j] = y[i + j] / (1.0f + t[j]);
      } else {
        for (int64_t j = 0; j < m; ++j) y[i + j] = 1.0f / (1.0f + t[j]);
      }
    }
  });
  return kOk;
}

int lx_silu_mul(const float* x, int ld_x, void* out, lx_dtype out_dtype, int rows, int cols, void* stream) {
  LX_PROFILE("silu_mul");
  (void)stream;
  if (rows <= 0 || cols <= 0) return kOk;
  if (ld_x < 2 * cols || !x || !out || !valid_dtype(out_dtype)) return fail(kInvalid);
  const Kern& k = kern();
  parallel_for(rows, row_grain(rows, cols), [&](int64_t r0, int64_t r1) {
    float* t = static_cast<float*>(scratch(sizeof(float) * cols));
    for (int64_t r = r0; r < r1; ++r) {
      const float* xr = x + r * ld_x;
      for (int c = 0; c < cols; ++c) t[c] = -xr[c];
      k.exp_f32(t, t, cols);
      for (int c = 0; c < cols; ++c) t[c] = xr[c] / (1.0f + t[c]) * xr[cols + c];
      store_row(out, out_dtype, r * cols, t, cols);
    }
  });
  return kOk;
}

int lx_add_row_bf16(float* y, const void* row, int rows, int cols, void* stream) {
  LX_PROFILE("add_row_bf16");
  (void)stream;
  if (rows <= 0 || cols <= 0) return kOk;
  if (!y || !row) return fail(kInvalid);
  const uint16_t* rb = static_cast<const uint16_t*>(row);
  parallel_for(rows, row_grain(rows, cols), [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r)
      for (int c = 0; c < cols; ++c) y[r * cols + c] += bf16_to_f32(rb[c]);
  });
  return kOk;
}

// ---- Embedding and the residual streams ---------------------------------------------------------------

int lx_embed_streams(const void* table, const int32_t* ids, const void* stream_emb, float* streams, int T, int S, int D,
                     void* stream) {
  LX_PROFILE("embed_streams");
  (void)stream;
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (!table || !ids || !stream_emb || !streams) return fail(kInvalid);
  const uint16_t* tb = static_cast<const uint16_t*>(table);
  const uint16_t* se = static_cast<const uint16_t*>(stream_emb);
  parallel_for(T, row_grain(T, static_cast<int64_t>(S) * D), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const uint16_t* row = tb + static_cast<int64_t>(ids[t]) * D;
      float* out = streams + t * S * D;
      for (int s = 0; s < S; ++s)
        for (int d = 0; d < D; ++d) out[s * D + d] = bf16_to_f32(row[d]) + bf16_to_f32(se[s * D + d]);
    }
  });
  return kOk;
}

int lx_stream_mean(const float* streams, float* mean, int T, int S, int D, void* stream) {
  LX_PROFILE("stream_mean");
  (void)stream;
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (!streams || !mean) return fail(kInvalid);
  parallel_for(T, row_grain(T, static_cast<int64_t>(S) * D), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const float* in = streams + t * S * D;
      float* out = mean + t * D;
      for (int d = 0; d < D; ++d) out[d] = in[d];
      for (int s = 1; s < S; ++s)
        for (int d = 0; d < D; ++d) out[d] += in[s * D + d];
      for (int d = 0; d < D; ++d) out[d] /= S;
    }
  });
  return kOk;
}

int lx_mhc_read(const float* streams, const float* matrix, const float* read, const void* norm_w, float eps,
                float* mixed, void* h, int T, int S, int D, void* stream) {
  LX_PROFILE("mhc_read");
  (void)stream;
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (!streams || !matrix || !read || !norm_w || !mixed || !h) return fail(kInvalid);
  const uint16_t* nw = static_cast<const uint16_t*>(norm_w);
  uint16_t* hb = static_cast<uint16_t*>(h);
  parallel_for(T, row_grain(T, static_cast<int64_t>(S) * D), [&](int64_t t0, int64_t t1) {
    float* source = static_cast<float*>(scratch(sizeof(float) * D));
    for (int64_t t = t0; t < t1; ++t) {
      const float* in = streams + t * S * D;
      float* mix = mixed + t * S * D;
      for (int o = 0; o < S; ++o) {
        float* mo = mix + o * D;
        const float m0 = matrix[o * S];
        for (int d = 0; d < D; ++d) mo[d] = m0 * in[d];
        for (int i = 1; i < S; ++i) {
          const float mi = matrix[o * S + i];
          const float* xi = in + i * D;
          for (int d = 0; d < D; ++d) mo[d] += mi * xi[d];
        }
      }
      const float r0 = read[0];
      for (int d = 0; d < D; ++d) source[d] = r0 * in[d];
      for (int i = 1; i < S; ++i) {
        const float ri = read[i];
        const float* xi = in + i * D;
        for (int d = 0; d < D; ++d) source[d] += ri * xi[d];
      }
      const float scale = 1.0f / std::sqrt(sum_squares(source, D) / D + eps);
      uint16_t* ho = hb + t * D;
      for (int d = 0; d < D; ++d) ho[d] = f32_to_bf16(source[d] * scale * bf16_to_f32(nw[d]));
    }
  });
  return kOk;
}

int lx_mhc_write(const float* mixed, const float* write, const float* update, float* streams, int T, int S, int D,
                 void* stream) {
  LX_PROFILE("mhc_write");
  (void)stream;
  if (T <= 0 || S <= 0 || D <= 0) return kOk;
  if (!mixed || !write || !update || !streams) return fail(kInvalid);
  parallel_for(T, row_grain(T, static_cast<int64_t>(S) * D), [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const float* up = update + t * D;
      for (int s = 0; s < S; ++s) {
        const float w = write[s];
        const float* mi = mixed + (t * S + s) * D;
        float* so = streams + (t * S + s) * D;
        for (int d = 0; d < D; ++d) so[d] = mi[d] + w * up[d];
      }
    }
  });
  return kOk;
}

int lx_mhc_write_read(float* mixed, const float* write, const float* update, float* streams, const float* matrix,
                      const float* read, const void* norm_w, float eps, void* h, int T, int S, int D, void* stream) {
  const int status = lx_mhc_write(mixed, write, update, streams, T, S, D, stream);
  if (status != kOk) return status;
  return lx_mhc_read(streams, matrix, read, norm_w, eps, mixed, h, T, S, D, stream);
}

}  // extern "C"
