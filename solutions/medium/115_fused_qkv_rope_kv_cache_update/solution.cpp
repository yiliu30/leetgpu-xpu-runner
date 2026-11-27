#include <sycl/sycl.hpp>

namespace {

constexpr int kMaxHeadDim = 128;

struct QkvRopeCacheKernel {
  const float *x;
  const float *W_qkv;
  const float *cos_sin_cache;
  const int *positions;
  float *K_cache;
  float *V_cache;
  float *Q_out;
  int d_model;
  int H_q;
  int H_kv;
  int D;
  int S_max;

  void operator()(sycl::id<1> idx) const {
    const int heads_per_batch = H_q + 2 * H_kv;
    const int b = idx[0] / heads_per_batch;
    const int lane = idx[0] - b * heads_per_batch;
    const int out_cols = heads_per_batch * D;
    const int half = D / 2;
    float values[kMaxHeadDim];

    int col_base = 0;
    int head = 0;
    int kind = 0;
    if (lane < H_q) {
      kind = 0;
      head = lane;
      col_base = head * D;
    } else if (lane < H_q + H_kv) {
      kind = 1;
      head = lane - H_q;
      col_base = (H_q + head) * D;
    } else {
      kind = 2;
      head = lane - H_q - H_kv;
      col_base = (H_q + H_kv + head) * D;
    }

    const float *x_row = x + static_cast<long>(b) * d_model;
    for (int j = 0; j < D; ++j) {
      double sum = 0.0;
      for (int k = 0; k < d_model; ++k) {
        sum += static_cast<double>(x_row[k]) *
               static_cast<double>(W_qkv[static_cast<long>(k) * out_cols + col_base + j]);
      }
      values[j] = static_cast<float>(sum);
    }

    if (kind == 0 || kind == 1) {
      const int pos = positions[b];
      const float *cos_row = cos_sin_cache + static_cast<long>(pos) * D;
      float rotated[kMaxHeadDim];
      for (int j = 0; j < half; ++j) {
        const float a = values[j];
        const float bv = values[j + half];
        const float c = cos_row[j];
        const float s = cos_row[half + j];
        rotated[j] = a * c - bv * s;
        rotated[j + half] = bv * c + a * s;
      }
      if (kind == 0) {
        float *dst = Q_out + (static_cast<long>(b) * H_q + head) * D;
        for (int j = 0; j < D; ++j) {
          dst[j] = rotated[j];
        }
      } else {
        const int pos_k = positions[b];
        float *dst = K_cache + ((static_cast<long>(b) * H_kv + head) * S_max + pos_k) * D;
        for (int j = 0; j < D; ++j) {
          dst[j] = rotated[j];
        }
      }
    } else {
      const int pos_v = positions[b];
      float *dst = V_cache + ((static_cast<long>(b) * H_kv + head) * S_max + pos_v) * D;
      for (int j = 0; j < D; ++j) {
        dst[j] = values[j];
      }
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *W_qkv, const float *cos_sin_cache,
                      const int *positions, float *K_cache, float *V_cache, float *Q_out, int B,
                      int d_model, int H_q, int H_kv, int D, int S_max) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total_heads = B * (H_q + 2 * H_kv);
  q.parallel_for(sycl::range<1>(total_heads),
                 QkvRopeCacheKernel{x, W_qkv, cos_sin_cache, positions, K_cache, V_cache, Q_out,
                                    d_model, H_q, H_kv, D, S_max});
  q.wait();
}
