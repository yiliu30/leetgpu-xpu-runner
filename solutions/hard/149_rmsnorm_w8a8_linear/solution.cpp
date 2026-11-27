#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

struct QuantizeActivationsKernel {
  const float *x;
  const float *gamma;
  int8_t *q;
  double *x_scale;
  int M;
  int K;

  void operator()(sycl::id<1> idx) const {
    const int row = idx[0];
    if (row >= M) {
      return;
    }
    const float *x_row = x + static_cast<size_t>(row) * K;
    double sum_sq = 0.0;
    for (int k = 0; k < K; ++k) {
      const double xv = static_cast<double>(x_row[k]);
      sum_sq += xv * xv;
    }
    const double inv_rms =
        1.0 / sycl::sqrt(sum_sq / static_cast<double>(K) + 1e-6);

    double max_abs = 0.0;
    for (int k = 0; k < K; ++k) {
      const double y = static_cast<double>(x_row[k]) * inv_rms *
                       static_cast<double>(gamma[k]);
      const double ay = y < 0.0 ? -y : y;
      if (ay > max_abs) {
        max_abs = ay;
      }
    }

    const double scale = max_abs / 127.0;
    x_scale[row] = scale;
    int8_t *q_row = q + static_cast<size_t>(row) * K;
    if (scale > 0.0) {
      for (int k = 0; k < K; ++k) {
        const double y = static_cast<double>(x_row[k]) * inv_rms *
                         static_cast<double>(gamma[k]);
        double rounded = sycl::rint(y / scale);
        if (rounded > 127.0) {
          rounded = 127.0;
        } else if (rounded < -127.0) {
          rounded = -127.0;
        }
        q_row[k] = static_cast<int8_t>(rounded);
      }
    } else {
      for (int k = 0; k < K; ++k) {
        q_row[k] = static_cast<int8_t>(0);
      }
    }
  }
};

struct W8A8LinearKernel {
  const int8_t *q;
  const int8_t *W;
  const double *x_scale;
  const float *w_scale;
  float *output;
  int M;
  int N;
  int K;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    const size_t total = static_cast<size_t>(M) * N;
    if (linear >= total) {
      return;
    }
    const int row = static_cast<int>(linear / N);
    const int col = static_cast<int>(linear % N);
    const int8_t *q_row = q + static_cast<size_t>(row) * K;
    const int8_t *w_row = W + static_cast<size_t>(col) * K;
    int32_t acc = 0;
    for (int k = 0; k < K; ++k) {
      acc += static_cast<int32_t>(q_row[k]) * static_cast<int32_t>(w_row[k]);
    }
    output[linear] =
        static_cast<float>(static_cast<double>(acc) * x_scale[row] *
                           static_cast<double>(w_scale[col]));
  }
};

} // namespace

extern "C" void solve(const float *x, const float *gamma, const int8_t *W,
                      const float *w_scale, float *output, int M, int N,
                      int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  int8_t *q_act = sycl::malloc_device<int8_t>(static_cast<size_t>(M) * K, q);
  double *x_scale = sycl::malloc_device<double>(M, q);

  q.parallel_for(sycl::range<1>(M),
                 QuantizeActivationsKernel{x, gamma, q_act, x_scale, M, K});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * N),
                 W8A8LinearKernel{q_act, W, x_scale, w_scale, output, M, N, K});
  q.wait();

  sycl::free(q_act, q);
  sycl::free(x_scale, q);
}
