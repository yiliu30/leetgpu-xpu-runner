#include <sycl/sycl.hpp>

namespace {

struct LayerNormKernel {
  const float *x;
  const float *gamma;
  const float *beta;
  float *normalized;
  int K;
  float eps;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const size_t base = static_cast<size_t>(row) * K;
    double mean = 0.0;
    for (int col = 0; col < K; ++col) {
      mean += static_cast<double>(x[base + col]);
    }
    mean /= K;
    double variance = 0.0;
    for (int col = 0; col < K; ++col) {
      const double centered = static_cast<double>(x[base + col]) - mean;
      variance += centered * centered;
    }
    variance /= K;
    const double scale = 1.0 / sycl::sqrt(variance + static_cast<double>(eps));
    for (int col = 0; col < K; ++col) {
      normalized[base + col] = static_cast<float>(
          (static_cast<double>(x[base + col]) - mean) * scale *
              static_cast<double>(gamma[col]) +
          static_cast<double>(beta[col]));
    }
  }
};

struct LinearGeluKernel {
  const float *normalized;
  const float *W;
  const float *bias;
  float *y;
  int K;
  int N;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int row = static_cast<int>(linear / N);
    const int col = static_cast<int>(linear % N);
    const float *xr = normalized + static_cast<size_t>(row) * K;
    double sum = 0.0;
    for (int k = 0; k < K; ++k) {
      sum += static_cast<double>(xr[k]) * static_cast<double>(W[static_cast<size_t>(k) * N + col]);
    }
    const double h = sum + static_cast<double>(bias[col]);
    const double gelu = 0.5 * h * (1.0 + sycl::erf(h / sycl::sqrt(2.0)));
    y[linear] = static_cast<float>(gelu);
  }
};

} // namespace

extern "C" void solve(const float *x, const float *gamma, const float *beta, const float *W,
                      const float *bias, float *y, int M, int K, int N, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  float *normalized = sycl::malloc_device<float>(static_cast<size_t>(M) * K, q);
  q.parallel_for(sycl::range<1>(M), LayerNormKernel{x, gamma, beta, normalized, K, eps});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * N),
                 LinearGeluKernel{normalized, W, bias, y, K, N});
  q.wait();
  sycl::free(normalized, q);
}
