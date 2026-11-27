#include <sycl/sycl.hpp>

namespace {

struct LayerNormDxKernel {
  const float *x;
  const float *gamma;
  const float *mean;
  const float *rstd;
  const float *dy;
  float *dx;
  int C;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const size_t base = static_cast<size_t>(row) * C;
    const double row_mean = static_cast<double>(mean[row]);
    const double scale = static_cast<double>(rstd[row]);
    double sum_g = 0.0;
    double sum_g_xhat = 0.0;
    for (int col = 0; col < C; ++col) {
      const double xhat =
          (static_cast<double>(x[base + col]) - row_mean) * scale;
      const double g =
          static_cast<double>(dy[base + col]) * static_cast<double>(gamma[col]);
      sum_g += g;
      sum_g_xhat += g * xhat;
    }
    const double mean_g = sum_g / C;
    const double proj = sum_g_xhat / C;
    for (int col = 0; col < C; ++col) {
      const double xhat =
          (static_cast<double>(x[base + col]) - row_mean) * scale;
      const double g =
          static_cast<double>(dy[base + col]) * static_cast<double>(gamma[col]);
      dx[base + col] = static_cast<float>(scale * (g - mean_g - xhat * proj));
    }
  }
};

struct LayerNormParamGradKernel {
  const float *x;
  const float *mean;
  const float *rstd;
  const float *dy;
  float *dgamma;
  float *dbeta;
  int N;
  int C;

  void operator()(sycl::id<1> id) const {
    const int col = static_cast<int>(id[0]);
    if (col < C) {
      double gamma_sum = 0.0;
      double beta_sum = 0.0;
      for (int row = 0; row < N; ++row) {
        const size_t index = static_cast<size_t>(row) * C + col;
        const double xhat =
            (static_cast<double>(x[index]) - static_cast<double>(mean[row])) *
            static_cast<double>(rstd[row]);
        const double grad = static_cast<double>(dy[index]);
        gamma_sum += grad * xhat;
        beta_sum += grad;
      }
      dgamma[col] = static_cast<float>(gamma_sum);
      dbeta[col] = static_cast<float>(beta_sum);
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *gamma, const float *mean,
                      const float *rstd, const float *dy, float *dx,
                      float *dgamma, float *dbeta, int N, int C) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(N),
                 LayerNormDxKernel{x, gamma, mean, rstd, dy, dx, C});
  q.parallel_for(
      sycl::range<1>(C),
      LayerNormParamGradKernel{x, mean, rstd, dy, dgamma, dbeta, N, C});
  q.wait();
}
