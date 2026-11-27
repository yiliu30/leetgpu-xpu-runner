#include <sycl/sycl.hpp>

namespace {

struct RmsRstdKernel {
  const float *x;
  float *rstd;
  int C;
  float eps;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const float *xr = x + static_cast<size_t>(row) * C;
    double sum_sq = 0.0;
    for (int col = 0; col < C; ++col) {
      const double value = static_cast<double>(xr[col]);
      sum_sq += value * value;
    }
    rstd[row] = static_cast<float>(1.0 / sycl::sqrt(sum_sq / C + eps));
  }
};

struct RmsDxKernel {
  const float *x;
  const float *weight;
  const float *dy;
  const float *rstd;
  float *dx;
  int C;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const size_t base = static_cast<size_t>(row) * C;
    const double scale = static_cast<double>(rstd[row]);
    double proj = 0.0;
    for (int col = 0; col < C; ++col) {
      const double xhat = static_cast<double>(x[base + col]) * scale;
      proj += static_cast<double>(dy[base + col]) * static_cast<double>(weight[col]) * xhat;
    }
    proj /= C;
    for (int col = 0; col < C; ++col) {
      const double xhat = static_cast<double>(x[base + col]) * scale;
      const double g = static_cast<double>(dy[base + col]) * static_cast<double>(weight[col]);
      dx[base + col] = static_cast<float>(scale * (g - xhat * proj));
    }
  }
};

struct RmsDweightKernel {
  const float *x;
  const float *dy;
  const float *rstd;
  float *dweight;
  int N;
  int C;

  void operator()(sycl::id<1> id) const {
    const int col = static_cast<int>(id[0]);
    if (col < C) {
      double sum = 0.0;
      for (int row = 0; row < N; ++row) {
        const size_t index = static_cast<size_t>(row) * C + col;
        sum += static_cast<double>(dy[index]) * static_cast<double>(x[index]) *
               static_cast<double>(rstd[row]);
      }
      dweight[col] = static_cast<float>(sum);
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *weight, const float *dy, float *dx,
                      float *dweight, int N, int C, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  float *rstd = sycl::malloc_device<float>(N, q);
  q.parallel_for(sycl::range<1>(N), RmsRstdKernel{x, rstd, C, eps});
  q.parallel_for(sycl::range<1>(N), RmsDxKernel{x, weight, dy, rstd, dx, C});
  q.parallel_for(sycl::range<1>(C), RmsDweightKernel{x, dy, rstd, dweight, N, C});
  q.wait();
  sycl::free(rstd, q);
}
