#include <sycl/sycl.hpp>

namespace {

struct GroupNormKernel {
  const float *X;
  const float *gamma;
  const float *beta;
  float *Y;
  int C;
  int H;
  int W;
  int G;
  float eps;

  void operator()(sycl::id<1> idx) const {
    const int ng = idx[0];
    const int n = ng / G;
    const int g = ng - n * G;
    const int channels_per_group = C / G;
    const int spatial = H * W;
    const long group_elems = static_cast<long>(channels_per_group) * spatial;
    const long base =
        (static_cast<long>(n) * C + static_cast<long>(g) * channels_per_group) *
        spatial;

    double sum = 0.0;
    for (long k = 0; k < group_elems; ++k) {
      sum += static_cast<double>(X[base + k]);
    }
    const double mean = sum / static_cast<double>(group_elems);

    double var_sum = 0.0;
    for (long k = 0; k < group_elems; ++k) {
      const double d = static_cast<double>(X[base + k]) - mean;
      var_sum += d * d;
    }
    const double inv_std =
        1.0 / sycl::sqrt(var_sum / static_cast<double>(group_elems) +
                         static_cast<double>(eps));

    for (int c_local = 0; c_local < channels_per_group; ++c_local) {
      const int c = g * channels_per_group + c_local;
      const double scale = static_cast<double>(gamma[c]) * inv_std;
      const double shift = static_cast<double>(beta[c]) - mean * scale;
      const long channel_base = base + static_cast<long>(c_local) * spatial;
      for (int s = 0; s < spatial; ++s) {
        Y[channel_base + s] = static_cast<float>(
            static_cast<double>(X[channel_base + s]) * scale + shift);
      }
    }
  }
};

} // namespace

extern "C" void solve(const float *X, const float *gamma, const float *beta,
                      float *Y, int N, int C, int H, int W, int G, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(N * G),
                 GroupNormKernel{X, gamma, beta, Y, C, H, W, G, eps});
  q.wait();
}
