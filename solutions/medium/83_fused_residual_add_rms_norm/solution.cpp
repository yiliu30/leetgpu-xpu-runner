#include <sycl/sycl.hpp>

namespace {

struct FusedResidualAddRmsNormKernel {
  const float *x;
  const float *residual;
  const float *weight;
  float *out;
  int C;
  float eps;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const std::size_t base = static_cast<std::size_t>(row) * C;
    float sum_sq = 0.0f;
    for (int c = 0; c < C; ++c) {
      const float z = x[base + c] + residual[base + c];
      sum_sq += z * z;
    }
    const float rms = sycl::sqrt(sum_sq / static_cast<float>(C) + eps);
    for (int c = 0; c < C; ++c) {
      const float z = x[base + c] + residual[base + c];
      out[base + c] = z / rms * weight[c];
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *residual, const float *weight, float *out, int N,
                      int C, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(N)),
                 FusedResidualAddRmsNormKernel{x, residual, weight, out, C, eps});
  q.wait();
}
