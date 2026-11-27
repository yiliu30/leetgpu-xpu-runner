#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct FusedResidualLayerNormKernel {
  sycl::local_accessor<float, 1> part;
  sycl::local_accessor<float, 1> stats;
  const float *x;
  const float *residual;
  const float *gamma;
  const float *beta;
  float *y;
  float *residual_out;
  float *mean;
  float *rstd;
  int C;
  float eps;

  void operator()(sycl::nd_item<1> item) const {
    const int row = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int wg = static_cast<int>(item.get_local_range(0));
    const std::size_t base = static_cast<std::size_t>(row) * C;

    float sum = 0.0f;
    float comp = 0.0f;
    for (int j = tid; j < C; j += wg) {
      const float h = x[base + j] + residual[base + j];
      residual_out[base + j] = h;
      const float corrected = h - comp;
      const float next = sum + corrected;
      comp = (next - sum) - corrected;
      sum = next;
    }
    part[tid] = sum;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float total = 0.0f;
      float c = 0.0f;
      for (int i = 0; i < wg; ++i) {
        const float corrected = part[i] - c;
        const float next = total + corrected;
        c = (next - total) - corrected;
        total = next;
      }
      stats[0] = total / static_cast<float>(C);
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float mu = stats[0];
    float var_sum = 0.0f;
    float var_comp = 0.0f;
    for (int j = tid; j < C; j += wg) {
      const float diff = residual_out[base + j] - mu;
      const float term = diff * diff;
      const float corrected = term - var_comp;
      const float next = var_sum + corrected;
      var_comp = (next - var_sum) - corrected;
      var_sum = next;
    }
    part[tid] = var_sum;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float total = 0.0f;
      float c = 0.0f;
      for (int i = 0; i < wg; ++i) {
        const float corrected = part[i] - c;
        const float next = total + corrected;
        c = (next - total) - corrected;
        total = next;
      }
      const float variance = total / static_cast<float>(C);
      const float inv_std = 1.0f / sycl::sqrt(variance + eps);
      stats[1] = inv_std;
      mean[row] = mu;
      rstd[row] = inv_std;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float inv_std = stats[1];
    for (int j = tid; j < C; j += wg) {
      y[base + j] =
          (residual_out[base + j] - mu) * inv_std * gamma[j] + beta[j];
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *residual, const float *gamma,
                      const float *beta, float *y, float *residual_out,
                      float *mean, float *rstd, int N, int C, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (N > 0 && C > 0) {
    const sycl::range<1> local(kWorkGroupSize);
    const sycl::range<1> global(static_cast<std::size_t>(N) * kWorkGroupSize);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> part(sycl::range<1>(kWorkGroupSize), h);
      sycl::local_accessor<float, 1> stats(sycl::range<1>(2), h);
      h.parallel_for(sycl::nd_range<1>(global, local),
                     FusedResidualLayerNormKernel{part, stats, x, residual,
                                                  gamma, beta, y, residual_out,
                                                  mean, rstd, C, eps});
    });
  }
  q.wait();
}
