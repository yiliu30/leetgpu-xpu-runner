#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct BatchNormStatsKernel {
  const float *input;
  float *mean;
  float *invStd;
  int n;
  int c;
  float eps;

  void operator()(sycl::id<1> idx) const {
    const int channel = idx[0];
    if (channel >= c) {
      return;
    }

    float runningMean = 0.0f;
    float runningM2 = 0.0f;
    for (int i = 0; i < n; ++i) {
      const float value = input[i * c + channel];
      const float delta = value - runningMean;
      runningMean += delta / static_cast<float>(i + 1);
      runningM2 += delta * (value - runningMean);
    }

    const float variance = runningM2 / static_cast<float>(n);
    mean[channel] = runningMean;
    invStd[channel] = 1.0f / sycl::sqrt(variance + eps);
  }
};

struct BatchNormApplyKernel {
  const float *input;
  const float *gamma;
  const float *beta;
  const float *mean;
  const float *invStd;
  float *output;
  int total;
  int c;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i >= total) {
      return;
    }
    const int channel = i % c;
    output[i] =
        gamma[channel] * ((input[i] - mean[channel]) * invStd[channel]) +
        beta[channel];
  }
};

} // namespace

extern "C" void solve(const float *input, const float *gamma, const float *beta,
                      float *output, int N, int C, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int statsWorkGroups = (C + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  const int total = N * C;
  const int applyWorkGroups = (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);

  float *mean = sycl::malloc_device<float>(C, q);
  float *invStd = sycl::malloc_device<float>(C, q);

  q.parallel_for(sycl::range<1>(statsWorkGroups * WORK_GROUP_SIZE),
                 BatchNormStatsKernel{input, mean, invStd, N, C, eps});
  q.parallel_for(
      sycl::range<1>(applyWorkGroups * WORK_GROUP_SIZE),
      BatchNormApplyKernel{input, gamma, beta, mean, invStd, output, total, C});
  q.wait();

  sycl::free(mean, q);
  sycl::free(invStd, q);
}
