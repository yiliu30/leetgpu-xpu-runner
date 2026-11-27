#include <limits>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct AttentionWithLinearBiasKernel {
  sycl::local_accessor<float, 1> scores;
  sycl::local_accessor<float, 1> scratch;
  const float *q;
  const float *k;
  const float *v;
  float *output;
  int m;
  int n;
  int d;
  float alpha;

  void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    const int lid = item.get_local_id(0);
    const int localSize = item.get_local_range(0);
    const float scale = sycl::sqrt(static_cast<float>(d));

    for (int j = lid; j < n; j += localSize) {
      float dot = 0.0f;
      for (int t = 0; t < d; ++t) {
        dot += q[row * d + t] * k[j * d + t];
      }
      scores[j] = dot / scale + alpha * static_cast<float>(row - j);
    }
    item.barrier(sycl::access::fence_space::local_space);

    float localMax = -std::numeric_limits<float>::infinity();
    for (int j = lid; j < n; j += localSize) {
      localMax = sycl::max(localMax, scores[j]);
    }
    scratch[lid] = localMax;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] = sycl::max(scratch[lid], scratch[lid + stride]);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const float maxScore = scratch[0];

    float localSum = 0.0f;
    for (int j = lid; j < n; j += localSize) {
      const float value = sycl::exp(scores[j] - maxScore);
      scores[j] = value;
      localSum += value;
    }
    scratch[lid] = localSum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = localSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const float denominator = scratch[0];

    for (int col = lid; col < d; col += localSize) {
      float acc = 0.0f;
      for (int j = 0; j < n; ++j) {
        acc += (scores[j] / denominator) * v[j * d + col];
      }
      output[row * d + col] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int N, int d, float alpha) {
  sycl::queue q{sycl::property::queue::in_order()};
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(static_cast<std::size_t>(M) * WORK_GROUP_SIZE);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scores(sycl::range<1>(N), h);
    sycl::local_accessor<float, 1> scratch(sycl::range<1>(WORK_GROUP_SIZE), h);
    h.parallel_for(sycl::nd_range<1>(globalSize, workGroupSize),
                   AttentionWithLinearBiasKernel{scores, scratch, Q, K, V,
                                                 output, M, N, d, alpha});
  });
  q.wait();
}
