#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct MinPSamplingKernel {
  sycl::local_accessor<float, 1> max_scratch;
  sycl::local_accessor<double, 1> sum_scratch;
  const float *logits;
  float *probs;
  float min_p;
  int V;

  void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *logit_row = logits + static_cast<long>(row) * V;
    float *prob_row = probs + static_cast<long>(row) * V;

    float local_max = -3.402823466e+38F;
    for (int i = lid; i < V; i += kWorkGroupSize) {
      local_max = sycl::max(local_max, logit_row[i]);
    }
    max_scratch[lid] = local_max;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        max_scratch[lid] = sycl::max(max_scratch[lid], max_scratch[lid + stride]);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const float row_max = max_scratch[0];

    double local_sum = 0.0;
    for (int i = lid; i < V; i += kWorkGroupSize) {
      const float e = sycl::exp(logit_row[i] - row_max);
      if (e >= min_p) {
        local_sum += static_cast<double>(e);
      }
    }
    sum_scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        sum_scratch[lid] += sum_scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double denom = sum_scratch[0];

    for (int i = lid; i < V; i += kWorkGroupSize) {
      const float e = sycl::exp(logit_row[i] - row_max);
      prob_row[i] = (e >= min_p) ? static_cast<float>(static_cast<double>(e) / denom) : 0.0f;
    }
  }
};

} // namespace

extern "C" void solve(const float *logits, float *probs, float min_p, int B, int V) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> max_scratch(sycl::range<1>(kWorkGroupSize), h);
    sycl::local_accessor<double, 1> sum_scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(B * kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        MinPSamplingKernel{max_scratch, sum_scratch, logits, probs, min_p, V});
  });
  q.wait();
}
