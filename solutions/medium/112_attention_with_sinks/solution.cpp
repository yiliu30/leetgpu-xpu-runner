#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct AttentionSinksKernel {
  sycl::local_accessor<float, 1> weights;
  sycl::local_accessor<float, 1> max_scratch;
  sycl::local_accessor<double, 1> sum_scratch;
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int d;
  int num_sinks;
  int window_size;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *q_row = Q + static_cast<long>(i) * d;

    float local_max = -3.402823466e+38F;
    for (int j = lid; j <= i; j += kWorkGroupSize) {
      if (j < num_sinks || j >= i - window_size + 1) {
        const float *k_row = K + static_cast<long>(j) * d;
        float dot = 0.0f;
        for (int c = 0; c < d; ++c) {
          dot += q_row[c] * k_row[c];
        }
        local_max = sycl::max(local_max, dot * scale);
      }
    }
    max_scratch[lid] = local_max;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        max_scratch[lid] =
            sycl::max(max_scratch[lid], max_scratch[lid + stride]);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const float row_max = max_scratch[0];

    double local_sum = 0.0;
    for (int j = lid; j <= i; j += kWorkGroupSize) {
      if (j < num_sinks || j >= i - window_size + 1) {
        const float *k_row = K + static_cast<long>(j) * d;
        float dot = 0.0f;
        for (int c = 0; c < d; ++c) {
          dot += q_row[c] * k_row[c];
        }
        const float weight = sycl::exp(dot * scale - row_max);
        weights[j] = weight;
        local_sum += static_cast<double>(weight);
      } else {
        weights[j] = 0.0f;
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

    float *out_row = output + static_cast<long>(i) * d;
    for (int c = lid; c < d; c += kWorkGroupSize) {
      double acc = 0.0;
      for (int j = 0; j <= i; ++j) {
        acc += static_cast<double>(weights[j]) *
               static_cast<double>(V[static_cast<long>(j) * d + c]);
      }
      out_row[c] = static_cast<float>(acc / denom);
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int d, int num_sinks,
                      int window_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> weights(sycl::range<1>(M), h);
    sycl::local_accessor<float, 1> max_scratch(sycl::range<1>(kWorkGroupSize),
                                               h);
    sycl::local_accessor<double, 1> sum_scratch(sycl::range<1>(kWorkGroupSize),
                                                h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M * kWorkGroupSize),
                                     sycl::range<1>(kWorkGroupSize)),
                   AttentionSinksKernel{weights, max_scratch, sum_scratch, Q, K,
                                        V, output, d, num_sinks, window_size,
                                        scale});
  });
  q.wait();
}
