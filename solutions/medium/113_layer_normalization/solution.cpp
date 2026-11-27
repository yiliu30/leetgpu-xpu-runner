#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct LayerNormKernel {
  sycl::local_accessor<double, 1> scratch;
  const float *input;
  const float *weight;
  const float *bias;
  float *output;
  int C;
  float eps;

  void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *src = input + static_cast<long>(row) * C;
    float *dst = output + static_cast<long>(row) * C;

    double local_sum = 0.0;
    for (int c = lid; c < C; c += kWorkGroupSize) {
      local_sum += static_cast<double>(src[c]);
    }
    scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double mean = scratch[0] / static_cast<double>(C);

    double local_var = 0.0;
    for (int c = lid; c < C; c += kWorkGroupSize) {
      const double d = static_cast<double>(src[c]) - mean;
      local_var += d * d;
    }
    scratch[lid] = local_var;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double inv_std =
        1.0 / sycl::sqrt(scratch[0] / static_cast<double>(C) + static_cast<double>(eps));

    for (int c = lid; c < C; c += kWorkGroupSize) {
      const double normalized = (static_cast<double>(src[c]) - mean) * inv_std;
      dst[c] = static_cast<float>(static_cast<double>(weight[c]) * normalized +
                                  static_cast<double>(bias[c]));
    }
  }
};

} // namespace

extern "C" void solve(const float *input, const float *weight, const float *bias, float *output,
                      int N, int C, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(N * kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        LayerNormKernel{scratch, input, weight, bias, output, C, eps});
  });
  q.wait();
}
