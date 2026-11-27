#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct PpoLossKernel {
  sycl::local_accessor<double, 1> scratch;
  const float *advantages;
  const float *log_pi;
  const float *log_pi_old;
  float *output;
  float clip_eps;
  int total;

  void operator()(sycl::nd_item<1> item) const {
    const int lid = item.get_local_id(0);
    double local_sum = 0.0;
    for (int i = lid; i < total; i += kWorkGroupSize) {
      const float ratio = sycl::exp(log_pi[i] - log_pi_old[i]);
      const float clipped =
          sycl::min(sycl::max(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
      const float surrogate =
          sycl::min(ratio * advantages[i], clipped * advantages[i]);
      local_sum += static_cast<double>(surrogate);
    }
    scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0) {
      output[0] = static_cast<float>(-scratch[0] / static_cast<double>(total));
    }
  }
};

} // namespace

extern "C" void solve(const float *advantages, const float *log_pi,
                      const float *log_pi_old, float *output, float clip_eps,
                      int B, int S) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = B * S;
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWorkGroupSize),
                                     sycl::range<1>(kWorkGroupSize)),
                   PpoLossKernel{scratch, advantages, log_pi, log_pi_old,
                                 output, clip_eps, total});
  });
  q.wait();
}
