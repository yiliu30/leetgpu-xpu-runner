#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct DpoLossKernel {
  sycl::local_accessor<double, 1> scratch;
  const float *chosen_logps;
  const float *rejected_logps;
  const float *chosen_ref_logps;
  const float *rejected_ref_logps;
  float *output;
  float beta;
  int B;

  void operator()(sycl::nd_item<1> item) const {
    const int lid = item.get_local_id(0);
    double local_sum = 0.0;
    for (int i = lid; i < B; i += kWorkGroupSize) {
      const float chosen_margin = chosen_logps[i] - rejected_logps[i];
      const float ref_margin = chosen_ref_logps[i] - rejected_ref_logps[i];
      const float logits = beta * (chosen_margin - ref_margin);
      double log_sigmoid;
      if (logits >= 0.0f) {
        log_sigmoid = -sycl::log(1.0 + sycl::exp(-static_cast<double>(logits)));
      } else {
        log_sigmoid = static_cast<double>(logits) -
                      sycl::log(1.0 + sycl::exp(static_cast<double>(logits)));
      }
      local_sum += log_sigmoid;
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
      output[0] = static_cast<float>(-scratch[0] / static_cast<double>(B));
    }
  }
};

} // namespace

extern "C" void solve(const float *chosen_logps, const float *rejected_logps,
                      const float *chosen_ref_logps, const float *rejected_ref_logps, float *output,
                      float beta, int B) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWorkGroupSize),
                                     sycl::range<1>(kWorkGroupSize)),
                   DpoLossKernel{scratch, chosen_logps, rejected_logps, chosen_ref_logps,
                                 rejected_ref_logps, output, beta, B});
  });
  q.wait();
}
