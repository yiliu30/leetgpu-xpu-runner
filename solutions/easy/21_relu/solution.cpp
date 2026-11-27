#include <sycl/sycl.hpp>

namespace {

struct ReluKernel {
  const float *input;
  float *output;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      const float x = input[i];
      output[i] = x > 0.0f ? x : 0.0f;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, ReluKernel{input, output, N});
  q.wait();
}
