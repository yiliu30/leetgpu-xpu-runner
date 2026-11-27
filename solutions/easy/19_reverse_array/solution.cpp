#include <sycl/sycl.hpp>

namespace {

struct ReverseArrayKernel {
  float *input;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < N / 2) {
      const int j = N - 1 - i;
      const float tmp = input[i];
      input[i] = input[j];
      input[j] = tmp;
    }
  }
};

} // namespace

extern "C" void solve(float *input, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, ReverseArrayKernel{input, N});
  q.wait();
}
