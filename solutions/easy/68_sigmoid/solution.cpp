#include <sycl/sycl.hpp>

namespace {

struct SigmoidKernel {
  const float *X;
  float *Y;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      Y[i] = 1.0f / (1.0f + sycl::exp(-X[i]));
    }
  }
};

} // namespace

extern "C" void solve(const float *X, float *Y, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, SigmoidKernel{X, Y, N});
  q.wait();
}
