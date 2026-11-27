#include <sycl/sycl.hpp>

namespace {

struct GeluKernel {
  const float *X;
  float *Y;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      const float x = X[i];
      Y[i] = 0.5f * x * (1.0f + sycl::erf(x * 0.70710678118654752440f));
    }
  }
};

} // namespace

extern "C" void solve(const float *X, float *Y, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, GeluKernel{X, Y, N});
  q.wait();
}
