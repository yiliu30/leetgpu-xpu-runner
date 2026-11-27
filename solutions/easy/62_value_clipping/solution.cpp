#include <sycl/sycl.hpp>

namespace {

struct ValueClippingKernel {
  const float *input;
  float *output;
  float lo;
  float hi;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = static_cast<int>(item.get_global_id(0));
    if (i < N) {
      float value = input[i];
      value = value < lo ? lo : value;
      value = value > hi ? hi : value;
      output[i] = value;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, float lo, float hi, int N) {
  if (N <= 0) {
    return;
  }

  sycl::queue q{sycl::property::queue::in_order()};
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, ValueClippingKernel{input, output, lo, hi, N});
  q.wait();
}
