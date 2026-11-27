#include <sycl/sycl.hpp>

namespace {

struct GegluKernel {
  const float *input;
  float *output;
  int halfN;

  void operator()(sycl::nd_item<1> item) const {
    const int i = static_cast<int>(item.get_global_id(0));
    if (i < halfN) {
      const float x1 = input[i];
      const float x2 = input[halfN + i];
      constexpr float invSqrt2 = 0.70710678118654752440f;
      const float gelu = 0.5f * x2 * (1.0f + sycl::erf(x2 * invSqrt2));
      output[i] = x1 * gelu;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  const int halfN = N / 2;
  if (halfN <= 0) {
    return;
  }

  sycl::queue q{sycl::property::queue::in_order()};
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (halfN + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, GegluKernel{input, output, halfN});
  q.wait();
}
