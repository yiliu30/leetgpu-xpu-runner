#include <sycl/sycl.hpp>

namespace {

struct SwigluKernel {
  const float *input;
  float *output;
  int halfN;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < halfN) {
      const float x1 = input[i];
      const float x2 = input[i + halfN];
      output[i] = (x1 / (1.0f + sycl::exp(-x1))) * x2;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int halfN = N / 2;
  const int workGroupSize = 256;
  const int numWorkGroups = (halfN + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, SwigluKernel{input, output, halfN});
  q.wait();
}
