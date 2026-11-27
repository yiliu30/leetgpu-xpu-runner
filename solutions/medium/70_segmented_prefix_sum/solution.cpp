#include <sycl/sycl.hpp>

namespace {

struct SegmentedPrefixSumKernel {
  const float *values;
  const int *flags;
  float *output;
  int N;

  void operator()(sycl::item<1> item) const {
    float running = 0.0f;
    for (int i = 0; i < N; ++i) {
      if (flags[i] != 0) {
        running = 0.0f;
      }
      output[i] = running;
      running += values[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *values, const int *flags, float *output,
                      int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (N > 0) {
    q.parallel_for(sycl::range<1>(1),
                   SegmentedPrefixSumKernel{values, flags, output, N});
  }
  q.wait();
}
