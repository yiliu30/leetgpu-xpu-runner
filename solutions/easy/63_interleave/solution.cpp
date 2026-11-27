#include <sycl/sycl.hpp>

namespace {

struct InterleaveKernel {
  const float *A;
  const float *B;
  float *output;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int i = static_cast<int>(item.get_global_id(0));
    if (i < N) {
      output[2 * i] = A[i];
      output[2 * i + 1] = B[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *output, int N) {
  if (N <= 0) {
    return;
  }

  sycl::queue q{sycl::property::queue::in_order()};
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, InterleaveKernel{A, B, output, N});
  q.wait();
}
