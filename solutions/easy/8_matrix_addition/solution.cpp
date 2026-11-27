#include <sycl/sycl.hpp>

namespace {

struct MatrixAdditionKernel {
  const float *A;
  const float *B;
  float *C;
  int numElements;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < numElements) {
      C[i] = A[i] + B[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *C, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numElements = N * N;
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (numElements + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, MatrixAdditionKernel{A, B, C, numElements});
  q.wait();
}
