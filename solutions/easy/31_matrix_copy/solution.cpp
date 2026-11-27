#include <sycl/sycl.hpp>

namespace {

struct MatrixCopyKernel {
  const float *A;
  float *B;
  int total;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < total) {
      B[i] = A[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *A, float *B, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = N * N;
  const int workGroupSize = 256;
  const int numWorkGroups = (total + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, MatrixCopyKernel{A, B, total});
  q.wait();
}
