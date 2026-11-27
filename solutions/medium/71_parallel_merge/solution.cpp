#include <sycl/sycl.hpp>

namespace {

struct MergeFromAKernel {
  const float *A;
  const float *B;
  float *C;
  int N;

  void operator()(sycl::id<1> globalId) const {
    const int i = static_cast<int>(globalId[0]);
    const float value = A[i];
    int lo = 0;
    int hi = N;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (B[mid] < value) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    C[i + lo] = value;
  }
};

struct MergeFromBKernel {
  const float *A;
  const float *B;
  float *C;
  int M;

  void operator()(sycl::id<1> globalId) const {
    const int j = static_cast<int>(globalId[0]);
    const float value = B[j];
    int lo = 0;
    int hi = M;
    while (lo < hi) {
      const int mid = (lo + hi) / 2;
      if (A[mid] <= value) {
        lo = mid + 1;
      } else {
        hi = mid;
      }
    }
    C[j + lo] = value;
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *C, int M, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M > 0) {
    q.parallel_for(sycl::range<1>(M), MergeFromAKernel{A, B, C, N});
  }
  if (N > 0) {
    q.parallel_for(sycl::range<1>(N), MergeFromBKernel{A, B, C, M});
  }
  q.wait();
}
