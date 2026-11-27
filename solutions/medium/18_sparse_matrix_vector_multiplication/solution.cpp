#include <sycl/sycl.hpp>

namespace {

struct SpmvKernel {
  const float *A;
  const float *x;
  float *y;
  int M;
  int N;

  void operator()(sycl::id<1> idx) const {
    const int row = static_cast<int>(idx[0]);
    float sum = 0.0f;
    for (int col = 0; col < N; ++col) {
      sum += A[row * N + col] * x[col];
    }
    y[row] = sum;
  }
};

} // namespace

extern "C" void solve(const float *A, const float *x, float *y, int M, int N,
                      int nnz) {
  (void)nnz;
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(M), SpmvKernel{A, x, y, M, N});
  q.wait();
}
