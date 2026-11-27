#include <sycl/sycl.hpp>

namespace {

struct SparseDenseMatMulKernel {
  const float *A;
  const float *B;
  float *C;
  int N;
  int K;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / K;
    const int col = index % K;
    float acc = 0.0f;
    for (int inner = 0; inner < N; ++inner) {
      acc += A[row * N + inner] * B[inner * K + col];
    }
    C[index] = acc;
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *C, int M, int N, int K,
                      int nnz) {
  (void)nnz;
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = M * K;
  if (total > 0) {
    q.parallel_for(sycl::range<1>(total), SparseDenseMatMulKernel{A, B, C, N, K});
  }
  q.wait();
}
