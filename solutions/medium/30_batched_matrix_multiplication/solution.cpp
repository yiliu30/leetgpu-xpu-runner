#include <sycl/sycl.hpp>

namespace {

struct BatchedMatMulKernel {
  const float *A;
  const float *B;
  float *C;
  int BATCH;
  int M;
  int N;
  int K;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int batchSize = M * N;
    const int batch = linear / batchSize;
    const int rem = linear - batch * batchSize;
    const int row = rem / N;
    const int col = rem - row * N;
    const float *a = A + static_cast<std::size_t>(batch) * M * K;
    const float *b = B + static_cast<std::size_t>(batch) * K * N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += a[row * K + k] * b[k * N + col];
    }
    C[static_cast<std::size_t>(batch) * batchSize + rem] = sum;
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *C, int BATCH, int M,
                      int N, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(BATCH) * M * N),
                 BatchedMatMulKernel{A, B, C, BATCH, M, N, K});
  q.wait();
}
