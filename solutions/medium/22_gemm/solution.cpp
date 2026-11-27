#include <sycl/sycl.hpp>

namespace {

struct GemmKernel {
  const sycl::half *A;
  const sycl::half *B;
  sycl::half *C;
  int M;
  int N;
  int K;
  float alpha;
  float beta;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int row = linear / N;
    const int col = linear - row * N;
    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
      sum += static_cast<float>(A[row * K + k]) *
             static_cast<float>(B[k * N + col]);
    }
    const float value = alpha * sum + beta * static_cast<float>(C[linear]);
    C[linear] = static_cast<sycl::half>(value);
  }
};

} // namespace

extern "C" void solve(const sycl::half *A, const sycl::half *B, sycl::half *C,
                      int M, int N, int K, float alpha, float beta) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(M) * N),
                 GemmKernel{A, B, C, M, N, K, alpha, beta});
  q.wait();
}
