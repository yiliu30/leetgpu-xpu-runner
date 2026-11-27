#include <sycl/sycl.hpp>

#include <cstdint>

namespace {

struct Int8MatMulKernel {
  const int8_t *A;
  const int8_t *B;
  int8_t *C;
  int M;
  int N;
  int K;
  float scaleA;
  float scaleB;
  float scaleC;
  int zeroPointA;
  int zeroPointB;
  int zeroPointC;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int row = linear / N;
    const int col = linear - row * N;
    int32_t accum = 0;
    for (int k = 0; k < K; ++k) {
      accum += (static_cast<int32_t>(A[row * K + k]) - zeroPointA) *
               (static_cast<int32_t>(B[k * N + col]) - zeroPointB);
    }
    const float scaled =
        static_cast<float>(accum) * scaleA * scaleB / scaleC;
    int32_t quantized =
        static_cast<int32_t>(sycl::rint(scaled)) + zeroPointC;
    quantized = sycl::clamp(quantized, -128, 127);
    C[linear] = static_cast<int8_t>(quantized);
  }
};

} // namespace

extern "C" void solve(const int8_t *A, const int8_t *B, int8_t *C, int M, int N,
                      int K, float scale_A, float scale_B, float scale_C,
                      int zero_point_A, int zero_point_B, int zero_point_C) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(
      sycl::range<1>(static_cast<std::size_t>(M) * N),
      Int8MatMulKernel{A, B, C, M, N, K, scale_A, scale_B, scale_C,
                       zero_point_A, zero_point_B, zero_point_C});
  q.wait();
}
