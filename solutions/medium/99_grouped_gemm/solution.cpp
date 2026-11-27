#include <sycl/sycl.hpp>

namespace {

struct GroupedGemmKernel {
  const float *A;
  const float *B;
  const int *group_offsets;
  float *C;
  int G;
  int K;
  int N;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int row = static_cast<int>(idx / static_cast<std::size_t>(N));
    const int col = static_cast<int>(idx % static_cast<std::size_t>(N));

    int g = -1;
    for (int i = 0; i < G; ++i) {
      if (row >= group_offsets[i] && row < group_offsets[i + 1]) {
        g = i;
        break;
      }
    }
    if (g < 0) {
      C[idx] = 0.0f;
      return;
    }

    const float *arow = A + static_cast<std::size_t>(row) * K;
    const float *bg = B + static_cast<std::size_t>(g) * K * N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += arow[k] * bg[static_cast<std::size_t>(k) * N + col];
    }
    C[idx] = acc;
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, const int *group_offsets,
                      float *C, int G, int M_total, int K, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(M_total) * N),
                 GroupedGemmKernel{A, B, group_offsets, C, G, K, N});
  q.wait();
}
