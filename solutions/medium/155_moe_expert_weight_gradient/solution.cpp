#include <sycl/sycl.hpp>

namespace {

struct MoeWeightGradientKernel {
  const float *X;
  const float *dY;
  const int *groupOffsets;
  float *dW;
  int G;
  int K;
  int N;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int kn = K * N;
    const int group = linear / kn;
    const int rem = linear - group * kn;
    const int k = rem / N;
    const int n = rem - k * N;
    const int start = groupOffsets[group];
    const int end = groupOffsets[group + 1];
    float sum = 0.0f;
    for (int row = start; row < end; ++row) {
      sum += X[static_cast<std::size_t>(row) * K + k] *
             dY[static_cast<std::size_t>(row) * N + n];
    }
    dW[linear] = sum;
  }
};

} // namespace

extern "C" void solve(const float *X, const float *dY, const int *group_offsets,
                      float *dW, int G, int M_total, int K, int N) {
  (void)M_total;
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(G) * K * N),
                 MoeWeightGradientKernel{X, dY, group_offsets, dW, G, K, N});
  q.wait();
}
