#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

struct Int4MatMulKernel {
  const sycl::half *x;
  const uint8_t *w_q;
  const sycl::half *scales;
  sycl::half *y;
  int N;
  int K;
  int groupSize;
  int groupsPerRow;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / N;
    const int outCol = index % N;
    const int weightRow = outCol * (K / 2);
    const int scaleRow = outCol * groupsPerRow;
    float acc = 0.0f;
    for (int inner = 0; inner < K; ++inner) {
      const uint8_t packed = w_q[weightRow + inner / 2];
      const int nibble = (inner & 1) ? (packed & 0x0F) : ((packed >> 4) & 0x0F);
      const float scale = static_cast<float>(scales[scaleRow + inner / groupSize]);
      acc += static_cast<float>(x[row * K + inner]) * static_cast<float>(nibble - 8) *
             scale;
    }
    y[index] = static_cast<sycl::half>(acc);
  }
};

} // namespace

extern "C" void solve(const sycl::half *x, const uint8_t *w_q, const sycl::half *scales,
                      sycl::half *y, int M, int N, int K, int group_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = M * N;
  if (total > 0) {
    q.parallel_for(sycl::range<1>(total),
                   Int4MatMulKernel{x, w_q, scales, y, N, K, group_size,
                                    K / group_size});
  }
  q.wait();
}
