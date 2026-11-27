#include <sycl/sycl.hpp>

namespace {

struct WeightDequantizationKernel {
  const float *X;
  const float *S;
  float *Y;
  int M;
  int N;
  int TILE_SIZE;
  int scaleCols;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / N;
    const int col = index % N;
    const float scale = S[(row / TILE_SIZE) * scaleCols + (col / TILE_SIZE)];
    Y[index] = X[index] * scale;
  }
};

} // namespace

extern "C" void solve(const float *X, const float *S, float *Y, int M, int N,
                      int TILE_SIZE) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = M * N;
  const int scaleCols = (N + TILE_SIZE - 1) / TILE_SIZE;
  if (total > 0) {
    q.parallel_for(
        sycl::range<1>(total),
        WeightDequantizationKernel{X, S, Y, M, N, TILE_SIZE, scaleCols});
  }
  q.wait();
}
