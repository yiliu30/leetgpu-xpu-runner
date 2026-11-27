#include <sycl/sycl.hpp>

namespace {

struct IntegralImageRowsKernel {
  const float *input;
  float *output;
  int W;

  void operator()(sycl::id<1> globalId) const {
    const int row = static_cast<int>(globalId[0]);
    const int offset = row * W;
    float running = 0.0f;
    for (int col = 0; col < W; ++col) {
      running += input[offset + col];
      output[offset + col] = running;
    }
  }
};

struct IntegralImageColsKernel {
  float *output;
  int H;
  int W;

  void operator()(sycl::id<1> globalId) const {
    const int col = static_cast<int>(globalId[0]);
    float running = 0.0f;
    for (int row = 0; row < H; ++row) {
      const int index = row * W + col;
      running += output[index];
      output[index] = running;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int H, int W) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (H > 0 && W > 0) {
    q.parallel_for(sycl::range<1>(H), IntegralImageRowsKernel{input, output, W});
    q.parallel_for(sycl::range<1>(W), IntegralImageColsKernel{output, H, W});
  }
  q.wait();
}
