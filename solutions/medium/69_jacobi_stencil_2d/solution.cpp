#include <sycl/sycl.hpp>

namespace {

struct JacobiStencilKernel {
  const float *input;
  float *output;
  int rows;
  int cols;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / cols;
    const int col = index % cols;
    if (row == 0 || row == rows - 1 || col == 0 || col == cols - 1) {
      output[index] = input[index];
      return;
    }
    output[index] = 0.25f * (input[index - cols] + input[index + cols] +
                             input[index - 1] + input[index + 1]);
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int rows, int cols) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = rows * cols;
  if (total > 0) {
    q.parallel_for(sycl::range<1>(total), JacobiStencilKernel{input, output, rows, cols});
  }
  q.wait();
}
