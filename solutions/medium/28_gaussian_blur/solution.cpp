#include <sycl/sycl.hpp>

namespace {

struct GaussianBlurKernel {
  const float *input;
  const float *kernel;
  float *output;
  int inputRows;
  int inputCols;
  int kernelRows;
  int kernelCols;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int row = linear / inputCols;
    const int col = linear - row * inputCols;
    const int padH = kernelRows / 2;
    const int padW = kernelCols / 2;
    double sum = 0.0;
    for (int kr = 0; kr < kernelRows; ++kr) {
      const int inRow = row + kr - padH;
      if (inRow < 0 || inRow >= inputRows) {
        continue;
      }
      for (int kc = 0; kc < kernelCols; ++kc) {
        const int inCol = col + kc - padW;
        if (inCol < 0 || inCol >= inputCols) {
          continue;
        }
        sum += static_cast<double>(input[inRow * inputCols + inCol]) *
               static_cast<double>(kernel[kr * kernelCols + kc]);
      }
    }
    output[linear] = static_cast<float>(sum);
  }
};

} // namespace

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(input_rows) * input_cols),
                 GaussianBlurKernel{input, kernel, output, input_rows, input_cols,
                                    kernel_rows, kernel_cols});
  q.wait();
}
