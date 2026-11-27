#include <sycl/sycl.hpp>

namespace {

struct Conv3DKernel {
  const float *input;
  const float *kernel;
  float *output;
  int input_rows;
  int input_cols;
  int kernel_depth;
  int kernel_rows;
  int kernel_cols;
  int output_rows;
  int output_cols;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int out_col = static_cast<int>(linear % output_cols);
    const int out_row = static_cast<int>((linear / output_cols) % output_rows);
    const int out_depth = static_cast<int>(
        linear / (static_cast<size_t>(output_rows) * output_cols));
    double sum = 0.0;
    for (int kd = 0; kd < kernel_depth; ++kd) {
      for (int kr = 0; kr < kernel_rows; ++kr) {
        for (int kc = 0; kc < kernel_cols; ++kc) {
          const size_t input_index =
              (static_cast<size_t>(out_depth + kd) * input_rows + out_row +
               kr) *
                  input_cols +
              out_col + kc;
          const size_t kernel_index =
              (static_cast<size_t>(kd) * kernel_rows + kr) * kernel_cols + kc;
          sum += static_cast<double>(input[input_index]) *
                 static_cast<double>(kernel[kernel_index]);
        }
      }
    }
    output[linear] = static_cast<float>(sum);
  }
};

} // namespace

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_depth, int input_rows, int input_cols,
                      int kernel_depth, int kernel_rows, int kernel_cols) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int output_depth = input_depth - kernel_depth + 1;
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(output_depth) *
                                output_rows * output_cols),
                 Conv3DKernel{input, kernel, output, input_rows, input_cols,
                              kernel_depth, kernel_rows, kernel_cols,
                              output_rows, output_cols});
  q.wait();
}
