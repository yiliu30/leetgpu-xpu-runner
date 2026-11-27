#include <sycl/sycl.hpp>

namespace {

struct Conv2DKernel {
  const float *input;
  const float *kernel;
  float *output;
  int input_cols;
  int kernel_rows;
  int kernel_cols;
  int output_cols;

  void operator()(sycl::id<1> idx) const {
    const long out_idx = idx[0];
    const int oh = static_cast<int>(out_idx / output_cols);
    const int ow =
        static_cast<int>(out_idx - static_cast<long>(oh) * output_cols);
    double sum = 0.0;
    for (int kh = 0; kh < kernel_rows; ++kh) {
      for (int kw = 0; kw < kernel_cols; ++kw) {
        const long in_idx = static_cast<long>(oh + kh) * input_cols + (ow + kw);
        sum += static_cast<double>(input[in_idx]) *
               static_cast<double>(kernel[kh * kernel_cols + kw]);
      }
    }
    output[out_idx] = static_cast<float>(sum);
  }
};

} // namespace

extern "C" void solve(const float *input, const float *kernel, float *output,
                      int input_rows, int input_cols, int kernel_rows,
                      int kernel_cols) {
  (void)input_rows;
  sycl::queue q{sycl::property::queue::in_order()};
  const int output_rows = input_rows - kernel_rows + 1;
  const int output_cols = input_cols - kernel_cols + 1;
  q.parallel_for(sycl::range<1>(static_cast<long>(output_rows) * output_cols),
                 Conv2DKernel{input, kernel, output, input_cols, kernel_rows,
                              kernel_cols, output_cols});
  q.wait();
}
