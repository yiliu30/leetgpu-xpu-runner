#include <sycl/sycl.hpp>

namespace {

struct Convolution1DKernel {
  const float *input;
  const float *filter;
  float *output;
  int outputSize;
  int kernelSize;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < outputSize) {
      float sum = 0.0f;
      for (int j = 0; j < kernelSize; ++j) {
        sum += input[i + j] * filter[j];
      }
      output[i] = sum;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, const float *kernel, float *output, int input_size,
                      int kernel_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int outputSize = input_size - kernel_size + 1;
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (outputSize + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange,
                 Convolution1DKernel{input, kernel, output, outputSize, kernel_size});
  q.wait();
}
