#include <sycl/sycl.hpp>

namespace {

struct RgbToGrayscaleKernel {
  const float *input;
  float *output;
  int totalPixels;

  void operator()(sycl::nd_item<1> item) const {
    const int i = static_cast<int>(item.get_global_id(0));
    if (i < totalPixels) {
      const int base = 3 * i;
      const float red = input[base];
      const float green = input[base + 1];
      const float blue = input[base + 2];
      output[i] = 0.299f * red + 0.587f * green + 0.114f * blue;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int width, int height) {
  const int totalPixels = width * height;
  if (totalPixels <= 0) {
    return;
  }

  sycl::queue q{sycl::property::queue::in_order()};
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (totalPixels + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, RgbToGrayscaleKernel{input, output, totalPixels});
  q.wait();
}
