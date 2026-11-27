#include <sycl/sycl.hpp>

namespace {

struct ColorInversionKernel {
  unsigned char *image;
  int numPixels;

  void operator()(sycl::nd_item<1> item) const {
    const int pixel = item.get_global_id(0);
    if (pixel < numPixels) {
      const int base = pixel * 4;
      image[base] = static_cast<unsigned char>(255 - image[base]);
      image[base + 1] = static_cast<unsigned char>(255 - image[base + 1]);
      image[base + 2] = static_cast<unsigned char>(255 - image[base + 2]);
    }
  }
};

} // namespace

extern "C" void solve(unsigned char *image, int width, int height) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numPixels = width * height;
  constexpr int workGroupSize = 256;
  const int numWorkGroups = (numPixels + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));

  q.parallel_for(launchRange, ColorInversionKernel{image, numPixels});
  q.wait();
}
