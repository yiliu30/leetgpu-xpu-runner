#include <limits>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct MaxPool2DKernel {
  const float *input;
  float *output;
  int c;
  int h;
  int w;
  int kernelSize;
  int stride;
  int padding;
  int hOut;
  int wOut;
  int total;

  void operator()(sycl::id<1> idx) const {
    const int index = idx[0];
    if (index >= total) {
      return;
    }

    int remaining = index;
    const int ow = remaining % wOut;
    remaining /= wOut;
    const int oh = remaining % hOut;
    remaining /= hOut;
    const int channel = remaining % c;
    const int batch = remaining / c;

    const int hStart = oh * stride - padding;
    const int wStart = ow * stride - padding;
    const int inputBase = ((batch * c + channel) * h) * w;
    float best = -std::numeric_limits<float>::infinity();

    for (int kh = 0; kh < kernelSize; ++kh) {
      const int ih = hStart + kh;
      if (ih < 0 || ih >= h) {
        continue;
      }
      for (int kw = 0; kw < kernelSize; ++kw) {
        const int iw = wStart + kw;
        if (iw < 0 || iw >= w) {
          continue;
        }
        best = sycl::max(best, input[inputBase + ih * w + iw]);
      }
    }
    output[index] = best;
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N, int C, int H,
                      int W, int kernel_size, int stride, int padding) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int hOut = (H + 2 * padding - kernel_size) / stride + 1;
  const int wOut = (W + 2 * padding - kernel_size) / stride + 1;
  const int total = N * C * hOut * wOut;
  const int numWorkGroups = (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.parallel_for(globalSize,
                 MaxPool2DKernel{input, output, C, H, W, kernel_size, stride,
                                 padding, hOut, wOut, total});
  q.wait();
}
