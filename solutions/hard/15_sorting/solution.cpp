#include <sycl/sycl.hpp>

#include <limits>

namespace {

struct FillFloatKernel {
  float *data;
  float value;

  void operator()(sycl::id<1> idx) const { data[idx] = value; }
};

struct CopyFloatKernel {
  const float *src;
  float *dst;

  void operator()(sycl::id<1> idx) const { dst[idx] = src[idx]; }
};

struct BitonicFloatKernel {
  float *data;
  int k;
  int j;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0]);
    const int ixj = i ^ j;
    if (ixj <= i) {
      return;
    }
    const float a = data[i];
    const float b = data[ixj];
    const bool ascending = (i & k) == 0;
    const bool swap = ascending ? (a > b) : (a < b);
    if (swap) {
      data[i] = b;
      data[ixj] = a;
    }
  }
};

} // namespace

extern "C" void solve(float *data, int N) {
  if (N <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  int padded = 1;
  while (padded < N) {
    padded <<= 1;
  }

  float *scratch = sycl::malloc_device<float>(padded, q);
  q.parallel_for(
      sycl::range<1>(padded),
      FillFloatKernel{scratch, std::numeric_limits<float>::infinity()});
  q.parallel_for(sycl::range<1>(N), CopyFloatKernel{data, scratch});
  for (int k = 2; k <= padded; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      q.parallel_for(sycl::range<1>(padded), BitonicFloatKernel{scratch, k, j});
    }
  }
  q.parallel_for(sycl::range<1>(N), CopyFloatKernel{scratch, data});
  q.wait();
  sycl::free(scratch, q);
}
