#include <sycl/sycl.hpp>

namespace {

constexpr unsigned int kMaxKey = 0xFFFFFFFFu;

struct FillKeyKernel {
  unsigned int *data;
  unsigned int value;

  void operator()(sycl::id<1> idx) const { data[idx] = value; }
};

struct CopyKeyKernel {
  const unsigned int *src;
  unsigned int *dst;

  void operator()(sycl::id<1> idx) const { dst[idx] = src[idx]; }
};

struct BitonicKeyKernel {
  unsigned int *data;
  int k;
  int j;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0]);
    const int ixj = i ^ j;
    if (ixj <= i) {
      return;
    }
    const unsigned int a = data[i];
    const unsigned int b = data[ixj];
    const bool ascending = (i & k) == 0;
    const bool swap = ascending ? (a > b) : (a < b);
    if (swap) {
      data[i] = b;
      data[ixj] = a;
    }
  }
};

} // namespace

extern "C" void solve(const unsigned int *input, unsigned int *output, int N) {
  if (N <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  int padded = 1;
  while (padded < N) {
    padded <<= 1;
  }

  unsigned int *scratch = sycl::malloc_device<unsigned int>(padded, q);
  q.parallel_for(sycl::range<1>(padded), FillKeyKernel{scratch, kMaxKey});
  q.parallel_for(sycl::range<1>(N), CopyKeyKernel{input, scratch});
  for (int k = 2; k <= padded; k <<= 1) {
    for (int j = k >> 1; j > 0; j >>= 1) {
      q.parallel_for(sycl::range<1>(padded), BitonicKeyKernel{scratch, k, j});
    }
  }
  q.parallel_for(sycl::range<1>(N), CopyKeyKernel{scratch, output});
  q.wait();
  sycl::free(scratch, q);
}
