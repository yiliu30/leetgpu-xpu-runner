#include <sycl/sycl.hpp>

namespace {

struct LinearRecurrenceKernel {
  const float *a;
  const float *x;
  float *h;
  int L;

  void operator()(sycl::id<1> id) const {
    const int b = static_cast<int>(id[0]);
    const std::size_t base = static_cast<std::size_t>(b) * L;
    float prev = x[base];
    h[base] = prev;
    for (int t = 1; t < L; ++t) {
      prev = a[base + t] * prev + x[base + t];
      h[base + t] = prev;
    }
  }
};

} // namespace

extern "C" void solve(const float *a, const float *x, float *h, int B, int L) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(B)),
                 LinearRecurrenceKernel{a, x, h, L});
  q.wait();
}
