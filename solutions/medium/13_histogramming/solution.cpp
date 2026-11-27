#include <sycl/sycl.hpp>

namespace {

struct HistogramKernel {
  const int *input;
  int *histogram;
  int N;
  int num_bins;

  void operator()(sycl::id<1> id) const {
    const int i = static_cast<int>(id[0]);
    if (i < N) {
      const int bin = input[i];
      if (bin >= 0 && bin < num_bins) {
        sycl::atomic_ref<int, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            count(histogram[bin]);
        count.fetch_add(1);
      }
    }
  }
};

} // namespace

extern "C" void solve(const int *input, int *histogram, int N, int num_bins) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.memset(histogram, 0, static_cast<size_t>(num_bins) * sizeof(int));
  q.parallel_for(sycl::range<1>(N),
                 HistogramKernel{input, histogram, N, num_bins});
  q.wait();
}
