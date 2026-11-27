#include <sycl/sycl.hpp>

namespace {

struct ScatterAddKernel {
  const float *src;
  const int *index;
  float *output;
  int D;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int row = static_cast<int>(linear / D);
    const int col = static_cast<int>(linear % D);
    const int dst = index[row];
    sycl::atomic_ref<float, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::global_space>
        out(output[static_cast<size_t>(dst) * D + col]);
    out.fetch_add(src[linear]);
  }
};

} // namespace

extern "C" void solve(const float *src, const int *index, float *output, int N, int D, int M) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.memset(output, 0, static_cast<size_t>(M) * D * sizeof(float));
  q.parallel_for(sycl::range<1>(static_cast<size_t>(N) * D),
                 ScatterAddKernel{src, index, output, D});
  q.wait();
}
