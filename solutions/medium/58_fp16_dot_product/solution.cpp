#include <cstddef>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct Fp16DotProductKernel {
  const sycl::half *a;
  const sycl::half *b;
  std::size_t n;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const std::size_t i = item.get_global_id(0);
    if (i < n) {
      sumReducer.combine(static_cast<float>(a[i]) * static_cast<float>(b[i]));
    }
  }
};

struct Fp16DotProductFinalizeKernel {
  const float *sum;
  sycl::half *result;

  void operator()(sycl::id<1>) const {
    result[0] = static_cast<sycl::half>(sum[0]);
  }
};

} // namespace

extern "C" void solve(const sycl::half *A, const sycl::half *B,
                      sycl::half *result, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const std::size_t total = static_cast<std::size_t>(N);
  const std::size_t numWorkGroups =
      (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  float *sum = sycl::malloc_device<float>(1, q);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(sum, sycl::plus<float>(),
                        sycl::property::reduction::initialize_to_identity()),
        Fp16DotProductKernel{A, B, total});
  });
  q.parallel_for(sycl::range<1>(1), Fp16DotProductFinalizeKernel{sum, result});
  q.wait();

  sycl::free(sum, q);
}
