#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct MonteCarloSumKernel {
  const float *y_samples;
  int n_samples;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const int i = item.get_global_id(0);
    if (i < n_samples) {
      sumReducer.combine(y_samples[i]);
    }
  }
};

struct MonteCarloFinalizeKernel {
  const float *sum;
  float *result;
  float a;
  float b;
  int n_samples;

  void operator()(sycl::id<1>) const {
    result[0] = (b - a) * sum[0] / static_cast<float>(n_samples);
  }
};

} // namespace

extern "C" void solve(const float *y_samples, float *result, float a, float b,
                      int n_samples) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (n_samples + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  float *sum = sycl::malloc_device<float>(1, q);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(sum, sycl::plus<float>(),
                        sycl::property::reduction::initialize_to_identity()),
        MonteCarloSumKernel{y_samples, n_samples});
  });
  q.parallel_for(sycl::range<1>(1),
                 MonteCarloFinalizeKernel{sum, result, a, b, n_samples});
  q.wait();

  sycl::free(sum, q);
}
