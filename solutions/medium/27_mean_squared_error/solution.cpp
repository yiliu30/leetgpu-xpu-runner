#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct MseKernel {
  const float *predictions;
  const float *targets;
  int N;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      const double diff = static_cast<double>(predictions[i]) - targets[i];
      reducer.combine(diff * diff);
    }
  }
};

struct MseFinalizeKernel {
  const double *sum;
  float *mse;
  int N;

  void operator()() const { mse[0] = static_cast<float>(*sum / N); }
};

} // namespace

extern "C" void solve(const float *predictions, const float *targets,
                      float *mse, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  double *sum = sycl::malloc_device<double>(1, q);
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  if (numWorkGroups > 0) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(numWorkGroups * WORK_GROUP_SIZE),
                            sycl::range<1>(WORK_GROUP_SIZE)),
          sycl::reduction(sum, sycl::plus<double>(),
                          sycl::property::reduction::initialize_to_identity()),
          MseKernel{predictions, targets, N});
    });
  } else {
    q.single_task([=]() { *sum = 0.0; });
  }
  q.single_task(MseFinalizeKernel{sum, mse, N});
  q.wait();
  sycl::free(sum, q);
}
