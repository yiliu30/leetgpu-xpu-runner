#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct DotProductKernel {
  const float *A;
  const float *B;
  int N;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      reducer.combine(static_cast<double>(A[i]) * static_cast<double>(B[i]));
    }
  }
};

struct DotProductFinalizeKernel {
  const double *sum;
  float *result;

  void operator()() const { result[0] = static_cast<float>(*sum); }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *result, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  double *sum = sycl::malloc_device<double>(1, q);
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  if (numWorkGroups > 0) {
    q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<1>(
              sycl::range<1>(numWorkGroups * WORK_GROUP_SIZE),
              sycl::range<1>(WORK_GROUP_SIZE)),
          sycl::reduction(sum, sycl::plus<double>(),
                          sycl::property::reduction::initialize_to_identity()),
          DotProductKernel{A, B, N});
    });
  } else {
    q.single_task([=]() { *sum = 0.0; });
  }
  q.single_task(DotProductFinalizeKernel{sum, result});
  q.wait();
  sycl::free(sum, q);
}
