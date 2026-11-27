#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct SoftmaxMaxKernel {
  const float *input;
  int N;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &maxReducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      maxReducer.combine(input[i]);
    }
  }
};

struct SoftmaxSumKernel {
  const float *input;
  const float *maxValue;
  int N;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      sumReducer.combine(
          static_cast<double>(sycl::exp(input[i] - maxValue[0])));
    }
  }
};

struct SoftmaxNormalizeKernel {
  const float *input;
  const float *maxValue;
  const double *expSum;
  float *output;

  void operator()(sycl::id<1> i) const {
    output[i] = static_cast<float>(
        static_cast<double>(sycl::exp(input[i] - maxValue[0])) / expSum[0]);
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  float *maxValue = sycl::malloc_device<float>(1, q);
  double *expSum = sycl::malloc_device<double>(1, q);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(maxValue, sycl::maximum<float>(),
                        sycl::property::reduction::initialize_to_identity()),
        SoftmaxMaxKernel{input, N});
  });

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(expSum, sycl::plus<double>(),
                        sycl::property::reduction::initialize_to_identity()),
        SoftmaxSumKernel{input, maxValue, N});
  });

  q.parallel_for(sycl::range<1>(N),
                 SoftmaxNormalizeKernel{input, maxValue, expSum, output});
  q.wait();

  sycl::free(maxValue, q);
  sycl::free(expSum, q);
}
