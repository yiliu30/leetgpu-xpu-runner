#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct RmsSumSquaresKernel {
  const float *input;
  int n;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const int i = item.get_global_id(0);
    if (i < n) {
      sumReducer.combine(input[i] * input[i]);
    }
  }
};

struct RmsNormalizeKernel {
  const float *input;
  const float *sumSquares;
  float *output;
  int n;
  float gamma;
  float beta;
  float eps;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i >= n) {
      return;
    }
    const float rms =
        sycl::sqrt(sumSquares[0] / static_cast<float>(n) + eps);
    output[i] = gamma * (input[i] / rms) + beta;
  }
};

} // namespace

extern "C" void solve(const float *input, float gamma, float beta, float *output,
                      int N, float eps) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  float *sumSquares = sycl::malloc_device<float>(1, q);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(sumSquares, sycl::plus<float>(),
                        sycl::property::reduction::initialize_to_identity()),
        RmsSumSquaresKernel{input, N});
  });
  q.parallel_for(globalSize, RmsNormalizeKernel{input, sumSquares, output, N,
                                                gamma, beta, eps});
  q.wait();

  sycl::free(sumSquares, q);
}
