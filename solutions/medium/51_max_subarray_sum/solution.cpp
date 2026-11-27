#include <sycl/sycl.hpp>

namespace {

struct MaxSubarraySumKernel {
  const int *input;
  int *output;
  int n;
  int windowSize;

  void operator()() const {
    int currentSum = 0;
    for (int i = 0; i < windowSize; ++i) {
      currentSum += input[i];
    }
    int maxSum = currentSum;
    for (int i = windowSize; i < n; ++i) {
      currentSum += input[i] - input[i - windowSize];
      maxSum = sycl::max(maxSum, currentSum);
    }
    output[0] = maxSum;
  }
};

} // namespace

extern "C" void solve(const int *input, int *output, int N, int window_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.single_task(MaxSubarraySumKernel{input, output, N, window_size});
  q.wait();
}
