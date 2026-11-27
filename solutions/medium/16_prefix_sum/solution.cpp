#include <sycl/sycl.hpp>

namespace {

struct PrefixSumKernel {
  const float *input;
  float *output;
  int N;

  void operator()() const {
    float running = 0.0f;
    for (int i = 0; i < N; ++i) {
      running += input[i];
      output[i] = running;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.single_task(PrefixSumKernel{input, output, N});
  q.wait();
}
