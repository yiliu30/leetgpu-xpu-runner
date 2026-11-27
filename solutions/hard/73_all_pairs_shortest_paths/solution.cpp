#include <sycl/sycl.hpp>

namespace {

struct FloydWarshallStepKernel {
  float *dist;
  int N;
  int k;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int i = linear / N;
    const int j = linear % N;
    const float candidate = dist[i * N + k] + dist[k * N + j];
    if (candidate < dist[linear]) {
      dist[linear] = candidate;
    }
  }
};

} // namespace

extern "C" void solve(const float *dist, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (N <= 0) {
    q.wait();
    return;
  }

  const size_t count = static_cast<size_t>(N) * N;
  q.memcpy(output, dist, count * sizeof(float));
  for (int k = 0; k < N; ++k) {
    q.parallel_for(sycl::range<1>(count),
                   FloydWarshallStepKernel{output, N, k});
  }
  q.wait();
}
