#include <sycl/sycl.hpp>

namespace {

struct MultiAgentSimKernel {
  const float *agents;
  float *agents_next;
  int N;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i >= N) {
      return;
    }
    const float xi = agents[static_cast<size_t>(i) * 4];
    const float yi = agents[static_cast<size_t>(i) * 4 + 1];
    const float vxi = agents[static_cast<size_t>(i) * 4 + 2];
    const float vyi = agents[static_cast<size_t>(i) * 4 + 3];

    float sum_vx = 0.0f;
    float sum_vy = 0.0f;
    int count = 0;
    for (int j = 0; j < N; ++j) {
      if (j == i) {
        continue;
      }
      const float dx = xi - agents[static_cast<size_t>(j) * 4];
      const float dy = yi - agents[static_cast<size_t>(j) * 4 + 1];
      if (dx * dx + dy * dy < 25.0f) {
        sum_vx += agents[static_cast<size_t>(j) * 4 + 2];
        sum_vy += agents[static_cast<size_t>(j) * 4 + 3];
        ++count;
      }
    }

    const float avg_vx = count > 0 ? sum_vx / static_cast<float>(count) : vxi;
    const float avg_vy = count > 0 ? sum_vy / static_cast<float>(count) : vyi;
    const float new_vx = vxi + 0.05f * (avg_vx - vxi);
    const float new_vy = vyi + 0.05f * (avg_vy - vyi);

    agents_next[static_cast<size_t>(i) * 4] = xi + new_vx;
    agents_next[static_cast<size_t>(i) * 4 + 1] = yi + new_vy;
    agents_next[static_cast<size_t>(i) * 4 + 2] = new_vx;
    agents_next[static_cast<size_t>(i) * 4 + 3] = new_vy;
  }
};

} // namespace

extern "C" void solve(const float *agents, float *agents_next, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(N),
                 MultiAgentSimKernel{agents, agents_next, N});
  q.wait();
}
