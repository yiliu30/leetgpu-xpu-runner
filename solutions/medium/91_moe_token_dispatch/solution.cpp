#include <sycl/sycl.hpp>

namespace {

struct MoeDispatchKernel {
  const float *x;
  const int *expert_idx;
  float *dispatched_x;
  int *token_counts;
  int T;
  int D;
  int capacity;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int e = static_cast<int>(idx / static_cast<std::size_t>(D));
    const int d = static_cast<int>(idx % static_cast<std::size_t>(D));

    int count = 0;
    for (int t = 0; t < T; ++t) {
      if (expert_idx[t] == e) {
        dispatched_x[(static_cast<std::size_t>(e) * capacity + count) * D + d] =
            x[static_cast<std::size_t>(t) * D + d];
        ++count;
      }
    }
    if (d == 0) {
      token_counts[e] = count;
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const int *expert_idx, float *dispatched_x, int *token_counts,
                      int T, int D, int E, int capacity) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.fill(dispatched_x, 0.0f, static_cast<std::size_t>(E) * capacity * D);
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(E) * D),
                 MoeDispatchKernel{x, expert_idx, dispatched_x, token_counts, T, D, capacity});
  q.wait();
}
