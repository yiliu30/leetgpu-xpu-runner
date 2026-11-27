#include <sycl/sycl.hpp>

namespace {

struct RowArgmaxKernel {
  const float *logits;
  int *output;
  int V;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const float *values = logits + static_cast<size_t>(row) * V;
    float best = values[0];
    int best_col = 0;
    for (int col = 1; col < V; ++col) {
      const float value = values[col];
      if (value > best || (value == best && col < best_col)) {
        best = value;
        best_col = col;
      }
    }
    output[row] = best_col;
  }
};

} // namespace

extern "C" void solve(const float *logits, int *output, int B, int V) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(B), RowArgmaxKernel{logits, output, V});
  q.wait();
}
