#include <sycl/sycl.hpp>

namespace {

struct SplitKVMergeKernel {
  const float *partial_out;
  const float *partial_lse;
  float *output;
  int num_splits;
  int num_heads;
  int head_dim;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int head = static_cast<int>(linear / head_dim);
    const int dim = static_cast<int>(linear % head_dim);

    double lse_max = static_cast<double>(partial_lse[head]);
    for (int split = 1; split < num_splits; ++split) {
      lse_max = sycl::fmax(
          lse_max, static_cast<double>(partial_lse[split * num_heads + head]));
    }

    double weight_sum = 0.0;
    for (int split = 0; split < num_splits; ++split) {
      weight_sum += sycl::exp(
          static_cast<double>(partial_lse[split * num_heads + head]) - lse_max);
    }

    double merged = 0.0;
    for (int split = 0; split < num_splits; ++split) {
      const double weight =
          sycl::exp(static_cast<double>(partial_lse[split * num_heads + head]) -
                    lse_max) /
          weight_sum;
      const size_t out_index =
          (static_cast<size_t>(split) * num_heads + head) * head_dim + dim;
      merged += weight * static_cast<double>(partial_out[out_index]);
    }
    output[linear] = static_cast<float>(merged);
  }
};

} // namespace

extern "C" void solve(const float *partial_out, const float *partial_lse,
                      float *output, int num_splits, int num_heads,
                      int head_dim) {
  sycl::queue q{sycl::property::queue::in_order()};
  const size_t total = static_cast<size_t>(num_heads) * head_dim;
  q.parallel_for(sycl::range<1>(total),
                 SplitKVMergeKernel{partial_out, partial_lse, output,
                                    num_splits, num_heads, head_dim});
  q.wait();
}
