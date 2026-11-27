#include <sycl/sycl.hpp>

namespace {

struct CausalDepthwiseConv1dKernel {
  const float *x;
  const float *weight;
  const float *bias;
  float *output;
  int L;
  int D;
  int K;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int d = static_cast<int>(idx % static_cast<std::size_t>(D));
    const std::size_t bl = idx / static_cast<std::size_t>(D);
    const int l = static_cast<int>(bl % static_cast<std::size_t>(L));
    const int b = static_cast<int>(bl / static_cast<std::size_t>(L));

    const float *xrow = x + static_cast<std::size_t>(b) * L * D;
    const float *wrow = weight + static_cast<std::size_t>(d) * K;
    float acc = bias[d];
    for (int k = 0; k < K; ++k) {
      const int pos = l - k;
      if (pos >= 0) {
        acc += wrow[k] * xrow[static_cast<std::size_t>(pos) * D + d];
      }
    }
    output[idx] = acc;
  }
};

} // namespace

extern "C" void solve(const float *x, const float *weight, const float *bias,
                      float *output, int B, int L, int D, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(B) * L * D),
                 CausalDepthwiseConv1dKernel{x, weight, bias, output, L, D, K});
  q.wait();
}
