#include <sycl/sycl.hpp>

namespace {

struct SwigluHiddenKernel {
  const float *x;
  const float *W_gate;
  const float *W_up;
  float *hidden;
  int d_model;
  int d_ffn;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int m = static_cast<int>(idx / static_cast<std::size_t>(d_ffn));
    const int j = static_cast<int>(idx % static_cast<std::size_t>(d_ffn));
    const float *xrow = x + static_cast<std::size_t>(m) * d_model;
    float gate = 0.0f;
    float up = 0.0f;
    for (int k = 0; k < d_model; ++k) {
      const float v = xrow[k];
      const std::size_t wk = static_cast<std::size_t>(k) * d_ffn + j;
      gate += v * W_gate[wk];
      up += v * W_up[wk];
    }
    const float sigmoid = 1.0f / (1.0f + sycl::exp(-gate));
    hidden[idx] = gate * sigmoid * up;
  }
};

struct SwigluOutKernel {
  const float *hidden;
  const float *W_down;
  float *output;
  int d_model;
  int d_ffn;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int m = static_cast<int>(idx / static_cast<std::size_t>(d_model));
    const int j = static_cast<int>(idx % static_cast<std::size_t>(d_model));
    const float *hrow = hidden + static_cast<std::size_t>(m) * d_ffn;
    float acc = 0.0f;
    for (int k = 0; k < d_ffn; ++k) {
      acc += hrow[k] * W_down[static_cast<std::size_t>(k) * d_model + j];
    }
    output[idx] = acc;
  }
};

} // namespace

extern "C" void solve(const float *x, const float *W_gate, const float *W_up,
                      const float *W_down, float *output, int M, int d_model,
                      int d_ffn) {
  sycl::queue q{sycl::property::queue::in_order()};
  const std::size_t hidden_count = static_cast<std::size_t>(M) * d_ffn;
  float *hidden = sycl::malloc_device<float>(hidden_count, q);

  q.parallel_for(sycl::range<1>(hidden_count),
                 SwigluHiddenKernel{x, W_gate, W_up, hidden, d_model, d_ffn});

  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(M) * d_model),
                 SwigluOutKernel{hidden, W_down, output, d_model, d_ffn});
  q.wait();
  sycl::free(hidden, q);
}
