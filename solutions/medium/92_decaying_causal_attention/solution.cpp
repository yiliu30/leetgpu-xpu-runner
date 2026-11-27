#include <sycl/sycl.hpp>

namespace {

struct DecayingCausalAttentionKernel {
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int seq_len;
  int d_model;
  float gamma;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int n = static_cast<int>(idx / static_cast<std::size_t>(d_model));
    const int d = static_cast<int>(idx % static_cast<std::size_t>(d_model));
    const float *qrow = Q + static_cast<std::size_t>(n) * d_model;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(d_model));

    float acc = 0.0f;
    for (int m = 0; m <= n; ++m) {
      const float *krow = K + static_cast<std::size_t>(m) * d_model;
      float dot = 0.0f;
      for (int k = 0; k < d_model; ++k) {
        dot += qrow[k] * krow[k];
      }
      const float decay = sycl::pow(gamma, static_cast<float>(n - m));
      acc += dot * scale * decay * V[static_cast<std::size_t>(m) * d_model + d];
    }
    output[idx] = acc;
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int seq_len, int d_model, float gamma) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(
      sycl::range<1>(static_cast<std::size_t>(seq_len) * d_model),
      DecayingCausalAttentionKernel{Q, K, V, output, seq_len, d_model, gamma});
  q.wait();
}
