#include <sycl/sycl.hpp>

namespace {

struct KimiDeltaAttentionKernel {
  const float *Q;
  const float *K;
  const float *V;
  const float *alpha;
  const float *beta;
  float *output;
  float *state;
  float *key_state;
  int seq_len;
  int d;

  void operator()(sycl::id<1> id) const {
    const int batch_index = static_cast<int>(id[0]);
    float *S = state + static_cast<size_t>(batch_index) * d * d;
    float *kS = key_state + static_cast<size_t>(batch_index) * d;

    for (int t = 0; t < seq_len; ++t) {
      const size_t token_base = (static_cast<size_t>(batch_index) * seq_len + t) * d;
      const float *q = Q + token_base;
      const float *k = K + token_base;
      const float *v = V + token_base;
      const float *a = alpha + token_base;
      const float b = beta[static_cast<size_t>(batch_index) * seq_len + t];

      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
          S[static_cast<size_t>(i) * d + j] *= a[i];
        }
      }

      for (int j = 0; j < d; ++j) {
        double sum = 0.0;
        for (int i = 0; i < d; ++i) {
          sum += static_cast<double>(k[i]) * static_cast<double>(S[static_cast<size_t>(i) * d + j]);
        }
        kS[j] = static_cast<float>(sum);
      }

      for (int i = 0; i < d; ++i) {
        for (int j = 0; j < d; ++j) {
          S[static_cast<size_t>(i) * d + j] +=
              b * k[i] * (v[j] - kS[j]);
        }
      }

      for (int j = 0; j < d; ++j) {
        double sum = 0.0;
        for (int i = 0; i < d; ++i) {
          sum += static_cast<double>(q[i]) * static_cast<double>(S[static_cast<size_t>(i) * d + j]);
        }
        output[token_base + j] = static_cast<float>(sum);
      }
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V, const float *alpha,
                      const float *beta, float *output, int batch, int seq_len, int d) {
  sycl::queue q{sycl::property::queue::in_order()};
  const size_t state_size = static_cast<size_t>(batch) * d * d;
  float *state = sycl::malloc_device<float>(state_size, q);
  float *key_state = sycl::malloc_device<float>(static_cast<size_t>(batch) * d, q);
  q.memset(state, 0, state_size * sizeof(float));
  q.parallel_for(sycl::range<1>(batch), KimiDeltaAttentionKernel{Q, K, V, alpha, beta, output,
                                                                state, key_state, seq_len, d});
  q.wait();
  sycl::free(state, q);
  sycl::free(key_state, q);
}
