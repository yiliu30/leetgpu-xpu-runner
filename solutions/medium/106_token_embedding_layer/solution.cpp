#include <sycl/sycl.hpp>

namespace {

struct TokenEmbeddingKernel {
  const int *token_ids;
  const int *position_ids;
  const float *token_embeddings;
  const float *position_embeddings;
  const float *gamma;
  const float *beta;
  float *output;
  int T;
  int D;
  float eps;

  void operator()(sycl::id<1> idx) const {
    const int row = idx[0];
    const int b = row / T;
    const int t = row - b * T;
    const int tok = token_ids[static_cast<long>(b) * T + t];
    const int pos = position_ids[t];
    const float *tok_row = token_embeddings + static_cast<long>(tok) * D;
    const float *pos_row = position_embeddings + static_cast<long>(pos) * D;
    float *out_row = output + static_cast<long>(row) * D;

    double sum = 0.0;
    for (int d = 0; d < D; ++d) {
      sum += static_cast<double>(tok_row[d]) + static_cast<double>(pos_row[d]);
    }
    const double mean = sum / static_cast<double>(D);

    double var_sum = 0.0;
    for (int d = 0; d < D; ++d) {
      const double v = static_cast<double>(tok_row[d]) + static_cast<double>(pos_row[d]) - mean;
      var_sum += v * v;
    }
    const double inv_std = 1.0 / sycl::sqrt(var_sum / static_cast<double>(D) +
                                            static_cast<double>(eps));

    for (int d = 0; d < D; ++d) {
      const double normalized =
          (static_cast<double>(tok_row[d]) + static_cast<double>(pos_row[d]) - mean) * inv_std;
      out_row[d] = static_cast<float>(normalized * static_cast<double>(gamma[d]) +
                                      static_cast<double>(beta[d]));
    }
  }
};

} // namespace

extern "C" void solve(const int *token_ids, const int *position_ids, const float *token_embeddings,
                      const float *position_embeddings, const float *gamma, const float *beta,
                      float *output, int B, int T, int V, int P, int D, float eps) {
  (void)V;
  (void)P;
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(B * T),
                 TokenEmbeddingKernel{token_ids, position_ids, token_embeddings,
                                      position_embeddings, gamma, beta, output, T, D, eps});
  q.wait();
}
