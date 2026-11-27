#include <sycl/sycl.hpp>

namespace {

constexpr double kDifferentialEps = 1e-5;

struct DifferentialAttentionWeightsKernel {
  const float *Q;
  const float *K;
  const float *lambdas;
  float *weights;
  int S;
  int H;
  int D;
  double scale;

  void operator()(sycl::id<1> idx) const {
    const int linear = idx[0];
    if (linear >= S * H) {
      return;
    }
    const int row = linear / H;
    const int head = linear % H;
    const size_t q_base = (static_cast<size_t>(row) * H + head) * (2 * D);
    const float *q1 = Q + q_base;
    const float *q2 = q1 + D;
    const double lambda = static_cast<double>(lambdas[head]);

    double max1 = -1.0e300;
    double max2 = -1.0e300;
    for (int key = 0; key <= row; ++key) {
      const float *k_row = K + (static_cast<size_t>(key) * H + head) * (2 * D);
      double dot1 = 0.0;
      double dot2 = 0.0;
      for (int d = 0; d < D; ++d) {
        dot1 += static_cast<double>(q1[d]) * static_cast<double>(k_row[d]);
        dot2 += static_cast<double>(q2[d]) * static_cast<double>(k_row[D + d]);
      }
      const double score1 = dot1 * scale;
      const double score2 = dot2 * scale;
      if (score1 > max1) {
        max1 = score1;
      }
      if (score2 > max2) {
        max2 = score2;
      }
    }

    double sum1 = 0.0;
    double sum2 = 0.0;
    for (int key = 0; key <= row; ++key) {
      const float *k_row = K + (static_cast<size_t>(key) * H + head) * (2 * D);
      double dot1 = 0.0;
      double dot2 = 0.0;
      for (int d = 0; d < D; ++d) {
        dot1 += static_cast<double>(q1[d]) * static_cast<double>(k_row[d]);
        dot2 += static_cast<double>(q2[d]) * static_cast<double>(k_row[D + d]);
      }
      sum1 += sycl::exp(dot1 * scale - max1);
      sum2 += sycl::exp(dot2 * scale - max2);
    }

    const size_t w_base = (static_cast<size_t>(row) * H + head) * S;
    for (int key = 0; key <= row; ++key) {
      const float *k_row = K + (static_cast<size_t>(key) * H + head) * (2 * D);
      double dot1 = 0.0;
      double dot2 = 0.0;
      for (int d = 0; d < D; ++d) {
        dot1 += static_cast<double>(q1[d]) * static_cast<double>(k_row[d]);
        dot2 += static_cast<double>(q2[d]) * static_cast<double>(k_row[D + d]);
      }
      const double a1 = sycl::exp(dot1 * scale - max1) / sum1;
      const double a2 = sycl::exp(dot2 * scale - max2) / sum2;
      weights[w_base + key] = static_cast<float>(a1 - lambda * a2);
    }
  }
};

struct DifferentialAttentionOutputKernel {
  const float *V;
  const float *weights;
  float *output;
  int S;
  int H;
  int D;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    if (linear >= static_cast<size_t>(S) * H * (2 * D)) {
      return;
    }
    const int d_out = 2 * D;
    const int row = static_cast<int>(linear / (H * d_out));
    const int rem = static_cast<int>(linear % (H * d_out));
    const int head = rem / d_out;
    const int d = rem - head * d_out;
    const size_t w_base = (static_cast<size_t>(row) * H + head) * S;
    double acc = 0.0;
    for (int key = 0; key <= row; ++key) {
      acc += static_cast<double>(weights[w_base + key]) *
             static_cast<double>(
                 V[(static_cast<size_t>(key) * H + head) * d_out + d]);
    }
    output[linear] = static_cast<float>(acc);
  }
};

struct DifferentialAttentionNormalizeKernel {
  float *output;
  int S;
  int H;
  int D;
  double lambda_init;

  void operator()(sycl::id<1> idx) const {
    const int linear = idx[0];
    if (linear >= S * H) {
      return;
    }
    const int d_out = 2 * D;
    const size_t base = static_cast<size_t>(linear) * d_out;
    double sum_sq = 0.0;
    for (int d = 0; d < d_out; ++d) {
      const double value = static_cast<double>(output[base + d]);
      sum_sq += value * value;
    }
    const double rms =
        sycl::sqrt(sum_sq / static_cast<double>(d_out) + kDifferentialEps);
    const double factor = (1.0 - lambda_init) / rms;
    for (int d = 0; d < d_out; ++d) {
      output[base + d] =
          static_cast<float>(static_cast<double>(output[base + d]) * factor);
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      const float *lambdas, float *output, int seq_len,
                      int num_heads, int head_dim, float lambda_init) {
  sycl::queue q{sycl::property::queue::in_order()};
  const double scale = 1.0 / sycl::sqrt(static_cast<double>(head_dim));
  float *weights = sycl::malloc_device<float>(
      static_cast<size_t>(seq_len) * num_heads * seq_len, q);

  q.parallel_for(sycl::range<1>(seq_len * num_heads),
                 DifferentialAttentionWeightsKernel{Q, K, lambdas, weights,
                                                    seq_len, num_heads,
                                                    head_dim, scale});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(seq_len) * num_heads * 2 * head_dim),
      DifferentialAttentionOutputKernel{V, weights, output, seq_len, num_heads,
                                        head_dim});
  q.parallel_for(
      sycl::range<1>(seq_len * num_heads),
      DifferentialAttentionNormalizeKernel{output, seq_len, num_heads, head_dim,
                                           static_cast<double>(lambda_init)});
  q.wait();

  sycl::free(weights, q);
}
