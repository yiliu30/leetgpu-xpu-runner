#include <sycl/sycl.hpp>

namespace {

struct GatedAttentionGateKernel {
  const float *X;
  const float *W_g;
  float *gate;
  int M;
  int d_model;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    if (linear >= static_cast<size_t>(M) * d_model) {
      return;
    }
    const int row = static_cast<int>(linear / d_model);
    const int col = static_cast<int>(linear % d_model);
    const float *x_row = X + static_cast<size_t>(row) * d_model;
    double acc = 0.0;
    for (int k = 0; k < d_model; ++k) {
      acc += static_cast<double>(x_row[k]) *
             static_cast<double>(W_g[static_cast<size_t>(k) * d_model + col]);
    }
    double sigmoid;
    if (acc >= 0.0) {
      sigmoid = 1.0 / (1.0 + sycl::exp(-acc));
    } else {
      const double e = sycl::exp(acc);
      sigmoid = e / (1.0 + e);
    }
    gate[linear] = static_cast<float>(sigmoid);
  }
};

struct GatedAttentionWeightsKernel {
  const float *Q;
  const float *K;
  float *weights;
  int M;
  int d_model;
  int h;
  int d_k;
  double scale;

  void operator()(sycl::id<1> idx) const {
    const int linear = idx[0];
    if (linear >= M * h) {
      return;
    }
    const int row = linear / h;
    const int head = linear % h;
    const int offset = head * d_k;
    const float *q_row = Q + static_cast<size_t>(row) * d_model + offset;

    float max_score = -3.402823466e+38F;
    for (int key = 0; key <= row; ++key) {
      const float *k_row = K + static_cast<size_t>(key) * d_model + offset;
      double dot = 0.0;
      for (int d = 0; d < d_k; ++d) {
        dot += static_cast<double>(q_row[d]) * static_cast<double>(k_row[d]);
      }
      const float score = static_cast<float>(dot * scale);
      if (score > max_score) {
        max_score = score;
      }
    }

    double sum = 0.0;
    const size_t base = (static_cast<size_t>(row) * h + head) * M;
    for (int key = 0; key <= row; ++key) {
      const float *k_row = K + static_cast<size_t>(key) * d_model + offset;
      double dot = 0.0;
      for (int d = 0; d < d_k; ++d) {
        dot += static_cast<double>(q_row[d]) * static_cast<double>(k_row[d]);
      }
      const float score = static_cast<float>(dot * scale);
      const float e = sycl::exp(score - max_score);
      weights[base + key] = e;
      sum += static_cast<double>(e);
    }
    const float inv_sum = static_cast<float>(1.0 / sum);
    for (int key = 0; key <= row; ++key) {
      weights[base + key] *= inv_sum;
    }
  }
};

struct GatedAttentionOutputKernel {
  const float *V;
  const float *weights;
  const float *gate;
  float *output;
  int M;
  int d_model;
  int h;
  int d_k;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    if (linear >= static_cast<size_t>(M) * d_model) {
      return;
    }
    const int row = static_cast<int>(linear / d_model);
    const int col = static_cast<int>(linear % d_model);
    const int head = col / d_k;
    const int d = col - head * d_k;
    const int offset = head * d_k;
    const size_t weight_base = (static_cast<size_t>(row) * h + head) * M;
    double acc = 0.0;
    for (int key = 0; key <= row; ++key) {
      acc += static_cast<double>(weights[weight_base + key]) *
             static_cast<double>(
                 V[static_cast<size_t>(key) * d_model + offset + d]);
    }
    output[linear] =
        static_cast<float>(acc * static_cast<double>(gate[linear]));
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      const float *X, const float *W_g, float *output, int M,
                      int d_model, int h) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int d_k = d_model / h;
  const double scale = 1.0 / sycl::sqrt(static_cast<double>(d_k));
  float *gate = sycl::malloc_device<float>(static_cast<size_t>(M) * d_model, q);
  float *weights =
      sycl::malloc_device<float>(static_cast<size_t>(M) * h * M, q);

  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * d_model),
                 GatedAttentionGateKernel{X, W_g, gate, M, d_model});
  q.parallel_for(
      sycl::range<1>(M * h),
      GatedAttentionWeightsKernel{Q, K, weights, M, d_model, h, d_k, scale});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(M) * d_model),
      GatedAttentionOutputKernel{V, weights, gate, output, M, d_model, h, d_k});
  q.wait();

  sycl::free(gate, q);
  sycl::free(weights, q);
}
