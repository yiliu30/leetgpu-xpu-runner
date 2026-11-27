#include <sycl/sycl.hpp>

namespace {

struct SoftcappedSlidingWindowWeightsKernel {
  const float *Q;
  const float *K;
  float *weights;
  float *row_sums;
  int num_q_heads;
  int num_kv_heads;
  int S;
  int D;
  int window_size;
  float softcap;
  float scale;
  int group_size;

  void operator()(sycl::id<1> idx) const {
    const int linear = idx[0];
    if (linear >= num_q_heads * S) {
      return;
    }
    const int head = linear / S;
    const int row = linear % S;
    const int kv_head = head / group_size;
    const int start = row >= window_size ? row - window_size + 1 : 0;
    const float *q_row = Q + (static_cast<size_t>(head) * S + row) * D;
    const size_t w_base = (static_cast<size_t>(head) * S + row) * S;

    float max_score = -3.402823466e+38F;
    for (int key = start; key <= row; ++key) {
      const float *k_row = K + (static_cast<size_t>(kv_head) * S + key) * D;
      float dot = 0.0f;
      for (int d = 0; d < D; ++d) {
        dot += q_row[d] * k_row[d];
      }
      const float score = softcap * sycl::tanh((dot * scale) / softcap);
      weights[w_base + key] = score;
      if (score > max_score) {
        max_score = score;
      }
    }

    double sum = 0.0;
    for (int key = start; key <= row; ++key) {
      const float e = sycl::exp(weights[w_base + key] - max_score);
      weights[w_base + key] = e;
      sum += static_cast<double>(e);
    }
    row_sums[static_cast<size_t>(head) * S + row] = static_cast<float>(sum);
  }
};

struct SoftcappedSlidingWindowOutputKernel {
  const float *V;
  const float *weights;
  const float *row_sums;
  float *output;
  int num_q_heads;
  int S;
  int D;
  int window_size;
  int group_size;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    if (linear >= static_cast<size_t>(num_q_heads) * S * D) {
      return;
    }
    const int head = static_cast<int>(linear / (static_cast<size_t>(S) * D));
    const int rem = static_cast<int>(linear % (static_cast<size_t>(S) * D));
    const int row = rem / D;
    const int d = rem % D;
    const int kv_head = head / group_size;
    const int start = row >= window_size ? row - window_size + 1 : 0;
    const float *w_row = weights + (static_cast<size_t>(head) * S + row) * S;

    double acc = 0.0;
    for (int key = start; key <= row; ++key) {
      acc += static_cast<double>(w_row[key]) *
             static_cast<double>(
                 V[(static_cast<size_t>(kv_head) * S + key) * D + d]);
    }
    const double denom =
        static_cast<double>(row_sums[static_cast<size_t>(head) * S + row]);
    output[linear] = static_cast<float>(acc / denom);
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim, int window_size,
                      float softcap) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int group_size = num_q_heads / num_kv_heads;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  float *weights = sycl::malloc_device<float>(
      static_cast<size_t>(num_q_heads) * seq_len * seq_len, q);
  float *row_sums =
      sycl::malloc_device<float>(static_cast<size_t>(num_q_heads) * seq_len, q);

  q.parallel_for(sycl::range<1>(num_q_heads * seq_len),
                 SoftcappedSlidingWindowWeightsKernel{
                     Q, K, weights, row_sums, num_q_heads, num_kv_heads,
                     seq_len, head_dim, window_size, softcap, scale,
                     group_size});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(num_q_heads) * seq_len * head_dim),
      SoftcappedSlidingWindowOutputKernel{V, weights, row_sums, output,
                                          num_q_heads, seq_len, head_dim,
                                          window_size, group_size});
  q.wait();

  sycl::free(weights, q);
  sycl::free(row_sums, q);
}
