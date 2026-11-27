#include <sycl/sycl.hpp>

namespace {

struct MultiHeadAttentionKernel {
  const float *q;
  const float *k;
  const float *v;
  float *output;
  int n;
  int d_model;
  int heads;
  int head_dim;

  void operator()(sycl::id<1> id) const {
    const int d = static_cast<int>(id[0] % static_cast<size_t>(head_dim));
    const int rest = static_cast<int>(id[0] / static_cast<size_t>(head_dim));
    const int row = rest % n;
    const int head = rest / n;
    const int offset = head * head_dim;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const float *q_row = q + static_cast<size_t>(row) * d_model + offset;

    float max_score = -3.402823466e+38f;
    for (int j = 0; j < n; ++j) {
      const float *k_row = k + static_cast<size_t>(j) * d_model + offset;
      float score = 0.0f;
      for (int x = 0; x < head_dim; ++x) {
        score += q_row[x] * k_row[x];
      }
      score *= scale;
      max_score = score > max_score ? score : max_score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int j = 0; j < n; ++j) {
      const float *k_row = k + static_cast<size_t>(j) * d_model + offset;
      float score = 0.0f;
      for (int x = 0; x < head_dim; ++x) {
        score += q_row[x] * k_row[x];
      }
      const float weight = sycl::exp(score * scale - max_score);
      denom += weight;
      acc += weight * v[static_cast<size_t>(j) * d_model + offset + d];
    }
    output[static_cast<size_t>(row) * d_model + offset + d] = acc / denom;
  }
};

} // namespace

extern "C" void solve(const float *q, const float *k, const float *v,
                      float *output, int n, int d_model, int heads) {
  if (n <= 0 || d_model <= 0 || heads <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  const int head_dim = d_model / heads;
  queue.parallel_for(
      sycl::range<1>(static_cast<size_t>(n) * d_model),
      MultiHeadAttentionKernel{q, k, v, output, n, d_model, heads, head_dim});
  queue.wait();
}
