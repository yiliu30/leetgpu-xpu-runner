#include <sycl/sycl.hpp>

namespace {

struct SharedPrefixAttentionKernel {
  const float *Q;
  const float *K_prefix;
  const float *V_prefix;
  const float *K_suffix;
  const float *V_suffix;
  float *output;
  int num_heads;
  int prefix_len;
  int suffix_len;
  int head_dim;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int d = static_cast<int>(idx % static_cast<std::size_t>(head_dim));
    const std::size_t rest = idx / static_cast<std::size_t>(head_dim);
    const int i = static_cast<int>(rest % static_cast<std::size_t>(suffix_len));
    const std::size_t bh = rest / static_cast<std::size_t>(suffix_len);
    const int h = static_cast<int>(bh % static_cast<std::size_t>(num_heads));
    const int b = static_cast<int>(bh / static_cast<std::size_t>(num_heads));

    const std::size_t bh_base =
        (static_cast<std::size_t>(b) * num_heads + h) * suffix_len * head_dim;
    const std::size_t row = bh_base + static_cast<std::size_t>(i) * head_dim;
    const float *qrow = Q + row;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int nkeys = prefix_len + i + 1;

    float max_score = -1.0e38f;
    for (int j = 0; j < nkeys; ++j) {
      const float *krow;
      if (j < prefix_len) {
        krow = K_prefix + (static_cast<std::size_t>(h) * prefix_len + j) * head_dim;
      } else {
        krow = K_suffix + bh_base +
               static_cast<std::size_t>(j - prefix_len) * head_dim;
      }
      float s = 0.0f;
      for (int k = 0; k < head_dim; ++k) {
        s += qrow[k] * krow[k];
      }
      s *= scale;
      max_score = s > max_score ? s : max_score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int j = 0; j < nkeys; ++j) {
      const float *krow;
      const float *vrow;
      if (j < prefix_len) {
        const std::size_t p =
            (static_cast<std::size_t>(h) * prefix_len + j) * head_dim;
        krow = K_prefix + p;
        vrow = V_prefix + p;
      } else {
        const std::size_t so =
            bh_base + static_cast<std::size_t>(j - prefix_len) * head_dim;
        krow = K_suffix + so;
        vrow = V_suffix + so;
      }
      float s = 0.0f;
      for (int k = 0; k < head_dim; ++k) {
        s += qrow[k] * krow[k];
      }
      s *= scale;
      const float w = sycl::exp(s - max_score);
      denom += w;
      acc += w * vrow[d];
    }
    output[row + d] = acc / denom;
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K_prefix, const float *V_prefix,
                      const float *K_suffix, const float *V_suffix, float *output, int batch_size,
                      int num_heads, int prefix_len, int suffix_len, int head_dim) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(batch_size) * num_heads * suffix_len *
                                head_dim),
                 SharedPrefixAttentionKernel{Q, K_prefix, V_prefix, K_suffix, V_suffix, output,
                                             num_heads, prefix_len, suffix_len, head_dim});
  q.wait();
}
