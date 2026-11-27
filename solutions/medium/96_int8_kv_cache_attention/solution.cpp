#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

struct Int8KvCacheAttentionKernel {
  const float *Q;
  const int8_t *K_int8;
  const int8_t *V_int8;
  const float *k_scale;
  const float *v_scale;
  float *output;
  int seq_len;
  int head_dim;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int d = static_cast<int>(idx % static_cast<std::size_t>(head_dim));
    const int h = static_cast<int>(idx / static_cast<std::size_t>(head_dim));
    const float *qrow = Q + static_cast<std::size_t>(h) * head_dim;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const int8_t *khead = K_int8 + static_cast<std::size_t>(h) * seq_len * head_dim;
    const int8_t *vhead = V_int8 + static_cast<std::size_t>(h) * seq_len * head_dim;
    const float *kscale = k_scale + static_cast<std::size_t>(h) * seq_len;
    const float *vscale = v_scale + static_cast<std::size_t>(h) * seq_len;

    float max_score = -1.0e38f;
    for (int s = 0; s < seq_len; ++s) {
      const int8_t *krow = khead + static_cast<std::size_t>(s) * head_dim;
      float dot = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        dot += qrow[i] * (static_cast<float>(krow[i]) * kscale[s]);
      }
      dot *= scale;
      max_score = dot > max_score ? dot : max_score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int s = 0; s < seq_len; ++s) {
      const int8_t *krow = khead + static_cast<std::size_t>(s) * head_dim;
      float dot = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        dot += qrow[i] * (static_cast<float>(krow[i]) * kscale[s]);
      }
      dot *= scale;
      const float w = sycl::exp(dot - max_score);
      denom += w;
      const int8_t *vrow = vhead + static_cast<std::size_t>(s) * head_dim;
      acc += w * (static_cast<float>(vrow[d]) * vscale[s]);
    }
    output[idx] = acc / denom;
  }
};

} // namespace

extern "C" void solve(const float *Q, const int8_t *K_int8, const int8_t *V_int8,
                      const float *k_scale, const float *v_scale, float *output, int num_heads,
                      int seq_len, int head_dim) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(num_heads) * head_dim),
                 Int8KvCacheAttentionKernel{Q, K_int8, V_int8, k_scale, v_scale, output, seq_len,
                                            head_dim});
  q.wait();
}
