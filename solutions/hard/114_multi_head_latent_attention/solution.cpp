#include <sycl/sycl.hpp>

namespace {

struct MhlaDecodeKernel {
  const float *q;
  const float *kv_cache;
  const float *w_uk;
  const float *w_uv;
  float *output;
  float *latent;
  float *scores;
  int num_heads;
  int seq_len;
  int kv_lora_rank;
  int head_dim;
  int rope_dim;

  void operator()(sycl::id<1> id) const {
    const int head = static_cast<int>(id[0]);
    const int q_stride = head_dim + rope_dim;
    const int kv_stride = kv_lora_rank + rope_dim;
    const float *q_row = q + static_cast<size_t>(head) * q_stride;
    const float *w_uk_head =
        w_uk + static_cast<size_t>(head) * head_dim * kv_lora_rank;
    float *latent_row = latent + static_cast<size_t>(head) * kv_lora_rank;
    float *score_row = scores + static_cast<size_t>(head) * seq_len;

    for (int r = 0; r < kv_lora_rank; ++r) {
      float acc = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        acc += q_row[d] * w_uk_head[static_cast<size_t>(d) * kv_lora_rank + r];
      }
      latent_row[r] = acc;
    }

    const float scale = 1.0f / sycl::sqrt(static_cast<float>(q_stride));
    float max_score = -3.402823466e+38f;
    for (int t = 0; t < seq_len; ++t) {
      const float *cache_row = kv_cache + static_cast<size_t>(t) * kv_stride;
      float acc = 0.0f;
      for (int r = 0; r < kv_lora_rank; ++r) {
        acc += latent_row[r] * cache_row[r];
      }
      for (int k = 0; k < rope_dim; ++k) {
        acc += q_row[head_dim + k] * cache_row[kv_lora_rank + k];
      }
      acc *= scale;
      score_row[t] = acc;
      max_score = acc > max_score ? acc : max_score;
    }

    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      const float weight = sycl::exp(score_row[t] - max_score);
      score_row[t] = weight;
      denom += weight;
    }
    const float inv_denom = 1.0f / denom;

    for (int r = 0; r < kv_lora_rank; ++r) {
      latent_row[r] = 0.0f;
    }
    for (int t = 0; t < seq_len; ++t) {
      const float weight = score_row[t] * inv_denom;
      const float *cache_row = kv_cache + static_cast<size_t>(t) * kv_stride;
      for (int r = 0; r < kv_lora_rank; ++r) {
        latent_row[r] += weight * cache_row[r];
      }
    }

    const float *w_uv_head =
        w_uv + static_cast<size_t>(head) * kv_lora_rank * head_dim;
    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int r = 0; r < kv_lora_rank; ++r) {
        acc += latent_row[r] * w_uv_head[static_cast<size_t>(r) * head_dim + d];
      }
      output[static_cast<size_t>(head) * head_dim + d] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *q, const float *kv_cache, const float *w_uk,
                      const float *w_uv, float *output, int num_heads,
                      int seq_len, int kv_lora_rank, int head_dim,
                      int rope_dim) {
  if (num_heads <= 0 || seq_len <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  float *latent = sycl::malloc_device<float>(
      static_cast<size_t>(num_heads) * kv_lora_rank, queue);
  float *scores = sycl::malloc_device<float>(
      static_cast<size_t>(num_heads) * seq_len, queue);
  queue.parallel_for(sycl::range<1>(static_cast<size_t>(num_heads)),
                     MhlaDecodeKernel{q, kv_cache, w_uk, w_uv, output, latent,
                                      scores, num_heads, seq_len, kv_lora_rank,
                                      head_dim, rope_dim});
  queue.wait();
  sycl::free(scores, queue);
  sycl::free(latent, queue);
}
