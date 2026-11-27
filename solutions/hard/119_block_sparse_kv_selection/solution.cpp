#include <sycl/sycl.hpp>

namespace {

struct BlockSparseKvKernel {
  const float *q;
  const float *k;
  const float *v;
  float *output;
  float *block_scores;
  unsigned char *selected;
  int num_heads;
  int seq_len;
  int head_dim;
  int block_size;
  int num_selected;

  void operator()(sycl::id<1> id) const {
    const int head = static_cast<int>(id[0]);
    const int num_blocks = (seq_len + block_size - 1) / block_size;
    const int blocks_to_keep =
        num_selected < num_blocks ? num_selected : num_blocks;
    const float *q_row = q + static_cast<size_t>(head) * head_dim;
    const float *k_head = k + static_cast<size_t>(head) * seq_len * head_dim;
    const float *v_head = v + static_cast<size_t>(head) * seq_len * head_dim;
    float *scores = block_scores + static_cast<size_t>(head) * num_blocks;
    unsigned char *mask = selected + static_cast<size_t>(head) * num_blocks;

    for (int block = 0; block < num_blocks; ++block) {
      const int first = block * block_size;
      const int count =
          (seq_len - first) < block_size ? (seq_len - first) : block_size;
      const float inv_count = 1.0f / static_cast<float>(count);
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        float pooled = 0.0f;
        for (int t = 0; t < count; ++t) {
          pooled += k_head[static_cast<size_t>(first + t) * head_dim + d];
        }
        score += q_row[d] * pooled * inv_count;
      }
      scores[block] = score;
    }

    for (int block = 0; block < num_blocks; ++block) {
      int rank = 0;
      const float value = scores[block];
      for (int other = 0; other < num_blocks; ++other) {
        const float candidate = scores[other];
        if (candidate > value || (candidate == value && other < block)) {
          ++rank;
        }
      }
      mask[block] = rank < blocks_to_keep ? 1 : 0;
    }

    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    float max_score = -3.402823466e+38f;
    for (int t = 0; t < seq_len; ++t) {
      if (!mask[t / block_size]) {
        continue;
      }
      const float *k_row = k_head + static_cast<size_t>(t) * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        score += q_row[d] * k_row[d];
      }
      score *= scale;
      max_score = score > max_score ? score : max_score;
    }

    float denom = 0.0f;
    for (int t = 0; t < seq_len; ++t) {
      if (!mask[t / block_size]) {
        continue;
      }
      const float *k_row = k_head + static_cast<size_t>(t) * head_dim;
      float score = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        score += q_row[d] * k_row[d];
      }
      denom += sycl::exp(score * scale - max_score);
    }
    const float inv_denom = 1.0f / denom;

    for (int d = 0; d < head_dim; ++d) {
      float acc = 0.0f;
      for (int t = 0; t < seq_len; ++t) {
        if (!mask[t / block_size]) {
          continue;
        }
        const float *k_row = k_head + static_cast<size_t>(t) * head_dim;
        float score = 0.0f;
        for (int x = 0; x < head_dim; ++x) {
          score += q_row[x] * k_row[x];
        }
        const float weight = sycl::exp(score * scale - max_score) * inv_denom;
        acc += weight * v_head[static_cast<size_t>(t) * head_dim + d];
      }
      output[static_cast<size_t>(head) * head_dim + d] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *q, const float *k, const float *v,
                      float *output, int num_heads, int seq_len, int head_dim,
                      int block_size, int num_selected) {
  if (num_heads <= 0 || seq_len <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  const int num_blocks = (seq_len + block_size - 1) / block_size;
  float *block_scores = sycl::malloc_device<float>(
      static_cast<size_t>(num_heads) * num_blocks, queue);
  unsigned char *selected = sycl::malloc_device<unsigned char>(
      static_cast<size_t>(num_heads) * num_blocks, queue);
  queue.parallel_for(sycl::range<1>(static_cast<size_t>(num_heads)),
                     BlockSparseKvKernel{q, k, v, output, block_scores,
                                         selected, num_heads, seq_len, head_dim,
                                         block_size, num_selected});
  queue.wait();
  sycl::free(selected, queue);
  sycl::free(block_scores, queue);
}
