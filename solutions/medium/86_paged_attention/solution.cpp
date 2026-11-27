#include <sycl/sycl.hpp>

namespace {

struct PagedAttentionKernel {
  const float *Q;
  const float *K_cache;
  const float *V_cache;
  const int *block_table;
  const int *context_lens;
  float *output;
  int num_heads;
  int head_dim;
  int block_size;
  int max_blocks_per_seq;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int d = static_cast<int>(idx % static_cast<std::size_t>(head_dim));
    const std::size_t head_idx = idx / static_cast<std::size_t>(head_dim);
    const int h =
        static_cast<int>(head_idx % static_cast<std::size_t>(num_heads));
    const int b =
        static_cast<int>(head_idx / static_cast<std::size_t>(num_heads));

    const int ctx = context_lens[b];
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    const float *qrow =
        Q + (static_cast<std::size_t>(b) * num_heads + h) * head_dim;
    const int *table_row =
        block_table + static_cast<std::size_t>(b) * max_blocks_per_seq;

    float max_score = -1.0e38f;
    for (int t = 0; t < ctx; ++t) {
      const int phys = table_row[t / block_size];
      const int off = t % block_size;
      const std::size_t kbase =
          (static_cast<std::size_t>(phys) * block_size + off) * num_heads *
              head_dim +
          static_cast<std::size_t>(h) * head_dim;
      const float *krow = K_cache + kbase;
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        s += qrow[i] * krow[i];
      }
      s *= scale;
      max_score = s > max_score ? s : max_score;
    }

    float denom = 0.0f;
    float acc = 0.0f;
    for (int t = 0; t < ctx; ++t) {
      const int phys = table_row[t / block_size];
      const int off = t % block_size;
      const std::size_t kvbase =
          (static_cast<std::size_t>(phys) * block_size + off) * num_heads *
              head_dim +
          static_cast<std::size_t>(h) * head_dim;
      const float *krow = K_cache + kvbase;
      float s = 0.0f;
      for (int i = 0; i < head_dim; ++i) {
        s += qrow[i] * krow[i];
      }
      s *= scale;
      const float w = sycl::exp(s - max_score);
      denom += w;
      acc += w * V_cache[kvbase + d];
    }
    output[idx] = acc / denom;
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K_cache,
                      const float *V_cache, const int *block_table,
                      const int *context_lens, float *output, int batch_size,
                      int num_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(batch_size) *
                                num_heads * head_dim),
                 PagedAttentionKernel{Q, K_cache, V_cache, block_table,
                                      context_lens, output, num_heads, head_dim,
                                      block_size, max_blocks_per_seq});
  q.wait();
}
