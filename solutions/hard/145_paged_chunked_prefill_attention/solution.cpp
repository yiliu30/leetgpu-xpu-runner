#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 128;
constexpr int kMaxContext = 8192;

struct PagedChunkedPrefillKernel {
  sycl::local_accessor<float, 1> qloc;
  sycl::local_accessor<float, 1> scores;
  sycl::local_accessor<float, 1> part;
  sycl::local_accessor<float, 1> scal;
  const float *Q;
  const float *K_cache;
  const float *V_cache;
  const int *block_table;
  const int *context_lens;
  const int *cu_seqlens_q;
  float *output;
  int num_seqs;
  int num_q_heads;
  int num_kv_heads;
  int head_dim;
  int block_size;
  int max_blocks_per_seq;
  int group_size;

  void operator()(sycl::nd_item<1> item) const {
    const int work = static_cast<int>(item.get_group(0));
    const int row = work / num_q_heads;
    const int head = work - row * num_q_heads;
    const int tid = static_cast<int>(item.get_local_id(0));
    const int wg = static_cast<int>(item.get_local_range(0));

    int seq = num_seqs - 1;
    for (int i = 0; i < num_seqs; ++i) {
      if (row < cu_seqlens_q[i + 1]) {
        seq = i;
        break;
      }
    }
    const int q_start = cu_seqlens_q[seq];
    const int q_end = cu_seqlens_q[seq + 1];
    const int context = context_lens[seq];
    const int q_len = q_end - q_start;
    const int position = context - q_len + (row - q_start);
    const int num_keys = position + 1;
    const int kv_head = head / group_size;
    const int *table =
        block_table + static_cast<std::size_t>(seq) * max_blocks_per_seq;

    const std::size_t q_base =
        (static_cast<std::size_t>(row) * num_q_heads + head) * head_dim;
    for (int d = tid; d < head_dim; d += wg) {
      qloc[d] = Q[q_base + d];
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
    for (int k = tid; k < num_keys; k += wg) {
      const int block = table[k / block_size];
      const std::size_t slot =
          static_cast<std::size_t>(block) * block_size + (k % block_size);
      const float *krow = K_cache + (slot * num_kv_heads + kv_head) * head_dim;
      float dot = 0.0f;
      for (int d = 0; d < head_dim; ++d) {
        dot += qloc[d] * krow[d];
      }
      scores[k] = dot * scale;
    }
    item.barrier(sycl::access::fence_space::local_space);

    float local_max = -1.0e38f;
    for (int k = tid; k < num_keys; k += wg) {
      local_max = sycl::max(local_max, scores[k]);
    }
    part[tid] = local_max;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float m = part[0];
      for (int i = 1; i < wg; ++i) {
        m = sycl::max(m, part[i]);
      }
      scal[0] = m;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float row_max = scal[0];
    float local_sum = 0.0f;
    for (int k = tid; k < num_keys; k += wg) {
      const float e = sycl::exp(scores[k] - row_max);
      scores[k] = e;
      local_sum += e;
    }
    part[tid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float sum = 0.0f;
      for (int i = 0; i < wg; ++i) {
        sum += part[i];
      }
      scal[1] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float inv_sum = scal[1];
    for (int d = tid; d < head_dim; d += wg) {
      float acc = 0.0f;
      for (int k = 0; k < num_keys; ++k) {
        const int block = table[k / block_size];
        const std::size_t slot =
            static_cast<std::size_t>(block) * block_size + (k % block_size);
        acc +=
            scores[k] * V_cache[(slot * num_kv_heads + kv_head) * head_dim + d];
      }
      output[q_base + d] = acc * inv_sum;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K_cache,
                      const float *V_cache, const int *block_table,
                      const int *context_lens, const int *cu_seqlens_q,
                      float *output, int num_seqs, int total_q, int num_q_heads,
                      int num_kv_heads, int head_dim, int block_size,
                      int max_blocks_per_seq) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (num_seqs > 0 && total_q > 0 && num_q_heads > 0 && num_kv_heads > 0 &&
      head_dim > 0 && block_size > 0) {
    const int group_size = num_q_heads / num_kv_heads;
    const sycl::range<1> local(kWorkGroupSize);
    const sycl::range<1> global(static_cast<std::size_t>(total_q) *
                                num_q_heads * kWorkGroupSize);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> qloc(sycl::range<1>(head_dim), h);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(kMaxContext), h);
      sycl::local_accessor<float, 1> part(sycl::range<1>(kWorkGroupSize), h);
      sycl::local_accessor<float, 1> scal(sycl::range<1>(2), h);
      h.parallel_for(sycl::nd_range<1>(global, local),
                     PagedChunkedPrefillKernel{
                         qloc, scores, part, scal, Q, K_cache, V_cache,
                         block_table, context_lens, cu_seqlens_q, output,
                         num_seqs, num_q_heads, num_kv_heads, head_dim,
                         block_size, max_blocks_per_seq, group_size});
    });
  }
  q.wait();
}
