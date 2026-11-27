#include <sycl/sycl.hpp>

namespace {

struct GroupedQueryAttentionKernel {
  sycl::local_accessor<float, 1> scores;
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int groups;
  int seqLen;
  int headDim;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int task = static_cast<int>(item.get_global_id(0));
    const int queryHead = task / seqLen;
    const int queryPos = task % seqLen;
    const int kvHead = queryHead / groups;
    const int qOffset = (queryHead * seqLen + queryPos) * headDim;
    const int kvHeadOffset = kvHead * seqLen * headDim;

    float maxScore = -3.402823466e+38F;
    for (int keyPos = 0; keyPos < seqLen; ++keyPos) {
      const int kOffset = kvHeadOffset + keyPos * headDim;
      float dot = 0.0f;
      for (int j = 0; j < headDim; ++j) {
        dot += Q[qOffset + j] * K[kOffset + j];
      }
      const float score = dot * scale;
      scores[keyPos] = score;
      maxScore = sycl::max(maxScore, score);
    }

    float sum = 0.0f;
    for (int keyPos = 0; keyPos < seqLen; ++keyPos) {
      scores[keyPos] = sycl::exp(scores[keyPos] - maxScore);
      sum += scores[keyPos];
    }
    const float invSum = 1.0f / sum;

    for (int j = 0; j < headDim; ++j) {
      float acc = 0.0f;
      for (int keyPos = 0; keyPos < seqLen; ++keyPos) {
        acc += scores[keyPos] * invSum * V[kvHeadOffset + keyPos * headDim + j];
      }
      output[qOffset + j] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int num_q_heads, int num_kv_heads,
                      int seq_len, int head_dim) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (num_q_heads <= 0 || num_kv_heads <= 0 || seq_len <= 0 || head_dim <= 0) {
    return;
  }
  const int groups = num_q_heads / num_kv_heads;
  const int tasks = num_q_heads * seq_len;
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(head_dim));
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scores(sycl::range<1>(seq_len), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(tasks), sycl::range<1>(1)),
                   GroupedQueryAttentionKernel{scores, Q, K, V, output, groups,
                                               seq_len, head_dim, scale});
  });
  q.wait();
}
