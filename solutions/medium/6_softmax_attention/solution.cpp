#include <sycl/sycl.hpp>

namespace {

struct SoftmaxAttentionKernel {
  sycl::local_accessor<float, 1> scores;
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int N;
  int d;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int row = static_cast<int>(item.get_group(0));
    const int qOffset = row * d;

    float maxScore = -3.402823466e+38F;
    for (int n = 0; n < N; ++n) {
      const int kOffset = n * d;
      float dot = 0.0f;
      for (int j = 0; j < d; ++j) {
        dot += Q[qOffset + j] * K[kOffset + j];
      }
      const float score = dot * scale;
      scores[n] = score;
      maxScore = sycl::max(maxScore, score);
    }

    float sum = 0.0f;
    for (int n = 0; n < N; ++n) {
      scores[n] = sycl::exp(scores[n] - maxScore);
      sum += scores[n];
    }
    const float invSum = 1.0f / sum;

    for (int j = 0; j < d; ++j) {
      float acc = 0.0f;
      for (int n = 0; n < N; ++n) {
        acc += scores[n] * invSum * V[n * d + j];
      }
      output[qOffset + j] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V, float *output,
                      int M, int N, int d) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M > 0 && N > 0 && d > 0) {
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> scores(sycl::range<1>(N), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M), sycl::range<1>(1)),
                     SoftmaxAttentionKernel{scores, Q, K, V, output, N, d, scale});
    });
  }
  q.wait();
}
