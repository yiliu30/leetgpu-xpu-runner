#include <sycl/sycl.hpp>

namespace {

struct SlidingWindowAttentionKernel {
  sycl::local_accessor<float, 1> weights;
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int M;
  int d;
  int window;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int row = static_cast<int>(item.get_global_id(0));
    const int start = sycl::max(0, row - window);
    const int end = sycl::min(M - 1, row + window);
    const int qOffset = row * d;

    float maxScore = -3.402823466e+38F;
    for (int key = start; key <= end; ++key) {
      const int kOffset = key * d;
      float dot = 0.0f;
      for (int col = 0; col < d; ++col) {
        dot += Q[qOffset + col] * K[kOffset + col];
      }
      const float score = dot * scale;
      weights[key - start] = score;
      maxScore = sycl::max(maxScore, score);
    }

    float sum = 0.0f;
    for (int key = start; key <= end; ++key) {
      const int index = key - start;
      const float value = sycl::exp(weights[index] - maxScore);
      weights[index] = value;
      sum += value;
    }
    const float invSum = 1.0f / sum;

    for (int col = 0; col < d; ++col) {
      float acc = 0.0f;
      for (int key = start; key <= end; ++key) {
        acc += weights[key - start] * V[key * d + col];
      }
      output[qOffset + col] = acc * invSum;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int d, int window_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M > 0 && d > 0) {
    const int window = sycl::max(0, window_size);
    const int capacity = sycl::min(M, 2 * window + 1);
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> weights(sycl::range<1>(capacity), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(M), sycl::range<1>(1)),
                     SlidingWindowAttentionKernel{weights, Q, K, V, output, M,
                                                  d, window, scale});
    });
  }
  q.wait();
}
