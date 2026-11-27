#include <sycl/sycl.hpp>

namespace {

inline float phi(float x) { return x > 0.0f ? x + 1.0f : sycl::exp(x); }

struct LinearAttentionStateKernel {
  const float *K;
  const float *V;
  float *state;
  int M;
  int d;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / d;
    const int col = linear % d;
    float acc = 0.0f;
    for (int key = 0; key < M; ++key) {
      acc += phi(K[key * d + row]) * V[key * d + col];
    }
    state[linear] = acc;
  }
};

struct LinearAttentionSumKernel {
  const float *K;
  float *sums;
  int M;
  int d;

  void operator()(sycl::id<1> id) const {
    const int col = static_cast<int>(id[0]);
    float acc = 0.0f;
    for (int key = 0; key < M; ++key) {
      acc += phi(K[key * d + col]);
    }
    sums[col] = acc;
  }
};

struct LinearAttentionOutputKernel {
  const float *Q;
  const float *state;
  const float *sums;
  float *output;
  int d;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / d;
    const int col = linear % d;
    float numerator = 0.0f;
    float denominator = 0.0f;
    for (int feature = 0; feature < d; ++feature) {
      const float qValue = phi(Q[row * d + feature]);
      numerator += qValue * state[feature * d + col];
      denominator += qValue * sums[feature];
    }
    output[linear] = numerator / denominator;
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int d) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M <= 0 || d <= 0) {
    q.wait();
    return;
  }

  const size_t stateSize = static_cast<size_t>(d) * d;
  float *state = sycl::malloc_device<float>(stateSize, q);
  float *sums = sycl::malloc_device<float>(d, q);

  q.parallel_for(sycl::range<1>(stateSize),
                 LinearAttentionStateKernel{K, V, state, M, d});
  q.parallel_for(sycl::range<1>(d), LinearAttentionSumKernel{K, sums, M, d});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * d),
                 LinearAttentionOutputKernel{Q, state, sums, output, d});
  q.wait();

  sycl::free(sums, q);
  sycl::free(state, q);
}
