#include <sycl/sycl.hpp>

namespace {

struct LoraShrinkKernel {
  const float *x;
  const float *A;
  const int *adapter_ids;
  float *shrink;
  int K;
  int R;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int token = static_cast<int>(linear / R);
    const int rank = static_cast<int>(linear % R);
    const int adapter = adapter_ids[token];
    if (adapter < 0) {
      shrink[linear] = 0.0f;
      return;
    }
    float sum = 0.0f;
    const float *xr = x + static_cast<size_t>(token) * K;
    const float *Ar = A + static_cast<size_t>(adapter) * K * R;
    for (int k = 0; k < K; ++k) {
      sum += xr[k] * Ar[static_cast<size_t>(k) * R + rank];
    }
    shrink[linear] = sum;
  }
};

struct LoraOutputKernel {
  const float *x;
  const float *W;
  const float *B;
  const int *adapter_ids;
  const float *shrink;
  float *y;
  int K;
  int N;
  int R;
  float scale;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int token = static_cast<int>(linear / N);
    const int col = static_cast<int>(linear % N);
    const float *xr = x + static_cast<size_t>(token) * K;
    float value = 0.0f;
    for (int k = 0; k < K; ++k) {
      value += xr[k] * W[static_cast<size_t>(k) * N + col];
    }
    const int adapter = adapter_ids[token];
    if (adapter >= 0) {
      const float *shrink_row = shrink + static_cast<size_t>(token) * R;
      const float *Br = B + static_cast<size_t>(adapter) * R * N;
      float adapter_value = 0.0f;
      for (int rank = 0; rank < R; ++rank) {
        adapter_value +=
            shrink_row[rank] * Br[static_cast<size_t>(rank) * N + col];
      }
      value += scale * adapter_value;
    }
    y[linear] = value;
  }
};

} // namespace

extern "C" void solve(const float *x, const float *W, const float *A,
                      const float *B, const int *adapter_ids, float *y, int T,
                      int K, int N, int R, int L, float scale) {
  sycl::queue q{sycl::property::queue::in_order()};
  float *shrink = sycl::malloc_device<float>(static_cast<size_t>(T) * R, q);
  q.parallel_for(sycl::range<1>(static_cast<size_t>(T) * R),
                 LoraShrinkKernel{x, A, adapter_ids, shrink, K, R});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(T) * N),
      LoraOutputKernel{x, W, B, adapter_ids, shrink, y, K, N, R, scale});
  q.wait();
  sycl::free(shrink, q);
}
