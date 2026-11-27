#include <sycl/sycl.hpp>

namespace {

struct LoraLinearKernel {
  const float *x;
  const float *W;
  const float *A;
  const float *B;
  float *output;
  int d_in;
  int d_out;
  int rank;
  float lora_scale;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int row = static_cast<int>(idx / static_cast<std::size_t>(d_out));
    const int col = static_cast<int>(idx % static_cast<std::size_t>(d_out));

    const float *xrow = x + static_cast<std::size_t>(row) * d_in;
    float base = 0.0f;
    const float *wrow = W + static_cast<std::size_t>(col) * d_in;
    for (int k = 0; k < d_in; ++k) {
      base += xrow[k] * wrow[k];
    }

    float delta = 0.0f;
    for (int r = 0; r < rank; ++r) {
      const float *arow = A + static_cast<std::size_t>(r) * d_in;
      float hidden = 0.0f;
      for (int k = 0; k < d_in; ++k) {
        hidden += xrow[k] * arow[k];
      }
      delta += hidden * B[static_cast<std::size_t>(col) * rank + r];
    }

    output[idx] = base + lora_scale * delta;
  }
};

} // namespace

extern "C" void solve(const float *x, const float *W, const float *A, const float *B, float *output,
                      int batch, int d_in, int d_out, int rank, float lora_scale) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(batch) * d_out),
                 LoraLinearKernel{x, W, A, B, output, d_in, d_out, rank, lora_scale});
  q.wait();
}
