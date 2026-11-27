#include <sycl/sycl.hpp>

namespace {

struct FusedGateUpSiluKernel {
  const float *x;
  const float *W_gate_up;
  float *output;
  int M;
  int H;
  int I;

  void operator()(sycl::id<1> idx) const {
    const size_t linear = idx[0];
    const size_t total = static_cast<size_t>(M) * I;
    if (linear >= total) {
      return;
    }
    const int row = static_cast<int>(linear / I);
    const int col = static_cast<int>(linear % I);
    const float *x_row = x + static_cast<size_t>(row) * H;
    double gate = 0.0;
    double up = 0.0;
    for (int h = 0; h < H; ++h) {
      const double xv = static_cast<double>(x_row[h]);
      gate += xv * static_cast<double>(
                       W_gate_up[static_cast<size_t>(h) * (2 * I) + col]);
      up += xv * static_cast<double>(
                     W_gate_up[static_cast<size_t>(h) * (2 * I) + I + col]);
    }
    double sigmoid;
    if (gate >= 0.0) {
      sigmoid = 1.0 / (1.0 + sycl::exp(-gate));
    } else {
      const double e = sycl::exp(gate);
      sigmoid = e / (1.0 + e);
    }
    output[linear] = static_cast<float>(gate * sigmoid * up);
  }
};

} // namespace

extern "C" void solve(const float *x, const float *W_gate_up, float *output,
                      int M, int H, int I) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * I),
                 FusedGateUpSiluKernel{x, W_gate_up, output, M, H, I});
  q.wait();
}
