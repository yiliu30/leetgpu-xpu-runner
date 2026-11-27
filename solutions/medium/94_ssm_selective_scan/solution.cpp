#include <sycl/sycl.hpp>

namespace {

constexpr int kMaxState = 64;

struct SsmSelectiveScanKernel {
  const float *u;
  const float *delta;
  const float *A;
  const float *B;
  const float *C;
  const float *skip;
  float *y;
  int seq_len;
  int d_model;
  int d_state;

  void operator()(sycl::id<1> id) const {
    const std::size_t idx = id[0];
    const int d = static_cast<int>(idx % static_cast<std::size_t>(d_model));
    const int b = static_cast<int>(idx / static_cast<std::size_t>(d_model));

    float state[kMaxState];
    for (int s = 0; s < d_state; ++s) {
      state[s] = 0.0f;
    }

    const float *Arow = A + static_cast<std::size_t>(d) * d_state;
    for (int t = 0; t < seq_len; ++t) {
      const std::size_t bd =
          (static_cast<std::size_t>(b) * seq_len + t) * d_model + d;
      const std::size_t bs =
          (static_cast<std::size_t>(b) * seq_len + t) * d_state;
      const float dt = delta[bd];
      const float ut = u[bd];

      float acc = 0.0f;
      for (int s = 0; s < d_state; ++s) {
        const float a_bar = sycl::exp(dt * Arow[s]);
        const float b_bar = dt * B[bs + s];
        state[s] = a_bar * state[s] + b_bar * ut;
        acc += C[bs + s] * state[s];
      }
      y[bd] = acc + skip[d] * ut;
    }
  }
};

} // namespace

extern "C" void solve(const float *u, const float *delta, const float *A,
                      const float *B, const float *C, const float *skip,
                      float *y, int batch, int seq_len, int d_model,
                      int d_state) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(batch) * d_model),
                 SsmSelectiveScanKernel{u, delta, A, B, C, skip, y, seq_len,
                                        d_model, d_state});
  q.wait();
}
