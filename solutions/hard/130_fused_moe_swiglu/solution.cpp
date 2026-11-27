#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 128;

struct MoeSwigluKernel {
  sycl::local_accessor<float, 1> hidden;
  sycl::local_accessor<float, 1> accum;
  const float *x;
  const int *topk_ids;
  const float *topk_weights;
  const float *w_gate;
  const float *w_up;
  const float *w_down;
  float *output;
  int H;
  int I;
  int K;

  void operator()(sycl::nd_item<1> item) const {
    const int t = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int wg = static_cast<int>(item.get_local_range(0));
    const float *xrow = x + static_cast<std::size_t>(t) * H;

    for (int j = tid; j < H; j += wg) {
      accum[j] = 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int k = 0; k < K; ++k) {
      const int e = topk_ids[static_cast<std::size_t>(t) * K + k];
      const float w = topk_weights[static_cast<std::size_t>(t) * K + k];
      const float *gateE = w_gate + static_cast<std::size_t>(e) * H * I;
      const float *upE = w_up + static_cast<std::size_t>(e) * H * I;
      const float *downE = w_down + static_cast<std::size_t>(e) * I * H;

      for (int i = tid; i < I; i += wg) {
        float gate = 0.0f;
        float up = 0.0f;
        for (int h = 0; h < H; ++h) {
          const float xv = xrow[h];
          gate += xv * gateE[static_cast<std::size_t>(h) * I + i];
          up += xv * upE[static_cast<std::size_t>(h) * I + i];
        }
        const float sigmoid = 1.0f / (1.0f + sycl::exp(-gate));
        hidden[i] = gate * sigmoid * up;
      }
      item.barrier(sycl::access::fence_space::local_space);

      for (int j = tid; j < H; j += wg) {
        float sum = 0.0f;
        for (int i = 0; i < I; ++i) {
          sum += hidden[i] * downE[static_cast<std::size_t>(i) * H + j];
        }
        accum[j] += w * sum;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    for (int j = tid; j < H; j += wg) {
      output[static_cast<std::size_t>(t) * H + j] = accum[j];
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const int *topk_ids,
                      const float *topk_weights, const float *w_gate,
                      const float *w_up, const float *w_down, float *output,
                      int T, int H, int I, int E, int K) {
  (void)E;
  sycl::queue q{sycl::property::queue::in_order()};
  if (T > 0 && H > 0 && I > 0 && K > 0) {
    const sycl::range<1> local(kWorkGroupSize);
    const sycl::range<1> global(static_cast<std::size_t>(T) * kWorkGroupSize);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> hidden(sycl::range<1>(I), h);
      sycl::local_accessor<float, 1> accum(sycl::range<1>(H), h);
      h.parallel_for(sycl::nd_range<1>(global, local),
                     MoeSwigluKernel{hidden, accum, x, topk_ids, topk_weights,
                                     w_gate, w_up, w_down, output, H, I, K});
    });
  }
  q.wait();
}
