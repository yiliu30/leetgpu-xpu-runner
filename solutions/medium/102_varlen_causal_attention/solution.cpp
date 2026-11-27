#include <sycl/sycl.hpp>

namespace {

struct VarlenAttentionKernel {
  const float *Q;
  const float *K;
  const float *V;
  const int *cu_seqlens;
  float *output;
  int T;
  int d;
  int S;
  float scale;

  void operator()(sycl::id<1> idx) const {
    const int t = idx[0];
    if (t >= T) {
      return;
    }
    int seq = 0;
    while (seq + 1 < S && t >= cu_seqlens[seq + 1]) {
      ++seq;
    }
    const int start = cu_seqlens[seq];
    const float *q_row = Q + static_cast<long>(t) * d;

    float row_max = -3.402823466e+38F;
    for (int u = start; u <= t; ++u) {
      const float *k_row = K + static_cast<long>(u) * d;
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) {
        dot += q_row[c] * k_row[c];
      }
      row_max = sycl::max(row_max, dot * scale);
    }

    double row_sum = 0.0;
    for (int u = start; u <= t; ++u) {
      const float *k_row = K + static_cast<long>(u) * d;
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) {
        dot += q_row[c] * k_row[c];
      }
      row_sum += sycl::exp(dot * scale - row_max);
    }

    float *out_row = output + static_cast<long>(t) * d;
    for (int c = 0; c < d; ++c) {
      double acc = 0.0;
      for (int u = start; u <= t; ++u) {
        const float *k_row = K + static_cast<long>(u) * d;
        float dot = 0.0f;
        for (int kc = 0; kc < d; ++kc) {
          dot += q_row[kc] * k_row[kc];
        }
        const double weight = sycl::exp(dot * scale - row_max);
        acc += weight * static_cast<double>(V[static_cast<long>(u) * d + c]);
      }
      out_row[c] = static_cast<float>(acc / row_sum);
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V, const int *cu_seqlens,
                      float *output, int T, int d, int S) {
  sycl::queue q{sycl::property::queue::in_order()};
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));
  q.parallel_for(sycl::range<1>(T),
                 VarlenAttentionKernel{Q, K, V, cu_seqlens, output, T, d, S, scale});
  q.wait();
}
