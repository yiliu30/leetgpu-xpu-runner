#include <sycl/sycl.hpp>

namespace {

struct SpeculativeVerificationKernel {
  const int *draft_tokens;
  const float *draft_probs;
  const float *target_probs;
  const float *uniform_samples;
  int *output_tokens;
  int T;
  int V;

  void operator()(sycl::id<1> id) const {
    const int b = static_cast<int>(id[0]);
    int *out = output_tokens + static_cast<std::size_t>(b) * (T + 1);
    for (int i = 0; i <= T; ++i) {
      out[i] = 0;
    }

    const float *urow = uniform_samples + static_cast<std::size_t>(b) * (T + 1);
    const float r = urow[T];
    bool rejected = false;

    for (int i = 0; i < T; ++i) {
      const std::size_t row =
          (static_cast<std::size_t>(b) * T + i) * static_cast<std::size_t>(V);
      const int tok = draft_tokens[static_cast<std::size_t>(b) * T + i];
      const float p = draft_probs[row + tok];
      const float q = target_probs[row + tok];
      const double ratio = static_cast<double>(q) / static_cast<double>(p);
      const double alpha = ratio < 1.0 ? ratio : 1.0;

      if (static_cast<double>(urow[i]) < alpha) {
        out[i] = tok;
        continue;
      }

      const float *dp = draft_probs + row;
      const float *tp = target_probs + row;
      float total = 0.0f;
      for (int v = 0; v < V; ++v) {
        const float diff = tp[v] - dp[v];
        if (diff > 0.0f) {
          total += diff;
        }
      }

      float cum = 0.0f;
      int count = 0;
      if (total > 0.0f) {
        for (int v = 0; v < V; ++v) {
          const float diff = tp[v] - dp[v];
          if (diff > 0.0f) {
            cum += diff / total;
          }
          if (cum < r) {
            ++count;
          }
        }
      } else {
        const float step = 1.0f / static_cast<float>(V);
        for (int v = 0; v < V; ++v) {
          cum += step;
          if (cum < r) {
            ++count;
          }
        }
      }
      out[i] = count < V ? count : V - 1;
      rejected = true;
      break;
    }

    if (!rejected) {
      const std::size_t row = (static_cast<std::size_t>(b) * T + (T - 1)) *
                              static_cast<std::size_t>(V);
      const float *tp = target_probs + row;
      float cum = 0.0f;
      int count = 0;
      for (int v = 0; v < V; ++v) {
        cum += tp[v];
        if (cum < r) {
          ++count;
        }
      }
      out[T] = count < V ? count : V - 1;
    }
  }
};

} // namespace

extern "C" void solve(const int *draft_tokens, const float *draft_probs, const float *target_probs,
                      const float *uniform_samples, int *output_tokens, int B, int T, int V) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(B)),
                 SpeculativeVerificationKernel{draft_tokens, draft_probs, target_probs,
                                               uniform_samples, output_tokens, T, V});
  q.wait();
}
