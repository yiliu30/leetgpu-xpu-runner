#include <sycl/sycl.hpp>

namespace {

constexpr int kMaxBeams = 64;

struct BeamSearchStepKernel {
  const float *beam_scores;
  const float *token_logprobs;
  float *new_beam_scores;
  int *parent_beam_indices;
  int *next_tokens;
  int K;
  int V;

  void operator()(sycl::id<1> id) const {
    const int b = static_cast<int>(id[0]);
    const float *scores = beam_scores + static_cast<std::size_t>(b) * K;
    const float *logprobs =
        token_logprobs + static_cast<std::size_t>(b) * K * V;

    float top_vals[kMaxBeams];
    int top_idx[kMaxBeams];
    int count = 0;
    const long long total = static_cast<long long>(K) * V;

    for (long long c = 0; c < total; ++c) {
      const int k = static_cast<int>(c / V);
      const int v = static_cast<int>(c % V);
      const float val =
          scores[k] + logprobs[static_cast<std::size_t>(k) * V + v];

      if (count < K) {
        int pos = count;
        while (pos > 0 && top_vals[pos - 1] < val) {
          top_vals[pos] = top_vals[pos - 1];
          top_idx[pos] = top_idx[pos - 1];
          --pos;
        }
        top_vals[pos] = val;
        top_idx[pos] = static_cast<int>(c);
        ++count;
      } else if (val > top_vals[K - 1]) {
        int pos = K - 1;
        while (pos > 0 && top_vals[pos - 1] < val) {
          top_vals[pos] = top_vals[pos - 1];
          top_idx[pos] = top_idx[pos - 1];
          --pos;
        }
        top_vals[pos] = val;
        top_idx[pos] = static_cast<int>(c);
      }
    }

    float *out_scores = new_beam_scores + static_cast<std::size_t>(b) * K;
    int *out_parent = parent_beam_indices + static_cast<std::size_t>(b) * K;
    int *out_tok = next_tokens + static_cast<std::size_t>(b) * K;
    for (int k = 0; k < K; ++k) {
      out_scores[k] = top_vals[k];
      out_parent[k] = top_idx[k] / V;
      out_tok[k] = top_idx[k] % V;
    }
  }
};

} // namespace

extern "C" void solve(const float *beam_scores, const float *token_logprobs,
                      float *new_beam_scores, int *parent_beam_indices,
                      int *next_tokens, int B, int K, int V) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(static_cast<std::size_t>(B)),
                 BeamSearchStepKernel{beam_scores, token_logprobs,
                                      new_beam_scores, parent_beam_indices,
                                      next_tokens, K, V});
  q.wait();
}
