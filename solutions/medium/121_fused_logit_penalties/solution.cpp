#include <sycl/sycl.hpp>

namespace {

struct CountTokensKernel {
  const int *tokens;
  int *counts;
  int V;
  int tokens_per_row;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int row = static_cast<int>(linear / tokens_per_row);
    const int token = tokens[linear];
    if (token >= 0) {
      int *count = counts + static_cast<size_t>(row) * V + token;
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          atomic_count(*count);
      atomic_count.fetch_add(1);
    }
  }
};

struct ApplyPenaltiesKernel {
  const float *logits;
  const int *prompt_counts;
  const int *output_counts;
  const float *presence_penalty;
  const float *frequency_penalty;
  const float *repetition_penalty;
  float *output;
  int V;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int row = static_cast<int>(linear / V);
    const int col = static_cast<int>(linear % V);
    const size_t count_index = static_cast<size_t>(row) * V + col;
    const int prompt_count = prompt_counts[count_index];
    const int output_count = output_counts[count_index];
    const bool seen = prompt_count + output_count > 0;
    const bool generated = output_count > 0;

    float value = logits[linear];
    if (seen) {
      const float repetition = repetition_penalty[row];
      value = value > 0.0f ? value / repetition : value * repetition;
    }
    value -= frequency_penalty[row] * static_cast<float>(output_count);
    if (generated) {
      value -= presence_penalty[row];
    }
    output[linear] = value;
  }
};

} // namespace

extern "C" void solve(const float *logits, const int *prompt_tokens,
                      const int *output_tokens, const float *presence_penalty,
                      const float *frequency_penalty,
                      const float *repetition_penalty, float *output, int B,
                      int V, int P, int G) {
  sycl::queue q{sycl::property::queue::in_order()};
  const size_t count_size = static_cast<size_t>(B) * V;
  int *prompt_counts = sycl::malloc_device<int>(count_size, q);
  int *output_counts = sycl::malloc_device<int>(count_size, q);
  q.memset(prompt_counts, 0, count_size * sizeof(int));
  q.memset(output_counts, 0, count_size * sizeof(int));
  q.parallel_for(sycl::range<1>(static_cast<size_t>(B) * P),
                 CountTokensKernel{prompt_tokens, prompt_counts, V, P});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(B) * G),
                 CountTokensKernel{output_tokens, output_counts, V, G});
  q.parallel_for(sycl::range<1>(count_size),
                 ApplyPenaltiesKernel{logits, prompt_counts, output_counts,
                                      presence_penalty, frequency_penalty,
                                      repetition_penalty, output, V});
  q.wait();
  sycl::free(prompt_counts, q);
  sycl::free(output_counts, q);
}
