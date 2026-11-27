#include <sycl/sycl.hpp>

namespace {

constexpr int MAX_SELECTED = 128;

struct MoeTopKGatingKernel {
  const float *logits;
  float *topkWeights;
  int *topkIndices;
  int E;
  int k;

  void operator()(sycl::id<1> globalId) const {
    const int row = static_cast<int>(globalId[0]);
    const int rowOffset = row * E;
    int selected[MAX_SELECTED];
    float selectedValues[MAX_SELECTED];

    for (int rank = 0; rank < k; ++rank) {
      float bestValue = -3.402823466e+38F;
      int bestIndex = -1;
      for (int expert = 0; expert < E; ++expert) {
        bool alreadySelected = false;
        for (int j = 0; j < rank; ++j) {
          if (selected[j] == expert) {
            alreadySelected = true;
            break;
          }
        }
        if (alreadySelected) {
          continue;
        }
        const float value = logits[rowOffset + expert];
        if (value > bestValue) {
          bestValue = value;
          bestIndex = expert;
        }
      }
      selected[rank] = bestIndex;
      selectedValues[rank] = bestValue;
    }

    float sum = 0.0f;
    for (int rank = 0; rank < k; ++rank) {
      sum += sycl::exp(selectedValues[rank] - selectedValues[0]);
    }
    const int outOffset = row * k;
    for (int rank = 0; rank < k; ++rank) {
      topkWeights[outOffset + rank] =
          sycl::exp(selectedValues[rank] - selectedValues[0]) / sum;
      topkIndices[outOffset + rank] = selected[rank];
    }
  }
};

} // namespace

extern "C" void solve(const float *logits, float *topk_weights, int *topk_indices,
                      int M, int E, int k) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M > 0 && E > 0 && k > 0) {
    q.parallel_for(sycl::range<1>(M),
                   MoeTopKGatingKernel{logits, topk_weights, topk_indices, E, k});
  }
  q.wait();
}
