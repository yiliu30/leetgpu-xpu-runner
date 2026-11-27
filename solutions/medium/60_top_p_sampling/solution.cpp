#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

struct TopPSamplingKernel {
  const float *logits;
  const float *p;
  const int *seed;
  int *sampledToken;
  float *probs;
  int *order;
  int vocabSize;

  static uint32_t pcgHash(uint32_t value) {
    const uint32_t state = value * 747796405u + 2891336453u;
    const uint32_t shifted = state >> (((state >> 28) + 4u) & 31u);
    const uint32_t word = ((shifted ^ state) * 277803737u);
    return (word >> 22) ^ word;
  }

  void operator()(sycl::item<1>) const {
    if (vocabSize <= 0) {
      return;
    }

    float maxLogit = logits[0];
    for (int i = 1; i < vocabSize; ++i) {
      maxLogit = sycl::max(maxLogit, logits[i]);
    }

    float sumExp = 0.0f;
    for (int i = 0; i < vocabSize; ++i) {
      probs[i] = sycl::exp(logits[i] - maxLogit);
      sumExp += probs[i];
      order[i] = i;
    }
    const float invSum = 1.0f / sumExp;
    for (int i = 0; i < vocabSize; ++i) {
      probs[i] *= invSum;
    }

    for (int i = 1; i < vocabSize; ++i) {
      const int keyIndex = order[i];
      const float keyProb = probs[keyIndex];
      int j = i - 1;
      while (j >= 0) {
        const int currentIndex = order[j];
        const float currentProb = probs[currentIndex];
        if (currentProb < keyProb ||
            (currentProb == keyProb && currentIndex > keyIndex)) {
          order[j + 1] = currentIndex;
          --j;
        } else {
          break;
        }
      }
      order[j + 1] = keyIndex;
    }

    const float pValue = p[0];
    float cumulative = 0.0f;
    int cutoff = vocabSize;
    for (int i = 0; i < vocabSize; ++i) {
      cumulative += probs[order[i]];
      if (cumulative >= pValue) {
        cutoff = i + 1;
        break;
      }
    }

    float nucleusSum = 0.0f;
    for (int i = 0; i < cutoff; ++i) {
      nucleusSum += probs[order[i]];
    }

    const uint32_t seedHash = pcgHash(static_cast<uint32_t>(seed[0]));
    double bestScore = -1.0e300;
    int bestToken = order[0];
    for (int i = 0; i < cutoff; ++i) {
      const int token = order[i];
      const uint32_t bits = pcgHash(seedHash + static_cast<uint32_t>(token));
      const double uniform = (static_cast<double>(bits) + 0.5) / 4294967296.0;
      const double score = (static_cast<double>(probs[token]) /
                            static_cast<double>(nucleusSum)) /
                           (-sycl::log(uniform));
      if (score > bestScore) {
        bestScore = score;
        bestToken = token;
      }
    }
    sampledToken[0] = bestToken;
  }
};

} // namespace

extern "C" void solve(const float *logits, const float *p, const int *seed,
                      int *sampled_token, int vocab_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (vocab_size > 0) {
    float *probs = sycl::malloc_device<float>(vocab_size, q);
    int *order = sycl::malloc_device<int>(vocab_size, q);
    q.parallel_for(sycl::range<1>(1),
                   TopPSamplingKernel{logits, p, seed, sampled_token, probs,
                                      order, vocab_size});
    q.wait();
    sycl::free(order, q);
    sycl::free(probs, q);
  }
}
