#include <sycl/sycl.hpp>

namespace {

constexpr int VOCAB_SIZE = 10;
constexpr int PROMPT_LEN = 31;
constexpr int OUTPUT_DIGITS = 11;
constexpr int MAX_SEQ_LEN = PROMPT_LEN + OUTPUT_DIGITS;
constexpr float RMS_EPS = 1e-6f;
constexpr float EMBED_CONST = 1000.0f;

struct AdderTransformerKernel {
  const int *prompts;
  float *output;
  const float *weights;

  void operator()(sycl::id<1> globalId) const {
    const int batch = static_cast<int>(globalId[0]);
    int tokens[MAX_SEQ_LEN];
    for (int i = 0; i < PROMPT_LEN; ++i) {
      tokens[i] = prompts[batch * PROMPT_LEN + i];
    }

    const float embedW0 = weights[0];
    const float embedW1 = weights[1];
    const float qW0 = weights[2];
    const float qW1 = weights[3];
    const float vW = weights[4];
    const float gateW0 = weights[5];
    const float gateW1 = weights[6];
    const float carryW = weights[7];
    const float normW0 = weights[8];
    const float normW1 = weights[9];

    constexpr double omega = 0.33069396353576769;
    constexpr float attentionScale = 52.917649884949959f;
    const float omegaF = static_cast<float>(omega);

    float qRot0[MAX_SEQ_LEN];
    float qRot1[MAX_SEQ_LEN];
    float kRot0[MAX_SEQ_LEN];
    float kRot1[MAX_SEQ_LEN];
    float value0[MAX_SEQ_LEN];
    float value1[MAX_SEQ_LEN];
    float scores[MAX_SEQ_LEN];

    for (int step = 0; step < OUTPUT_DIGITS; ++step) {
      const int seqLen = PROMPT_LEN + step;

      for (int pos = 0; pos < seqLen; ++pos) {
        const float digit = static_cast<float>(tokens[pos]);
        const float h0 = embedW0 - embedW1 * digit * digit;
        const float h1 = -digit;
        float invNorm = sycl::rsqrt(0.5f * (h0 * h0 + h1 * h1) + RMS_EPS);
        const float hn0 = h0 * invNorm;
        const float hn1 = h1 * invNorm;

        const float q0 = hn0 * qW0;
        const float q1 = hn0 * qW1;
        invNorm = sycl::rsqrt(0.5f * (q0 * q0 + q1 * q1) + RMS_EPS);
        const float qn0 = q0 * invNorm;
        const float qn1 = q1 * invNorm;

        const float k0 = hn0;
        invNorm = sycl::rsqrt(0.5f * (k0 * k0) + RMS_EPS);
        const float kn0 = k0 * invNorm;

        const float v0 = hn1 * vW;
        const float angle = static_cast<float>(pos) * omegaF;
        const float cosine = sycl::cos(angle);
        const float sine = sycl::sin(angle);

        qRot0[pos] = qn0 * cosine - qn1 * sine;
        qRot1[pos] = qn0 * sine + qn1 * cosine;
        kRot0[pos] = kn0 * cosine;
        kRot1[pos] = kn0 * sine;
        value0[pos] = v0;
        value1[pos] = 0.0f;
      }

      for (int pos = 0; pos < seqLen; ++pos) {
        float maxScore = -3.402823466e+38F;
        for (int key = 0; key <= pos; ++key) {
          const float score =
              (qRot0[pos] * kRot0[key] + qRot1[pos] * kRot1[key]) *
              attentionScale;
          scores[key] = score;
          maxScore = sycl::max(maxScore, score);
        }
        float sum = 0.0f;
        for (int key = 0; key <= pos; ++key) {
          scores[key] = sycl::exp(scores[key] - maxScore);
          sum += scores[key];
        }
        const float invSum = 1.0f / sum;
        float attention0 = 0.0f;
        float attention1 = 0.0f;
        for (int key = 0; key <= pos; ++key) {
          const float probability = scores[key] * invSum;
          attention0 += probability * value0[key];
          attention1 += probability * value1[key];
        }

        const float digit = static_cast<float>(tokens[pos]);
        float h0 = embedW0 - embedW1 * digit * digit;
        float h1 = -digit;
        h1 += attention0;

        float invNorm = sycl::rsqrt(0.5f * (h0 * h0 + h1 * h1) + RMS_EPS);
        const float hn0 = h0 * invNorm;
        const float hn1 = h1 * invNorm;
        const float g0 = hn0 * gateW0 + hn1 * gateW1;
        const float g1 = hn0 * (gateW0 - gateW1 / EMBED_CONST) + hn1 * gateW1;
        const float base = hn0;
        const float mix0 = (g0 / (1.0f + sycl::exp(-g0))) * base;
        const float mix1 = (g1 / (1.0f + sycl::exp(-g1))) * base;
        h1 += carryW * (mix1 - mix0);

        const float rms = sycl::sqrt(0.5f * (h0 * h0 + h1 * h1) + RMS_EPS);
        h0 = (h0 / rms) * normW0;
        h1 = (h1 / rms) * normW1;

        if (pos == seqLen - 1) {
          int bestToken = 0;
          float bestLogit = -3.402823466e+38F;
          for (int token = 0; token < VOCAB_SIZE; ++token) {
            const float tokenValue = static_cast<float>(token);
            const float table0 = embedW0 - embedW1 * tokenValue * tokenValue;
            const float table1 = -tokenValue;
            const float logit = h0 * table0 + h1 * table1;
            output[(batch * OUTPUT_DIGITS + step) * VOCAB_SIZE + token] = logit;
            if (logit > bestLogit) {
              bestLogit = logit;
              bestToken = token;
            }
          }
          if (step + 1 < OUTPUT_DIGITS) {
            tokens[seqLen] = bestToken;
          }
        }
      }
    }
  }
};

} // namespace

extern "C" void solve(const int *prompts, float *output, const float *weights,
                      int batch_size) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (batch_size > 0) {
    q.parallel_for(sycl::range<1>(batch_size),
                   AdderTransformerKernel{prompts, output, weights});
  }
  q.wait();
}
