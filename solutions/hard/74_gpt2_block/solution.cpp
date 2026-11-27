#include <sycl/sycl.hpp>

namespace {

constexpr int kD = 768;
constexpr int kHeads = 12;
constexpr int kHeadDim = 64;
constexpr int kFfn = 3072;

constexpr int kOLn1W = 0;
constexpr int kOLn1B = kOLn1W + kD;
constexpr int kOWqkv = kOLn1B + kD;
constexpr int kOBqkv = kOWqkv + kD * 3 * kD;
constexpr int kOWAttn = kOBqkv + 3 * kD;
constexpr int kOBAttn = kOWAttn + kD * kD;
constexpr int kOLn2W = kOBAttn + kD;
constexpr int kOLn2B = kOLn2W + kD;
constexpr int kOWFc = kOLn2B + kD;
constexpr int kOBFc = kOWFc + kD * kFfn;
constexpr int kOWProj = kOBFc + kFfn;
constexpr int kOBProj = kOWProj + kFfn * kD;

struct LayerNormKernel {
  const float *input;
  const float *weight;
  const float *bias;
  float *output;
  int cols;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const float *src = input + static_cast<size_t>(row) * cols;
    float *dst = output + static_cast<size_t>(row) * cols;
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sum += src[col];
    }
    const float mean = sum / static_cast<float>(cols);
    float variance = 0.0f;
    for (int col = 0; col < cols; ++col) {
      const float delta = src[col] - mean;
      variance += delta * delta;
    }
    const float invStd =
        sycl::rsqrt(variance / static_cast<float>(cols) + 1.0e-5f);
    for (int col = 0; col < cols; ++col) {
      dst[col] = (src[col] - mean) * invStd * weight[col] + bias[col];
    }
  }
};

struct MatmulKernel {
  const float *a;
  const float *b;
  const float *bias;
  float *c;
  int inner;
  int cols;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / cols;
    const int col = linear % cols;
    float acc = 0.0f;
    for (int k = 0; k < inner; ++k) {
      acc += a[row * inner + k] * b[k * cols + col];
    }
    c[linear] = acc + bias[col];
  }
};

struct GeluKernel {
  float *values;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const float x = values[linear];
    const float cubic = x * x * x;
    values[linear] =
        0.5f * x *
        (1.0f + sycl::tanh(0.7978845608028654f * (x + 0.044715f * cubic)));
  }
};

struct AttentionKernel {
  sycl::local_accessor<float, 1> weights;
  const float *qkv;
  float *attnOut;
  int seqLen;

  void operator()(sycl::nd_item<1> item) const {
    const int task = static_cast<int>(item.get_global_id(0));
    const int head = task / seqLen;
    const int row = task % seqLen;
    const int qBase = row * (3 * kD) + head * kHeadDim;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(kHeadDim));

    float maxScore = -3.402823466e+38F;
    for (int key = 0; key < seqLen; ++key) {
      const int kBase = key * (3 * kD) + kD + head * kHeadDim;
      float dot = 0.0f;
      for (int col = 0; col < kHeadDim; ++col) {
        dot += qkv[qBase + col] * qkv[kBase + col];
      }
      const float score = dot * scale;
      weights[key] = score;
      maxScore = sycl::max(maxScore, score);
    }

    float sum = 0.0f;
    for (int key = 0; key < seqLen; ++key) {
      const float value = sycl::exp(weights[key] - maxScore);
      weights[key] = value;
      sum += value;
    }
    const float invSum = 1.0f / sum;

    for (int col = 0; col < kHeadDim; ++col) {
      float acc = 0.0f;
      for (int key = 0; key < seqLen; ++key) {
        const int vIndex = key * (3 * kD) + 2 * kD + head * kHeadDim + col;
        acc += weights[key] * qkv[vIndex];
      }
      attnOut[row * kD + head * kHeadDim + col] = acc * invSum;
    }
  }
};

struct AddKernel {
  const float *a;
  const float *b;
  float *output;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    output[linear] = a[linear] + b[linear];
  }
};

} // namespace

extern "C" void solve(const float *x, float *output, const float *weights,
                      int seq_len) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (seq_len <= 0) {
    q.wait();
    return;
  }

  const size_t tokenCount = static_cast<size_t>(seq_len) * kD;
  const size_t qkvCount = static_cast<size_t>(seq_len) * 3 * kD;
  const size_t ffnCount = static_cast<size_t>(seq_len) * kFfn;

  float *xNorm = sycl::malloc_device<float>(tokenCount, q);
  float *qkv = sycl::malloc_device<float>(qkvCount, q);
  float *attnOut = sycl::malloc_device<float>(tokenCount, q);
  float *attnProj = sycl::malloc_device<float>(tokenCount, q);
  float *hidden = sycl::malloc_device<float>(tokenCount, q);
  float *hNorm = sycl::malloc_device<float>(tokenCount, q);
  float *fc = sycl::malloc_device<float>(ffnCount, q);
  float *proj = sycl::malloc_device<float>(tokenCount, q);

  q.parallel_for(
      sycl::range<1>(seq_len),
      LayerNormKernel{x, weights + kOLn1W, weights + kOLn1B, xNorm, kD});
  q.parallel_for(
      sycl::range<1>(qkvCount),
      MatmulKernel{xNorm, weights + kOWqkv, weights + kOBqkv, qkv, kD, 3 * kD});
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scores(sycl::range<1>(seq_len), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(kHeads * seq_len), sycl::range<1>(1)),
        AttentionKernel{scores, qkv, attnOut, seq_len});
  });
  q.parallel_for(sycl::range<1>(tokenCount),
                 MatmulKernel{attnOut, weights + kOWAttn, weights + kOBAttn,
                              attnProj, kD, kD});
  q.parallel_for(sycl::range<1>(tokenCount), AddKernel{x, attnProj, hidden});
  q.parallel_for(
      sycl::range<1>(seq_len),
      LayerNormKernel{hidden, weights + kOLn2W, weights + kOLn2B, hNorm, kD});
  q.parallel_for(
      sycl::range<1>(ffnCount),
      MatmulKernel{hNorm, weights + kOWFc, weights + kOBFc, fc, kD, kFfn});
  q.parallel_for(sycl::range<1>(ffnCount), GeluKernel{fc});
  q.parallel_for(
      sycl::range<1>(tokenCount),
      MatmulKernel{fc, weights + kOWProj, weights + kOBProj, proj, kFfn, kD});
  q.parallel_for(sycl::range<1>(tokenCount), AddKernel{hidden, proj, output});
  q.wait();

  sycl::free(proj, q);
  sycl::free(fc, q);
  sycl::free(hNorm, q);
  sycl::free(hidden, q);
  sycl::free(attnProj, q);
  sycl::free(attnOut, q);
  sycl::free(qkv, q);
  sycl::free(xNorm, q);
}
