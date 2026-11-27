#include <sycl/sycl.hpp>

namespace {

constexpr int kD = 512;
constexpr int kQHeads = 8;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 64;
constexpr int kQDim = 512;
constexpr int kKvDim = 128;
constexpr int kGqaGroups = 4;
constexpr int kFfn = 1408;

constexpr int kORms1W = 0;
constexpr int kOWq = kORms1W + kD;
constexpr int kOWk = kOWq + kQDim * kD;
constexpr int kOWv = kOWk + kKvDim * kD;
constexpr int kOWo = kOWv + kKvDim * kD;
constexpr int kORms2W = kOWo + kD * kD;
constexpr int kOWGate = kORms2W + kD;
constexpr int kOWUp = kOWGate + kFfn * kD;
constexpr int kOWDown = kOWUp + kFfn * kD;

struct RmsNormKernel {
  const float *input;
  const float *weight;
  float *output;
  int cols;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const float *src = input + static_cast<size_t>(row) * cols;
    float *dst = output + static_cast<size_t>(row) * cols;
    float sumSquares = 0.0f;
    for (int col = 0; col < cols; ++col) {
      sumSquares += src[col] * src[col];
    }
    const float invRms =
        sycl::rsqrt(sumSquares / static_cast<float>(cols) + 1.0e-5f);
    for (int col = 0; col < cols; ++col) {
      dst[col] = src[col] * invRms * weight[col];
    }
  }
};

struct ProjectionKernel {
  const float *input;
  const float *weight;
  float *output;
  int inner;
  int cols;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / cols;
    const int col = linear % cols;
    float acc = 0.0f;
    for (int k = 0; k < inner; ++k) {
      acc += input[row * inner + k] * weight[col * inner + k];
    }
    output[linear] = acc;
  }
};

struct RopeKernel {
  float *data;
  const float *cos;
  const float *sin;
  int seqLen;
  int heads;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / (heads * (kHeadDim / 2));
    const int remainder = linear % (heads * (kHeadDim / 2));
    const int head = remainder / (kHeadDim / 2);
    const int pair = remainder % (kHeadDim / 2);
    const int base = (row * heads + head) * kHeadDim;
    const float cosine = cos[row * (kHeadDim / 2) + pair];
    const float sine = sin[row * (kHeadDim / 2) + pair];
    const float first = data[base + pair];
    const float second = data[base + pair + kHeadDim / 2];
    data[base + pair] = first * cosine - second * sine;
    data[base + pair + kHeadDim / 2] = first * sine + second * cosine;
  }
};

struct GqaAttentionKernel {
  sycl::local_accessor<float, 1> weights;
  const float *q;
  const float *k;
  const float *v;
  float *attnOut;
  int seqLen;

  void operator()(sycl::nd_item<1> item) const {
    const int task = static_cast<int>(item.get_global_id(0));
    const int head = task / seqLen;
    const int row = task % seqLen;
    const int kvHead = head / kGqaGroups;
    const int qBase = row * kQDim + head * kHeadDim;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(kHeadDim));

    float maxScore = -3.402823466e+38F;
    for (int key = 0; key <= row; ++key) {
      const int kBase = key * kKvDim + kvHead * kHeadDim;
      float dot = 0.0f;
      for (int col = 0; col < kHeadDim; ++col) {
        dot += q[qBase + col] * k[kBase + col];
      }
      const float score = dot * scale;
      weights[key] = score;
      maxScore = sycl::max(maxScore, score);
    }

    float sum = 0.0f;
    for (int key = 0; key <= row; ++key) {
      const float value = sycl::exp(weights[key] - maxScore);
      weights[key] = value;
      sum += value;
    }
    const float invSum = 1.0f / sum;

    for (int col = 0; col < kHeadDim; ++col) {
      float acc = 0.0f;
      for (int key = 0; key <= row; ++key) {
        const int vIndex = key * kKvDim + kvHead * kHeadDim + col;
        acc += weights[key] * v[vIndex];
      }
      attnOut[row * kD + head * kHeadDim + col] = acc * invSum;
    }
  }
};

struct SwiGluKernel {
  float *gate;
  const float *up;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const float x = gate[linear];
    gate[linear] = (x / (1.0f + sycl::exp(-x))) * up[linear];
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
                      const float *cos, const float *sin, int seq_len) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (seq_len <= 0) {
    q.wait();
    return;
  }

  const size_t tokenCount = static_cast<size_t>(seq_len) * kD;
  const size_t qCount = static_cast<size_t>(seq_len) * kQDim;
  const size_t kvCount = static_cast<size_t>(seq_len) * kKvDim;
  const size_t ffnCount = static_cast<size_t>(seq_len) * kFfn;

  float *xNorm = sycl::malloc_device<float>(tokenCount, q);
  float *qTensor = sycl::malloc_device<float>(qCount, q);
  float *kTensor = sycl::malloc_device<float>(kvCount, q);
  float *vTensor = sycl::malloc_device<float>(kvCount, q);
  float *attnOut = sycl::malloc_device<float>(tokenCount, q);
  float *attnProj = sycl::malloc_device<float>(tokenCount, q);
  float *hidden = sycl::malloc_device<float>(tokenCount, q);
  float *hNorm = sycl::malloc_device<float>(tokenCount, q);
  float *gate = sycl::malloc_device<float>(ffnCount, q);
  float *up = sycl::malloc_device<float>(ffnCount, q);
  float *down = sycl::malloc_device<float>(tokenCount, q);

  q.parallel_for(sycl::range<1>(seq_len),
                 RmsNormKernel{x, weights + kORms1W, xNorm, kD});
  q.parallel_for(sycl::range<1>(qCount),
                 ProjectionKernel{xNorm, weights + kOWq, qTensor, kD, kQDim});
  q.parallel_for(sycl::range<1>(kvCount),
                 ProjectionKernel{xNorm, weights + kOWk, kTensor, kD, kKvDim});
  q.parallel_for(sycl::range<1>(kvCount),
                 ProjectionKernel{xNorm, weights + kOWv, vTensor, kD, kKvDim});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(seq_len) * kQHeads * (kHeadDim / 2)),
      RopeKernel{qTensor, cos, sin, seq_len, kQHeads});
  q.parallel_for(
      sycl::range<1>(static_cast<size_t>(seq_len) * kKvHeads * (kHeadDim / 2)),
      RopeKernel{kTensor, cos, sin, seq_len, kKvHeads});
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> scores(sycl::range<1>(seq_len), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(kQHeads * seq_len), sycl::range<1>(1)),
        GqaAttentionKernel{scores, qTensor, kTensor, vTensor, attnOut,
                           seq_len});
  });
  q.parallel_for(sycl::range<1>(tokenCount),
                 ProjectionKernel{attnOut, weights + kOWo, attnProj, kD, kD});
  q.parallel_for(sycl::range<1>(tokenCount), AddKernel{x, attnProj, hidden});
  q.parallel_for(sycl::range<1>(seq_len),
                 RmsNormKernel{hidden, weights + kORms2W, hNorm, kD});
  q.parallel_for(sycl::range<1>(ffnCount),
                 ProjectionKernel{hNorm, weights + kOWGate, gate, kD, kFfn});
  q.parallel_for(sycl::range<1>(ffnCount),
                 ProjectionKernel{hNorm, weights + kOWUp, up, kD, kFfn});
  q.parallel_for(sycl::range<1>(ffnCount), SwiGluKernel{gate, up});
  q.parallel_for(sycl::range<1>(tokenCount),
                 ProjectionKernel{gate, weights + kOWDown, down, kFfn, kD});
  q.parallel_for(sycl::range<1>(tokenCount), AddKernel{hidden, down, output});
  q.wait();

  sycl::free(down, q);
  sycl::free(up, q);
  sycl::free(gate, q);
  sycl::free(hNorm, q);
  sycl::free(hidden, q);
  sycl::free(attnProj, q);
  sycl::free(attnOut, q);
  sycl::free(vTensor, q);
  sycl::free(kTensor, q);
  sycl::free(qTensor, q);
  sycl::free(xNorm, q);
}
