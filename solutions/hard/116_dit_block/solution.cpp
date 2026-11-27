#include <sycl/sycl.hpp>

namespace {

constexpr int kDim = 512;
constexpr int kHeads = 8;
constexpr int kHeadDim = 64;
constexpr int kMlp = 2048;
constexpr float kEps = 1.0e-6f;
constexpr float kGeluScale = 0.7978845608028654f;
constexpr float kGeluCubic = 0.044715f;

constexpr size_t kOffWAda = 0;
constexpr size_t kOffBAda = kOffWAda + 6 * kDim * kDim;
constexpr size_t kOffWQkv = kOffBAda + 6 * kDim;
constexpr size_t kOffBQkv = kOffWQkv + 3 * kDim * kDim;
constexpr size_t kOffWO = kOffBQkv + 3 * kDim;
constexpr size_t kOffBO = kOffWO + kDim * kDim;
constexpr size_t kOffWFc1 = kOffBO + kDim;
constexpr size_t kOffBFc1 = kOffWFc1 + kMlp * kDim;
constexpr size_t kOffWFc2 = kOffBFc1 + kMlp;
constexpr size_t kOffBFc2 = kOffWFc2 + kDim * kMlp;

inline float silu(float x) { return x / (1.0f + sycl::exp(-x)); }

inline float gelu_tanh(float x) {
  return 0.5f * x *
         (1.0f + sycl::tanh(kGeluScale * (x + kGeluCubic * x * x * x)));
}

struct DitModulationKernel {
  const float *conditioning;
  const float *weights;
  float *modulation;
  int batch_size;

  void operator()(sycl::id<1> id) const {
    const int batch = static_cast<int>(id[0] / (6 * kDim));
    const int row = static_cast<int>(id[0] % (6 * kDim));
    const float *c_row = conditioning + static_cast<size_t>(batch) * kDim;
    const float *w_row = weights + kOffWAda + static_cast<size_t>(row) * kDim;
    float acc = weights[kOffBAda + row];
    for (int d = 0; d < kDim; ++d) {
      acc += silu(c_row[d]) * w_row[d];
    }
    modulation[static_cast<size_t>(batch) * 6 * kDim + row] = acc;
  }
};

struct DitLayerNormModulateKernel {
  const float *input;
  const float *modulation;
  float *output;
  int seq_len;
  int shift_index;
  int scale_index;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const int batch = row / seq_len;
    const float *source = input + static_cast<size_t>(row) * kDim;
    float *destination = output + static_cast<size_t>(row) * kDim;
    float mean = 0.0f;
    for (int d = 0; d < kDim; ++d) {
      mean += source[d];
    }
    mean /= static_cast<float>(kDim);
    float variance = 0.0f;
    for (int d = 0; d < kDim; ++d) {
      const float diff = source[d] - mean;
      variance += diff * diff;
    }
    variance /= static_cast<float>(kDim);
    const float inv_std = sycl::rsqrt(variance + kEps);
    const float *shift =
        modulation + (static_cast<size_t>(batch) * 6 + shift_index) * kDim;
    const float *scale =
        modulation + (static_cast<size_t>(batch) * 6 + scale_index) * kDim;
    for (int d = 0; d < kDim; ++d) {
      destination[d] =
          (source[d] - mean) * inv_std * (1.0f + scale[d]) + shift[d];
    }
  }
};

struct DitQkvKernel {
  const float *hidden;
  const float *weights;
  float *qkv;
  int tokens;

  void operator()(sycl::id<1> id) const {
    const int token = static_cast<int>(id[0] / (3 * kDim));
    const int output_col = static_cast<int>(id[0] % (3 * kDim));
    const float *hidden_row = hidden + static_cast<size_t>(token) * kDim;
    const float *w_row =
        weights + kOffWQkv + static_cast<size_t>(output_col) * kDim;
    float acc = weights[kOffBQkv + output_col];
    for (int d = 0; d < kDim; ++d) {
      acc += hidden_row[d] * w_row[d];
    }
    qkv[static_cast<size_t>(token) * 3 * kDim + output_col] = acc;
  }
};

struct DitAttentionScoresKernel {
  const float *qkv;
  float *attention;
  int seq_len;

  void operator()(sycl::id<1> id) const {
    const int query_row = static_cast<int>(id[0]);
    const int query_pos = query_row % seq_len;
    const int head = (query_row / seq_len) % kHeads;
    const int batch = query_row / (seq_len * kHeads);
    const size_t query_token =
        static_cast<size_t>(batch) * seq_len + static_cast<size_t>(query_pos);
    const float *q_row = qkv + query_token * 3 * kDim + head * kHeadDim;
    float *scores = attention + static_cast<size_t>(query_row) * seq_len;
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(kHeadDim));
    float max_score = -3.402823466e+38f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      const size_t key_token =
          static_cast<size_t>(batch) * seq_len + static_cast<size_t>(key_pos);
      const float *k_row = qkv + key_token * 3 * kDim + kDim + head * kHeadDim;
      float score = 0.0f;
      for (int d = 0; d < kHeadDim; ++d) {
        score += q_row[d] * k_row[d];
      }
      score *= scale;
      scores[key_pos] = score;
      max_score = score > max_score ? score : max_score;
    }
    float denominator = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      const float weight = sycl::exp(scores[key_pos] - max_score);
      scores[key_pos] = weight;
      denominator += weight;
    }
    const float inverse = 1.0f / denominator;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      scores[key_pos] *= inverse;
    }
  }
};

struct DitAttentionContextKernel {
  const float *qkv;
  const float *attention;
  float *context;
  int seq_len;

  void operator()(sycl::id<1> id) const {
    const int d = static_cast<int>(id[0] % kDim);
    const size_t token = id[0] / kDim;
    const int query_pos = static_cast<int>(token % seq_len);
    const int batch = static_cast<int>(token / seq_len);
    const int head = d / kHeadDim;
    const int head_col = d % kHeadDim;
    const size_t score_row =
        (static_cast<size_t>(batch) * kHeads + head) * seq_len + query_pos;
    const float *scores = attention + score_row * seq_len;
    float acc = 0.0f;
    for (int key_pos = 0; key_pos < seq_len; ++key_pos) {
      const size_t key_token = static_cast<size_t>(batch) * seq_len + key_pos;
      acc += scores[key_pos] *
             qkv[key_token * 3 * kDim + 2 * kDim + head * kHeadDim + head_col];
    }
    context[token * kDim + d] = acc;
  }
};

struct DitOutProjectionKernel {
  const float *context;
  const float *input;
  const float *modulation;
  const float *weights;
  float *output;
  int seq_len;

  void operator()(sycl::id<1> id) const {
    const int d = static_cast<int>(id[0] % kDim);
    const int token = static_cast<int>(id[0] / kDim);
    const int batch = token / seq_len;
    const float *context_row = context + static_cast<size_t>(token) * kDim;
    const float *w_row = weights + kOffWO + static_cast<size_t>(d) * kDim;
    float acc = weights[kOffBO + d];
    for (int k = 0; k < kDim; ++k) {
      acc += context_row[k] * w_row[k];
    }
    const float gate =
        modulation[(static_cast<size_t>(batch) * 6 + 2) * kDim + d];
    output[static_cast<size_t>(token) * kDim + d] =
        input[static_cast<size_t>(token) * kDim + d] + gate * acc;
  }
};

struct DitFirstFcKernel {
  const float *hidden;
  const float *weights;
  float *projected;
  int tokens;

  void operator()(sycl::id<1> id) const {
    const int token = static_cast<int>(id[0] / kMlp);
    const int output_col = static_cast<int>(id[0] % kMlp);
    const float *hidden_row = hidden + static_cast<size_t>(token) * kDim;
    const float *w_row =
        weights + kOffWFc1 + static_cast<size_t>(output_col) * kDim;
    float acc = weights[kOffBFc1 + output_col];
    for (int d = 0; d < kDim; ++d) {
      acc += hidden_row[d] * w_row[d];
    }
    projected[static_cast<size_t>(token) * kMlp + output_col] = gelu_tanh(acc);
  }
};

struct DitSecondFcKernel {
  const float *projected;
  const float *residual;
  const float *modulation;
  const float *weights;
  float *output;
  int seq_len;

  void operator()(sycl::id<1> id) const {
    const int d = static_cast<int>(id[0] % kDim);
    const int token = static_cast<int>(id[0] / kDim);
    const int batch = token / seq_len;
    const float *projected_row = projected + static_cast<size_t>(token) * kMlp;
    const float *w_row = weights + kOffWFc2 + static_cast<size_t>(d) * kMlp;
    float acc = weights[kOffBFc2 + d];
    for (int k = 0; k < kMlp; ++k) {
      acc += projected_row[k] * w_row[k];
    }
    const float gate =
        modulation[(static_cast<size_t>(batch) * 6 + 5) * kDim + d];
    output[static_cast<size_t>(token) * kDim + d] =
        residual[static_cast<size_t>(token) * kDim + d] + gate * acc;
  }
};

} // namespace

extern "C" void solve(const float *x, const float *c, float *output,
                      const float *weights, int batch_size, int seq_len) {
  if (batch_size <= 0 || seq_len <= 0) {
    return;
  }
  const size_t tokens = static_cast<size_t>(batch_size) * seq_len;
  sycl::queue queue{sycl::property::queue::in_order()};
  float *modulation = sycl::malloc_device<float>(
      static_cast<size_t>(batch_size) * 6 * kDim, queue);
  float *hidden = sycl::malloc_device<float>(tokens * kDim, queue);
  float *qkv = sycl::malloc_device<float>(tokens * 3 * kDim, queue);
  float *attention = sycl::malloc_device<float>(
      static_cast<size_t>(batch_size) * kHeads * seq_len * seq_len, queue);
  float *context = sycl::malloc_device<float>(tokens * kDim, queue);
  float *residual = sycl::malloc_device<float>(tokens * kDim, queue);
  float *hidden2 = sycl::malloc_device<float>(tokens * kDim, queue);
  float *projected = sycl::malloc_device<float>(tokens * kMlp, queue);

  queue.parallel_for(sycl::range<1>(static_cast<size_t>(batch_size) * 6 * kDim),
                     DitModulationKernel{c, weights, modulation, batch_size});
  queue.parallel_for(
      sycl::range<1>(tokens),
      DitLayerNormModulateKernel{x, modulation, hidden, seq_len, 0, 1});
  queue.parallel_for(
      sycl::range<1>(tokens * 3 * kDim),
      DitQkvKernel{hidden, weights, qkv, static_cast<int>(tokens)});
  queue.parallel_for(
      sycl::range<1>(static_cast<size_t>(batch_size) * kHeads * seq_len),
      DitAttentionScoresKernel{qkv, attention, seq_len});
  queue.parallel_for(
      sycl::range<1>(tokens * kDim),
      DitAttentionContextKernel{qkv, attention, context, seq_len});
  queue.parallel_for(sycl::range<1>(tokens * kDim),
                     DitOutProjectionKernel{context, x, modulation, weights,
                                            residual, seq_len});
  queue.parallel_for(
      sycl::range<1>(tokens),
      DitLayerNormModulateKernel{residual, modulation, hidden2, seq_len, 3, 4});
  queue.parallel_for(
      sycl::range<1>(tokens * kMlp),
      DitFirstFcKernel{hidden2, weights, projected, static_cast<int>(tokens)});
  queue.parallel_for(sycl::range<1>(tokens * kDim),
                     DitSecondFcKernel{projected, residual, modulation, weights,
                                       output, seq_len});
  queue.wait();

  sycl::free(projected, queue);
  sycl::free(hidden2, queue);
  sycl::free(residual, queue);
  sycl::free(context, queue);
  sycl::free(attention, queue);
  sycl::free(qkv, queue);
  sycl::free(hidden, queue);
  sycl::free(modulation, queue);
}
