#include <sycl/sycl.hpp>

#include <limits>

namespace {

constexpr int kWorkGroupSize = 256;

struct CrossAttentionKernel {
  const float *Q;
  const float *K;
  const float *V;
  float *output;
  int M;
  int N;
  int H;
  int D;
  int total_heads;
  sycl::local_accessor<float, 1> query;
  sycl::local_accessor<float, 1> scores;
  sycl::local_accessor<float, 1> reduce;

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    const int group_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_size = static_cast<int>(item.get_local_range(0));

    if (group_id >= total_heads) {
      return;
    }
    const int m = group_id / H;
    const int h = group_id - m * H;
    const long long head_offset_q = (static_cast<long long>(m) * H + h) * D;

    for (int d = tid; d < D; d += local_size) {
      query[d] = Q[head_offset_q + d];
    }
    sycl::group_barrier(group);

    const float scale = 1.0f / sycl::sqrt(static_cast<float>(D));
    for (int n = tid; n < N; n += local_size) {
      const long long offset = (static_cast<long long>(n) * H + h) * D;
      float acc = 0.0f;
      for (int d = 0; d < D; ++d) {
        acc += query[d] * K[offset + d];
      }
      scores[n] = acc * scale;
    }
    sycl::group_barrier(group);

    float local_max = -std::numeric_limits<float>::infinity();
    for (int n = tid; n < N; n += local_size) {
      local_max = sycl::fmax(local_max, scores[n]);
    }
    reduce[tid] = local_max;
    sycl::group_barrier(group);
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] = sycl::fmax(reduce[tid], reduce[tid + stride]);
      }
      sycl::group_barrier(group);
    }
    const float max_score = reduce[0];
    sycl::group_barrier(group);

    float local_sum = 0.0f;
    for (int n = tid; n < N; n += local_size) {
      const float value = sycl::exp(scores[n] - max_score);
      scores[n] = value;
      local_sum += value;
    }
    reduce[tid] = local_sum;
    sycl::group_barrier(group);
    for (int stride = local_size / 2; stride > 0; stride >>= 1) {
      if (tid < stride) {
        reduce[tid] += reduce[tid + stride];
      }
      sycl::group_barrier(group);
    }
    const float inverse_sum = 1.0f / reduce[0];
    sycl::group_barrier(group);

    for (int n = tid; n < N; n += local_size) {
      scores[n] *= inverse_sum;
    }
    sycl::group_barrier(group);

    for (int d = tid; d < D; d += local_size) {
      float acc = 0.0f;
      for (int n = 0; n < N; ++n) {
        const long long offset = (static_cast<long long>(n) * H + h) * D;
        acc += scores[n] * V[offset + d];
      }
      output[head_offset_q + d] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int M, int N, int H, int D) {
  if (M <= 0 || N <= 0 || H <= 0 || D <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> query(sycl::range<1>(D), cgh);
    sycl::local_accessor<float, 1> scores(sycl::range<1>(N), cgh);
    sycl::local_accessor<float, 1> reduce(sycl::range<1>(kWorkGroupSize), cgh);
    const size_t total_heads = static_cast<size_t>(M) * H;
    const size_t total_work_items = total_heads * kWorkGroupSize;
    CrossAttentionKernel kernel{
        Q,     K,      V,     output, M, N, H, D, static_cast<int>(total_heads),
        query, scores, reduce};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_work_items),
                                       sycl::range<1>(kWorkGroupSize)),
                     kernel);
  });
  q.wait();
}
