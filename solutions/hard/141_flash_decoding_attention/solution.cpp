#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 128;

struct FlashDecodeScoresKernel {
  const float *Q;
  const float *K;
  float *scores;
  int S;
  int H;
  int D;
  std::size_t total;
  float scale;

  void operator()(sycl::id<1> id) const {
    const std::size_t index = id[0];
    if (index >= total) {
      return;
    }
    const int s = static_cast<int>(index % static_cast<std::size_t>(S));
    const std::size_t rest = index / static_cast<std::size_t>(S);
    const int h = static_cast<int>(rest % static_cast<std::size_t>(H));
    const int t = static_cast<int>(rest / static_cast<std::size_t>(H));
    const float *qrow = Q + (static_cast<std::size_t>(t) * H + h) * D;
    const float *krow = K + (static_cast<std::size_t>(s) * H + h) * D;
    float dot = 0.0f;
    for (int d = 0; d < D; ++d) {
      dot += qrow[d] * krow[d];
    }
    scores[index] = dot * scale;
  }
};

struct FlashDecodeOutputKernel {
  sycl::local_accessor<float, 1> part;
  sycl::local_accessor<float, 1> scal;
  const float *V;
  float *scores;
  float *output;
  int S;
  int H;
  int D;

  void operator()(sycl::nd_item<1> item) const {
    const int group = static_cast<int>(item.get_group(0));
    const int t = group / H;
    const int h = group - t * H;
    const int tid = static_cast<int>(item.get_local_id(0));
    const int wg = static_cast<int>(item.get_local_range(0));

    float *row = scores + static_cast<std::size_t>(group) * S;

    float local_max = -1.0e38f;
    for (int s = tid; s < S; s += wg) {
      local_max = sycl::max(local_max, row[s]);
    }
    part[tid] = local_max;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float m = part[0];
      for (int i = 1; i < wg; ++i) {
        m = sycl::max(m, part[i]);
      }
      scal[0] = m;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float row_max = scal[0];
    float local_sum = 0.0f;
    for (int s = tid; s < S; s += wg) {
      const float e = sycl::exp(row[s] - row_max);
      row[s] = e;
      local_sum += e;
    }
    part[tid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      float sum = 0.0f;
      for (int i = 0; i < wg; ++i) {
        sum += part[i];
      }
      scal[1] = sum > 0.0f ? 1.0f / sum : 0.0f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    const float inv_sum = scal[1];
    for (int d = tid; d < D; d += wg) {
      float acc = 0.0f;
      for (int s = 0; s < S; ++s) {
        acc += row[s] * V[(static_cast<std::size_t>(s) * H + h) * D + d];
      }
      output[(static_cast<std::size_t>(t) * H + h) * D + d] = acc * inv_sum;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V,
                      float *output, int T, int S, int H, int D) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (T > 0 && S > 0 && H > 0 && D > 0) {
    const std::size_t count = static_cast<std::size_t>(T) * H * S;
    float *scores = sycl::malloc_device<float>(count, q);
    if (scores != nullptr) {
      const float scale = 1.0f / sycl::sqrt(static_cast<float>(D));
      const std::size_t score_groups =
          (count + kWorkGroupSize - 1) / kWorkGroupSize;
      q.parallel_for(
          sycl::range<1>(score_groups * kWorkGroupSize),
          FlashDecodeScoresKernel{Q, K, scores, S, H, D, count, scale});
      const sycl::range<1> local(kWorkGroupSize);
      const sycl::range<1> global(static_cast<std::size_t>(T) * H *
                                  kWorkGroupSize);
      q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(kWorkGroupSize), h);
        sycl::local_accessor<float, 1> scal(sycl::range<1>(2), h);
        h.parallel_for(
            sycl::nd_range<1>(global, local),
            FlashDecodeOutputKernel{part, scal, V, scores, output, S, H, D});
      });
      q.wait();
      sycl::free(scores, q);
      return;
    }
  }
  q.wait();
}
