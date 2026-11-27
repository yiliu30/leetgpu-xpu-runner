#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 128;

struct SpecTreeAttentionKernel {
  sycl::local_accessor<float, 1> qloc;
  sycl::local_accessor<float, 1> scores;
  sycl::local_accessor<int, 1> ancestor;
  sycl::local_accessor<float, 1> part;
  sycl::local_accessor<float, 1> scal;
  const float *Q;
  const float *K_cache;
  const float *V_cache;
  const float *K_tree;
  const float *V_tree;
  const int *parents;
  float *output;
  int P;
  int T;
  int H;
  int D;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int group = static_cast<int>(item.get_group(0));
    const int t = group / H;
    const int h = group - t * H;
    const int tid = static_cast<int>(item.get_local_id(0));
    const int wg = static_cast<int>(item.get_local_range(0));
    const int total = P + T;

    for (int i = tid; i < T; i += wg) {
      ancestor[i] = 0;
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (tid == 0) {
      int cur = t;
      while (cur >= 0) {
        ancestor[cur] = 1;
        cur = parents[cur];
      }
    }
    for (int d = tid; d < D; d += wg) {
      qloc[d] = Q[(static_cast<std::size_t>(t) * H + h) * D + d];
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int s = tid; s < total; s += wg) {
      const float *kptr = nullptr;
      bool allowed = true;
      if (s < P) {
        kptr = K_cache + (static_cast<std::size_t>(s) * H + h) * D;
      } else {
        const int node = s - P;
        kptr = K_tree + (static_cast<std::size_t>(node) * H + h) * D;
        allowed = ancestor[node] != 0;
      }
      float dot = 0.0f;
      if (allowed) {
        for (int d = 0; d < D; ++d) {
          dot += qloc[d] * kptr[d];
        }
      }
      scores[s] = allowed ? dot * scale : -1.0e38f;
    }
    item.barrier(sycl::access::fence_space::local_space);

    float local_max = -1.0e38f;
    for (int s = tid; s < total; s += wg) {
      local_max = sycl::max(local_max, scores[s]);
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
    for (int s = tid; s < total; s += wg) {
      const float e = sycl::exp(scores[s] - row_max);
      scores[s] = e;
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
      for (int s = 0; s < P; ++s) {
        const float e = scores[s];
        if (e != 0.0f) {
          acc += e * V_cache[(static_cast<std::size_t>(s) * H + h) * D + d];
        }
      }
      for (int node = 0; node < T; ++node) {
        const float e = scores[P + node];
        if (e != 0.0f) {
          acc += e * V_tree[(static_cast<std::size_t>(node) * H + h) * D + d];
        }
      }
      output[(static_cast<std::size_t>(t) * H + h) * D + d] = acc * inv_sum;
    }
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K_cache,
                      const float *V_cache, const float *K_tree,
                      const float *V_tree, const int *parents, float *output,
                      int P, int T, int H, int D) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (P >= 0 && T > 0 && H > 0 && D > 0) {
    const float scale = 1.0f / sycl::sqrt(static_cast<float>(D));
    const sycl::range<1> local(kWorkGroupSize);
    const sycl::range<1> global(static_cast<std::size_t>(T) * H *
                                kWorkGroupSize);
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 1> qloc(sycl::range<1>(D), h);
      sycl::local_accessor<float, 1> scores(sycl::range<1>(P + T), h);
      sycl::local_accessor<int, 1> ancestor(sycl::range<1>(T), h);
      sycl::local_accessor<float, 1> part(sycl::range<1>(kWorkGroupSize), h);
      sycl::local_accessor<float, 1> scal(sycl::range<1>(2), h);
      h.parallel_for(sycl::nd_range<1>(global, local),
                     SpecTreeAttentionKernel{qloc, scores, ancestor, part, scal,
                                             Q, K_cache, V_cache, K_tree,
                                             V_tree, parents, output, P, T, H,
                                             D, scale});
    });
  }
  q.wait();
}
