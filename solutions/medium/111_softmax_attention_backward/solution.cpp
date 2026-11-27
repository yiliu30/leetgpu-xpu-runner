#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct AttentionBackwardRowsKernel {
  sycl::local_accessor<float, 1> max_scratch;
  sycl::local_accessor<double, 1> sum_scratch;
  const float *Q;
  const float *K;
  const float *V;
  const float *dO;
  float *P;
  float *dS;
  int N;
  int d;
  float scale;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *q_row = Q + static_cast<long>(i) * d;
    const float *do_row = dO + static_cast<long>(i) * d;
    float *p_row = P + static_cast<long>(i) * N;
    float *ds_row = dS + static_cast<long>(i) * N;

    float local_max = -3.402823466e+38F;
    for (int j = lid; j < N; j += kWorkGroupSize) {
      const float *k_row = K + static_cast<long>(j) * d;
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) {
        dot += q_row[c] * k_row[c];
      }
      local_max = sycl::max(local_max, dot * scale);
    }
    max_scratch[lid] = local_max;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        max_scratch[lid] = sycl::max(max_scratch[lid], max_scratch[lid + stride]);
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const float row_max = max_scratch[0];

    double local_sum = 0.0;
    for (int j = lid; j < N; j += kWorkGroupSize) {
      const float *k_row = K + static_cast<long>(j) * d;
      float dot = 0.0f;
      for (int c = 0; c < d; ++c) {
        dot += q_row[c] * k_row[c];
      }
      const float e = sycl::exp(dot * scale - row_max);
      p_row[j] = e;
      local_sum += static_cast<double>(e);
    }
    sum_scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        sum_scratch[lid] += sum_scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double denom = sum_scratch[0];

    double local_dot = 0.0;
    for (int j = lid; j < N; j += kWorkGroupSize) {
      const float p = static_cast<float>(static_cast<double>(p_row[j]) / denom);
      p_row[j] = p;
      const float *v_row = V + static_cast<long>(j) * d;
      float dp = 0.0f;
      for (int c = 0; c < d; ++c) {
        dp += do_row[c] * v_row[c];
      }
      local_dot += static_cast<double>(p) * static_cast<double>(dp);
    }
    sum_scratch[lid] = local_dot;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        sum_scratch[lid] += sum_scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double row_dot = sum_scratch[0];

    for (int j = lid; j < N; j += kWorkGroupSize) {
      const float *v_row = V + static_cast<long>(j) * d;
      float dp = 0.0f;
      for (int c = 0; c < d; ++c) {
        dp += do_row[c] * v_row[c];
      }
      ds_row[j] = static_cast<float>(static_cast<double>(p_row[j]) *
                                     (static_cast<double>(dp) - row_dot) * scale);
    }
  }
};

struct AttentionDqKernel {
  const float *K;
  const float *dS;
  float *dQ;
  int N;
  int d;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0] / d);
    const int c = static_cast<int>(idx[0] - static_cast<long>(i) * d);
    const float *ds_row = dS + static_cast<long>(i) * N;
    double sum = 0.0;
    for (int j = 0; j < N; ++j) {
      sum += static_cast<double>(ds_row[j]) *
             static_cast<double>(K[static_cast<long>(j) * d + c]);
    }
    dQ[idx[0]] = static_cast<float>(sum);
  }
};

struct AttentionDkDvKernel {
  const float *Q;
  const float *dO;
  const float *P;
  const float *dS;
  float *dK;
  float *dV;
  int M;
  int N;
  int d;

  void operator()(sycl::id<1> idx) const {
    const int j = static_cast<int>(idx[0] / d);
    const int c = static_cast<int>(idx[0] - static_cast<long>(j) * d);
    double dk_sum = 0.0;
    double dv_sum = 0.0;
    for (int i = 0; i < M; ++i) {
      const long pidx = static_cast<long>(i) * N + j;
      dk_sum += static_cast<double>(dS[pidx]) *
                static_cast<double>(Q[static_cast<long>(i) * d + c]);
      dv_sum += static_cast<double>(P[pidx]) *
                static_cast<double>(dO[static_cast<long>(i) * d + c]);
    }
    dK[idx[0]] = static_cast<float>(dk_sum);
    dV[idx[0]] = static_cast<float>(dv_sum);
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *K, const float *V, const float *dO, float *dQ,
                      float *dK, float *dV, int M, int N, int d) {
  sycl::queue q{sycl::property::queue::in_order()};
  float *P = sycl::malloc_device<float>(static_cast<long>(M) * N, q);
  float *dS = sycl::malloc_device<float>(static_cast<long>(M) * N, q);
  const float scale = 1.0f / sycl::sqrt(static_cast<float>(d));

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> max_scratch(sycl::range<1>(kWorkGroupSize), h);
    sycl::local_accessor<double, 1> sum_scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(M * kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        AttentionBackwardRowsKernel{max_scratch, sum_scratch, Q, K, V, dO, P, dS, N, d,
                                    scale});
  });

  q.parallel_for(sycl::range<1>(static_cast<long>(M) * d),
                 AttentionDqKernel{K, dS, dQ, N, d});

  q.parallel_for(sycl::range<1>(static_cast<long>(N) * d),
                 AttentionDkDvKernel{Q, dO, P, dS, dK, dV, M, N, d});
  q.wait();
  sycl::free(P, q);
  sycl::free(dS, q);
}
