#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct GrpoGroupKernel {
  sycl::local_accessor<double, 1> scratch;
  const float *rewards;
  const float *log_pi;
  const float *log_pi_old;
  const float *log_ref;
  double *partial;
  float clip_eps;
  float beta;
  int G;
  int S;

  void operator()(sycl::nd_item<1> item) const {
    const int b = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *reward_row = rewards + static_cast<long>(b) * G;

    double local_sum = 0.0;
    for (int g = lid; g < G; g += kWorkGroupSize) {
      local_sum += static_cast<double>(reward_row[g]);
    }
    scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double mean = scratch[0] / static_cast<double>(G);

    double local_var = 0.0;
    for (int g = lid; g < G; g += kWorkGroupSize) {
      const double d = static_cast<double>(reward_row[g]) - mean;
      local_var += d * d;
    }
    scratch[lid] = local_var;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    const double stddev = sycl::sqrt(scratch[0] / static_cast<double>(G));

    const long base = static_cast<long>(b) * G * S;
    const long total = static_cast<long>(G) * S;
    double local_loss = 0.0;
    for (long k = lid; k < total; k += kWorkGroupSize) {
      const int g = static_cast<int>(k / S);
      const double advantage = (static_cast<double>(reward_row[g]) - mean) / (stddev + 1.0e-8);
      const float diff = log_pi[base + k] - log_pi_old[base + k];
      const float ratio = sycl::exp(diff);
      const float clipped = sycl::min(sycl::max(ratio, 1.0f - clip_eps), 1.0f + clip_eps);
      const double surrogate = static_cast<double>(
          sycl::min(ratio * static_cast<float>(advantage),
                    clipped * static_cast<float>(advantage)));
      const float kl_diff = log_ref[base + k] - log_pi[base + k];
      const double kl_penalty = sycl::exp(static_cast<double>(kl_diff)) -
                                static_cast<double>(kl_diff) - 1.0;
      local_loss += surrogate - static_cast<double>(beta) * kl_penalty;
    }
    scratch[lid] = local_loss;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0) {
      partial[b] = scratch[0];
    }
  }
};

struct GrpoFinalizeKernel {
  sycl::local_accessor<double, 1> scratch;
  const double *partial;
  float *output;
  int B;
  double denom;

  void operator()(sycl::nd_item<1> item) const {
    const int lid = item.get_local_id(0);
    double local_sum = 0.0;
    for (int b = lid; b < B; b += kWorkGroupSize) {
      local_sum += partial[b];
    }
    scratch[lid] = local_sum;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = kWorkGroupSize / 2; stride > 0; stride >>= 1) {
      if (lid < stride) {
        scratch[lid] += scratch[lid + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (lid == 0) {
      output[0] = static_cast<float>(-scratch[0] / denom);
    }
  }
};

} // namespace

extern "C" void solve(const float *rewards, const float *log_pi, const float *log_pi_old,
                      const float *log_ref, float *output, float clip_eps, float beta, int B, int G,
                      int S) {
  sycl::queue q{sycl::property::queue::in_order()};
  double *partial = sycl::malloc_device<double>(B, q);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(B * kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        GrpoGroupKernel{scratch, rewards, log_pi, log_pi_old, log_ref, partial, clip_eps, beta,
                        G, S});
  });

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(kWorkGroupSize), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        GrpoFinalizeKernel{scratch, partial, output, B,
                           static_cast<double>(B) * G * S});
  });
  q.wait();
  sycl::free(partial, q);
}
