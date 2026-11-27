#include <sycl/sycl.hpp>

#include <cmath>

namespace {

struct AdamWKernel {
  float *params;
  const float *grad;
  float *m;
  float *v;
  int N;
  float lr;
  float beta1;
  float beta2;
  float eps;
  float weight_decay;
  float bias_correction1;
  float bias_correction2;

  void operator()(sycl::id<1> id) const {
    const int i = static_cast<int>(id[0]);
    if (i < N) {
      const float g = grad[i];
      const float p = params[i] * (1.0f - lr * weight_decay);
      const float m_new = beta1 * m[i] + (1.0f - beta1) * g;
      const float v_new = beta2 * v[i] + (1.0f - beta2) * g * g;
      m[i] = m_new;
      v[i] = v_new;
      const float denom = sycl::sqrt(v_new / bias_correction2) + eps;
      params[i] = p - lr * (m_new / bias_correction1) / denom;
    }
  }
};

} // namespace

extern "C" void solve(float *params, const float *grad, float *m, float *v, int N, float lr,
                      float beta1, float beta2, float eps, float weight_decay, int t) {
  sycl::queue q{sycl::property::queue::in_order()};
  const double bc1 = 1.0 - std::pow(static_cast<double>(beta1), static_cast<double>(t));
  const double bc2 = 1.0 - std::pow(static_cast<double>(beta2), static_cast<double>(t));
  q.parallel_for(sycl::range<1>(N),
                 AdamWKernel{params, grad, m, v, N, lr, beta1, beta2, eps, weight_decay,
                             static_cast<float>(bc1), static_cast<float>(bc2)});
  q.wait();
}
