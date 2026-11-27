#include <sycl/sycl.hpp>

namespace {

struct GaeScanKernel {
  const float *rewards;
  const float *values;
  float *advantages;
  float gamma;
  float decay;
  int S;

  void operator()(sycl::id<1> idx) const {
    const int b = idx[0];
    const float *reward_row = rewards + static_cast<long>(b) * S;
    const float *value_row = values + static_cast<long>(b) * S;
    float *adv_row = advantages + static_cast<long>(b) * S;
    float carry = 0.0f;
    for (int t = S - 1; t >= 0; --t) {
      const float next_value = (t + 1 < S) ? value_row[t + 1] : 0.0f;
      const float delta = reward_row[t] + gamma * next_value - value_row[t];
      carry = delta + decay * carry;
      adv_row[t] = carry;
    }
  }
};

} // namespace

extern "C" void solve(const float *rewards, const float *values, float *advantages, float gamma,
                      float lam, int B, int S) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(B), GaeScanKernel{rewards, values, advantages, gamma, gamma * lam,
                                                   S});
  q.wait();
}
