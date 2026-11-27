#include <sycl/sycl.hpp>

namespace {

struct DistillationKLKernel {
  const float *teacher_logits;
  const float *student_logits;
  float *kl;
  int V;
  float temperature;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    const size_t base = static_cast<size_t>(row) * V;
    const double inv_temperature = 1.0 / static_cast<double>(temperature);

    double teacher_max =
        static_cast<double>(teacher_logits[base]) * inv_temperature;
    double student_max =
        static_cast<double>(student_logits[base]) * inv_temperature;
    for (int col = 1; col < V; ++col) {
      teacher_max = sycl::fmax(teacher_max,
                               static_cast<double>(teacher_logits[base + col]) *
                                   inv_temperature);
      student_max = sycl::fmax(student_max,
                               static_cast<double>(student_logits[base + col]) *
                                   inv_temperature);
    }

    double teacher_sum = 0.0;
    double student_sum = 0.0;
    for (int col = 0; col < V; ++col) {
      teacher_sum += sycl::exp(static_cast<double>(teacher_logits[base + col]) *
                                   inv_temperature -
                               teacher_max);
      student_sum += sycl::exp(static_cast<double>(student_logits[base + col]) *
                                   inv_temperature -
                               student_max);
    }
    const double teacher_lse = teacher_max + sycl::log(teacher_sum);
    const double student_lse = student_max + sycl::log(student_sum);

    double result = 0.0;
    for (int col = 0; col < V; ++col) {
      const double log_p =
          static_cast<double>(teacher_logits[base + col]) * inv_temperature -
          teacher_lse;
      const double log_q =
          static_cast<double>(student_logits[base + col]) * inv_temperature -
          student_lse;
      result += sycl::exp(log_p) * (log_p - log_q);
    }
    kl[row] = static_cast<float>(result);
  }
};

} // namespace

extern "C" void solve(const float *teacher_logits, const float *student_logits,
                      float *kl, int N, int V, float temperature) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(
      sycl::range<1>(N),
      DistillationKLKernel{teacher_logits, student_logits, kl, V, temperature});
  q.wait();
}
