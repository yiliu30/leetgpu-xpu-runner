#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct CrossEntropyRowKernel {
  const float *x;
  const float *w;
  const int *targets;
  int n;
  int d;
  int vocab;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int row = static_cast<int>(item.get_global_id(0));
    if (row >= n) {
      return;
    }
    const float *x_row = x + static_cast<size_t>(row) * d;
    const int target = targets[row];
    float max_logit = -3.402823466e+38f;
    float target_logit = 0.0f;
    for (int token = 0; token < vocab; ++token) {
      const float *w_row = w + static_cast<size_t>(token) * d;
      float logit = 0.0f;
      for (int k = 0; k < d; ++k) {
        logit += x_row[k] * w_row[k];
      }
      if (logit > max_logit) {
        max_logit = logit;
      }
      if (token == target) {
        target_logit = logit;
      }
    }

    double exp_sum = 0.0;
    for (int token = 0; token < vocab; ++token) {
      const float *w_row = w + static_cast<size_t>(token) * d;
      float logit = 0.0f;
      for (int k = 0; k < d; ++k) {
        logit += x_row[k] * w_row[k];
      }
      exp_sum += static_cast<double>(sycl::exp(logit - max_logit));
    }
    const double row_loss = sycl::log(exp_sum) +
                            static_cast<double>(max_logit) -
                            static_cast<double>(target_logit);
    reducer.combine(row_loss);
  }
};

struct CrossEntropyFinalizeKernel {
  const double *sum;
  float *loss;
  int n;

  void operator()() const {
    loss[0] = static_cast<float>(*sum / static_cast<double>(n));
  }
};

} // namespace

extern "C" void solve(const float *x, const float *w, const int *targets,
                      float *loss, int n, int d, int vocab) {
  if (n <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  double *sum = sycl::malloc_device<double>(1, queue);
  const int num_work_groups = (n + kWorkGroupSize - 1) / kWorkGroupSize;
  queue.submit([&](sycl::handler &handler) {
    handler.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(num_work_groups) *
                                         kWorkGroupSize),
                          sycl::range<1>(kWorkGroupSize)),
        sycl::reduction(sum, sycl::plus<double>(),
                        sycl::property::reduction::initialize_to_identity()),
        CrossEntropyRowKernel{x, w, targets, n, d, vocab});
  });
  queue.single_task(CrossEntropyFinalizeKernel{sum, loss, n});
  queue.wait();
  sycl::free(sum, queue);
}
