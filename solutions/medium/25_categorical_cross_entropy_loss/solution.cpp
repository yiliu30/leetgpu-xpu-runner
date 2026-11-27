#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct CrossEntropyRowsKernel {
  const float *logits;
  const int *trueLabels;
  float *rowLosses;
  int N;
  int C;

  void operator()(sycl::id<1> idx) const {
    const int row = static_cast<int>(idx[0]);
    const float *rowLogits = logits + static_cast<std::size_t>(row) * C;
    float maxLogit = rowLogits[0];
    for (int col = 1; col < C; ++col) {
      maxLogit = sycl::fmax(maxLogit, rowLogits[col]);
    }
    float expSum = 0.0f;
    for (int col = 0; col < C; ++col) {
      expSum += sycl::exp(rowLogits[col] - maxLogit);
    }
    rowLosses[row] = (maxLogit + sycl::log(expSum)) -
                     rowLogits[trueLabels[row]];
  }
};

struct CrossEntropySumKernel {
  const float *rowLosses;
  int N;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      reducer.combine(static_cast<double>(rowLosses[i]));
    }
  }
};

struct CrossEntropyMeanKernel {
  const double *total;
  float *loss;
  int N;

  void operator()() const { loss[0] = static_cast<float>(*total / N); }
};

} // namespace

extern "C" void solve(const float *logits, const int *true_labels, float *loss,
                      int N, int C) {
  sycl::queue q{sycl::property::queue::in_order()};
  float *rowLosses = sycl::malloc_device<float>(N, q);
  double *total = sycl::malloc_device<double>(1, q);
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

  q.parallel_for(sycl::range<1>(N),
                 CrossEntropyRowsKernel{logits, true_labels, rowLosses, N, C});
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(
            sycl::range<1>(numWorkGroups * WORK_GROUP_SIZE),
            sycl::range<1>(WORK_GROUP_SIZE)),
        sycl::reduction(total, sycl::plus<double>(),
                        sycl::property::reduction::initialize_to_identity()),
        CrossEntropySumKernel{rowLosses, N});
  });
  q.single_task(CrossEntropyMeanKernel{total, loss, N});
  q.wait();

  sycl::free(rowLosses, q);
  sycl::free(total, q);
}
