#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct CountArrayElementKernel {
  const int *input;
  int n;
  int key;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &countReducer) const {
    const int i = item.get_global_id(0);
    if (i < n) {
      countReducer.combine(input[i] == key ? 1 : 0);
    }
  }
};

} // namespace

extern "C" void solve(const int *input, int *output, int N, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(output, sycl::plus<int>(),
                        sycl::property::reduction::initialize_to_identity()),
        CountArrayElementKernel{input, N, K});
  });
  q.wait();
}
