#include <cstddef>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct SubarraySumKernel {
  const int *input;
  std::size_t count;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const std::size_t i = item.get_global_id(0);
    if (i < count) {
      sumReducer.combine(input[i]);
    }
  }
};

} // namespace

extern "C" void solve(const int *input, int *output, int N, int S, int E) {
  sycl::queue q{sycl::property::queue::in_order()};
  const std::size_t count = static_cast<std::size_t>(E - S + 1);
  const std::size_t numWorkGroups =
      (count + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(output, sycl::plus<int>(),
                        sycl::property::reduction::initialize_to_identity()),
        SubarraySumKernel{input + S, count});
  });
  q.wait();
}
