#include <cstddef>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct SubarraySum2DKernel {
  const int *input;
  int m;
  int sRow;
  int sCol;
  std::size_t cols;
  std::size_t total;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &sumReducer) const {
    const std::size_t index = item.get_global_id(0);
    if (index >= total) {
      return;
    }
    const std::size_t row = index / cols;
    const std::size_t col = index % cols;
    const std::size_t inputIndex =
        (static_cast<std::size_t>(sRow) + row) * m +
        static_cast<std::size_t>(sCol) + col;
    sumReducer.combine(input[inputIndex]);
  }
};

} // namespace

extern "C" void solve(const int *input, int *output, int N, int M, int S_ROW,
                      int E_ROW, int S_COL, int E_COL) {
  sycl::queue q{sycl::property::queue::in_order()};
  const std::size_t rows = static_cast<std::size_t>(E_ROW - S_ROW + 1);
  const std::size_t cols = static_cast<std::size_t>(E_COL - S_COL + 1);
  const std::size_t total = rows * cols;
  const std::size_t numWorkGroups =
      (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(globalSize, workGroupSize),
        sycl::reduction(output, sycl::plus<int>(),
                        sycl::property::reduction::initialize_to_identity()),
        SubarraySum2DKernel{input, M, S_ROW, S_COL, cols, total});
  });
  q.wait();
}
