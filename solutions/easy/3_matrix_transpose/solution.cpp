#include <sycl/sycl.hpp>

namespace {

constexpr int TILE = 16;

struct TransposeKernel {
  sycl::local_accessor<float, 2> tile;
  const float *input;
  float *output;
  int rows;
  int cols;

  void operator()(sycl::nd_item<2> item) const {
    const int col = item.get_global_id(0);
    const int row = item.get_global_id(1);
    const int localCol = item.get_local_id(0);
    const int localRow = item.get_local_id(1);
    if (row < rows && col < cols) {
      tile[localRow][localCol] = input[row * cols + col];
    }
    item.barrier(sycl::access::fence_space::local_space);
    if (col < cols && row < rows) {
      output[col * rows + row] = tile[localRow][localCol];
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int rows, int cols) {
  sycl::queue q{sycl::property::queue::in_order()};
  sycl::range<2> workGroupSize(TILE, TILE);
  sycl::range<2> numWorkGroups((cols + TILE - 1) / TILE,
                               (rows + TILE - 1) / TILE);

  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 2> tile(sycl::range<2>(TILE, TILE), h);
     h.parallel_for(
         sycl::nd_range<2>(numWorkGroups * workGroupSize, workGroupSize),
         TransposeKernel{tile, input, output, rows, cols});
   }).wait();
}
