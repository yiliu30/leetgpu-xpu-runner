#include <sycl/sycl.hpp>

namespace {

constexpr int TILE = 16;

struct MatrixMultiplyKernel {
  sycl::local_accessor<float, 2> tileA;
  sycl::local_accessor<float, 2> tileB;
  const float *A;
  const float *B;
  float *C;
  int M;
  int N;
  int K;
  int numKTiles;

  void operator()(sycl::nd_item<2> item) const {
    const int col = item.get_global_id(0);
    const int row = item.get_global_id(1);
    const int localCol = item.get_local_id(0);
    const int localRow = item.get_local_id(1);
    float acc = 0.0f;
    for (int t = 0; t < numKTiles; ++t) {
      const int aCol = t * TILE + localCol;
      const int bRow = t * TILE + localRow;
      tileA[localRow][localCol] =
          (row < M && aCol < N) ? A[row * N + aCol] : 0.0f;
      tileB[localRow][localCol] =
          (bRow < N && col < K) ? B[bRow * K + col] : 0.0f;
      item.barrier(sycl::access::fence_space::local_space);
      for (int k = 0; k < TILE; ++k) {
        acc += tileA[localRow][k] * tileB[k][localCol];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (row < M && col < K) {
      C[row * K + col] = acc;
    }
  }
};

} // namespace

extern "C" void solve(const float *A, const float *B, float *C, int M, int N,
                      int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numKTiles = (N + TILE - 1) / TILE;
  sycl::range<2> workGroupSize(TILE, TILE);
  sycl::range<2> numWorkGroups((K + TILE - 1) / TILE, (M + TILE - 1) / TILE);

  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 2> tileA(sycl::range<2>(TILE, TILE), h);
     sycl::local_accessor<float, 2> tileB(sycl::range<2>(TILE, TILE), h);
     h.parallel_for(
         sycl::nd_range<2>(numWorkGroups * workGroupSize, workGroupSize),
         MatrixMultiplyKernel{tileA, tileB, A, B, C, M, N, K, numKTiles});
   }).wait();
}
