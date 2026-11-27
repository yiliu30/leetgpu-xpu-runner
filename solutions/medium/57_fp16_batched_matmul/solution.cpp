#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct Fp16BatchedMatmulKernel {
  const sycl::half *a;
  const sycl::half *b;
  sycl::half *c;
  int m;
  int n;
  int k;
  int total;

  void operator()(sycl::id<1> idx) const {
    const int index = idx[0];
    if (index >= total) {
      return;
    }

    int remaining = index;
    const int col = remaining % n;
    remaining /= n;
    const int row = remaining % m;
    const int batch = remaining / m;

    const int aBase = (batch * m + row) * k;
    const int bBase = batch * k * n + col;
    float acc = 0.0f;
    for (int kk = 0; kk < k; ++kk) {
      acc += static_cast<float>(a[aBase + kk]) *
             static_cast<float>(b[bBase + kk * n]);
    }
    c[index] = static_cast<sycl::half>(acc);
  }
};

} // namespace

extern "C" void solve(const sycl::half *A, const sycl::half *B, sycl::half *C,
                      int BATCH, int M, int N, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = BATCH * M * N;
  const int numWorkGroups = (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.parallel_for(globalSize, Fp16BatchedMatmulKernel{A, B, C, M, N, K, total});
  q.wait();
}
