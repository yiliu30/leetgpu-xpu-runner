#include <sycl/sycl.hpp>

namespace {

struct RopeEmbeddingKernel {
  const float *Q;
  const float *cosine;
  const float *sine;
  float *output;
  int D;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / D;
    const int col = index % D;
    const int half = D / 2;
    const int rowOffset = row * D;
    const int rotatedCol = (col < half) ? col + half : col - half;
    const float rotated = (col < half) ? -Q[rowOffset + rotatedCol]
                                       : Q[rowOffset + rotatedCol];
    output[index] = Q[index] * cosine[index] + rotated * sine[index];
  }
};

} // namespace

extern "C" void solve(const float *Q, const float *cosine, const float *sine,
                      float *output, int M, int D) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = M * D;
  if (total > 0) {
    q.parallel_for(sycl::range<1>(total), RopeEmbeddingKernel{Q, cosine, sine, output, D});
  }
  q.wait();
}
