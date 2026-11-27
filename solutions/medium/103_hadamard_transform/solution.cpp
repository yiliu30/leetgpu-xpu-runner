#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct HadamardKernel {
  sycl::local_accessor<float, 1> tile;
  const float *input;
  float *output;
  int N;
  float inv_sqrt_n;

  void operator()(sycl::nd_item<1> item) const {
    const int row = item.get_group(0);
    const int lid = item.get_local_id(0);
    const float *src = input + row * N;
    float *dst = output + row * N;

    for (int i = lid; i < N; i += kWorkGroupSize) {
      tile[i] = src[i];
    }
    item.barrier(sycl::access::fence_space::local_space);

    for (int h = 1; h < N; h <<= 1) {
      for (int p = lid; p < N / 2; p += kWorkGroupSize) {
        const int block = p / h;
        const int offset = p - block * h;
        const int i0 = block * 2 * h + offset;
        const int i1 = i0 + h;
        const float a = tile[i0];
        const float b = tile[i1];
        tile[i0] = a + b;
        tile[i1] = a - b;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }

    for (int i = lid; i < N; i += kWorkGroupSize) {
      dst[i] = tile[i] * inv_sqrt_n;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int B, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const float inv_sqrt_n = 1.0f / sycl::sqrt(static_cast<float>(N));
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<float, 1> tile(sycl::range<1>(N), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(B * kWorkGroupSize),
                                     sycl::range<1>(kWorkGroupSize)),
                   HadamardKernel{tile, input, output, N, inv_sqrt_n});
  });
  q.wait();
}
