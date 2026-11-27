#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

constexpr int kBlockSize = 128;
constexpr int kWorkGroupSize = 256;

inline float e4m3ToFloat(std::uint8_t bits) {
  const std::uint32_t sign = (static_cast<std::uint32_t>(bits) & 0x80u) << 24;
  const std::uint32_t exponent =
      (static_cast<std::uint32_t>(bits) >> 3) & 0x0Fu;
  const std::uint32_t mantissa = static_cast<std::uint32_t>(bits) & 0x07u;
  float value;
  if (exponent == 0) {
    value = static_cast<float>(mantissa) * 0.001953125f; // mantissa * 2^-9
  } else {
    value = (1.0f + 0.125f * static_cast<float>(mantissa)) *
            sycl::exp2(static_cast<float>(exponent) - 7.0f);
  }
  return sign != 0 ? -value : value;
}

struct Fp8BlockScaledGemmKernel {
  const std::uint8_t *x_q;
  const float *x_scales;
  const std::uint8_t *w_q;
  const float *w_scales;
  sycl::half *y;
  int M;
  int N;
  int K;
  int num_blocks;
  int total;

  void operator()(sycl::id<1> id) const {
    const int index = static_cast<int>(id[0]);
    if (index >= total) {
      return;
    }
    const int m = index / N;
    const int n = index - m * N;
    const std::uint8_t *xrow = x_q + static_cast<std::size_t>(m) * K;
    const std::uint8_t *wrow = w_q + static_cast<std::size_t>(n) * K;
    const float *xscale = x_scales + static_cast<std::size_t>(m) * num_blocks;
    const float *wscale = w_scales + static_cast<std::size_t>(n) * num_blocks;
    float acc = 0.0f;
    for (int b = 0; b < num_blocks; ++b) {
      const std::size_t offset = static_cast<std::size_t>(b) * kBlockSize;
      float block_sum = 0.0f;
      for (int k = 0; k < kBlockSize; ++k) {
        block_sum +=
            e4m3ToFloat(xrow[offset + k]) * e4m3ToFloat(wrow[offset + k]);
      }
      acc += block_sum * (xscale[b] * wscale[b]);
    }
    y[index] = static_cast<sycl::half>(acc);
  }
};

} // namespace

extern "C" void solve(const std::uint8_t *x_q, const float *x_scales,
                      const std::uint8_t *w_q, const float *w_scales,
                      sycl::half *y, int M, int N, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = M * N;
  if (total > 0 && K > 0) {
    const int num_blocks = K / kBlockSize;
    const std::size_t groups =
        (static_cast<std::size_t>(total) + kWorkGroupSize - 1) / kWorkGroupSize;
    q.parallel_for(sycl::range<1>(groups * kWorkGroupSize),
                   Fp8BlockScaledGemmKernel{x_q, x_scales, w_q, w_scales, y, M,
                                            N, K, num_blocks, total});
  }
  q.wait();
}
