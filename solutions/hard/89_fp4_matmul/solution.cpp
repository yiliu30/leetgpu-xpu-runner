#include <cstdint>
#include <sycl/sycl.hpp>

namespace {

inline float fp4Value(unsigned int value) {
  switch (value & 0xFu) {
  case 0:
    return 0.0f;
  case 1:
    return 0.5f;
  case 2:
    return 1.0f;
  case 3:
    return 1.5f;
  case 4:
    return 2.0f;
  case 5:
    return 3.0f;
  case 6:
    return 4.0f;
  case 7:
    return 6.0f;
  case 8:
    return -0.0f;
  case 9:
    return -0.5f;
  case 10:
    return -1.0f;
  case 11:
    return -1.5f;
  case 12:
    return -2.0f;
  case 13:
    return -3.0f;
  case 14:
    return -4.0f;
  default:
    return -6.0f;
  }
}

inline float e4m3Value(unsigned int bits) {
  const unsigned int sign = (bits >> 7) & 1u;
  const int exponent = static_cast<int>((bits >> 3) & 0xFu);
  const int mantissa = static_cast<int>(bits & 0x7u);
  float value;
  if (exponent == 0) {
    value = (static_cast<float>(mantissa) / 8.0f) * 0.015625f;
  } else {
    value = (1.0f + static_cast<float>(mantissa) / 8.0f) *
            sycl::exp2(static_cast<float>(exponent - 7));
  }
  return sign != 0 ? -value : value;
}

struct Fp4DecodeKernel {
  const uint8_t *packed;
  const uint8_t *scales;
  float *decoded;
  int rows;
  int cols;
  int blocksPerRow;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / cols;
    const int col = linear % cols;
    const uint8_t byte = packed[row * (cols / 2) + col / 2];
    const unsigned int nibble =
        (col & 1) == 0 ? (byte >> 4) & 0xFu : byte & 0xFu;
    const float scale = e4m3Value(scales[row * blocksPerRow + col / 16]);
    decoded[linear] = fp4Value(nibble) * scale;
  }
};

struct Fp4MatmulKernel {
  const float *x;
  const float *w;
  sycl::half *y;
  int M;
  int N;
  int K;
  float alpha;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int row = linear / N;
    const int col = linear % N;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) {
      acc += x[row * K + k] * w[col * K + k];
    }
    y[linear] = static_cast<sycl::half>(alpha * acc);
  }
};

} // namespace

extern "C" void solve(const uint8_t *x_q, const uint8_t *x_scales,
                      const uint8_t *w_q, const uint8_t *w_scales, float alpha,
                      sycl::half *y, int M, int N, int K) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M <= 0 || N <= 0 || K <= 0) {
    q.wait();
    return;
  }

  const int blocks = K / 16;
  float *x = sycl::malloc_device<float>(static_cast<size_t>(M) * K, q);
  float *w = sycl::malloc_device<float>(static_cast<size_t>(N) * K, q);

  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * K),
                 Fp4DecodeKernel{x_q, x_scales, x, M, K, blocks});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(N) * K),
                 Fp4DecodeKernel{w_q, w_scales, w, N, K, blocks});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(M) * N),
                 Fp4MatmulKernel{x, w, y, M, N, K, alpha});
  q.wait();

  sycl::free(w, q);
  sycl::free(x, q);
}
