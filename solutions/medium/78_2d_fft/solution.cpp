#include <sycl/sycl.hpp>

namespace {

constexpr double PI = 3.14159265358979323846;

struct FftTableKernel {
  float *table;
  int size;

  void operator()(sycl::id<1> globalId) const {
    const int i = static_cast<int>(globalId[0]);
    const double angle =
        -2.0 * PI * static_cast<double>(i) / static_cast<double>(size);
    table[2 * i] = static_cast<float>(sycl::cos(angle));
    table[2 * i + 1] = static_cast<float>(sycl::sin(angle));
  }
};

struct FftRowsKernel {
  const float *signal;
  float *scratch;
  const float *table;
  int M;
  int N;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int row = index / N;
    const int frequency = index % N;
    float sumRe = 0.0f;
    float sumIm = 0.0f;
    const int rowOffset = row * N;
    for (int n = 0; n < N; ++n) {
      const int tableIndex = (n * frequency) % N;
      const float cosine = table[2 * tableIndex];
      const float sine = table[2 * tableIndex + 1];
      const float re = signal[2 * (rowOffset + n)];
      const float im = signal[2 * (rowOffset + n) + 1];
      sumRe += re * cosine - im * sine;
      sumIm += re * sine + im * cosine;
    }
    scratch[2 * index] = sumRe;
    scratch[2 * index + 1] = sumIm;
  }
};

struct FftColsKernel {
  const float *scratch;
  float *spectrum;
  const float *table;
  int M;
  int N;

  void operator()(sycl::id<1> globalId) const {
    const int index = static_cast<int>(globalId[0]);
    const int col = index / M;
    const int frequency = index % M;
    float sumRe = 0.0f;
    float sumIm = 0.0f;
    for (int r = 0; r < M; ++r) {
      const int tableIndex = (r * frequency) % M;
      const float cosine = table[2 * tableIndex];
      const float sine = table[2 * tableIndex + 1];
      const int sourceIndex = r * N + col;
      const float re = scratch[2 * sourceIndex];
      const float im = scratch[2 * sourceIndex + 1];
      sumRe += re * cosine - im * sine;
      sumIm += re * sine + im * cosine;
    }
    const int destinationIndex = frequency * N + col;
    spectrum[2 * destinationIndex] = sumRe;
    spectrum[2 * destinationIndex + 1] = sumIm;
  }
};

} // namespace

extern "C" void solve(const float *signal, float *spectrum, int M, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (M <= 0 || N <= 0) {
    return;
  }

  const int total = M * N;
  float *tableM = sycl::malloc_device<float>(2 * M, q);
  float *tableN = sycl::malloc_device<float>(2 * N, q);
  float *scratch = sycl::malloc_device<float>(2 * total, q);

  q.parallel_for(sycl::range<1>(M), FftTableKernel{tableM, M});
  q.parallel_for(sycl::range<1>(N), FftTableKernel{tableN, N});
  q.parallel_for(sycl::range<1>(total),
                 FftRowsKernel{signal, scratch, tableN, M, N});
  q.parallel_for(sycl::range<1>(total),
                 FftColsKernel{scratch, spectrum, tableM, M, N});
  q.wait();

  sycl::free(scratch, q);
  sycl::free(tableN, q);
  sycl::free(tableM, q);
}
