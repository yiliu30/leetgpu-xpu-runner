#include <sycl/sycl.hpp>

namespace {

constexpr double kPi = 3.14159265358979323846;

struct BitReverseCopyKernel {
  const float *signal;
  float *spectrum;
  int bits;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0]);
    int reversed = 0;
    int value = i;
    for (int b = 0; b < bits; ++b) {
      reversed = (reversed << 1) | (value & 1);
      value >>= 1;
    }
    spectrum[2 * reversed] = signal[2 * i];
    spectrum[2 * reversed + 1] = signal[2 * i + 1];
  }
};

struct ButterflyKernel {
  float *spectrum;
  int length;

  void operator()(sycl::id<1> idx) const {
    const int butterfly = static_cast<int>(idx[0]);
    const int half = length >> 1;
    const int group = butterfly / half;
    const int position = butterfly - group * half;
    const int i = group * length + position;
    const int j = i + half;

    const double angle = -2.0 * kPi * static_cast<double>(position) /
                         static_cast<double>(length);
    const float wr = static_cast<float>(sycl::cos(angle));
    const float wi = static_cast<float>(sycl::sin(angle));

    const float ar = spectrum[2 * i];
    const float ai = spectrum[2 * i + 1];
    const float br = spectrum[2 * j];
    const float bi = spectrum[2 * j + 1];

    const float tr = wr * br - wi * bi;
    const float ti = wr * bi + wi * br;

    spectrum[2 * i] = ar + tr;
    spectrum[2 * i + 1] = ai + ti;
    spectrum[2 * j] = ar - tr;
    spectrum[2 * j + 1] = ai - ti;
  }
};

struct NaiveTransformKernel {
  const float *signal;
  float *spectrum;
  int n;

  void operator()(sycl::id<1> idx) const {
    const int k = static_cast<int>(idx[0]);
    double real = 0.0;
    double imag = 0.0;
    for (int t = 0; t < n; ++t) {
      const long long product = static_cast<long long>(k) * t;
      const long long wrapped = product % n;
      const double angle =
          -2.0 * kPi * static_cast<double>(wrapped) / static_cast<double>(n);
      const double c = sycl::cos(angle);
      const double s = sycl::sin(angle);
      const double xr = static_cast<double>(signal[2 * t]);
      const double xi = static_cast<double>(signal[2 * t + 1]);
      real += xr * c - xi * s;
      imag += xr * s + xi * c;
    }
    spectrum[2 * k] = static_cast<float>(real);
    spectrum[2 * k + 1] = static_cast<float>(imag);
  }
};

} // namespace

extern "C" void solve(const float *signal, float *spectrum, int N) {
  if (N <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  const bool power_of_two = (N & (N - 1)) == 0;
  if (!power_of_two) {
    q.parallel_for(sycl::range<1>(N),
                   NaiveTransformKernel{signal, spectrum, N});
    q.wait();
    return;
  }

  int bits = 0;
  while ((1 << bits) < N) {
    ++bits;
  }

  q.parallel_for(sycl::range<1>(N),
                 BitReverseCopyKernel{signal, spectrum, bits});
  for (int length = 2; length <= N; length <<= 1) {
    q.parallel_for(sycl::range<1>(N / 2), ButterflyKernel{spectrum, length});
  }
  q.wait();
}
