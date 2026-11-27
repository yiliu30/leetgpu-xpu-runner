#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct MatrixPowerInitKernel {
  const float *input;
  float *base;
  float *identity;
  int n;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    const int total = n * n;
    if (i < total) {
      base[i] = input[i];
      identity[i] = (i / n == i % n) ? 1.0f : 0.0f;
    }
  }
};

struct MatrixPowerMatmulKernel {
  const float *a;
  const float *b;
  float *c;
  int n;

  void operator()(sycl::id<1> idx) const {
    const int index = idx[0];
    const int total = n * n;
    if (index >= total) {
      return;
    }
    const int row = index / n;
    const int col = index % n;
    float acc = 0.0f;
    for (int k = 0; k < n; ++k) {
      acc += a[row * n + k] * b[k * n + col];
    }
    c[index] = acc;
  }
};

struct MatrixPowerCopyKernel {
  const float *input;
  float *output;
  int total;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i < total) {
      output[i] = input[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N, int P) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int total = N * N;
  const int numWorkGroups = (total + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  if (P == 1) {
    q.parallel_for(globalSize, MatrixPowerCopyKernel{input, output, total});
    q.wait();
    return;
  }

  float *base = sycl::malloc_device<float>(total, q);
  float *baseTmp = sycl::malloc_device<float>(total, q);
  float *result = sycl::malloc_device<float>(total, q);
  float *resultTmp = sycl::malloc_device<float>(total, q);

  q.parallel_for(globalSize, MatrixPowerInitKernel{input, base, result, N});

  int exponent = P;
  while (exponent > 0) {
    if ((exponent & 1) != 0) {
      q.parallel_for(globalSize,
                     MatrixPowerMatmulKernel{result, base, resultTmp, N});
      float *swap = result;
      result = resultTmp;
      resultTmp = swap;
    }
    exponent >>= 1;
    if (exponent > 0) {
      q.parallel_for(globalSize,
                     MatrixPowerMatmulKernel{base, base, baseTmp, N});
      float *swap = base;
      base = baseTmp;
      baseTmp = swap;
    }
  }

  q.parallel_for(globalSize, MatrixPowerCopyKernel{result, output, total});
  q.wait();

  sycl::free(base, q);
  sycl::free(baseTmp, q);
  sycl::free(result, q);
  sycl::free(resultTmp, q);
}
