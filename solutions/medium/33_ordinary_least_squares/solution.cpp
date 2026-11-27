#include <sycl/sycl.hpp>

namespace {

struct OlsKernel {
  const float *X;
  const float *y;
  float *beta;
  float *work;
  int nSamples;
  int nFeatures;

  void operator()() const {
    float *xtx = work;
    float *xty = xtx + nFeatures * nFeatures;
    float *z = xty + nFeatures;
    float *result = z + nFeatures;

    for (int i = 0; i < nFeatures; ++i) {
      for (int j = 0; j < nFeatures; ++j) {
        float sum = 0.0f;
        for (int row = 0; row < nSamples; ++row) {
          sum += X[static_cast<std::size_t>(row) * nFeatures + i] *
                 X[static_cast<std::size_t>(row) * nFeatures + j];
        }
        xtx[i * nFeatures + j] = sum;
      }
      float sum = 0.0f;
      for (int row = 0; row < nSamples; ++row) {
        sum += X[static_cast<std::size_t>(row) * nFeatures + i] * y[row];
      }
      xty[i] = sum;
    }

    for (int i = 0; i < nFeatures; ++i) {
      for (int j = 0; j <= i; ++j) {
        float sum = xtx[i * nFeatures + j];
        for (int k = 0; k < j; ++k) {
          sum -= xtx[i * nFeatures + k] * xtx[j * nFeatures + k];
        }
        if (i == j) {
          xtx[i * nFeatures + i] = sycl::sqrt(sum);
        } else {
          xtx[i * nFeatures + j] = sum / xtx[j * nFeatures + j];
        }
      }
    }

    for (int i = 0; i < nFeatures; ++i) {
      float sum = xty[i];
      for (int j = 0; j < i; ++j) {
        sum -= xtx[i * nFeatures + j] * z[j];
      }
      z[i] = sum / xtx[i * nFeatures + i];
    }

    for (int i = nFeatures - 1; i >= 0; --i) {
      float sum = z[i];
      for (int j = i + 1; j < nFeatures; ++j) {
        sum -= xtx[j * nFeatures + i] * result[j];
      }
      result[i] = sum / xtx[i * nFeatures + i];
    }

    for (int i = 0; i < nFeatures; ++i) {
      beta[i] = result[i];
    }
  }
};

} // namespace

extern "C" void solve(const float *X, const float *y, float *beta,
                      int n_samples, int n_features) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int scratchSize = 2 * n_features * n_features + 3 * n_features;
  float *work = sycl::malloc_device<float>(scratchSize, q);
  q.single_task(OlsKernel{X, y, beta, work, n_samples, n_features});
  q.wait();
  sycl::free(work, q);
}
