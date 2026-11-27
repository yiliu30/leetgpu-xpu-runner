#include <sycl/sycl.hpp>

namespace {

struct LogisticRegressionKernel {
  const float *X;
  const float *y;
  float *beta;
  float *work;
  int nSamples;
  int nFeatures;

  void operator()() const {
    float *z = work;
    float *p = z + nSamples;
    float *w = p + nSamples;
    float *gradient = w + nSamples;
    float *hessian = gradient + nFeatures;
    float *delta = hessian + nFeatures * nFeatures;

    const int maxIter = 1000;
    const float tolerance = 1e-8f;
    const float l2Reg = 1e-6f;

    for (int i = 0; i < nFeatures; ++i) {
      beta[i] = 0.0f;
    }

    for (int iter = 0; iter < maxIter; ++iter) {
      for (int row = 0; row < nSamples; ++row) {
        float value = 0.0f;
        for (int i = 0; i < nFeatures; ++i) {
          value += X[static_cast<std::size_t>(row) * nFeatures + i] * beta[i];
        }
        z[row] = value;
        float prob;
        if (value >= 0.0f) {
          prob = 1.0f / (1.0f + sycl::exp(-value));
        } else {
          const float e = sycl::exp(value);
          prob = e / (1.0f + e);
        }
        p[row] = prob;
        w[row] = sycl::fmax(prob * (1.0f - prob), 1e-8f);
      }

      for (int i = 0; i < nFeatures; ++i) {
        float sum = 0.0f;
        for (int row = 0; row < nSamples; ++row) {
          sum += X[static_cast<std::size_t>(row) * nFeatures + i] *
                 (p[row] - y[row]);
        }
        gradient[i] = sum + l2Reg * beta[i];
      }

      for (int i = 0; i < nFeatures; ++i) {
        for (int j = 0; j < nFeatures; ++j) {
          float sum = 0.0f;
          for (int row = 0; row < nSamples; ++row) {
            sum += X[static_cast<std::size_t>(row) * nFeatures + i] *
                   X[static_cast<std::size_t>(row) * nFeatures + j] * w[row];
          }
          hessian[i * nFeatures + j] = sum + ((i == j) ? l2Reg : 0.0f);
        }
      }

      for (int i = 0; i < nFeatures; ++i) {
        for (int j = 0; j <= i; ++j) {
          float sum = hessian[i * nFeatures + j];
          for (int k = 0; k < j; ++k) {
            sum -= hessian[i * nFeatures + k] * hessian[j * nFeatures + k];
          }
          if (i == j) {
            hessian[i * nFeatures + i] = sycl::sqrt(sum);
          } else {
            hessian[i * nFeatures + j] = sum / hessian[j * nFeatures + j];
          }
        }
      }

      for (int i = 0; i < nFeatures; ++i) {
        float sum = gradient[i];
        for (int j = 0; j < i; ++j) {
          sum -= hessian[i * nFeatures + j] * delta[j];
        }
        delta[i] = sum / hessian[i * nFeatures + i];
      }
      for (int i = nFeatures - 1; i >= 0; --i) {
        float sum = delta[i];
        for (int j = i + 1; j < nFeatures; ++j) {
          sum -= hessian[j * nFeatures + i] * delta[j];
        }
        delta[i] = sum / hessian[i * nFeatures + i];
      }

      float normSquared = 0.0f;
      for (int i = 0; i < nFeatures; ++i) {
        beta[i] -= delta[i];
        normSquared += delta[i] * delta[i];
      }
      if (sycl::sqrt(normSquared) < tolerance) {
        break;
      }
    }
  }
};

} // namespace

extern "C" void solve(const float *X, const float *y, float *beta,
                      int n_samples, int n_features) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int scratchSize =
      3 * n_samples + 2 * n_features + n_features * n_features;
  float *work = sycl::malloc_device<float>(scratchSize, q);
  q.single_task(
      LogisticRegressionKernel{X, y, beta, work, n_samples, n_features});
  q.wait();
  sycl::free(work, q);
}
