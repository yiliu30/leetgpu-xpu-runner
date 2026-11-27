#include <sycl/sycl.hpp>

namespace {

struct TurboQuantDecodeKernel {
  const unsigned char *K_idx;
  const signed char *qjl_signs;
  const float *gamma;
  const float *Pi;
  const float *S_mat;
  const float *codebook;
  float *decoded;
  int S;
  int D;
  int C;
  float qjlScale;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int key = linear / D;
    const int col = linear % D;
    float mse = 0.0f;
    float qjl = 0.0f;
    for (int feature = 0; feature < D; ++feature) {
      const int index = static_cast<int>(K_idx[key * D + feature]);
      const float centroid = index < C ? codebook[index] : 0.0f;
      mse += centroid * Pi[col * D + feature];
      qjl += static_cast<float>(qjl_signs[key * D + feature]) *
             S_mat[feature * D + col];
    }
    decoded[linear] = mse + qjlScale * gamma[key] * qjl;
  }
};

struct TurboQuantScoreKernel {
  const float *Q;
  const float *decoded;
  float *scores;
  int S;
  int D;

  void operator()(sycl::id<1> id) const {
    const int linear = static_cast<int>(id[0]);
    const int batch = linear / S;
    const int key = linear % S;
    float acc = 0.0f;
    for (int col = 0; col < D; ++col) {
      acc += Q[batch * D + col] * decoded[key * D + col];
    }
    scores[linear] = acc;
  }
};

} // namespace

extern "C" void solve(const float *Q, const unsigned char *K_idx,
                      const signed char *qjl_signs, const float *gamma,
                      const float *Pi, const float *S_mat,
                      const float *codebook, float *scores, int B, int S, int D,
                      int C) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (B <= 0 || S <= 0 || D <= 0 || C <= 0) {
    q.wait();
    return;
  }

  const size_t decodedSize = static_cast<size_t>(S) * D;
  float *decoded = sycl::malloc_device<float>(decodedSize, q);
  const float qjlScale =
      sycl::sqrt(3.14159265358979323846f / 2.0f) / static_cast<float>(D);
  q.parallel_for(sycl::range<1>(decodedSize),
                 TurboQuantDecodeKernel{K_idx, qjl_signs, gamma, Pi, S_mat,
                                        codebook, decoded, S, D, C, qjlScale});
  q.parallel_for(sycl::range<1>(static_cast<size_t>(B) * S),
                 TurboQuantScoreKernel{Q, decoded, scores, S, D});
  q.wait();
  sycl::free(decoded, q);
}
