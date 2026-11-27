#include <sycl/sycl.hpp>

namespace {

struct NBodyForceKernel {
  const float *positions;
  const float *masses;
  float *forces;
  int N;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i >= N) {
      return;
    }
    const float *pi = positions + 3 * i;
    float fx = 0.0f;
    float fy = 0.0f;
    float fz = 0.0f;
    for (int j = 0; j < N; ++j) {
      const float dx = positions[3 * j] - pi[0];
      const float dy = positions[3 * j + 1] - pi[1];
      const float dz = positions[3 * j + 2] - pi[2];
      const float dist_sq = dx * dx + dy * dy + dz * dz;
      const float softened = dist_sq + 1.0e-6f;
      const float inv_denom = masses[j] / (softened * sycl::sqrt(softened));
      fx += dx * inv_denom;
      fy += dy * inv_denom;
      fz += dz * inv_denom;
    }
    forces[3 * i] = fx;
    forces[3 * i + 1] = fy;
    forces[3 * i + 2] = fz;
  }
};

} // namespace

extern "C" void solve(const float *positions, const float *masses, float *forces, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  q.parallel_for(sycl::range<1>(N), NBodyForceKernel{positions, masses, forces, N});
  q.wait();
}
