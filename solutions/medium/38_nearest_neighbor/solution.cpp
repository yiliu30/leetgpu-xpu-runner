#include <limits>
#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct NearestNeighborKernel {
  const float *points;
  int *indices;
  int n;

  void operator()(sycl::id<1> idx) const {
    const int i = idx[0];
    if (i >= n) {
      return;
    }

    const float px = points[3 * i];
    const float py = points[3 * i + 1];
    const float pz = points[3 * i + 2];
    float bestDistance = std::numeric_limits<float>::infinity();
    int bestIndex = 0;

    for (int j = 0; j < n; ++j) {
      if (j == i) {
        continue;
      }
      const float dx = px - points[3 * j];
      const float dy = py - points[3 * j + 1];
      const float dz = pz - points[3 * j + 2];
      const float distance = dx * dx + dy * dy + dz * dz;
      if (distance < bestDistance) {
        bestDistance = distance;
        bestIndex = j;
      }
    }
    indices[i] = bestIndex;
  }
};

} // namespace

extern "C" void solve(const float *points, int *indices, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  sycl::range<1> workGroupSize(WORK_GROUP_SIZE);
  sycl::range<1> globalSize(numWorkGroups * WORK_GROUP_SIZE);

  q.parallel_for(globalSize, NearestNeighborKernel{points, indices, N});
  q.wait();
}
