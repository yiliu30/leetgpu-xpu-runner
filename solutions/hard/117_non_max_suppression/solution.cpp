#include <sycl/sycl.hpp>

namespace {

struct NmsKernel {
  const float *boxes;
  const float *scores;
  int *keep;
  int *order;
  unsigned char *taken;
  unsigned char *alive;
  int n;
  float iou_threshold;

  void operator()() const {
    for (int i = 0; i < n; ++i) {
      taken[i] = 0;
    }
    for (int rank = 0; rank < n; ++rank) {
      int best = -1;
      float best_score = -3.402823466e+38f;
      for (int i = 0; i < n; ++i) {
        if (!taken[i] && (best < 0 || scores[i] > best_score)) {
          best = i;
          best_score = scores[i];
        }
      }
      taken[best] = 1;
      order[rank] = best;
    }

    for (int i = 0; i < n; ++i) {
      alive[i] = 1;
    }
    for (int rank = 0; rank < n; ++rank) {
      const int i = order[rank];
      if (!alive[i]) {
        continue;
      }
      const float ix1 = boxes[static_cast<size_t>(i) * 4 + 0];
      const float iy1 = boxes[static_cast<size_t>(i) * 4 + 1];
      const float ix2 = boxes[static_cast<size_t>(i) * 4 + 2];
      const float iy2 = boxes[static_cast<size_t>(i) * 4 + 3];
      const float area_i = (ix2 - ix1) * (iy2 - iy1);
      for (int later = rank + 1; later < n; ++later) {
        const int j = order[later];
        if (!alive[j]) {
          continue;
        }
        const float jx1 = boxes[static_cast<size_t>(j) * 4 + 0];
        const float jy1 = boxes[static_cast<size_t>(j) * 4 + 1];
        const float jx2 = boxes[static_cast<size_t>(j) * 4 + 2];
        const float jy2 = boxes[static_cast<size_t>(j) * 4 + 3];
        const float w =
            sycl::fmax(0.0f, sycl::fmin(ix2, jx2) - sycl::fmax(ix1, jx1));
        const float h =
            sycl::fmax(0.0f, sycl::fmin(iy2, jy2) - sycl::fmax(iy1, jy1));
        const float intersection = w * h;
        const float area_j = (jx2 - jx1) * (jy2 - jy1);
        const float iou = intersection / (area_i + area_j - intersection);
        if (iou > iou_threshold) {
          alive[j] = 0;
        }
      }
    }

    for (int i = 0; i < n; ++i) {
      keep[i] = alive[i] ? 1 : 0;
    }
  }
};

} // namespace

extern "C" void solve(const float *boxes, const float *scores, int *keep, int n,
                      float iou_threshold) {
  if (n <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  int *order = sycl::malloc_device<int>(static_cast<size_t>(n), queue);
  unsigned char *taken =
      sycl::malloc_device<unsigned char>(static_cast<size_t>(n), queue);
  unsigned char *alive =
      sycl::malloc_device<unsigned char>(static_cast<size_t>(n), queue);
  queue.single_task(
      NmsKernel{boxes, scores, keep, order, taken, alive, n, iou_threshold});
  queue.wait();
  sycl::free(alive, queue);
  sycl::free(taken, queue);
  sycl::free(order, queue);
}
