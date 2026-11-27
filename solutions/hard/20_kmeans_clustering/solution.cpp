#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct KMeansKernel {
  const float *data_x;
  const float *data_y;
  int *labels;
  float *initial_centroid_x;
  float *initial_centroid_y;
  float *final_centroid_x;
  float *final_centroid_y;
  int sample_size;
  int k;
  int max_iterations;
  sycl::local_accessor<float, 1> centroid_x;
  sycl::local_accessor<float, 1> centroid_y;

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_size = static_cast<int>(item.get_local_range(0));

    for (int j = tid; j < k; j += local_size) {
      centroid_x[j] = initial_centroid_x[j];
      centroid_y[j] = initial_centroid_y[j];
    }
    sycl::group_barrier(group);

    for (int iteration = 0; iteration < max_iterations; ++iteration) {
      for (int i = tid; i < sample_size; i += local_size) {
        const float x = data_x[i];
        const float y = data_y[i];
        int best = 0;
        float best_distance = 0.0f;
        for (int j = 0; j < k; ++j) {
          const float dx = x - centroid_x[j];
          const float dy = y - centroid_y[j];
          const float distance = dx * dx + dy * dy;
          if (j == 0 || distance < best_distance) {
            best_distance = distance;
            best = j;
          }
        }
        labels[i] = best;
      }
      sycl::group_barrier(group);

      for (int j = tid; j < k; j += local_size) {
        double sum_x = 0.0;
        double sum_y = 0.0;
        int count = 0;
        for (int i = 0; i < sample_size; ++i) {
          if (labels[i] == j) {
            sum_x += static_cast<double>(data_x[i]);
            sum_y += static_cast<double>(data_y[i]);
            ++count;
          }
        }
        if (count > 0) {
          centroid_x[j] =
              static_cast<float>(sum_x / static_cast<double>(count));
          centroid_y[j] =
              static_cast<float>(sum_y / static_cast<double>(count));
        }
      }
      sycl::group_barrier(group);
    }

    for (int j = tid; j < k; j += local_size) {
      final_centroid_x[j] = centroid_x[j];
      final_centroid_y[j] = centroid_y[j];
    }
  }
};

} // namespace

extern "C" void solve(const float *data_x, const float *data_y, int *labels,
                      float *initial_centroid_x, float *initial_centroid_y,
                      float *final_centroid_x, float *final_centroid_y,
                      int sample_size, int k, int max_iterations) {
  if (sample_size <= 0 || k <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> centroid_x(sycl::range<1>(k), cgh);
    sycl::local_accessor<float, 1> centroid_y(sycl::range<1>(k), cgh);
    KMeansKernel kernel{data_x,
                        data_y,
                        labels,
                        initial_centroid_x,
                        initial_centroid_y,
                        final_centroid_x,
                        final_centroid_y,
                        sample_size,
                        k,
                        max_iterations,
                        centroid_x,
                        centroid_y};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(kWorkGroupSize),
                                       sycl::range<1>(kWorkGroupSize)),
                     kernel);
  });
  q.wait();
}
