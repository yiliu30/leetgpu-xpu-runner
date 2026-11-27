#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct ReducePartialsKernel {
  sycl::local_accessor<double, 1> scratch;
  const float *input;
  double *partial;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int localId = item.get_local_id(0);
    const int i = item.get_global_id(0);
    scratch[localId] = (i < N) ? static_cast<double>(input[i]) : 0.0;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
      if (localId < stride) {
        scratch[localId] += scratch[localId + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (localId == 0) {
      partial[item.get_group(0)] = scratch[0];
    }
  }
};

struct ReduceFinalKernel {
  sycl::local_accessor<double, 1> scratch;
  const double *partial;
  float *output;
  int numWorkGroups;

  void operator()(sycl::nd_item<1> item) const {
    const int localId = item.get_local_id(0);
    double local = 0.0;
    for (int g = localId; g < numWorkGroups; g += WORK_GROUP_SIZE) {
      local += partial[g];
    }
    scratch[localId] = local;
    item.barrier(sycl::access::fence_space::local_space);
    for (int stride = WORK_GROUP_SIZE / 2; stride > 0; stride >>= 1) {
      if (localId < stride) {
        scratch[localId] += scratch[localId + stride];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (localId == 0) {
      output[0] = static_cast<float>(scratch[0]);
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  double *partial = sycl::malloc_device<double>(numWorkGroups, q);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(WORK_GROUP_SIZE), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(numWorkGroups * WORK_GROUP_SIZE),
                          sycl::range<1>(WORK_GROUP_SIZE)),
        ReducePartialsKernel{scratch, input, partial, N});
  });

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<double, 1> scratch(sycl::range<1>(WORK_GROUP_SIZE), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(WORK_GROUP_SIZE),
                                     sycl::range<1>(WORK_GROUP_SIZE)),
                   ReduceFinalKernel{scratch, partial, output, numWorkGroups});
  });
  q.wait();
  sycl::free(partial, q);
}
