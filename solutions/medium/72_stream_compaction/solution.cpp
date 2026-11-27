#include <sycl/sycl.hpp>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct StreamCompactionBlockSumsKernel {
  sycl::local_accessor<int, 1> scan;
  const float *A;
  int *blockSums;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int localId = static_cast<int>(item.get_local_id(0));
    const int globalId = static_cast<int>(item.get_global_id(0));
    const int flag = (globalId < N && A[globalId] > 0.0f) ? 1 : 0;
    scan[localId] = flag;
    item.barrier(sycl::access::fence_space::local_space);
    for (int offset = 1; offset < WORK_GROUP_SIZE; offset <<= 1) {
      int value = 0;
      if (localId >= offset) {
        value = scan[localId - offset];
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (localId >= offset) {
        scan[localId] += value;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (localId == WORK_GROUP_SIZE - 1) {
      blockSums[item.get_group(0)] = scan[localId];
    }
  }
};

struct StreamCompactionScanBlocksKernel {
  int *blockSums;
  int numBlocks;

  void operator()(sycl::item<1>) const {
    int running = 0;
    for (int block = 0; block < numBlocks; ++block) {
      const int count = blockSums[block];
      blockSums[block] = running;
      running += count;
    }
  }
};

struct StreamCompactionScatterKernel {
  sycl::local_accessor<int, 1> scan;
  const float *A;
  const int *blockOffsets;
  float *out;
  int N;

  void operator()(sycl::nd_item<1> item) const {
    const int localId = static_cast<int>(item.get_local_id(0));
    const int globalId = static_cast<int>(item.get_global_id(0));
    const int flag = (globalId < N && A[globalId] > 0.0f) ? 1 : 0;
    scan[localId] = flag;
    item.barrier(sycl::access::fence_space::local_space);
    for (int offset = 1; offset < WORK_GROUP_SIZE; offset <<= 1) {
      int value = 0;
      if (localId >= offset) {
        value = scan[localId - offset];
      }
      item.barrier(sycl::access::fence_space::local_space);
      if (localId >= offset) {
        scan[localId] += value;
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (flag) {
      const int localExclusive = scan[localId] - 1;
      const int outputIndex = blockOffsets[item.get_group(0)] + localExclusive;
      out[outputIndex] = A[globalId];
    }
  }
};

} // namespace

extern "C" void solve(const float *A, int N, float *out) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (N <= 0) {
    return;
  }

  const int numBlocks = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;
  const int globalSize = numBlocks * WORK_GROUP_SIZE;
  int *blockOffsets = sycl::malloc_device<int>(numBlocks, q);

  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int, 1> scan(sycl::range<1>(WORK_GROUP_SIZE), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(globalSize),
                                     sycl::range<1>(WORK_GROUP_SIZE)),
                   StreamCompactionBlockSumsKernel{scan, A, blockOffsets, N});
  });
  q.parallel_for(sycl::range<1>(1),
                 StreamCompactionScanBlocksKernel{blockOffsets, numBlocks});
  q.submit([&](sycl::handler &h) {
    sycl::local_accessor<int, 1> scan(sycl::range<1>(WORK_GROUP_SIZE), h);
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(globalSize),
                          sycl::range<1>(WORK_GROUP_SIZE)),
        StreamCompactionScatterKernel{scan, A, blockOffsets, out, N});
  });
  q.wait();
  sycl::free(blockOffsets, q);
}
