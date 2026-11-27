#include <sycl/sycl.hpp>

namespace {

unsigned int fnv1a_hash(unsigned int input) {
  constexpr unsigned int FNV_PRIME = 16777619u;
  constexpr unsigned int OFFSET_BASIS = 2166136261u;

  unsigned int hash = OFFSET_BASIS;
  for (int bytePos = 0; bytePos < 4; ++bytePos) {
    const unsigned char byte =
        static_cast<unsigned char>((input >> (bytePos * 8)) & 0xFFu);
    hash = (hash ^ byte) * FNV_PRIME;
  }
  return hash;
}

struct RainbowTableKernel {
  const int *input;
  unsigned int *output;
  int N;
  int R;

  void operator()(sycl::nd_item<1> item) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      unsigned int hash = static_cast<unsigned int>(input[i]);
      for (int round = 0; round < R; ++round) {
        hash = fnv1a_hash(hash);
      }
      output[i] = hash;
    }
  }
};

} // namespace

extern "C" void solve(const int *input, unsigned int *output, int N, int R) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int workGroupSize = 256;
  const int numWorkGroups = (N + workGroupSize - 1) / workGroupSize;
  sycl::nd_range<1> launchRange(sycl::range<1>(numWorkGroups * workGroupSize),
                                sycl::range<1>(workGroupSize));
  q.parallel_for(launchRange, RainbowTableKernel{input, output, N, R});
  q.wait();
}
