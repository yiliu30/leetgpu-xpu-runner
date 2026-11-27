#include <sycl/sycl.hpp>

#include <limits>

namespace {

int hamming_distance(const int *codes, const int *query) {
  int distance = 0;
  for (int word = 0; word < 8; ++word) {
    unsigned int diff = static_cast<unsigned int>(codes[word] ^ query[word]);
    while (diff != 0u) {
      distance += static_cast<int>(diff & 1u);
      diff >>= 1u;
    }
  }
  return distance;
}

struct HammingDistanceKernel {
  const int *codes;
  const int *query;
  int *distances;
  int N;

  void operator()(sycl::id<1> id) const {
    const int row = static_cast<int>(id[0]);
    if (row < N) {
      distances[row] = hamming_distance(codes + static_cast<size_t>(row) * 8, query);
    }
  }
};

struct TopKSelectKernel {
  const int *distances;
  int *output;
  int N;
  int k;

  void operator()() const {
    long long previous_score = -1;
    for (int rank = 0; rank < k; ++rank) {
      long long best_score = std::numeric_limits<long long>::max();
      int best_index = -1;
      for (int row = 0; row < N; ++row) {
        const long long score = static_cast<long long>(distances[row]) * N + row;
        if (score > previous_score && score < best_score) {
          best_score = score;
          best_index = row;
        }
      }
      output[rank] = best_index;
      previous_score = best_score;
    }
  }
};

} // namespace

extern "C" void solve(const int *codes, const int *query, int *output, int N, int k) {
  sycl::queue q{sycl::property::queue::in_order()};
  int *distances = sycl::malloc_device<int>(N, q);
  q.parallel_for(sycl::range<1>(N), HammingDistanceKernel{codes, query, distances, N});
  q.single_task(TopKSelectKernel{distances, output, N, k});
  q.wait();
  sycl::free(distances, q);
}
