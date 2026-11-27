#include <sycl/sycl.hpp>

namespace {

struct ConnectedComponentsKernel {
  const int *src;
  const int *dst;
  int *labels;
  int *parent;
  int N;
  int M;

  int findRoot(int vertex) const {
    int root = vertex;
    while (parent[root] != root) {
      root = parent[root];
    }
    while (parent[vertex] != vertex) {
      const int next = parent[vertex];
      parent[vertex] = root;
      vertex = next;
    }
    return root;
  }

  void operator()(sycl::item<1>) const {
    for (int i = 0; i < N; ++i) {
      parent[i] = i;
    }
    for (int edge = 0; edge < M; ++edge) {
      const int a = src[edge];
      const int b = dst[edge];
      if (a < 0 || a >= N || b < 0 || b >= N) {
        continue;
      }
      const int rootA = findRoot(a);
      const int rootB = findRoot(b);
      if (rootA != rootB) {
        const int minimum = (rootA < rootB) ? rootA : rootB;
        const int maximum = (rootA < rootB) ? rootB : rootA;
        parent[maximum] = minimum;
        parent[minimum] = minimum;
      }
    }
    for (int i = 0; i < N; ++i) {
      labels[i] = findRoot(i);
    }
  }
};

} // namespace

extern "C" void solve(const int *src, const int *dst, int *labels, int N, int M) {
  sycl::queue q{sycl::property::queue::in_order()};
  if (N <= 0) {
    return;
  }
  int *parent = sycl::malloc_device<int>(N, q);
  q.parallel_for(sycl::range<1>(1),
                 ConnectedComponentsKernel{src, dst, labels, parent, N, M});
  q.wait();
  sycl::free(parent, q);
}
