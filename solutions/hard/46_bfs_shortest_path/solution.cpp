#include <sycl/sycl.hpp>

namespace {

struct BfsKernel {
  const int *grid;
  int *result;
  int *distance;
  int *queue;
  int rows;
  int cols;
  int start_row;
  int start_col;
  int end_row;
  int end_col;

  void operator()() const {
    const int total = rows * cols;
    for (int i = 0; i < total; ++i) {
      distance[i] = -1;
    }

    const int start = start_row * cols + start_col;
    const int target = end_row * cols + end_col;
    distance[start] = 0;

    int head = 0;
    int tail = 0;
    queue[tail++] = start;

    while (head < tail) {
      const int cur = queue[head++];
      if (cur == target) {
        break;
      }
      const int row = cur / cols;
      const int col = cur - row * cols;
      const int next_distance = distance[cur] + 1;

      for (int step = 0; step < 4; ++step) {
        int next_row = row;
        int next_col = col;
        if (step == 0) {
          --next_row;
        } else if (step == 1) {
          ++next_row;
        } else if (step == 2) {
          --next_col;
        } else {
          ++next_col;
        }
        if (next_row < 0 || next_row >= rows || next_col < 0 ||
            next_col >= cols) {
          continue;
        }
        const int next = next_row * cols + next_col;
        if (distance[next] < 0 && grid[next] == 0) {
          distance[next] = next_distance;
          queue[tail++] = next;
        }
      }
    }

    result[0] = distance[target];
  }
};

} // namespace

extern "C" void solve(const int *grid, int *result, int rows, int cols,
                      int start_row, int start_col, int end_row, int end_col) {
  if (rows <= 0 || cols <= 0) {
    result[0] = -1;
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  const int total = rows * cols;
  int *distance = sycl::malloc_device<int>(total, q);
  int *queue = sycl::malloc_device<int>(total, q);

  q.submit([&](sycl::handler &cgh) {
    cgh.single_task(BfsKernel{grid, result, distance, queue, rows, cols,
                              start_row, start_col, end_row, end_col});
  });
  q.wait();

  sycl::free(queue, q);
  sycl::free(distance, q);
}
