#include <sycl/sycl.hpp>

namespace {

constexpr int kWorkGroupSize = 256;

struct Mamba2ScanKernel {
  const float *x;
  const float *dt;
  const float *dt_bias;
  const float *A;
  const float *B;
  const float *C;
  const float *D;
  const float *z;
  float *y;
  int batch;
  int seq_len;
  int num_heads;
  int head_dim;
  int num_groups;
  int state_dim;
  int total_heads;
  sycl::local_accessor<float, 1> state;
  sycl::local_accessor<float, 1> b_row;
  sycl::local_accessor<float, 1> c_row;
  sycl::local_accessor<float, 1> x_row;

  void operator()(sycl::nd_item<1> item) const {
    auto group = item.get_group();
    const int group_id = static_cast<int>(item.get_group(0));
    const int tid = static_cast<int>(item.get_local_id(0));
    const int local_size = static_cast<int>(item.get_local_range(0));

    if (group_id >= total_heads) {
      return;
    }
    const int batch_index = group_id / num_heads;
    const int head = group_id - batch_index * num_heads;
    const int heads_per_group = num_heads / num_groups;
    const int group_index = head / heads_per_group;

    const int P = head_dim;
    const int N = state_dim;

    for (int i = tid; i < P * N; i += local_size) {
      state[i] = 0.0f;
    }
    const float decay_rate = A[head];
    const float skip = D[head];
    const float bias = dt_bias[head];
    sycl::group_barrier(group);

    for (int t = 0; t < seq_len; ++t) {
      const float dt_value =
          dt[(static_cast<long long>(batch_index) * seq_len + t) * num_heads +
             head];
      const float shifted = dt_value + bias;
      const float delta =
          (shifted > 20.0f) ? shifted : sycl::log1p(sycl::exp(shifted));
      const float decay = sycl::exp(delta * decay_rate);

      const long long bc_offset =
          (static_cast<long long>(batch_index) * seq_len + t) * num_groups * N +
          static_cast<long long>(group_index) * N;
      const long long y_offset =
          (static_cast<long long>(batch_index) * seq_len + t) * num_heads * P +
          static_cast<long long>(head) * P;

      for (int n = tid; n < N; n += local_size) {
        b_row[n] = B[bc_offset + n];
        c_row[n] = C[bc_offset + n];
      }
      for (int p = tid; p < P; p += local_size) {
        x_row[p] = x[y_offset + p];
      }
      sycl::group_barrier(group);

      for (int i = tid; i < P * N; i += local_size) {
        const int p = i / N;
        const int n = i - p * N;
        state[i] = state[i] * decay + (x_row[p] * delta) * b_row[n];
      }
      sycl::group_barrier(group);

      for (int p = tid; p < P; p += local_size) {
        const int base = p * N;
        float acc = 0.0f;
        for (int n = 0; n < N; ++n) {
          acc += state[base + n] * c_row[n];
        }
        const float gate = z[y_offset + p];
        const float silu = gate * (1.0f / (1.0f + sycl::exp(-gate)));
        y[y_offset + p] = (acc + skip * x_row[p]) * silu;
      }
      sycl::group_barrier(group);
    }
  }
};

} // namespace

extern "C" void solve(const float *x, const float *dt, const float *dt_bias,
                      const float *A, const float *B, const float *C,
                      const float *D, const float *z, float *y, int batch,
                      int seq_len, int num_heads, int head_dim, int num_groups,
                      int state_dim) {
  if (batch <= 0 || seq_len <= 0 || num_heads <= 0 || head_dim <= 0 ||
      num_groups <= 0 || state_dim <= 0) {
    return;
  }
  sycl::queue q{sycl::property::queue::in_order()};

  q.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 1> state(
        sycl::range<1>(static_cast<size_t>(head_dim) * state_dim), cgh);
    sycl::local_accessor<float, 1> b_row(sycl::range<1>(state_dim), cgh);
    sycl::local_accessor<float, 1> c_row(sycl::range<1>(state_dim), cgh);
    sycl::local_accessor<float, 1> x_row(sycl::range<1>(head_dim), cgh);
    const size_t total_heads = static_cast<size_t>(batch) * num_heads;
    const size_t total_work_items = total_heads * kWorkGroupSize;
    Mamba2ScanKernel kernel{
        x,        dt,         dt_bias,   A,
        B,        C,          D,         z,
        y,        batch,      seq_len,   num_heads,
        head_dim, num_groups, state_dim, static_cast<int>(total_heads),
        state,    b_row,      c_row,     x_row};
    cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(total_work_items),
                                       sycl::range<1>(kWorkGroupSize)),
                     kernel);
  });
  q.wait();
}
