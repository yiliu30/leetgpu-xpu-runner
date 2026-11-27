#include <sycl/sycl.hpp>

namespace {

struct MsDeformAttentionKernel {
  const float *value;
  const int *spatial_shapes;
  const float *sampling_loc;
  const float *attn_weight;
  float *output;
  int num_queries;
  int num_heads;
  int head_dim;
  int num_levels;
  int num_points;

  void operator()(sycl::id<1> id) const {
    const int d = static_cast<int>(id[0] % static_cast<size_t>(head_dim));
    const size_t rest = id[0] / static_cast<size_t>(head_dim);
    const int head = static_cast<int>(rest % static_cast<size_t>(num_heads));
    const int query = static_cast<int>(rest / static_cast<size_t>(num_heads));

    float acc = 0.0f;
    int level_start = 0;
    for (int level = 0; level < num_levels; ++level) {
      const int level_h = spatial_shapes[level * 2];
      const int level_w = spatial_shapes[level * 2 + 1];
      for (int point = 0; point < num_points; ++point) {
        const size_t loc_base =
            ((static_cast<size_t>(query) * num_heads + head) * num_levels +
             level) *
                num_points * 2 +
            static_cast<size_t>(point) * 2;
        const float x =
            sampling_loc[loc_base] * static_cast<float>(level_w) - 0.5f;
        const float y =
            sampling_loc[loc_base + 1] * static_cast<float>(level_h) - 0.5f;
        const float x_floor = sycl::floor(x);
        const float y_floor = sycl::floor(y);
        const float dx = x - x_floor;
        const float dy = y - y_floor;
        const int x0 = static_cast<int>(x_floor);
        const int y0 = static_cast<int>(y_floor);
        const float weight = attn_weight[(
            ((static_cast<size_t>(query) * num_heads + head) * num_levels +
             level) *
                num_points +
            point)];

        for (int corner = 0; corner < 4; ++corner) {
          const int xi = x0 + (corner & 1);
          const int yi = y0 + (corner >> 1);
          if (xi < 0 || xi >= level_w || yi < 0 || yi >= level_h) {
            continue;
          }
          const float wx = (corner & 1) ? dx : (1.0f - dx);
          const float wy = (corner >> 1) ? dy : (1.0f - dy);
          const size_t value_index =
              (static_cast<size_t>(level_start + yi * level_w + xi) *
                   num_heads +
               head) *
                  head_dim +
              d;
          acc += value[value_index] * (weight * wx * wy);
        }
      }
      level_start += level_h * level_w;
    }
    output[(static_cast<size_t>(query) * num_heads + head) * head_dim + d] =
        acc;
  }
};

} // namespace

extern "C" void solve(const float *value, const int *spatial_shapes,
                      const float *sampling_loc, const float *attn_weight,
                      float *output, int num_queries, int num_heads,
                      int head_dim, int num_levels, int num_points) {
  if (num_queries <= 0 || num_heads <= 0) {
    return;
  }
  sycl::queue queue{sycl::property::queue::in_order()};
  queue.parallel_for(
      sycl::range<1>(static_cast<size_t>(num_queries) * num_heads * head_dim),
      MsDeformAttentionKernel{value, spatial_shapes, sampling_loc, attn_weight,
                              output, num_queries, num_heads, head_dim,
                              num_levels, num_points});
  queue.wait();
}
