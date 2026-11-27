#include <sycl/sycl.hpp>

namespace {

struct VitPatchEmbeddingKernel {
  const float *images;
  const float *patch_weight;
  const float *patch_bias;
  const float *cls_token;
  const float *pos_embed;
  float *output;
  int C;
  int H;
  int W;
  int P;
  int D;
  int N;
  int tokens;

  void operator()(sycl::id<1> id) const {
    const size_t linear = id[0];
    const int token_in_batch = static_cast<int>((linear / D) % tokens);
    const int dim = static_cast<int>(linear % D);
    const size_t batch = linear / (static_cast<size_t>(tokens) * D);

    double value = static_cast<double>(pos_embed[static_cast<size_t>(token_in_batch) * D + dim]);
    if (token_in_batch == 0) {
      value += static_cast<double>(cls_token[dim]);
    } else {
      const int patch = token_in_batch - 1;
      const int grid_w = W / P;
      const int patch_row = patch / grid_w;
      const int patch_col = patch % grid_w;
      const int patch_size = P * P;
      const int patch_elements = C * patch_size;
      value += static_cast<double>(patch_bias[dim]);
      for (int channel = 0; channel < C; ++channel) {
        for (int row = 0; row < P; ++row) {
          for (int col = 0; col < P; ++col) {
            const size_t image_index =
                ((static_cast<size_t>(batch) * C + channel) * H + patch_row * P + row) * W +
                patch_col * P + col;
            const int patch_index = channel * patch_size + row * P + col;
            value += static_cast<double>(images[image_index]) *
                     static_cast<double>(patch_weight[static_cast<size_t>(dim) * patch_elements +
                                                      patch_index]);
          }
        }
      }
    }
    output[linear] = static_cast<float>(value);
  }
};

} // namespace

extern "C" void solve(const float *images, const float *patch_weight, const float *patch_bias,
                      const float *cls_token, const float *pos_embed, float *output, int B, int C,
                      int H, int W, int P, int D) {
  sycl::queue q{sycl::property::queue::in_order()};
  const int N = (H / P) * (W / P);
  const int tokens = N + 1;
  q.parallel_for(sycl::range<1>(static_cast<size_t>(B) * tokens * D),
                 VitPatchEmbeddingKernel{images, patch_weight, patch_bias, cls_token, pos_embed,
                                         output, C, H, W, P, D, N, tokens});
  q.wait();
}
