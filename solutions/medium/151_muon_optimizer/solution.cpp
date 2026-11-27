#include <sycl/sycl.hpp>

#include <algorithm>
#include <cmath>

namespace {

constexpr int TILE = 16;
constexpr int REDUCTION_WORK_GROUP = 256;

struct MuonNormKernel {
  const float *G;
  const float *Buf;
  int total;
  float momentum;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int i = item.get_global_id(0);
    if (i < total) {
      const float value = momentum * Buf[i] + G[i];
      reducer.combine(static_cast<double>(value) * static_cast<double>(value));
    }
  }
};

struct MuonNormSqrtKernel {
  double *norm;

  void operator()() const { *norm = sycl::sqrt(*norm); }
};

struct MuonInitKernel {
  const float *G;
  const float *Buf;
  const double *norm;
  float *X;
  int M;
  int N;
  int R;
  int C;
  float momentum;
  bool transposed;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int xRow = linear / C;
    const int xCol = linear - xRow * C;
    const int sourceRow = transposed ? xCol : xRow;
    const int sourceCol = transposed ? xRow : xCol;
    const std::size_t source =
        static_cast<std::size_t>(sourceRow) * N + sourceCol;
    const float value = momentum * Buf[source] + G[source];
    X[linear] = value / (static_cast<float>(*norm) + 1e-7f);
  }
};

struct MuonGramKernel {
  sycl::local_accessor<float, 2> tileX;
  sycl::local_accessor<float, 2> tileXt;
  const float *X;
  float *out;
  int R;
  int C;

  void operator()(sycl::nd_item<2> item) const {
    const int localRow = static_cast<int>(item.get_local_id(0));
    const int localCol = static_cast<int>(item.get_local_id(1));
    const int row = static_cast<int>(item.get_global_id(0));
    const int col = static_cast<int>(item.get_global_id(1));
    float acc = 0.0f;
    for (int k0 = 0; k0 < C; k0 += TILE) {
      const int kRow = k0 + localRow;
      const int kCol = k0 + localCol;
      tileX[localRow][localCol] =
          (row < R && kCol < C) ? X[row * C + kCol] : 0.0f;
      tileXt[localRow][localCol] =
          (col < R && kRow < C) ? X[col * C + kRow] : 0.0f;
      item.barrier(sycl::access::fence_space::local_space);
      for (int t = 0; t < TILE; ++t) {
        acc += tileX[localRow][t] * tileXt[t][localCol];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (row < R && col < R) {
      out[row * R + col] = acc;
    }
  }
};

struct MuonSquareScaleKernel {
  sycl::local_accessor<float, 2> tileLeft;
  sycl::local_accessor<float, 2> tileRight;
  const float *A;
  float *B;
  int R;
  float linearScale;
  float squareScale;

  void operator()(sycl::nd_item<2> item) const {
    const int localRow = static_cast<int>(item.get_local_id(0));
    const int localCol = static_cast<int>(item.get_local_id(1));
    const int row = static_cast<int>(item.get_global_id(0));
    const int col = static_cast<int>(item.get_global_id(1));
    float acc = 0.0f;
    for (int k0 = 0; k0 < R; k0 += TILE) {
      tileLeft[localRow][localCol] =
          (row < R && k0 + localCol < R) ? A[row * R + k0 + localCol] : 0.0f;
      tileRight[localRow][localCol] =
          (col < R && k0 + localRow < R) ? A[(k0 + localRow) * R + col] : 0.0f;
      item.barrier(sycl::access::fence_space::local_space);
      for (int t = 0; t < TILE; ++t) {
        acc += tileLeft[localRow][t] * tileRight[t][localCol];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (row < R && col < R) {
      const int index = row * R + col;
      B[index] = linearScale * A[index] + squareScale * acc;
    }
  }
};

struct MuonMatMulKernel {
  sycl::local_accessor<float, 2> tileLeft;
  sycl::local_accessor<float, 2> tileRight;
  const float *left;
  const float *right;
  float *out;
  int R;
  int C;

  void operator()(sycl::nd_item<2> item) const {
    const int localRow = static_cast<int>(item.get_local_id(0));
    const int localCol = static_cast<int>(item.get_local_id(1));
    const int row = static_cast<int>(item.get_global_id(0));
    const int col = static_cast<int>(item.get_global_id(1));
    float acc = 0.0f;
    for (int k0 = 0; k0 < R; k0 += TILE) {
      tileLeft[localRow][localCol] =
          (row < R && k0 + localCol < R) ? left[row * R + k0 + localCol] : 0.0f;
      tileRight[localRow][localCol] = (col < C && k0 + localRow < R)
                                          ? right[(k0 + localRow) * C + col]
                                          : 0.0f;
      item.barrier(sycl::access::fence_space::local_space);
      for (int t = 0; t < TILE; ++t) {
        acc += tileLeft[localRow][t] * tileRight[t][localCol];
      }
      item.barrier(sycl::access::fence_space::local_space);
    }
    if (row < R && col < C) {
      out[row * C + col] = acc;
    }
  }
};

struct MuonUpdateKernel {
  const float *T;
  float *X;
  int total;
  float linearScale;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0]);
    X[i] = linearScale * X[i] + T[i];
  }
};

struct MuonWriteKernel {
  const float *X;
  float *output;
  int M;
  int N;
  int R;
  int C;
  float scale;
  bool transposed;

  void operator()(sycl::id<1> idx) const {
    const int linear = static_cast<int>(idx[0]);
    const int row = linear / N;
    const int col = linear - row * N;
    const int xIndex = transposed ? col * C + row : row * C + col;
    output[linear] = X[xIndex] * scale;
  }
};

inline int roundUp(int value, int multiple) {
  return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

extern "C" void solve(const float *G, const float *Buf, float *output, int M,
                      int N, float momentum) {
  sycl::queue q{sycl::property::queue::in_order()};
  const bool transposed = M > N;
  const int R = std::min(M, N);
  const int C = std::max(M, N);
  const int total = M * N;

  float *X = sycl::malloc_device<float>(R * C, q);
  float *A = sycl::malloc_device<float>(R * R, q);
  float *B = sycl::malloc_device<float>(R * R, q);
  float *T = sycl::malloc_device<float>(R * C, q);
  double *norm = sycl::malloc_device<double>(1, q);

  const int normGroups =
      (total + REDUCTION_WORK_GROUP - 1) / REDUCTION_WORK_GROUP;
  q.submit([&](sycl::handler &h) {
    h.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(normGroups * REDUCTION_WORK_GROUP),
                          sycl::range<1>(REDUCTION_WORK_GROUP)),
        sycl::reduction(norm, sycl::plus<double>(),
                        sycl::property::reduction::initialize_to_identity()),
        MuonNormKernel{G, Buf, total, momentum});
  });
  q.single_task(MuonNormSqrtKernel{norm});
  q.parallel_for(
      sycl::range<1>(R * C),
      MuonInitKernel{G, Buf, norm, X, M, N, R, C, momentum, transposed});

  const sycl::range<2> localRange(TILE, TILE);
  const sycl::range<2> gramGlobal(roundUp(R, TILE), roundUp(R, TILE));
  const sycl::range<2> matmulGlobal(roundUp(R, TILE), roundUp(C, TILE));

  for (int step = 0; step < 5; ++step) {
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 2> tileX(sycl::range<2>(TILE, TILE), h);
      sycl::local_accessor<float, 2> tileXt(sycl::range<2>(TILE, TILE), h);
      h.parallel_for(sycl::nd_range<2>(gramGlobal, localRange),
                     MuonGramKernel{tileX, tileXt, X, A, R, C});
    });
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 2> tileLeft(sycl::range<2>(TILE, TILE), h);
      sycl::local_accessor<float, 2> tileRight(sycl::range<2>(TILE, TILE), h);
      h.parallel_for(sycl::nd_range<2>(gramGlobal, localRange),
                     MuonSquareScaleKernel{tileLeft, tileRight, A, B, R,
                                           -4.7750f, 2.0315f});
    });
    q.submit([&](sycl::handler &h) {
      sycl::local_accessor<float, 2> tileLeft(sycl::range<2>(TILE, TILE), h);
      sycl::local_accessor<float, 2> tileRight(sycl::range<2>(TILE, TILE), h);
      h.parallel_for(sycl::nd_range<2>(matmulGlobal, localRange),
                     MuonMatMulKernel{tileLeft, tileRight, B, X, T, R, C});
    });
    q.parallel_for(sycl::range<1>(R * C),
                   MuonUpdateKernel{T, X, R * C, 3.4445f});
  }

  const float scale =
      static_cast<float>(std::sqrt(static_cast<double>(std::max(M, N))));
  q.parallel_for(sycl::range<1>(total),
                 MuonWriteKernel{X, output, M, N, R, C, scale, transposed});
  q.wait();

  sycl::free(X, q);
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(T, q);
  sycl::free(norm, q);
}
