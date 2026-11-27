#include <sycl/sycl.hpp>

#include <cstdint>

namespace {

constexpr int WORK_GROUP_SIZE = 256;

struct SelectionState {
  unsigned int prefix;
  int kth;
};

inline unsigned int orderedKey(float value) {
  union {
    float f;
    unsigned int u;
  } bits;
  bits.f = value;
  return (bits.u & 0x80000000u) ? ~bits.u : (bits.u ^ 0x80000000u);
}

inline float valueFromKey(unsigned int key) {
  union {
    float f;
    unsigned int u;
  } bits;
  bits.u = (key & 0x80000000u) ? (key ^ 0x80000000u) : ~key;
  return bits.f;
}

struct InitSelectionKernel {
  SelectionState *state;
  int k;

  void operator()() const { *state = SelectionState{0u, k}; }
};

struct CountSelectionBitsKernel {
  const float *input;
  const SelectionState *state;
  int N;
  unsigned int highMask;
  unsigned int bitMask;

  template <typename Reducer>
  void operator()(sycl::nd_item<1> item, Reducer &reducer) const {
    const int i = item.get_global_id(0);
    if (i < N) {
      const unsigned int key = orderedKey(input[i]);
      if ((key & highMask) == state->prefix && (key & bitMask) != 0u) {
        reducer.combine(static_cast<unsigned long long>(1));
      }
    }
  }
};

struct SelectBitKernel {
  SelectionState *state;
  const unsigned long long *count;
  unsigned int bitMask;

  void operator()() const {
    const unsigned long long bitCount = *count;
    if (bitCount >= static_cast<unsigned long long>(state->kth)) {
      state->prefix |= bitMask;
    } else {
      state->kth -= static_cast<int>(bitCount);
    }
  }
};

struct InitCounterKernel {
  int *counter;

  void operator()() const { *counter = 0; }
};

struct CollectTopKKernel {
  const float *input;
  const SelectionState *state;
  float *output;
  int *counter;
  int N;
  int k;

  void operator()(sycl::id<1> idx) const {
    const int i = static_cast<int>(idx[0]);
    const unsigned int key = orderedKey(input[i]);
    if (key > state->prefix) {
      sycl::atomic_ref<int, sycl::memory_order::relaxed,
                       sycl::memory_scope::device,
                       sycl::access::address_space::global_space>
          writeCounter(*counter);
      const int position = writeCounter.fetch_add(1);
      if (position < k) {
        output[position] = input[i];
      }
    }
  }
};

struct FinalizeTopKKernel {
  float *output;
  const int *counter;
  const SelectionState *state;
  int k;

  void operator()() const {
    int count = sycl::min(*counter, k);
    const float threshold = valueFromKey(state->prefix);
    for (int i = count; i < k; ++i) {
      output[i] = threshold;
    }
    for (int i = 1; i < k; ++i) {
      const float value = output[i];
      int j = i - 1;
      while (j >= 0 && output[j] < value) {
        output[j + 1] = output[j];
        --j;
      }
      output[j + 1] = value;
    }
  }
};

} // namespace

extern "C" void solve(const float *input, float *output, int N, int k) {
  sycl::queue q{sycl::property::queue::in_order()};
  SelectionState *state = sycl::malloc_device<SelectionState>(1, q);
  unsigned long long *count = sycl::malloc_device<unsigned long long>(1, q);
  int *counter = sycl::malloc_device<int>(1, q);
  const int numWorkGroups = (N + WORK_GROUP_SIZE - 1) / WORK_GROUP_SIZE;

  q.single_task(InitSelectionKernel{state, k});
  for (int bit = 31; bit >= 0; --bit) {
    const unsigned int bitMask = 1u << bit;
    const unsigned int highMask = (bit == 31) ? 0u : (0xFFFFFFFFu << (bit + 1));
    q.submit([&](sycl::handler &h) {
      h.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(numWorkGroups * WORK_GROUP_SIZE),
                            sycl::range<1>(WORK_GROUP_SIZE)),
          sycl::reduction(count, sycl::plus<unsigned long long>(),
                          sycl::property::reduction::initialize_to_identity()),
          CountSelectionBitsKernel{input, state, N, highMask, bitMask});
    });
    q.single_task(SelectBitKernel{state, count, bitMask});
  }

  q.single_task(InitCounterKernel{counter});
  q.parallel_for(sycl::range<1>(N),
                 CollectTopKKernel{input, state, output, counter, N, k});
  q.single_task(FinalizeTopKKernel{output, counter, state, k});
  q.wait();

  sycl::free(state, q);
  sycl::free(count, q);
  sycl::free(counter, q);
}
