# Quickstart

Everything below runs on a host with an Intel GPU, `icpx`, and a torch-XPU
environment.

```bash
cd /path/to/leetgpu-xpu-runner
PY=/path/to/venv/bin/python   # must have torch with XPU support
```

## Grade one submission

```bash
$PY -m tools.grade easy/1_vector_add \
  --solution solution.cpp \
  --repo /path/to/leetgpu-challenges
```

```
status: success
challenge: easy/1_vector_add
device: 0
compile: 1654 ms
functional: 13/13 passed
perf: median 0.585 ms (min 0.578, max 0.595, spread 1.2%), 300.0 MB @ 513.2 GB/s
```

The challenge can be an id (`1`), slug (`vector-add`), relative path
(`easy/1_vector_add`), or a directory. `--device N` pins to Level Zero device N.

## Machine-readable result

```bash
$PY -m tools.grade 5_softmax --solution sol.cpp --repo /path/to/leetgpu-challenges \
  --json result.json --quiet
```

Exit code is 0 on `success`, 1 otherwise. `status` is one of
`success | error | timeout | oom | interrupted`.

## Write your own SYCL solution

Start from the challenge's starter:

```bash
cp /path/to/leetgpu-challenges/challenges/easy/1_vector_add/starter/starter.sycl.cpp my_solution.cpp
```

Rules the runner relies on:

- Keep the `extern "C" void solve(...)` signature exactly as in `challenge.py`.
- Pointers are USM device pointers from torch XPU tensors; a plain `sycl::queue` can use them.
- Call `q.wait()` inside `solve`; the runner cannot synchronize your queue from Python.
- Use an in-order queue (`sycl::queue q{sycl::property::queue::in_order()};`) for multi-kernel solutions.
- Give each kernel a unique named functor; unnamed lambdas collide across modules and segfault.

Minimal example:

```cpp
#include <sycl/sycl.hpp>

namespace {
struct VectorAddKernel {
    const float *A, *B;
    float *C;
    int N;
    void operator()(sycl::nd_item<1> item) const {
        const int i = item.get_global_id(0);
        if (i < N) C[i] = A[i] + B[i];
    }
};
}  // namespace

extern "C" void solve(const float *A, const float *B, float *C, int N) {
    sycl::queue q{sycl::property::queue::in_order()};
    const int wg = 256;
    const int groups = (N + wg - 1) / wg;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(groups * wg), sycl::range<1>(wg)),
                   VectorAddKernel{A, B, C, N})
        .wait();
}
```

## Benchmark tuning

```bash
$PY -m tools.grade ... --runs 15 --warmups 2            # platform parity (default)
$PY -m tools.grade ... --inner-repeats 10               # stable median for sub-0.2 ms kernels
$PY -m tools.grade ... --timeout 120                    # raise the wall-clock budget
$PY -m tools.grade ... --atol 1e-4 --rtol 1e-4          # override tolerances
$PY -m tools.grade ... --keep-build                     # keep the .so and job JSON
```

## Web UI

Start the service (see below), then open `http://<xpu-host>:8080/` in a browser.
Click **API token**, paste `change-me`, pick a challenge, and press **Run on XPU**
(or Ctrl/Cmd+Enter). Deep links: `/#/c/1` (problem), `/#/c/1/code` (editor).

## Service mode

Controller on the XPU host:

```bash
LEETGPU_CHALLENGES=/path/to/leetgpu-challenges \
XPU_RUNNER_API_KEY=change-me \
XPU_RUNNER_DEVICES=0-3 \
$PY -m service
```

Unmodified LeetGPU client (same host):

```bash
cd /path/to/leetgpu-challenges
mkdir -p challenges/easy/1_vector_add/solution
cp solution.cpp challenges/easy/1_vector_add/solution/solution.cpp
SERVICE_URL=http://localhost:8080 LEETGPU_API_KEY=change-me \
  $PY scripts/run_challenge.py challenges/easy/1_vector_add \
  --language sycl --accelerator B70 --action run
```

From another machine, point the client at the host:

```bash
SERVICE_URL=http://<xpu-host>:8080 LEETGPU_API_KEY=change-me \
  python scripts/run_challenge.py ...
```

Endpoints: `/healthz`, `/api/v1/accelerators`, `/api/v1/challenges`,
`/api/v1/challenges/{id}`, `WS /api/v1/ws/submit`. The service rejects a
`challengeCode` that differs from the repo's `challenge.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `icpx not found` | `--icpx /opt/intel/oneapi/compiler/2026.1/bin/icpx` or `source /opt/intel/oneapi/setvars.sh` |
| `worker died with SIGSEGV` | kernel bug: null USM pointer, or two modules with the same kernel name |
| `functional: 0/N passed` | missing `q.wait()`, race on an out-of-order queue, or wrong index math |
| `status: timeout` | raise `--timeout`, or reduce the test size |
| `spread_pct` high | add `--inner-repeats 10` for small kernels |

## Challenges with SYCL starters today

`easy/1_vector_add`, `easy/2_matrix_multiplication`, `easy/3_matrix_transpose`,
`medium/4_reduction`, `medium/5_softmax`.
