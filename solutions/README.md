# SYCL reference kernels

Working SYCL / oneAPI DPC++ solutions for LeetGPU challenges, used to validate the
grader end to end and to serve as reference kernels for XPU users.

Layout mirrors the challenge repo:

```
solutions/<difficulty>/<id>_<name>/solution.cpp
```

Each file compiles with `icpx -fsycl -O2 -fPIC -shared` and exports
`extern "C" solve(...)` with the same ABI as the challenge's starter.

Validate one:

```bash
python -m tools.grade easy/1_vector_add \
  --solution solutions/easy/1_vector_add/solution.cpp \
  --repo /path/to/leetgpu-challenges
```

Add `--json out.json` for the full result. Correctness is checked against the
challenge's own `reference_impl` and functional tests; the benchmark uses the
platform methodology (warmups, L2 flush, fresh inputs, median).
