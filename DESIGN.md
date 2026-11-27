# LeetGPU XPU Runner — Design and Build Plan

Goal: run, verify, and time LeetGPU challenge submissions on Intel XPU, using the
challenge definitions in `leetgpu-challenges` as the source of truth, behind an API
that the existing `scripts/run_challenge.py` client can already speak.

## Non-goals (v1)

- No leaderboard, accounts, or public web UI.
- No CUDA backend on XPU (the existing repo already covers NVIDIA).
- v1 is SYCL-only: no PyTorch/Triton submission backends, no service layer yet.
- No re-implementation of challenge content; `challenge.py` stays the oracle.

## Architecture

```
 client (scripts/run_challenge.py --language sycl --accelerator B70)
        │  WS /api/v1/ws/submit            HTTP /api/v1/challenges[/{id}]
        ▼
 ┌────────────────────────── service ──────────────────────────┐
 │ auth (API key) · job queue · per-GPU workers · limits        │
 └───────────────┬──────────────────────────────────────────────┘
                 ▼  one subprocess per submission (crash isolation)
 ┌────────────────────────── grader core ──────────────────────┐
 │ challenge loader → test materializer → ABI binder            │
 │ → sycl backend (icpx -fsycl) → correctness compare → bench   │
 └───────────────────────────────┬─────────────────────────────┘
                                 ▼
                        torch-xpu (tests + oracle + timing)
```

Layout:

```
leetgpu-xpu-runner/
├── runner/            # importable grader core (no service deps)
│   ├── challenge.py   # load challenge.py, materialize RandTensor/FullTensor/OutTensor
│   ├── abi.py         # get_solve_signature → ctypes argtypes, tensor/data_ptr marshalling
│   ├── verify.py      # atol/rtol comparison, per-case reporting
│   ├── bench.py       # warmups, L2 flush, median-of-15, wall-clock budget
│   ├── exec.py        # subprocess isolation, timeouts, rlimits, result JSON
│   ├── devices.py     # ZE_AFFINITY_MASK pinning, discovery, memory headroom
│   └── backends/
│       ├── sycl.py    # icpx -fsycl -shared → dlopen → extern "C" solve
│       ├── pytorch.py # python solve() on torch.xpu
│       └── triton.py  # triton XPU backend
├── service/           # FastAPI: HTTP + WS, queue, workers
├── tools/grade.py     # one-shot CLI grader (same core, no service)
├── tests/             # unit + integration (integration gated on GPU presence)
├── docker/            # icpx 2026.x + torch-xpu + runner
└── DESIGN.md
```

## Submission contract (compatible with the existing client)

`scripts/run_challenge.py` already sends:

```json
{"action": "run|submit", "token": "<api key>",
 "submission": {"files": [{"name": "...", "content": "..."}],
                "language": "sycl", "accelerator": "B70",
                "mode": "accelerated", "public": false,
                "challengeCode": "<challenge.py source>"}}
```

and reads `{"status": ..., "output": ...}` until one of
`success | error | timeout | oom | interrupted`. We implement exactly those.

HTTP mirrors: `GET /api/v1/challenges`, `GET /api/v1/challenges/{id}` (numeric,
matching `api.leetgpu.com`), plus `GET /api/v1/accelerators` and `/healthz`.

The submitted `challengeCode` is the oracle, but we cross-check it against the
first-party `challenge.py` in the challenge repo and refuse mismatches (otherwise a
submitter could grade themselves with a weakened reference).

## Execution model

1. Resolve challenge by id/slug from the local challenge repo.
2. Materialize example/functional/performance tests with `device=<xpu device>`.
3. Load submission file, compile if needed (`icpx -fsycl -O2 -shared -fPIC`),
   under a wall-clock budget (~40 s, matching the platform).
4. Bind `solve` via ctypes from `get_solve_signature()`; tensors become
   `data_ptr()` pointers, which are USM device allocations on torch XPU.
5. Run functional cases; compare outputs against `reference_impl` with
   `atol`/`rtol`. First failure returns `error` with case name and max error.
6. Run perf test: 2 warmups, then 15 timed runs with fresh inputs and an L2 flush
   before each; report the median. Compile time is excluded from kernel timing.
7. Emit JSON: status, per-case results, median ms, derived GFLOPS/GB/s,
   compile ms, wall ms.

Each submission runs in its own subprocess pinned with `ZE_AFFINITY_MASK=<gpu>`,
so a SYCL segfault (kernel-name collision, bad USM, driver abort) cannot take down
the worker. Timeouts use process-group kill; `oom` is detected from device memory
headroom before launch and from Level Zero error codes after.

## Timing details

- Timing unit: `torch.xpu.Event` around `solve` + `torch.xpu.synchronize()`,
  matching the platform's "includes launch overhead, excludes compilation".
- L2 flush: last-level cache is 24 MB on Arc Pro B70. Before each timed run, write a
  buffer of `max(64 MB, 2 × last_level_cache_size)` and synchronize. Confirm the
  flush is effective by measuring a small working set (should stay flat across
  repeats) and a >L2 working set (should not).
- Fresh inputs per timed run, matching the platform's stated methodology.
- Record min/median/max and relative spread; warn above 5% spread.

## Backends

- `sycl` (primary): compile `starter.cpp`/`solution.cpp` with `icpx -fsycl`,
  `dlopen`, call `extern "C" solve`. Kernel naming and in-order queue rules are
  documented in the challenge repo's CLAUDE.md and must be followed by solutions.
- `pytorch` and `triton` submissions: deferred to v2; torch-xpu is still used for
  test generation, the reference implementation, and synchronization.
- Optional later: `sycl-graph`, `onednn`, and a CUDA-compat path.

## Phases and gates

| Phase | Deliverable | Gate |
|---|---|---|
| P0 | challenge loader, test materializer, ABI binder, SYCL backend, CLI grader | **done** — 5 SYCL challenges, 57/57 cases |
| P1 | subprocess isolation, failure taxonomy | **done** — crash and wrong-answer both return `error` |
| P2 | benchmark harness (warmup, L2 flush, median-15, inner repeats) | **done** — IQR spread <2% with inner repeats; compile excluded |
| P3 | service: HTTP + WS compatible with `run_challenge.py`, queue, per-GPU workers | **done** — unmodified `scripts/run_challenge.py --language sycl --accelerator B70` succeeds against the local service |
| P3.5 | web UI (challenge list, editor, live results) | **done** — headless-browser run reaches `success` with 13/13 cases |
| P4 | container image, limits, multi-GPU scheduling | all 16 GPUs run concurrently; timeout and OOM paths tested |
| P5 | coverage: SYCL starters for more challenges, docs, CI | starter coverage matrix published; CI green on a single GPU runner |

## Risks and mitigations

1. **Cross-context USM** (torch XPU allocations used by a plain `sycl::queue`) works
   empirically on B70 but is not guaranteed by the SYCL spec. Mitigation: keep the
   interop shim isolated in `backends/sycl.py`, add a Level Zero context-sharing path
   later, and fail loudly on USM allocation errors.
2. **Kernel-name collisions** across modules segfault the process (observed).
   Mitigation: subprocess per submission; named functors documented in CLAUDE.md.
3. **Out-of-order default queue** silently races multi-kernel solutions (observed
   with softmax). Mitigation: starters use in-order queues; the runner cannot detect
   this, so document it.
4. **L2 flush fidelity**. Mitigation: empirical calibration test in P2.
5. **Multi-tenant GPUs**: 16 devices shared with other users. Mitigation: pre-flight
   memory/utilisation check, skip busy devices, pin with `ZE_AFFINITY_MASK`.
6. **Compile cost**: IGC JIT of SPIR-V can dominate wall time. Mitigation: measure
   compile ms separately; wall-clock budget; persistent cache where safe.

## Environment (validated)

- Validated on Intel Arc Pro B70 (32 GB, 24 MB last-level cache, 256 Xe cores,
  driver 1.15.38646), oneAPI DPC++ 2026.1, torch 2.13.0+xpu, 16 devices.
- Compiler: any `icpx` with `-fsycl` (auto-discovered under `/opt/intel/oneapi`,
  or pass `--icpx` / `LEETGPU_XPU_ICPX`).
- Runtime: a Python environment with torch XPU support.
- Reference SYCL kernels used for validation live outside this repo.
