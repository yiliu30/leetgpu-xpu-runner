# leetgpu-xpu-runner

Local grader for LeetGPU challenges on Intel XPU (SYCL / oneAPI DPC++).

It loads a challenge from the `leetgpu-challenges` repo, uses its `challenge.py`
as the correctness oracle, compiles a SYCL submission with `icpx -fsycl`, runs the
functional tests on a torch-XPU device, and benchmarks with the platform's
methodology (2 warmups, 15 timed runs, fresh inputs, L2 flush, median).

## Requirements

- Intel GPU visible through Level Zero (`sycl-ls`, `lsgpu`, or `xpu-smi discovery`)
- oneAPI DPC++ compiler (`icpx` with `-fsycl`)
- PyTorch with XPU support
- A checkout of the challenge repo (`leetgpu-challenges` or the XPU fork)

## Usage

```bash
python -m tools.grade easy/1_vector_add \
  --solution solution.cpp \
  --repo ~/leetgpu-challenges-xpu
```

```
status: success
challenge: easy/1_vector_add
device: 0
compile: 1669 ms
functional: 13/13 passed
perf: median 0.650 ms (min 0.649, max 0.660, spread 1.7%), 300.0 MB @ 461.5 GB/s
```

Challenges can be selected by id (`1`), slug (`vector-add`), relative path
(`easy/1_vector_add`), or directory.

Useful flags:

| Flag | Meaning |
|---|---|
| `--device N` | pin to Level Zero device N (sets `ZE_AFFINITY_MASK`) |
| `--runs N` | timed runs (default 15) |
| `--warmups N` | warmup runs (default 2) |
| `--inner-repeats N` | solve calls per timed sample, divided out (default 1; use 10-100 for sub-0.1 ms kernels) |
| `--timeout S` | wall-clock limit (default 40 s, matching the platform) |
| `--json PATH` | write the full result JSON |
| `--keep-build` | keep the compiled `.so` and job file |
| `--in-process` | skip subprocess isolation (debugging only) |

## Result schema

```json
{
  "status": "success | error | timeout | oom | interrupted",
  "challenge": "easy/1_vector_add",
  "language": "sycl",
  "device": 0,
  "compile_ms": 1669.0,
  "wall_ms": 4200.0,
  "functional": {"passed": 13, "total": 13, "cases": [...]},
  "performance": {"median_ms": 0.65, "min_ms": 0.649, "max_ms": 0.66,
                  "p25_ms": 0.649, "p75_ms": 0.651, "spread_pct": 1.7,
                  "runs": 15, "inner_repeats": 1,
                  "io_bytes": 300000000, "io_gbps": 461.5},
  "error": null,
  "logs": ""
}
```

`spread_pct` is the interquartile spread over the median. Each submission runs in
its own subprocess, so a crashing kernel reports `error` with the signal name
instead of killing the grader.

## Service

`python -m service` exposes the grader over the API the LeetGPU client already speaks,
so `scripts/run_challenge.py` works unmodified.

```bash
LEETGPU_CHALLENGES=~/leetgpu-challenges-xpu \
XPU_RUNNER_API_KEY=change-me \
XPU_RUNNER_DEVICES=0-3 \
python -m service
```

| Endpoint | Purpose |
|---|---|
| `GET /healthz` | status, challenge count, devices |
| `GET /api/v1/accelerators` | advertised accelerators |
| `GET /api/v1/challenges` | challenge list (id, title, spec, difficulty, tier) |
| `GET /api/v1/challenges/{id}` | detail incl. `starterCode` per language |
| `WS /api/v1/ws/submit` | grading requests, same payload as LeetGPU |

Environment:

| Variable | Default | Meaning |
|---|---|---|
| `LEETGPU_CHALLENGES` | auto-detect | challenge repo root |
| `XPU_RUNNER_API_KEY` | `LEETGPU_API_KEY` | required token in the submit payload |
| `XPU_RUNNER_DEVICES` | `0` | `0-3` or `0,2,5`; one job per device at a time |
| `XPU_RUNNER_RUNS` / `XPU_RUNNER_WARMUPS` | `15` / `2` | timing methodology |
| `XPU_RUNNER_INNER_REPEATS` | `1` | solve calls per timed sample |
| `XPU_RUNNER_TIMEOUT` | `40` | wall-clock budget per submission |
| `XPU_RUNNER_HOST` / `XPU_RUNNER_PORT` | `0.0.0.0` / `8080` | bind address |
| `XPU_RUNNER_PYTHON` | `python3` | interpreter with torch-XPU, fastapi, uvicorn |

The service rejects submissions whose `challengeCode` differs from the first-party
`challenge.py`, so a client cannot grade itself with a weakened reference.

Client:

```bash
mkdir -p challenges/easy/1_vector_add/solution
cp solution.cpp challenges/easy/1_vector_add/solution/solution.cpp
SERVICE_URL=http://localhost:8080 LEETGPU_API_KEY=change-me \
  python scripts/run_challenge.py challenges/easy/1_vector_add \
  --language sycl --accelerator B70 --action run
```


## Web UI

The service also serves a self-contained UI at `http://<host>:8080/` (no CDN, no build step).

- Challenge list with search and difficulty filters, 147 challenges
- Problem view rendered from each challenge's `challenge.html`
- SYCL editor seeded from `starter.sycl.cpp` (draft autosaved per challenge)
- Run button with live status, functional case list, median/min/max/spread and GB/s
- Per-run history in `localStorage`
- Deep links: `#/c/1` (problem) and `#/c/1/code` (editor)
- API token is entered via the header button (a dialog opens automatically on 401)
  and kept in `localStorage`; saving it retries the failed run
- `scripts/restart_service.sh` restarts the service; `XPU_RUNNER_API_KEY=` disables auth

Only challenges that ship a SYCL starter can be run; the rest are listed and marked
`no SYCL starter`.

## Deploying the UI

Two pieces with different constraints:

- **UI** (`web/`) is static and can be hosted anywhere, including GitHub Pages.
- **Verifier** (`python -m service`) must run on the XPU host, which may have no
  outbound internet access.

### GitHub Pages + SSH tunnel

Live UI: **https://yiliu30.github.io/leetgpu-xpu-runner/**

The `pages.yml` workflow publishes `web/` on every push to `main` (enable
Settings → Pages → Source: GitHub Actions once; this repo is already set up).

Users then connect the UI to the verifier through a tunnel:

```bash
ssh -L 8080:<xpu-host>:8080 <user>@<jump-host>
```

In the UI click **API token**, set **Service URL** to `http://localhost:8080`,
and paste the token. The setting is stored in `localStorage`.

Why the tunnel is required: the XPU host has no inbound internet, and a browser
blocks an HTTPS page from calling an HTTP backend (mixed content). `localhost`
is exempt, so the tunnel gives a working, private path without exposing the GPU.
For same-host use the Service URL can stay empty and the UI calls the service
that served it.

Alternative when the service is reachable on the internal network with TLS: set
`web/config.js` → `window.XPU_RUNNER_CONFIG.apiBase` to that URL before publishing.

### GitHub Actions self-hosted runner

Not usable when the XPU host has no outbound network: a self-hosted runner must
reach `github.com` over HTTPS. If outbound access is available, a
`workflow_dispatch` job could grade asynchronously, but remember that it executes
submitted C++ on the host, so restrict who can trigger it.

## Limitations

- SYCL submissions only. The reference implementation, test generation, and
  synchronization still use torch XPU.
- `solve` must complete its device work (use `sycl::queue::wait()`); the runner
  cannot synchronize an arbitrary user queue from Python.
- A plain `sycl::queue` can access torch-XPU USM pointers on the validated driver
  stack (Arc Pro B70, driver 1.15.38646), but SYCL does not guarantee cross-context
  USM. Keep this assumption in the backend, not in challenge code.
- `io_gbps` is bytes moved over time; it is a rough IO metric, not a compute
  metric (matmul is compute-bound and will look "slow" under it).

## Tests

```bash
PYTHONPATH=. python -m unittest discover -s tests
```

Unit tests cover signature parsing, argument marshalling, challenge resolution,
and compiler command construction. GPU integration runs are done on a B70 host.
