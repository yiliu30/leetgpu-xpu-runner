"""Grade one submission. Runs inside a dedicated subprocess for crash isolation."""

import argparse
import json
import time
import traceback
from pathlib import Path
from typing import Any, Dict

from . import abi, bench
from . import challenge as challenge_mod
from . import sycl
from .result import GradeResult
from .verify import run_case


def _is_oom(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return "out of memory" in text or "memory allocation" in text or "outofmemory" in text


def run_job(job: Dict[str, Any]) -> GradeResult:
    import torch

    started = time.perf_counter()
    deadline = started + float(job.get("timeout", 40.0))
    result = GradeResult(
        status="error",
        challenge=job["challenge"],
        device=int(job["device"]),
    )

    try:
        directory = challenge_mod.resolve(Path(job["repo"]), job["challenge"])
        loaded = challenge_mod.load(directory, device="xpu")
        result.challenge = loaded.key
        specs = abi.parse_signature(loaded.instance)
        atol = float(job.get("atol") if job.get("atol") is not None else loaded.atol)
        rtol = float(job.get("rtol") if job.get("rtol") is not None else loaded.rtol)

        icpx = sycl.find_icpx(job.get("icpx"))
        build_dir = Path(job["build_dir"])
        build_dir.mkdir(parents=True, exist_ok=True)
        library = build_dir / "submission.so"
        remaining = max(1.0, deadline - time.perf_counter())
        rc, log, compile_ms = sycl.compile_submission(
            Path(job["solution"]), library, icpx, timeout=min(180.0, remaining)
        )
        result.compile_ms = compile_ms
        if rc != 0:
            result.error = {"stage": "compile", "message": log[-4000:] or f"icpx exited with {rc}"}
            return result

        fn = abi.bind(sycl.load_library(library), specs)

        cases = challenge_mod.functional_cases(loaded)
        for index, case in enumerate(cases):
            if time.perf_counter() > deadline:
                result.error = {"stage": "functional", "message": "wall-clock limit exceeded"}
                result.status = "timeout"
                return result
            case_result = run_case(loaded, specs, fn, case, "xpu", atol, rtol, index=index)
            result.functional.append(case_result)
            if not case_result.passed:
                result.error = {
                    "stage": "functional",
                    "case": str(index),
                    "param": case_result.failed_param or "",
                    "message": case_result.detail[-2000:],
                }
                return result

        if time.perf_counter() > deadline:
            result.error = {"stage": "performance", "message": "wall-clock limit exceeded"}
            result.status = "timeout"
            return result
        result.performance = bench.run_bench(
            specs,
            fn,
            lambda: challenge_mod.performance_case(loaded),
            "xpu",
            runs=int(job.get("runs", 15)),
            warmups=int(job.get("warmups", 2)),
            inner_repeats=int(job.get("inner_repeats", 1)),
        )
        result.status = "success"
    except torch.OutOfMemoryError as exc:
        result.status = "oom"
        result.error = {"stage": "run", "message": str(exc)}
    except SystemExit as exc:
        result.error = {"stage": "setup", "message": str(exc)}
    except BaseException as exc:  # noqa: BLE001 - report, never crash the parent
        result.status = "oom" if _is_oom(exc) else "error"
        result.error = {
            "stage": "run",
            "message": f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=6)}",
        }
    finally:
        result.wall_ms = (time.perf_counter() - started) * 1000.0
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Grade one LeetGPU XPU submission.")
    parser.add_argument("--job", required=True, type=Path)
    args = parser.parse_args()
    job = json.loads(args.job.read_text())
    result = run_job(job)
    result.write(Path(job["result"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
