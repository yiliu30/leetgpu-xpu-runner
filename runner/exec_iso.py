"""Run one grading job in an isolated subprocess."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def _result_stub(
    job: Dict[str, Any], status: str, stage: str, message: str, logs: str = ""
) -> Dict[str, Any]:
    return {
        "status": status,
        "challenge": job.get("challenge", ""),
        "language": "sycl",
        "device": job.get("device"),
        "compile_ms": 0.0,
        "wall_ms": 0.0,
        "functional": {"passed": 0, "total": 0, "cases": []},
        "performance": None,
        "error": {"stage": stage, "message": message},
        "logs": logs[-4000:],
    }


def run_isolated(job: Dict[str, Any], hard_timeout: float) -> Dict[str, Any]:
    """Spawn `runner.worker` for `job` and return its result JSON.

    A crash, timeout, or missing result is converted into a terminal status so the
    caller never sees an exception from a bad submission.
    """
    result_path = Path(job["result"])
    job_path = result_path.with_suffix(".job.json")
    result_path.parent.mkdir(parents=True, exist_ok=True)
    job_path.write_text(json.dumps(job))

    env = dict(os.environ)
    env["ZE_AFFINITY_MASK"] = str(job["device"])
    env["PYTHONPATH"] = os.pathsep.join([str(PACKAGE_ROOT), env.get("PYTHONPATH", "")]).rstrip(
        os.pathsep
    )
    command = [sys.executable, "-m", "runner.worker", "--job", str(job_path)]
    proc = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    try:
        output, _ = proc.communicate(timeout=hard_timeout)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        output, _ = proc.communicate()
        return _result_stub(
            job, "timeout", "harness", f"worker killed after {hard_timeout:.0f}s", output or ""
        )

    if result_path.is_file():
        result = json.loads(result_path.read_text())
        if proc.returncode != 0:
            result["logs"] = ((result.get("logs") or "") + "\n" + (output or ""))[-4000:]
        return result

    if proc.returncode < 0:
        try:
            name = signal.Signals(-proc.returncode).name
        except ValueError:
            name = f"signal {-proc.returncode}"
        message = f"worker died with {name}"
    else:
        message = f"worker exited with {proc.returncode} and produced no result"
    return _result_stub(job, "error", "harness", message, output or "")


def summarize(result: Dict[str, Any]) -> str:
    lines = [
        f"status: {result['status']}",
        f"challenge: {result['challenge']}",
        f"device: {result['device']}",
        f"compile: {result['compile_ms']:.0f} ms",
    ]
    functional = result.get("functional") or {}
    if functional:
        lines.append(f"functional: {functional['passed']}/{functional['total']} passed")
    performance = result.get("performance")
    if performance:
        lines.append(
            "perf: median {:.3f} ms (min {:.3f}, max {:.3f}, spread {:.1f}%), "
            "{:.1f} MB @ {:.1f} GB/s".format(
                performance["median_ms"],
                performance["min_ms"],
                performance["max_ms"],
                performance["spread_pct"],
                performance["io_bytes"] / 1e6,
                performance["io_gbps"],
            )
        )
    if performance and performance["median_ms"] < 0.2 and performance["inner_repeats"] == 1:
        lines.append("hint: kernel is under 0.2 ms; use --inner-repeats 10 for a stable median")
    error = result.get("error")
    if error:
        lines.append(f"error: [{error.get('stage')}] {error.get('message', '')[:800]}")
    return "\n".join(lines)
