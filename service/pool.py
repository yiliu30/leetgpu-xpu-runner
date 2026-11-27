"""Per-device worker pool: one grading job per GPU at a time."""

import shutil
import tempfile
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Tuple

from runner.exec_iso import run_isolated


class DevicePool:
    def __init__(self, devices: List[int], repo: Path, defaults: Dict[str, Any]):
        if not devices:
            raise ValueError("no devices configured")
        self.devices = list(devices)
        self.repo = Path(repo)
        self.defaults = dict(defaults)
        self._executors = {
            device: ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"xpu-{device}")
            for device in self.devices
        }
        self._next = 0
        self._lock = threading.Lock()

    def _pick(self) -> int:
        with self._lock:
            device = self.devices[self._next % len(self.devices)]
            self._next += 1
            return device

    def submit(self, request: Dict[str, Any], on_start=None) -> Tuple[int, Future]:
        device = self._pick()

        def task() -> Dict[str, Any]:
            if on_start is not None:
                on_start(device)
            return self._run(device, request)

        return device, self._executors[device].submit(task)

    def _run(self, device: int, request: Dict[str, Any]) -> Dict[str, Any]:
        workdir = Path(tempfile.mkdtemp(prefix=f"xpu-job-{device}-"))
        try:
            solution = workdir / "solution.cpp"
            solution.write_text(request["content"])
            settings = {**self.defaults, **request.get("overrides", {})}
            job = {
                "repo": str(self.repo),
                "challenge": request["challenge"],
                "solution": str(solution),
                "device": device,
                "runs": settings["runs"],
                "warmups": settings["warmups"],
                "inner_repeats": settings["inner_repeats"],
                "timeout": settings["timeout"],
                "atol": settings.get("atol"),
                "rtol": settings.get("rtol"),
                "icpx": settings.get("icpx"),
                "build_dir": str(workdir / "build"),
                "result": str(workdir / "result.json"),
            }
            return run_isolated(job, hard_timeout=settings["timeout"] + 30.0)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    def stats(self) -> Dict[str, int]:
        return {"devices": len(self.devices)}

    def shutdown(self) -> None:
        for executor in self._executors.values():
            executor.shutdown(wait=False, cancel_futures=True)
