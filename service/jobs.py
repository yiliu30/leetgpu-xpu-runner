"""In-memory submission jobs for the web UI and REST clients."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .pool import DevicePool

TERMINAL = {"success", "error", "timeout", "oom", "interrupted"}


@dataclass
class Job:
    id: str
    challenge: str
    language: str
    device: int
    status: str = "queued"
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    result: Optional[Dict[str, Any]] = None

    def to_dict(self, include_result: bool = True) -> Dict[str, Any]:
        payload = {
            "id": self.id,
            "challenge": self.challenge,
            "language": self.language,
            "device": self.device,
            "status": self.status,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }
        if include_result and self.result is not None:
            payload["result"] = self.result
        return payload


class JobStore:
    def __init__(self, pool: DevicePool, max_jobs: int = 200):
        self.pool = pool
        self.max_jobs = max_jobs
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._lock = threading.Lock()

    def submit(self, request: Dict[str, Any]) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            challenge=request["challenge"],
            language=request.get("language", "sycl"),
            device=-1,
        )
        with self._lock:
            self._jobs[job.id] = job
            self._order.append(job.id)
            self._trim_locked()

        def on_start(device: int) -> None:
            with self._lock:
                job.device = device
                job.status = "running"
                job.updated_at = time.time()

        def on_done(future) -> None:
            try:
                result = future.result()
                status = result.get("status", "error")
            except Exception as exc:  # noqa: BLE001 - surface as a failed job
                result = {
                    "status": "error",
                    "challenge": job.challenge,
                    "language": job.language,
                    "device": job.device,
                    "compile_ms": 0.0,
                    "wall_ms": 0.0,
                    "functional": {"passed": 0, "total": 0, "cases": []},
                    "performance": None,
                    "error": {"stage": "harness", "message": str(exc)},
                    "logs": "",
                }
                status = "error"
            with self._lock:
                job.status = status if status in TERMINAL else "error"
                job.result = result
                job.updated_at = time.time()

        _, future = self.pool.submit(request, on_start=on_start)
        future.add_done_callback(on_done)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> List[Job]:
        with self._lock:
            ids = self._order[-limit:][::-1]
            return [self._jobs[job_id] for job_id in ids if job_id in self._jobs]

    def _trim_locked(self) -> None:
        while len(self._order) > self.max_jobs:
            oldest = self._order.pop(0)
            self._jobs.pop(oldest, None)
