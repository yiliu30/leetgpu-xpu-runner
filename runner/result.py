"""Result schema shared by the worker and the CLI."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

STATUSES = ("success", "error", "timeout", "oom", "interrupted")


@dataclass
class CaseResult:
    index: int
    passed: bool
    worst_abs_error: float = 0.0
    worst_rel_error: float = 0.0
    failed_param: Optional[str] = None
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "passed": self.passed,
            "worst_abs_error": self.worst_abs_error,
            "worst_rel_error": self.worst_rel_error,
            "failed_param": self.failed_param,
            "detail": self.detail,
        }


@dataclass
class BenchResult:
    median_ms: float
    min_ms: float
    max_ms: float
    p25_ms: float
    p75_ms: float
    spread_pct: float
    runs: int
    inner_repeats: int
    io_bytes: int
    io_gbps: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "median_ms": self.median_ms,
            "min_ms": self.min_ms,
            "max_ms": self.max_ms,
            "p25_ms": self.p25_ms,
            "p75_ms": self.p75_ms,
            "spread_pct": self.spread_pct,
            "runs": self.runs,
            "inner_repeats": self.inner_repeats,
            "io_bytes": self.io_bytes,
            "io_gbps": self.io_gbps,
        }


@dataclass
class GradeResult:
    status: str
    challenge: str = ""
    language: str = "sycl"
    device: Optional[int] = None
    compile_ms: float = 0.0
    wall_ms: float = 0.0
    functional: List[CaseResult] = field(default_factory=list)
    performance: Optional[BenchResult] = None
    error: Optional[Dict[str, str]] = None
    logs: str = ""

    def to_dict(self) -> Dict[str, Any]:
        passed = sum(case.passed for case in self.functional)
        return {
            "status": self.status,
            "challenge": self.challenge,
            "language": self.language,
            "device": self.device,
            "compile_ms": round(self.compile_ms, 3),
            "wall_ms": round(self.wall_ms, 3),
            "functional": {
                "passed": passed,
                "total": len(self.functional),
                "cases": [case.to_dict() for case in self.functional],
            },
            "performance": self.performance.to_dict() if self.performance else None,
            "error": self.error,
            "logs": self.logs,
        }

    def write(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=2))
