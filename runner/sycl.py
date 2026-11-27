"""Compile and load SYCL submissions."""

import ctypes
import glob
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import List, Optional, Tuple

_CANDIDATE_GLOBS = [
    "/opt/intel/oneapi/compiler/*/bin/icpx",
    "/opt/intel/oneapi/compiler/latest/bin/icpx",
]

DEFAULT_FLAGS = ["-fsycl", "-O2", "-fPIC", "-shared"]


def find_icpx(explicit: Optional[str] = None) -> str:
    if explicit:
        return explicit
    env = os.environ.get("LEETGPU_XPU_ICPX")
    if env:
        return env
    found = shutil.which("icpx")
    if found:
        return found
    for pattern in _CANDIDATE_GLOBS:
        matches = sorted(glob.glob(pattern))
        if matches:
            return matches[-1]
    raise SystemExit(
        "icpx not found: pass --icpx, set LEETGPU_XPU_ICPX, or source oneAPI setvars.sh"
    )


def build_command(
    source: Path, output: Path, icpx: str, extra_flags: Optional[List[str]] = None
) -> List[str]:
    flags = list(extra_flags) if extra_flags else []
    return [icpx, *DEFAULT_FLAGS, *flags, "-o", str(output), str(source)]


def compile_submission(
    source: Path,
    output: Path,
    icpx: str,
    timeout: float = 180.0,
    extra_flags: Optional[List[str]] = None,
) -> Tuple[int, str, float]:
    command = build_command(source, output, icpx, extra_flags)
    start = time.perf_counter()
    try:
        proc = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        log = (proc.stdout or "") + (proc.stderr or "")
        return proc.returncode, log.strip(), elapsed_ms
    except subprocess.TimeoutExpired:
        return (
            124,
            f"compilation timed out after {timeout:.0f}s",
            (time.perf_counter() - start) * 1000.0,
        )


def load_library(path: Path):
    return ctypes.CDLL(str(path))
