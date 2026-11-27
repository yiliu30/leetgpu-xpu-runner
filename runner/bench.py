"""Timing harness mirroring the platform methodology."""

import statistics
import time
from typing import Any, Callable, Dict, List

from .abi import ArgSpec, raw_args
from .challenge import materialize_case
from .result import BenchResult

MIN_FLUSH_BYTES = 64 * 1024 * 1024


def tensor_bytes(case: Dict[str, Any]) -> int:
    import torch

    total = 0
    for value in case.values():
        if isinstance(value, torch.Tensor):
            total += value.numel() * value.element_size()
    return total


def flush_bytes(device) -> int:
    import torch

    try:
        cache = torch.xpu.get_device_properties(device).last_level_cache_size
    except Exception:  # noqa: BLE001 - property name varies across torch builds
        cache = 24 * 1024 * 1024
    return max(MIN_FLUSH_BYTES, 2 * int(cache))


def flush_l2(scratch) -> None:
    scratch.fill_(0)
    import torch

    torch.xpu.synchronize()


def run_bench(
    specs: List[ArgSpec],
    fn: Callable[..., Any],
    case_factory: Callable[[], Dict[str, Any]],
    device: str,
    runs: int = 15,
    warmups: int = 2,
    inner_repeats: int = 1,
) -> BenchResult:
    import torch

    call = lambda kwargs: fn(*raw_args(specs, kwargs))  # noqa: E731

    for _ in range(warmups):
        call(materialize_case(case_factory(), device))
    torch.xpu.synchronize()

    scratch = torch.empty(flush_bytes(device), dtype=torch.uint8, device=device)
    times: List[float] = []
    io_bytes = 0
    for _ in range(runs):
        case = materialize_case(case_factory(), device)
        io_bytes = max(io_bytes, tensor_bytes(case))
        flush_l2(scratch)
        start = time.perf_counter()
        for _ in range(inner_repeats):
            call(case)
        torch.xpu.synchronize()
        times.append((time.perf_counter() - start) * 1000.0 / inner_repeats)

    median = statistics.median(times)
    ordered = sorted(times)
    p25 = statistics.quantiles(ordered, n=4)[0] if len(ordered) > 1 else ordered[0]
    p75 = statistics.quantiles(ordered, n=4)[2] if len(ordered) > 1 else ordered[0]
    return BenchResult(
        median_ms=median,
        min_ms=ordered[0],
        max_ms=ordered[-1],
        p25_ms=p25,
        p75_ms=p75,
        spread_pct=(p75 - p25) / median * 100.0 if median else 0.0,
        runs=runs,
        inner_repeats=inner_repeats,
        io_bytes=io_bytes,
        io_gbps=(io_bytes / (median / 1000.0) / 1e9) if median else 0.0,
    )
