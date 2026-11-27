"""Correctness verification against a challenge's reference implementation."""

import traceback
from typing import Any, Dict, List

from .abi import ArgSpec, output_names, raw_args
from .challenge import LoadedChallenge, describe_case, materialize_case
from .result import CaseResult


def _clone_inputs(case: Dict[str, Any]) -> Dict[str, Any]:
    import torch

    cloned = {}
    for name, value in case.items():
        cloned[name] = value.detach().clone() if isinstance(value, torch.Tensor) else value
    return cloned


def _errors(got, want):
    got = got.float()
    want = want.float()
    abs_error = (got - want).abs()
    rel_error = abs_error / want.abs().clamp_min(1e-12)
    return float(abs_error.max().item()) if abs_error.numel() else 0.0, (
        float(rel_error.max().item()) if rel_error.numel() else 0.0
    )


def run_case(
    challenge: LoadedChallenge,
    specs: List[ArgSpec],
    fn,
    case: Dict[str, Any],
    device: str,
    atol: float,
    rtol: float,
    index: int = 0,
) -> CaseResult:
    import torch

    kwargs = materialize_case(case, device)
    reference_kwargs = _clone_inputs(kwargs)

    try:
        challenge.instance.reference_impl(**reference_kwargs)
    except Exception as exc:  # noqa: BLE001 - surface challenge bugs verbatim
        return CaseResult(
            index=index,
            passed=False,
            detail=f"reference_impl failed: {exc}\n{traceback.format_exc(limit=3)}",
        )

    try:
        fn(*raw_args(specs, kwargs))
    except Exception as exc:  # noqa: BLE001
        return CaseResult(index=index, passed=False, detail=f"solve raised: {exc}")
    torch.xpu.synchronize()

    worst_abs = 0.0
    worst_rel = 0.0
    for name in output_names(specs):
        got = kwargs[name]
        want = reference_kwargs[name]
        if not isinstance(got, torch.Tensor) or not isinstance(want, torch.Tensor):
            return CaseResult(
                index=index,
                passed=False,
                failed_param=name,
                detail=f"{name}: output is not a tensor",
            )
        abs_error, rel_error = _errors(got, want)
        worst_abs = max(worst_abs, abs_error)
        worst_rel = max(worst_rel, rel_error)
        if not torch.allclose(got, want, atol=atol, rtol=rtol):
            return CaseResult(
                index=index,
                passed=False,
                worst_abs_error=worst_abs,
                worst_rel_error=worst_rel,
                failed_param=name,
                detail=f"{name}: mismatch ({describe_case(case)})",
            )
    return CaseResult(
        index=index,
        passed=True,
        worst_abs_error=worst_abs,
        worst_rel_error=worst_rel,
    )
