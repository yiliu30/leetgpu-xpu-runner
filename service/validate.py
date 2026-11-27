"""Submission validation shared by the WebSocket handler and unit tests."""

from typing import Any, Dict, Optional, Tuple

from .registry import Registry

MAX_FILE_BYTES = 2 * 1024 * 1024
SUPPORTED_LANGUAGES = {"sycl"}
SUPPORTED_ACCELERATORS = {"B70", "XPU", "XPU-B70", "AUTO"}
SOURCE_SUFFIXES = (".cpp", ".cc", ".cxx")


def error(output: str) -> Dict[str, str]:
    return {"status": "error", "output": output}


def validate_submission(
    payload: Dict[str, Any], api_key: Optional[str], registry: Registry
) -> Tuple[Dict[str, Any], Optional[Dict[str, str]]]:
    """Return (request, None) or ({}, error payload)."""
    if api_key and payload.get("token") != api_key:
        return {}, error("invalid API token")
    submission = payload.get("submission") or {}
    language = str(submission.get("language", ""))
    if language not in SUPPORTED_LANGUAGES:
        return {}, error(
            f"unsupported language {language!r}: this runner supports {sorted(SUPPORTED_LANGUAGES)}"
        )
    accelerator = str(submission.get("accelerator", "B70")).upper()
    if accelerator not in SUPPORTED_ACCELERATORS:
        return {}, error(
            f"unsupported accelerator {accelerator!r}: "
            f"choose one of {sorted(SUPPORTED_ACCELERATORS)}"
        )
    files = submission.get("files") or []
    source = next(
        (item for item in files if str(item.get("name", "")).lower().endswith(SOURCE_SUFFIXES)),
        None,
    )
    if source is None:
        return {}, error("no .cpp file in submission.files")
    content = str(source.get("content", ""))
    if len(content.encode()) > MAX_FILE_BYTES:
        return {}, error("submission file exceeds 2 MB")
    challenge_code = str(submission.get("challengeCode", ""))
    if not challenge_code:
        return {}, error("submission.challengeCode is required")
    entry = registry.find_by_source(challenge_code)
    if entry is None:
        return {}, error("challengeCode does not match any first-party challenge")
    if not registry.matches(entry, challenge_code):
        return {}, error(
            "submitted challengeCode differs from the first-party challenge; "
            "submissions are graded against the repo's reference implementation"
        )
    if "sycl" not in entry.starters:
        return {}, error(f"{entry.key} has no SYCL starter yet")
    return {
        "challenge": entry.key,
        "content": content,
        "file_name": source.get("name", "solution.cpp"),
        "action": str(payload.get("action", "run")),
    }, None
