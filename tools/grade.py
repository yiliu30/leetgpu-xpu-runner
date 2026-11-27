"""CLI: grade a SYCL submission against a LeetGPU challenge on Intel XPU."""

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

from runner import challenge as challenge_mod
from runner.exec_iso import run_isolated as run_iso
from runner.exec_iso import summarize

PACKAGE_ROOT = Path(__file__).resolve().parent.parent


def default_repo() -> Optional[Path]:
    env = os.environ.get("LEETGPU_CHALLENGES")
    if env:
        return Path(env).expanduser()
    for name in ("leetgpu-challenges-xpu", "leetgpu-challenges"):
        for base in (PACKAGE_ROOT, PACKAGE_ROOT.parent, Path.home()):
            candidate = base / name
            if (candidate / "challenges").is_dir():
                return candidate
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("challenge", help="challenge id, slug, relative path, or directory")
    parser.add_argument("--solution", required=True, type=Path, help="SYCL .cpp submission")
    parser.add_argument("--repo", type=Path, default=default_repo(), help="challenge repo root")
    parser.add_argument("--device", type=int, default=0, help="Level Zero device index")
    parser.add_argument("--runs", type=int, default=15, help="timed runs (default 15)")
    parser.add_argument("--warmups", type=int, default=2, help="warmup runs (default 2)")
    parser.add_argument(
        "--inner-repeats",
        type=int,
        default=1,
        help="solve calls per timed sample, divided out (default 1)",
    )
    parser.add_argument("--timeout", type=float, default=40.0, help="wall-clock limit (s)")
    parser.add_argument("--atol", type=float, default=None, help="override absolute tolerance")
    parser.add_argument("--rtol", type=float, default=None, help="override relative tolerance")
    parser.add_argument("--icpx", default=None, help="path to icpx")
    parser.add_argument("--json", type=Path, default=None, help="write the result JSON here")
    parser.add_argument("--in-process", action="store_true", help="skip subprocess isolation")
    parser.add_argument("--keep-build", action="store_true", help="keep the compiled library")
    parser.add_argument("--quiet", action="store_true", help="only print the final JSON")
    return parser.parse_args()


def build_job(
    args: argparse.Namespace, repo: Path, build_dir: Path, result_path: Path
) -> Dict[str, Any]:
    directory = challenge_mod.resolve(repo, args.challenge)
    return {
        "repo": str(repo),
        "challenge": f"{directory.parent.name}/{directory.name}",
        "solution": str(args.solution.resolve()),
        "device": args.device,
        "runs": args.runs,
        "warmups": args.warmups,
        "inner_repeats": args.inner_repeats,
        "timeout": args.timeout,
        "atol": args.atol,
        "rtol": args.rtol,
        "icpx": args.icpx,
        "build_dir": str(build_dir),
        "result": str(result_path),
    }


def main() -> int:
    args = parse_args()
    repo = args.repo or default_repo()
    if repo is None or not (Path(repo) / "challenges").is_dir():
        print(
            "error: challenge repo not found; pass --repo or set LEETGPU_CHALLENGES",
            file=sys.stderr,
        )
        return 2
    repo = Path(repo).expanduser().resolve()
    if not args.solution.is_file():
        print(f"error: solution not found: {args.solution}", file=sys.stderr)
        return 2

    workdir = Path(tempfile.mkdtemp(prefix="xpu-grade-"))
    result_path = workdir / "result.json"
    build_dir = workdir / "build"
    try:
        job = build_job(args, repo, build_dir, result_path)
        if args.in_process:
            from runner.worker import run_job

            result = run_job(job).to_dict()
        else:
            result = run_iso(job, args.timeout + 30.0)
    finally:
        if not args.keep_build:
            shutil.rmtree(workdir, ignore_errors=True)

    if args.json:
        args.json.write_text(json.dumps(result, indent=2))
    if args.quiet:
        print(json.dumps(result))
    else:
        print(summarize(result))
        if result.get("logs"):
            print("\nworker log:\n" + result["logs"])
        if args.keep_build:
            print(f"\nbuild kept in {build_dir}")
    return 0 if result["status"] == "success" else 1


if __name__ == "__main__":
    raise SystemExit(main())
