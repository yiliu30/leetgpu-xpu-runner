"""FastAPI service: LeetGPU client protocol, REST submission API, and web UI."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from runner.exec_iso import summarize

from .jobs import JobStore
from .pool import DevicePool
from .registry import Registry
from .validate import MAX_FILE_BYTES, SUPPORTED_ACCELERATORS, SUPPORTED_LANGUAGES
from .validate import error as error_payload
from .validate import validate_submission

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


def _parse_devices(value: str) -> List[int]:
    devices: List[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            devices.extend(range(int(start), int(end) + 1))
        else:
            devices.append(int(part))
    return sorted(set(devices))


def _find_repo() -> Path:
    env = os.environ.get("LEETGPU_CHALLENGES")
    if env:
        return Path(env).expanduser().resolve()
    for name in ("leetgpu-challenges-xpu", "leetgpu-challenges"):
        for base in (Path.cwd(), Path.home()):
            candidate = base / name
            if (candidate / "challenges").is_dir():
                return candidate.resolve()
    raise SystemExit("challenge repo not found: set LEETGPU_CHALLENGES")


class State:
    def __init__(self) -> None:
        self.repo = _find_repo()
        self.registry = Registry(self.repo)
        self.api_key = os.environ.get("XPU_RUNNER_API_KEY") or os.environ.get("LEETGPU_API_KEY")
        self.devices = _parse_devices(os.environ.get("XPU_RUNNER_DEVICES", "0"))
        self.pool = DevicePool(
            self.devices,
            self.repo,
            defaults={
                "runs": int(os.environ.get("XPU_RUNNER_RUNS", "15")),
                "warmups": int(os.environ.get("XPU_RUNNER_WARMUPS", "2")),
                "inner_repeats": int(os.environ.get("XPU_RUNNER_INNER_REPEATS", "1")),
                "timeout": float(os.environ.get("XPU_RUNNER_TIMEOUT", "40")),
                "atol": None,
                "rtol": None,
                "icpx": os.environ.get("LEETGPU_XPU_ICPX"),
            },
        )
        self.jobs = JobStore(self.pool)


_state: Optional[State] = None


def state() -> State:
    global _state
    if _state is None:
        _state = State()
    return _state


app = FastAPI(title="LeetGPU XPU Runner", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def require_auth(
    authorization: Optional[str] = Header(default=None),
    x_api_key: Optional[str] = Header(default=None),
) -> None:
    current = state()
    if not current.api_key:
        return
    token = x_api_key
    if authorization and authorization.lower().startswith("bearer "):
        token = authorization[7:]
    if token != current.api_key:
        raise HTTPException(status_code=401, detail="invalid API token")


class SubmissionRequest(BaseModel):
    challenge: Union[int, str]
    source: str
    language: str = "sycl"
    accelerator: str = "B70"
    runs: Optional[int] = None
    warmups: Optional[int] = None
    inner_repeats: Optional[int] = None
    timeout: Optional[float] = None


@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    current = state()
    return {
        "status": "ok",
        "repo": str(current.repo),
        "challenges": len(current.registry.by_id),
        "devices": current.devices,
        "auth_required": bool(current.api_key),
    }


@app.get("/api/v1/accelerators")
def accelerators() -> Dict[str, Any]:
    current = state()
    return {
        "accelerators": [
            {"name": "B70", "devices": current.devices, "language": "sycl"},
        ]
    }


@app.get("/api/v1/challenges")
def challenges() -> Dict[str, Any]:
    return {"challenges": state().registry.list_public()}


@app.get("/api/v1/challenges/{challenge_ref}")
def challenge_detail(challenge_ref: str):
    registry = state().registry
    entry = (
        registry.get(int(challenge_ref))
        if challenge_ref.isdigit()
        else registry.by_key.get(challenge_ref)
    )
    if entry is None:
        return JSONResponse(
            status_code=404, content={"error": f"unknown challenge {challenge_ref!r}"}
        )
    return entry.detail()


@app.post("/api/v1/submissions", status_code=202)
def create_submission(
    request: SubmissionRequest, _: None = Depends(require_auth)
) -> Dict[str, Any]:
    return _create_submission(request)


def _create_submission(request: SubmissionRequest) -> Dict[str, Any]:
    current = state()
    language = request.language.lower()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported language {language!r}: supported {sorted(SUPPORTED_LANGUAGES)}",
        )
    accelerator = request.accelerator.upper()
    if accelerator not in SUPPORTED_ACCELERATORS:
        raise HTTPException(status_code=400, detail=f"unsupported accelerator {accelerator!r}")
    if not request.source.strip():
        raise HTTPException(status_code=400, detail="source is empty")
    if len(request.source.encode()) > MAX_FILE_BYTES:
        raise HTTPException(status_code=400, detail="source exceeds 2 MB")

    registry = current.registry
    challenge_ref = str(request.challenge)
    entry = (
        registry.get(int(challenge_ref))
        if challenge_ref.isdigit()
        else registry.by_key.get(challenge_ref)
    )
    if entry is None:
        raise HTTPException(status_code=404, detail=f"unknown challenge {request.challenge!r}")
    if "sycl" not in entry.starters:
        raise HTTPException(status_code=400, detail=f"{entry.key} has no SYCL starter yet")

    overrides = {
        key: value
        for key, value in {
            "runs": request.runs,
            "warmups": request.warmups,
            "inner_repeats": request.inner_repeats,
            "timeout": request.timeout,
        }.items()
        if value is not None
    }
    job = current.jobs.submit(
        {
            "challenge": entry.key,
            "content": request.source,
            "file_name": "solution.cpp",
            "action": "run",
            "overrides": overrides,
        }
    )
    return job.to_dict(include_result=False)


@app.get("/api/v1/submissions")
def list_submissions(limit: int = 20) -> Dict[str, Any]:
    return {
        "submissions": [job.to_dict(include_result=False) for job in state().jobs.recent(limit)]
    }


@app.get("/api/v1/submissions/{job_id}")
def get_submission(job_id: str) -> Dict[str, Any]:
    job = state().jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown submission")
    return job.to_dict(include_result=True)


@app.websocket("/api/v1/ws/submit")
async def submit(ws: WebSocket) -> None:
    await ws.accept()
    try:
        raw = await ws.receive_text()
    except WebSocketDisconnect:
        return
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        await ws.send_json(error_payload("invalid JSON payload"))
        await ws.close()
        return

    current = state()
    if not isinstance(payload, dict):
        await ws.send_json(error_payload("payload must be an object"))
        await ws.close()
        return
    request, problem = validate_submission(payload, current.api_key, current.registry)
    if problem is not None:
        await ws.send_json(problem)
        await ws.close()
        return

    device, future = current.pool.submit(request)
    await ws.send_json(
        {"status": "running", "output": f"queued {request['file_name']} on device {device}"}
    )
    try:
        result = await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        return
    except Exception as exc:  # noqa: BLE001 - never leak a traceback to the client
        await ws.send_json(error_payload(f"grader failure: {exc}"))
        await ws.close()
        return

    await ws.send_json({"status": result["status"], "output": summarize(result)})
    try:
        await ws.close()
    except RuntimeError:
        pass


app.mount("/assets", StaticFiles(directory=str(WEB_DIR), check_dir=False), name="assets")


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


def main() -> None:
    import uvicorn

    host = os.environ.get("XPU_RUNNER_HOST", "0.0.0.0")
    port = int(os.environ.get("XPU_RUNNER_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level=os.environ.get("XPU_RUNNER_LOG", "info"))
