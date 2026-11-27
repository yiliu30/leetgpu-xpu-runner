"""Locate, load, and materialize LeetGPU challenges."""

import importlib.util
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

_ID_RE = re.compile(r"^(\d+)_")


@dataclass
class LoadedChallenge:
    key: str
    directory: Path
    module: Any
    instance: Any

    @property
    def atol(self) -> float:
        return float(self.instance.atol)

    @property
    def rtol(self) -> float:
        return float(self.instance.rtol)


def _challenge_dirs(challenges_root: Path) -> Iterable[Path]:
    for difficulty in ("easy", "medium", "hard"):
        directory = challenges_root / difficulty
        if directory.is_dir():
            yield from sorted(path for path in directory.iterdir() if path.is_dir())


def resolve(root: Path, spec: str) -> Path:
    """Resolve a challenge directory from an id, slug, relative path, or directory.

    `root` may be either the repository root or its `challenges/` directory.
    """
    root = Path(root).expanduser()
    challenges_root = root / "challenges" if (root / "challenges").is_dir() else root
    candidate = Path(spec).expanduser()
    if (candidate / "challenge.py").is_file():
        return candidate.resolve()
    if not candidate.is_absolute() and (challenges_root / candidate / "challenge.py").is_file():
        return (challenges_root / candidate).resolve()

    if spec.isdigit():
        matches = [
            p
            for p in _challenge_dirs(challenges_root)
            if _ID_RE.match(p.name) and int(_ID_RE.match(p.name).group(1)) == int(spec)
        ]
        if len(matches) == 1:
            return matches[0].resolve()
        raise SystemExit(f"challenge id {spec} matched {len(matches)} directories")

    slug = re.sub(r"[^a-z0-9]+", "_", spec.lower()).strip("_")
    matches = [p for p in _challenge_dirs(challenges_root) if re.sub(r"^\d+_", "", p.name) == slug]
    if len(matches) == 1:
        return matches[0].resolve()
    if not matches:
        raise SystemExit(f"no challenge matching {spec!r} under {challenges_root}")
    raise SystemExit(f"ambiguous challenge {spec!r}: {[str(p) for p in matches]}")


def challenge_key(directory: Path) -> str:
    return f"{directory.parent.name}/{directory.name}"


def _module_name(directory: Path) -> str:
    return "leetgpu_challenge_" + re.sub(r"[^a-zA-Z0-9_]", "_", challenge_key(directory))


def load(directory: Path, device: str = "xpu") -> LoadedChallenge:
    challenges_root = directory.parent.parent
    if str(challenges_root) not in sys.path:
        sys.path.insert(0, str(challenges_root))
    if "core" in sys.modules:
        del sys.modules["core"]
    module_name = _module_name(directory)
    spec = importlib.util.spec_from_file_location(module_name, directory / "challenge.py")
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {directory / 'challenge.py'}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return LoadedChallenge(
        challenge_key(directory), directory, module, module.Challenge(device=device)
    )


_DTYPE_NAMES = {
    "float32",
    "float16",
    "bfloat16",
    "float64",
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "bool",
}
_SPEC_NAMES = {"RandTensor", "RandnTensor", "RandIntTensor", "FullTensor", "OutTensor"}


def materialize(value: Any, device: str) -> Any:
    """Turn a challenge test spec into a concrete torch tensor on `device`."""
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device) if value.device.type != device else value
    if isinstance(value, (int, float, bool)):
        return value
    name = type(value).__name__
    if name not in _SPEC_NAMES:
        raise TypeError(f"unsupported test value {name}")
    dtype_name = getattr(value, "dtype", "float32")
    if dtype_name not in _DTYPE_NAMES:
        raise TypeError(f"unsupported dtype {dtype_name!r}")
    dtype = getattr(torch, dtype_name)
    if name == "RandTensor":
        return torch.empty(value.shape, device=device, dtype=dtype).uniform_(value.low, value.high)
    if name == "RandnTensor":
        return torch.randn(value.shape, device=device, dtype=dtype) * value.std + value.mean
    if name == "RandIntTensor":
        return torch.randint(value.low, value.high, value.shape, device=device, dtype=dtype)
    if name == "FullTensor":
        return torch.full(value.shape, value.value, device=device, dtype=dtype)
    return torch.empty(value.shape, device=device, dtype=dtype)


def materialize_case(case: Dict[str, Any], device: str) -> Dict[str, Any]:
    return {name: materialize(value, device) for name, value in case.items()}


def describe_case(case: Dict[str, Any]) -> str:
    parts = []
    for name, value in case.items():
        shape = getattr(value, "shape", None)
        parts.append(f"{name}{tuple(shape)}" if shape is not None else f"{name}={value}")
    return ", ".join(parts)


def functional_cases(challenge: LoadedChallenge) -> List[Dict[str, Any]]:
    return list(challenge.instance.generate_functional_test())


def performance_case(challenge: LoadedChallenge) -> Dict[str, Any]:
    return challenge.instance.generate_performance_test()


def signature_dict(challenge: LoadedChallenge) -> Dict[str, Any]:
    return challenge.instance.get_solve_signature()


def optional(value: Optional[str]) -> Optional[Path]:
    return Path(value).expanduser() if value else None
