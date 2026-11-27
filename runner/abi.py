"""Bind a challenge's `get_solve_signature()` to a ctypes callable."""

import ctypes
from dataclasses import dataclass
from typing import Any, Dict, List

_SCALARS = {
    ctypes.c_bool: bool,
    ctypes.c_int8: int,
    ctypes.c_int16: int,
    ctypes.c_int32: int,
    ctypes.c_int64: int,
    ctypes.c_int: int,
    ctypes.c_uint8: int,
    ctypes.c_uint16: int,
    ctypes.c_uint32: int,
    ctypes.c_uint64: int,
    ctypes.c_uint: int,
    ctypes.c_size_t: int,
    ctypes.c_float: float,
    ctypes.c_double: float,
}

DIRECTIONS = ("in", "out", "inout")


@dataclass(frozen=True)
class ArgSpec:
    name: str
    direction: str
    kind: str  # "tensor" | "scalar"
    ctype: Any

    def to_dict(self) -> Dict[str, str]:
        return {"name": self.name, "direction": self.direction, "kind": self.kind}


def _is_pointer(ctype: Any) -> bool:
    return isinstance(ctype, type) and issubclass(ctype, ctypes._Pointer)


def parse_signature(challenge) -> List[ArgSpec]:
    signature = challenge.get_solve_signature()
    specs = []
    for name, (ctype, direction) in signature.items():
        if direction not in DIRECTIONS:
            raise ValueError(f"{name}: invalid direction {direction!r}")
        if _is_pointer(ctype):
            specs.append(ArgSpec(name, direction, "tensor", ctype))
        elif ctype in _SCALARS:
            specs.append(ArgSpec(name, direction, "scalar", ctype))
        else:
            raise TypeError(f"{name}: unsupported ctypes type {ctype!r}")
    return specs


def bind(lib, specs: List[ArgSpec]):
    if not hasattr(lib, "solve"):
        raise AttributeError("submission does not export `solve` with C linkage")
    fn = lib.solve
    fn.restype = None
    fn.argtypes = [ctypes.c_void_p if spec.kind == "tensor" else spec.ctype for spec in specs]
    return fn


def raw_args(specs: List[ArgSpec], kwargs: Dict[str, Any]) -> List[Any]:
    args = []
    for spec in specs:
        value = kwargs[spec.name]
        if spec.kind == "tensor":
            ptr = getattr(value, "data_ptr", None)
            if ptr is None:
                raise TypeError(f"{spec.name}: expected a tensor")
            args.append(ctypes.c_void_p(ptr()))
        else:
            args.append(_SCALARS[spec.ctype](value))
    return args


def output_names(specs: List[ArgSpec]) -> List[str]:
    return [spec.name for spec in specs if spec.direction in ("out", "inout")]
