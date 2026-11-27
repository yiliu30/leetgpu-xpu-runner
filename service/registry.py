"""Challenge registry built from the challenge repo without importing torch."""

import ast
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

# starter filename -> (language, backend filename)
STARTERS = {
    "starter.cu": ("cuda", "starter.cu"),
    "starter.cute.py": ("cute", "starter.py"),
    "starter.jax.py": ("jax", "starter.py"),
    "starter.mojo": ("mojo", "starter.mojo"),
    "starter.pytorch.py": ("pytorch", "starter.py"),
    "starter.sycl.cpp": ("sycl", "starter.cpp"),
    "starter.triton.py": ("triton", "starter.py"),
}
_ID_RE = re.compile(r"^(\d+)_")
_DIFFICULTIES = ("easy", "medium", "hard")


@dataclass
class Entry:
    id: int
    key: str
    title: str
    difficulty: str
    access_tier: str
    path: Path
    spec: str
    source: str
    starters: Dict[str, Dict[str, str]] = field(default_factory=dict)

    def summary(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "title": self.title,
            "spec": self.spec,
            "difficultyLevel": self.difficulty,
            "accessTier": self.access_tier,
        }

    def detail(self) -> Dict[str, object]:
        payload = self.summary()
        payload["starterCode"] = [
            {"language": language, "fileName": filename, "fileContent": content}
            for language, (filename, content) in sorted(self.starters.items())
        ]
        return payload


def _class_metadata(source: str) -> Dict[str, str]:
    values: Dict[str, str] = {}
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return values
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "Challenge":
            for child in node.body:
                if isinstance(child, ast.Assign) and len(child.targets) == 1:
                    target = child.targets[0]
                    if isinstance(target, ast.Name) and isinstance(child.value, ast.Constant):
                        values[target.id] = str(child.value.value)
    return values


def _normalize(source: str) -> str:
    return source.replace("\r\n", "\n").rstrip()


class Registry:
    def __init__(self, repo: Path):
        self.repo = Path(repo).expanduser().resolve()
        self.root = self.repo / "challenges" if (self.repo / "challenges").is_dir() else self.repo
        self.by_id: Dict[int, Entry] = {}
        self.by_key: Dict[str, Entry] = {}
        self.refresh()

    def refresh(self) -> None:
        self.by_id.clear()
        self.by_key.clear()
        for difficulty in _DIFFICULTIES:
            directory = self.root / difficulty
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                match = _ID_RE.match(path.name)
                if not match or not (path / "challenge.py").is_file():
                    continue
                source = (path / "challenge.py").read_text(encoding="utf-8")
                metadata = _class_metadata(source)
                starters = {}
                for filename, (language, backend_name) in STARTERS.items():
                    starter_path = path / "starter" / filename
                    if starter_path.is_file():
                        starters[language] = (
                            backend_name,
                            starter_path.read_text(encoding="utf-8"),
                        )
                spec_path = path / "challenge.html"
                entry = Entry(
                    id=int(match.group(1)),
                    key=f"{difficulty}/{path.name}",
                    title=metadata.get("name", path.name),
                    difficulty=difficulty,
                    access_tier=metadata.get("access_tier", "free"),
                    path=path,
                    spec=spec_path.read_text(encoding="utf-8") if spec_path.is_file() else "",
                    source=source,
                    starters=starters,
                )
                self.by_id[entry.id] = entry
                self.by_key[entry.key] = entry

    def list_public(self) -> List[Dict[str, object]]:
        return [entry.summary() for entry in sorted(self.by_id.values(), key=lambda e: -e.id)]

    def get(self, challenge_id: int) -> Optional[Entry]:
        return self.by_id.get(challenge_id)

    def find_by_source(self, challenge_code: str) -> Optional[Entry]:
        needle = _normalize(challenge_code)
        for entry in self.by_id.values():
            if _normalize(entry.source) == needle:
                return entry
        title = _class_metadata(challenge_code).get("name")
        if title:
            matches = [entry for entry in self.by_id.values() if entry.title == title]
            if len(matches) == 1:
                return matches[0]
        return None

    def matches(self, entry: Entry, challenge_code: str) -> bool:
        """True when the submitted oracle is byte-identical to the first-party challenge."""
        return _normalize(entry.source) == _normalize(challenge_code)

    def supports_sycl(self, entry: Entry) -> bool:
        return "sycl" in entry.starters
