"""Small, deterministic delivery checks for explicit task artefacts.

This module does not decide whether code is *correct*. It only prevents a
self-verifying loop from treating a partial implementation as finished when a
user has explicitly named files that still do not exist in the local workspace.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


_PATH_PATTERN = re.compile(
    r"(?<![\w.-])(?P<path>(?:[A-Za-z0-9_.-]+[\\/])*[A-Za-z0-9_.-]+"
    r"\.(?:py|pyw|js|mjs|cjs|ts|tsx|jsx|html|css|json|toml|md|txt|yaml|yml))"
    r"(?![\w.-])",
    re.IGNORECASE,
)
_TEST_REQUEST = re.compile(r"\b(?:test|tests|pytest|unittest)\b|(?:测试|测验)", re.IGNORECASE)
_COMPILER_REQUEST = re.compile(
    r"\b(?:compile|compiler|py_compile|syntax\s+check)\b|(?:编译|语法检查)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True)
class DeliveryRequirements:
    """Explicit workspace-relative file deliverables parsed from the task."""

    paths: tuple[str, ...] = ()
    verification_kinds: tuple[str, ...] = ()

    @classmethod
    def from_task(cls, task: str) -> DeliveryRequirements:
        """Extract safe, explicitly named relative file paths in task order."""

        paths: list[str] = []
        for match in _PATH_PATTERN.finditer(task):
            normalized = _normalise_relative_path(match.group("path"))
            if normalized is not None and normalized not in paths:
                paths.append(normalized)
        kinds: list[str] = []
        if _TEST_REQUEST.search(task):
            kinds.append("test")
        if _COMPILER_REQUEST.search(task):
            kinds.append("compiler")
        return cls(tuple(paths), tuple(kinds))

    def missing(self, workspace: Path) -> tuple[str, ...]:
        """Return required files that do not currently exist as regular files."""

        result: list[str] = []
        for relative in self.paths:
            candidate = workspace / Path(relative)
            if not candidate.is_file():
                result.append(relative)
        return tuple(result)

    def missing_verification_kinds(
        self,
        completed_kinds: set[str] | frozenset[str],
    ) -> tuple[str, ...]:
        """Return explicitly requested deterministic evidence still absent."""

        return tuple(kind for kind in self.verification_kinds if kind not in completed_kinds)


def _normalise_relative_path(value: str) -> str | None:
    """Reject absolute/traversal paths before they can influence a prompt gate."""

    candidate = Path(value.replace("\\", "/"))
    if candidate.is_absolute() or any(part in {"", ".", ".."} for part in candidate.parts):
        return None
    if candidate.parts and candidate.parts[0].casefold() == ".dbagent":
        return None
    return candidate.as_posix()
