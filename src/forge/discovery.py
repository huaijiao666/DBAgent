"""Deterministic project-root discovery for the interactive launcher."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


_STRONG_MARKERS = (
    ".git",
    "pyproject.toml",
    "package.json",
    "Cargo.toml",
    "go.mod",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
)
_SUPPORTING_MARKERS = ("README.md", "README.rst", "README.txt", "tests", "src")


@dataclass(frozen=True, slots=True)
class WorkspaceDiscovery:
    """Explain which directory DBA selected and why."""

    start: Path
    root: Path
    markers: tuple[str, ...]

    @property
    def auto_detected(self) -> bool:
        return self.root != self.start


def discover_workspace(start: Path) -> WorkspaceDiscovery:
    """Choose the nearest plausible project root at or above ``start``.

    A strong build/VCS marker wins immediately. A nearer directory containing
    both documentation and tests/source layout is preferred over a more distant
    enclosing repository, which handles monorepos and nested demo projects.
    """

    resolved = start.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"workspace start is not a directory: {start}")

    fallback: tuple[Path, tuple[str, ...]] | None = None
    for candidate in (resolved, *resolved.parents):
        strong = tuple(name for name in _STRONG_MARKERS if (candidate / name).exists())
        supporting = tuple(
            name for name in _SUPPORTING_MARKERS if (candidate / name).exists()
        )
        if strong:
            return WorkspaceDiscovery(resolved, candidate, strong + supporting)
        has_docs = any(name.startswith("README") for name in supporting)
        has_layout = "tests" in supporting or "src" in supporting
        if has_docs and has_layout:
            return WorkspaceDiscovery(resolved, candidate, supporting)
        if fallback is None and supporting:
            fallback = (candidate, supporting)

    root, markers = fallback or (resolved, ())
    return WorkspaceDiscovery(resolved, root, markers)
