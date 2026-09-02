"""Deterministic repository scanning with explicit ignore rules."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from forge.repository.models import RepositoryFile, RepositoryScan
from forge.workspace import Workspace, is_local_secret_name

MAX_REPOSITORY_FILES = 20_000

_IGNORED_DIRECTORY_NAMES = frozenset(
    {
        ".git",
        ".forge",
        ".hg",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".tox",
        ".venv",
        "__pycache__",
        "build",
        "dist",
        "node_modules",
        "venv",
    }
)
_IGNORED_FILE_PATTERNS = ("*.pyc", "*.pyo", "*.so", "*.dll", "*.dylib")


@dataclass(frozen=True, slots=True)
class IgnoreRule:
    pattern: str
    negated: bool = False
    directory_only: bool = False

    def matches(self, relative_path: PurePosixPath, *, is_directory: bool) -> bool:
        if self.directory_only and not is_directory:
            return False
        path_text = relative_path.as_posix()
        if "/" in self.pattern:
            return fnmatch.fnmatchcase(path_text, self.pattern)
        return fnmatch.fnmatchcase(relative_path.name, self.pattern)


@dataclass(frozen=True, slots=True)
class IgnoreRules:
    rules: tuple[IgnoreRule, ...] = ()

    @classmethod
    def from_workspace(cls, workspace: Workspace) -> IgnoreRules:
        ignore_file = workspace.root / ".gitignore"
        if not ignore_file.is_file() or ignore_file.is_symlink():
            return cls()
        try:
            lines = ignore_file.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            return cls()
        parsed: list[IgnoreRule] = []
        for raw_line in lines:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            negated = line.startswith("!")
            if negated:
                line = line[1:]
            directory_only = line.endswith("/")
            pattern = line.rstrip("/").lstrip("/")
            if pattern:
                parsed.append(IgnoreRule(pattern, negated, directory_only))
        return cls(tuple(parsed))

    def is_ignored(self, relative_path: Path, *, is_directory: bool) -> bool:
        pure_path = PurePosixPath(relative_path.as_posix())
        if any(part.casefold() in _IGNORED_DIRECTORY_NAMES for part in pure_path.parts):
            return True
        if not is_directory and any(
            fnmatch.fnmatchcase(pure_path.name, pattern)
            for pattern in _IGNORED_FILE_PATTERNS
        ):
            return True
        if _is_local_environment_name(pure_path.name):
            return True

        ignored = False
        for rule in self.rules:
            if rule.matches(pure_path, is_directory=is_directory):
                ignored = not rule.negated
        return ignored


class RepositoryScanner:
    """List safe workspace files without following directory or file links."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        ignore_rules: IgnoreRules | None = None,
        max_files: int = MAX_REPOSITORY_FILES,
    ) -> None:
        if max_files <= 0:
            raise ValueError("max_files must be positive")
        self._workspace = workspace
        self._ignore_rules = ignore_rules or IgnoreRules.from_workspace(workspace)
        self._max_files = max_files

    def scan(self) -> RepositoryScan:
        files: list[RepositoryFile] = []
        for current_root, directory_names, file_names in os.walk(
            self._workspace.root, followlinks=False
        ):
            current = Path(current_root)
            kept_directories: list[str] = []
            for name in sorted(directory_names):
                path = current / name
                relative = path.relative_to(self._workspace.root)
                if path.is_symlink() or self._ignore_rules.is_ignored(
                    relative, is_directory=True
                ):
                    continue
                kept_directories.append(name)
            directory_names[:] = kept_directories

            for name in sorted(file_names):
                path = current / name
                relative = path.relative_to(self._workspace.root)
                if path.is_symlink() or self._ignore_rules.is_ignored(
                    relative, is_directory=False
                ):
                    continue
                if not self._workspace.contains(path):
                    continue
                files.append(
                    RepositoryFile(
                        path=path,
                        relative_path=relative.as_posix(),
                        size=path.stat().st_size,
                    )
                )
                if len(files) >= self._max_files:
                    ordered = tuple(sorted(files, key=lambda file: file.relative_path))
                    return RepositoryScan(ordered, truncated=True)
        ordered = tuple(sorted(files, key=lambda file: file.relative_path))
        return RepositoryScan(ordered)


def _is_local_environment_name(name: str) -> bool:
    return is_local_secret_name(name)
