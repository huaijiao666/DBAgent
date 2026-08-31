"""Canonical workspace path enforcement shared by all local tools."""

from __future__ import annotations

from pathlib import Path


class Workspace:
    """Resolve paths while enforcing one canonical workspace boundary."""

    def __init__(self, root: Path) -> None:
        resolved = root.resolve(strict=True)
        if not resolved.is_dir():
            raise ValueError(f"workspace is not a directory: {root}")
        self.root = resolved

    def resolve(self, user_path: str) -> Path:
        """Resolve an existing path contained by the workspace."""

        candidate = self._candidate(user_path)
        resolved = candidate.resolve(strict=True)
        self._validate_resolved(resolved, user_path)
        return resolved

    def resolve_for_create(self, user_path: str) -> Path:
        """Resolve a new file target, allowing missing in-workspace parents."""

        candidate = self._candidate(user_path)
        cursor = candidate.parent
        missing_parts: list[str] = []
        while not cursor.exists():
            if cursor.is_symlink():
                raise ValueError(f"path contains a dangling symlink: {user_path}")
            missing_parts.append(cursor.name)
            if cursor.parent == cursor:
                raise ValueError(f"path escapes workspace: {user_path}")
            cursor = cursor.parent
        resolved_parent = cursor.resolve(strict=True)
        self._validate_resolved(resolved_parent, user_path)
        for part in reversed(missing_parts):
            resolved_parent /= part
        self._validate_resolved(resolved_parent.resolve(strict=False), user_path)
        target = resolved_parent / candidate.name
        self._validate_resolved(target.resolve(strict=False), user_path)
        if target.exists() or target.is_symlink():
            raise FileExistsError(f"path already exists: {user_path}")
        return target

    def resolve_directory(self, user_path: str) -> Path:
        resolved = self.resolve(user_path)
        if not resolved.is_dir():
            raise ValueError(f"cwd is not a directory: {user_path}")
        return resolved

    def relative_name(self, path: Path) -> str:
        return path.resolve(strict=False).relative_to(self.root).as_posix() or "."

    def contains(self, path: Path) -> bool:
        try:
            path.resolve(strict=False).relative_to(self.root)
        except ValueError:
            return False
        return True

    def _candidate(self, user_path: str) -> Path:
        if not isinstance(user_path, str) or not user_path.strip():
            raise ValueError("path must be a non-empty string")
        candidate = Path(user_path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def _validate_resolved(self, resolved: Path, original: str) -> None:
        try:
            relative = resolved.relative_to(self.root)
        except ValueError as error:
            raise ValueError(f"path escapes workspace: {original}") from error
        if any(_is_local_environment_name(part) for part in relative.parts):
            raise PermissionError("access to local environment files is blocked")


def _is_local_environment_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered == ".env" or (
        lowered.startswith(".env.") and lowered != ".env.example"
    )
