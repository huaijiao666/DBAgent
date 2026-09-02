"""Exact-context, multi-file patching with rollback on handled failures."""

from __future__ import annotations

import hashlib
import os
import stat
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dbagent.workspace import Workspace

MAX_PATCH_FILES = 50
MAX_PATCH_HUNKS = 200
MAX_PATCH_FILE_BYTES = 2_000_000


class PatchError(Exception):
    """A patch could not be validated or committed safely."""


@dataclass(frozen=True, slots=True)
class _Line:
    content: str
    ending: str


@dataclass(frozen=True, slots=True)
class _PreparedChange:
    path: Path
    relative_path: str
    original: bytes
    updated: bytes
    mode: int
    hunks: int


@dataclass(slots=True)
class _StagedChange:
    change: _PreparedChange
    updated_temp: Path
    recovery_temp: Path


class PatchApplier:
    """Validate every hunk in memory before replacing any workspace file."""

    def __init__(self, workspace: Workspace) -> None:
        self._workspace = workspace

    def apply(self, file_patches: object) -> dict[str, object]:
        """Apply a structured patch and always return a structured result."""

        try:
            changes = self._prepare_changes(file_patches)
            self._commit_changes(changes)
        except (PatchError, OSError, UnicodeError, ValueError) as error:
            return {
                "applied": False,
                "changed_files": [],
                "hunks_applied": 0,
                "failure_reason": f"{type(error).__name__}: {error}",
            }

        return {
            "applied": True,
            "changed_files": [
                {
                    "path": change.relative_path,
                    "before_sha256": _sha256(change.original),
                    "after_sha256": _sha256(change.updated),
                }
                for change in changes
            ],
            "hunks_applied": sum(change.hunks for change in changes),
            "failure_reason": None,
        }

    def _prepare_changes(self, file_patches: object) -> list[_PreparedChange]:
        patches = _object_sequence(file_patches, "files")
        if not patches:
            raise PatchError("files must contain at least one file patch")
        if len(patches) > MAX_PATCH_FILES:
            raise PatchError(f"patch exceeds {MAX_PATCH_FILES} files")

        changes: list[_PreparedChange] = []
        seen_paths: set[Path] = set()
        total_hunks = 0
        for file_number, file_patch in enumerate(patches, start=1):
            requested_path = _required_string(file_patch, "path")
            path = self._workspace.resolve(requested_path)
            if path in seen_paths:
                raise PatchError(f"duplicate target path: {requested_path}")
            seen_paths.add(path)
            if not path.is_file():
                raise PatchError(f"target is not a regular file: {requested_path}")

            original = path.read_bytes()
            if len(original) > MAX_PATCH_FILE_BYTES:
                raise PatchError(
                    f"target exceeds {MAX_PATCH_FILE_BYTES} bytes: {requested_path}"
                )
            text = original.decode("utf-8")
            hunks = _object_sequence(file_patch.get("hunks"), "hunks")
            if not hunks:
                raise PatchError(f"file patch {file_number} has no hunks")
            total_hunks += len(hunks)
            if total_hunks > MAX_PATCH_HUNKS:
                raise PatchError(f"patch exceeds {MAX_PATCH_HUNKS} hunks")

            updated = _apply_hunks(text, hunks, requested_path)
            changes.append(
                _PreparedChange(
                    path=path,
                    relative_path=self._workspace.relative_name(path),
                    original=original,
                    updated=updated.encode("utf-8"),
                    mode=stat.S_IMODE(path.stat().st_mode),
                    hunks=len(hunks),
                )
            )
        return changes

    def _commit_changes(self, changes: list[_PreparedChange]) -> None:
        for change in changes:
            if change.path.read_bytes() != change.original:
                raise PatchError(
                    f"target changed while patch was prepared: {change.relative_path}"
                )

        staged: list[_StagedChange] = []
        try:
            for change in changes:
                staged.append(_stage_change(change))
            for change in changes:
                if change.path.read_bytes() != change.original:
                    raise PatchError(
                        "target changed while patch files were staged: "
                        f"{change.relative_path}"
                    )
        except BaseException:
            _clean_staged_files(staged)
            raise

        committed: list[_StagedChange] = []
        try:
            for item in staged:
                os.replace(item.updated_temp, item.change.path)
                committed.append(item)
        except BaseException as commit_error:
            rollback_errors: list[str] = []
            for item in reversed(committed):
                try:
                    os.replace(item.recovery_temp, item.change.path)
                except BaseException as rollback_error:
                    rollback_errors.append(
                        f"{item.change.relative_path}: {rollback_error}"
                    )
            _clean_staged_files(staged)
            if rollback_errors:
                raise PatchError(
                    "commit failed and rollback also failed: "
                    + "; ".join(rollback_errors)
                ) from commit_error
            raise PatchError(
                f"commit failed; original files restored: {commit_error}"
            ) from commit_error

        _clean_staged_files(staged)


def _apply_hunks(
    text: str, hunks: Sequence[Mapping[str, Any]], requested_path: str
) -> str:
    lines = _split_lines(text)
    preferred_ending = next((line.ending for line in lines if line.ending), "\n")
    for hunk_number, hunk in enumerate(hunks, start=1):
        old_lines = _line_sequence(hunk.get("old_lines"), "old_lines")
        new_lines = _line_sequence(hunk.get("new_lines"), "new_lines")
        if not old_lines:
            raise PatchError(
                f"{requested_path} hunk {hunk_number}: old_lines must not be empty"
            )
        if old_lines == new_lines:
            raise PatchError(
                f"{requested_path} hunk {hunk_number}: replacement makes no change"
            )

        matches = _find_matches(lines, old_lines)
        if not matches:
            raise PatchError(
                f"{requested_path} hunk {hunk_number}: context did not match"
            )
        if len(matches) > 1:
            raise PatchError(
                f"{requested_path} hunk {hunk_number}: context is ambiguous "
                f"({len(matches)} matches)"
            )

        start = matches[0]
        stop = start + len(old_lines)
        replaced_final_unterminated_line = (
            stop == len(lines) and lines[stop - 1].ending == ""
        )
        replacement = [
            _Line(
                content=line,
                ending=(
                    ""
                    if replaced_final_unterminated_line
                    and index == len(new_lines) - 1
                    else preferred_ending
                ),
            )
            for index, line in enumerate(new_lines)
        ]
        lines[start:stop] = replacement
    return "".join(line.content + line.ending for line in lines)


def _split_lines(text: str) -> list[_Line]:
    result: list[_Line] = []
    for raw_line in text.splitlines(keepends=True):
        if raw_line.endswith("\r\n"):
            result.append(_Line(raw_line[:-2], "\r\n"))
        elif raw_line.endswith(("\n", "\r")):
            result.append(_Line(raw_line[:-1], raw_line[-1]))
        else:
            result.append(_Line(raw_line, ""))
    return result


def _find_matches(lines: Sequence[_Line], expected: Sequence[str]) -> list[int]:
    last_start = len(lines) - len(expected)
    return [
        start
        for start in range(last_start + 1)
        if [line.content for line in lines[start : start + len(expected)]]
        == list(expected)
    ]


def _object_sequence(value: object, name: str) -> list[Mapping[str, Any]]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PatchError(f"{name} must be an array")
    if any(not isinstance(item, Mapping) for item in value):
        raise PatchError(f"every item in {name} must be an object")
    return list(value)  # type: ignore[arg-type]


def _line_sequence(value: object, name: str) -> list[str]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise PatchError(f"{name} must be an array of strings")
    if any(not isinstance(line, str) for line in value):
        raise PatchError(f"{name} must be an array of strings")
    return [_normalize_model_line(line, name) for line in value]


def _normalize_model_line(line: str, name: str) -> str:
    """Accept one accidental transport line ending, never embedded newlines.

    The patch protocol is line-oriented, so a provider that serializes an
    individual array entry as ``"return value\\n"`` has not supplied a
    different logical line.  Normalizing exactly one terminal CRLF/LF/CR makes
    native function-calling adapters more tolerant without turning this into a
    free-form diff parser.  A newline anywhere else remains invalid: it would
    make one array entry represent multiple patch lines and weaken exact-context
    matching.
    """

    if line.endswith("\r\n"):
        line = line[:-2]
    elif line.endswith(("\n", "\r")):
        line = line[:-1]
    if "\n" in line or "\r" in line:
        raise PatchError(f"{name} entries must not contain embedded line endings")
    return line


def _required_string(value: Mapping[str, Any], name: str) -> str:
    result = value.get(name)
    if not isinstance(result, str) or not result.strip():
        raise PatchError(f"{name} must be a non-empty string")
    return result


def _stage_bytes(path: Path, content: bytes, mode: int, label: str) -> Path:
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.dbagent-{label}-",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
        os.chmod(temporary_path, mode)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
    return temporary_path


def _stage_change(change: _PreparedChange) -> _StagedChange:
    updated_temp = _stage_bytes(
        change.path, change.updated, change.mode, "updated"
    )
    try:
        recovery_temp = _stage_bytes(
            change.path, change.original, change.mode, "recovery"
        )
    except BaseException:
        updated_temp.unlink(missing_ok=True)
        raise
    return _StagedChange(change, updated_temp, recovery_temp)


def _clean_staged_files(staged: Sequence[_StagedChange]) -> None:
    for item in staged:
        item.updated_temp.unlink(missing_ok=True)
        item.recovery_temp.unlink(missing_ok=True)


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()
