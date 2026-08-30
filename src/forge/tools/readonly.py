"""Bounded, read-only tools rooted in a local workspace."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from forge.llm import FunctionTool
from forge.tools.models import ToolDefinition, object_schema
from forge.tools.repository import register_repository_tools
from forge.tools.registry import ToolRegistry
from forge.workspace import Workspace

_EXCLUDED_DIRECTORIES = frozenset(
    {
        ".forge",
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "venv",
        "__pycache__",
        "node_modules",
    }
)
_MAX_LIST_ENTRIES = 2_000
_MAX_READ_CHARACTERS = 100_000
_MAX_SEARCH_FILE_BYTES = 1_000_000
_MAX_SEARCH_MATCHES = 200


def create_readonly_registry(workspace_root: Path) -> ToolRegistry:
    """Create generic and Python-aware read-only tools for one workspace."""

    workspace = Workspace(workspace_root)
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="list_files",
                    description=(
                        "Recursively list files under a workspace-relative path. "
                        "Use '.' for the workspace root."
                    ),
                    parameters=object_schema(
                        {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative directory or file path.",
                            }
                        },
                        required=["path"],
                    ),
                ),
                handler=lambda arguments: _list_files(workspace, arguments),
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="read_file",
                    description=(
                        "Read a UTF-8 text file inside the workspace with line numbers."
                    ),
                    parameters=object_schema(
                        {
                            "path": {
                                "type": "string",
                                "description": "Workspace-relative file path.",
                            }
                        },
                        required=["path"],
                    ),
                ),
                handler=lambda arguments: _read_file(workspace, arguments),
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="search_text",
                    description=(
                        "Case-insensitively search UTF-8 text files for a literal string."
                    ),
                    parameters=object_schema(
                        {
                            "query": {
                                "type": "string",
                                "description": "Literal text to search for.",
                            },
                            "path": {
                                "type": "string",
                                "description": (
                                    "Workspace-relative file or directory to search."
                                ),
                            },
                        },
                        required=["query", "path"],
                    ),
                ),
                handler=lambda arguments: _search_text(workspace, arguments),
            ),
        ]
    )
    register_repository_tools(registry, workspace)
    return registry


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _list_files(workspace: Workspace, arguments: Mapping[str, Any]) -> str:
    path = workspace.resolve(_required_string(arguments, "path"))
    _reject_excluded_path(workspace, path)
    if path.is_file():
        return workspace.relative_name(path)
    if not path.is_dir():
        raise ValueError("path is neither a file nor a directory")

    names: list[str] = []
    for file_path in _iter_files(workspace, path):
        names.append(workspace.relative_name(file_path))
        if len(names) >= _MAX_LIST_ENTRIES:
            names.append(f"[truncated after {_MAX_LIST_ENTRIES} files]")
            break
    return "\n".join(names) if names else "[no files]"


def _read_file(workspace: Workspace, arguments: Mapping[str, Any]) -> str:
    path = workspace.resolve(_required_string(arguments, "path"))
    _reject_excluded_path(workspace, path)
    if not path.is_file():
        raise ValueError("path is not a file")
    text = path.read_text(encoding="utf-8")
    numbered = "\n".join(
        f"{line_number}: {line}"
        for line_number, line in enumerate(text.splitlines(), start=1)
    )
    if not numbered:
        return "[empty file]"
    if len(numbered) > _MAX_READ_CHARACTERS:
        return (
            numbered[:_MAX_READ_CHARACTERS]
            + f"\n[truncated after {_MAX_READ_CHARACTERS} characters]"
        )
    return numbered


def _search_text(workspace: Workspace, arguments: Mapping[str, Any]) -> str:
    query = _required_string(arguments, "query")
    path = workspace.resolve(_required_string(arguments, "path"))
    _reject_excluded_path(workspace, path)
    files = [path] if path.is_file() else _iter_files(workspace, path)
    folded_query = query.casefold()
    matches: list[str] = []

    for file_path in files:
        if file_path.stat().st_size > _MAX_SEARCH_FILE_BYTES:
            continue
        try:
            lines = file_path.read_text(encoding="utf-8").splitlines()
        except UnicodeDecodeError:
            continue
        for line_number, line in enumerate(lines, start=1):
            if folded_query in line.casefold():
                matches.append(
                    f"{workspace.relative_name(file_path)}:{line_number}: {line}"
                )
                if len(matches) >= _MAX_SEARCH_MATCHES:
                    matches.append(
                        f"[truncated after {_MAX_SEARCH_MATCHES} matches]"
                    )
                    return "\n".join(matches)
    return "\n".join(matches) if matches else "[no matches]"


def _iter_files(workspace: Workspace, root: Path) -> Iterator[Path]:
    for current_root, directory_names, file_names in os.walk(
        root, followlinks=False
    ):
        current = Path(current_root)
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name.casefold() not in _EXCLUDED_DIRECTORIES
            and not (current / name).is_symlink()
        )
        for file_name in sorted(file_names):
            file_path = current / file_name
            if _is_local_environment_name(file_name):
                continue
            if file_path.is_symlink():
                try:
                    workspace.resolve(str(file_path))
                except (FileNotFoundError, ValueError):
                    continue
            yield file_path


def _is_local_environment_name(name: str) -> bool:
    lowered = name.casefold()
    return lowered == ".env" or (
        lowered.startswith(".env.") and lowered != ".env.example"
    )


def _reject_excluded_path(workspace: Workspace, path: Path) -> None:
    relative = path.relative_to(workspace.root)
    if any(part.casefold() in _EXCLUDED_DIRECTORIES for part in relative.parts):
        raise PermissionError("access to repository metadata/cache directories is blocked")
