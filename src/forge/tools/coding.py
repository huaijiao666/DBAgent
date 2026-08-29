"""Transitional write, command, and Git tools for coding tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.execution import CommandExecutor
from forge.llm import FunctionTool
from forge.tools.models import ToolDefinition, object_schema
from forge.tools.readonly import create_readonly_registry
from forge.tools.registry import ToolRegistry
from forge.workspace import Workspace

_MAX_WRITE_CHARACTERS = 200_000


def create_coding_registry(workspace_root: Path) -> ToolRegistry:
    """Create all current read-only and transitional coding tools."""

    workspace = Workspace(workspace_root)
    executor = CommandExecutor(workspace)
    registry = create_readonly_registry(workspace.root)
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="run_command",
                description=(
                    "Run one argument-vector command inside the workspace without a "
                    "shell. Returns stdout, stderr, return code, timeout, and "
                    "truncation metadata."
                ),
                parameters=object_schema(
                    {
                        "command": {
                            "type": "array",
                            "items": {"type": "string"},
                            "minItems": 1,
                            "description": (
                                "Executable and arguments, for example "
                                "['python', '-m', 'pytest', '-q']."
                            ),
                        },
                        "cwd": {
                            "type": "string",
                            "description": (
                                "Workspace-relative working directory; use '.'."
                            ),
                        },
                        "timeout_seconds": {
                            "type": "number",
                            "exclusiveMinimum": 0,
                            "maximum": 300,
                        },
                    },
                    required=["command", "cwd", "timeout_seconds"],
                ),
            ),
            handler=lambda arguments: _run_command(executor, arguments),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="create_file",
                description=(
                    "Create a new UTF-8 text file inside the workspace. Fails if the "
                    "file exists; its parent directory must already exist."
                ),
                parameters=_file_write_schema(),
            ),
            handler=lambda arguments: _create_file(workspace, arguments),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="write_file",
                description=(
                    "Replace the complete contents of an existing UTF-8 workspace file."
                ),
                parameters=_file_write_schema(),
            ),
            handler=lambda arguments: _write_file(workspace, arguments),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="git_diff",
                description=(
                    "Show workspace Git status and the current unstaged diff using "
                    "fixed read-only Git commands."
                ),
                parameters=object_schema({}, required=[]),
            ),
            handler=lambda _arguments: _git_diff(executor),
        )
    )
    return registry


def _file_write_schema() -> dict[str, Any]:
    return object_schema(
        {
            "path": {
                "type": "string",
                "description": "Workspace-relative file path.",
            },
            "content": {
                "type": "string",
                "description": "Complete UTF-8 file contents.",
            },
        },
        required=["path", "content"],
    )


def _run_command(
    executor: CommandExecutor, arguments: Mapping[str, Any]
) -> dict[str, object]:
    command = arguments.get("command")
    if isinstance(command, (str, bytes)) or not isinstance(command, Sequence):
        raise ValueError("command must be an array of strings")
    if any(not isinstance(part, str) for part in command):
        raise ValueError("command must be an array of strings")
    cwd = _required_string(arguments, "cwd")
    timeout = arguments.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    return executor.run(command, cwd=cwd, timeout_seconds=timeout).to_dict()


def _create_file(workspace: Workspace, arguments: Mapping[str, Any]) -> str:
    path = workspace.resolve_for_create(_required_string(arguments, "path"))
    content = _content(arguments)
    with path.open("x", encoding="utf-8", newline="") as file:
        file.write(content)
    return f"created {workspace.relative_name(path)} ({len(content)} characters)"


def _write_file(workspace: Workspace, arguments: Mapping[str, Any]) -> str:
    path = workspace.resolve(_required_string(arguments, "path"))
    if not path.is_file():
        raise ValueError("path is not a file")
    content = _content(arguments)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)
    return f"wrote {workspace.relative_name(path)} ({len(content)} characters)"


def _git_diff(executor: CommandExecutor) -> dict[str, object]:
    status = executor.run(
        ["git", "status", "--short"],
        cwd=".",
        timeout_seconds=30,
    )
    diff = executor.run(
        ["git", "diff", "--no-ext-diff", "--no-color", "--"],
        cwd=".",
        timeout_seconds=30,
    )
    return {"status": status.to_dict(), "diff": diff.to_dict()}


def _required_string(arguments: Mapping[str, Any], name: str) -> str:
    value = arguments.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _content(arguments: Mapping[str, Any]) -> str:
    content = arguments.get("content")
    if not isinstance(content, str):
        raise ValueError("content must be a string")
    if len(content) > _MAX_WRITE_CHARACTERS:
        raise ValueError(
            f"content exceeds {_MAX_WRITE_CHARACTERS} characters"
        )
    return content
