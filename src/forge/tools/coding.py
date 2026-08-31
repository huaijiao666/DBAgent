"""Patch, transitional write, command, and Git tools for coding tasks."""

from __future__ import annotations

import shlex
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from forge.execution import CommandExecutor
from forge.llm import FunctionTool
from forge.patching import PatchApplier
from forge.tools.models import ToolDefinition, ToolResult, object_schema
from forge.tools.readonly import create_readonly_registry
from forge.tools.registry import ToolRegistry
from forge.workspace import Workspace

_MAX_WRITE_CHARACTERS = 200_000


def create_coding_registry(workspace_root: Path) -> ToolRegistry:
    """Create all current local coding tools for one workspace."""

    workspace = Workspace(workspace_root)
    executor = CommandExecutor(workspace)
    patch_applier = PatchApplier(workspace)
    registry = create_readonly_registry(workspace.root)
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="apply_patch",
                description=(
                    "Apply exact, line-based hunks to one or more existing UTF-8 "
                    "workspace files. Each old_lines sequence must occur exactly once. "
                    "All files are validated before any change, and handled write "
                    "failures are rolled back. Prefer this over write_file."
                ),
                parameters=_patch_schema(),
            ),
            handler=lambda arguments: _apply_patch(patch_applier, arguments),
        )
    )
    registry.register(
        ToolDefinition(
            schema=FunctionTool(
                name="run_command",
                description=(
                    "Run one argument-vector command inside the workspace without a "
                    "shell. Prefer a JSON array of strings. A single command string "
                    "is accepted as a compatibility fallback and is parsed locally; "
                    "it never enables shell syntax. Returns stdout, stderr, return "
                    "code, timeout, and truncation metadata."
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
                    "Replace an entire existing UTF-8 workspace file. Use only when "
                    "apply_patch cannot express the change safely."
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


def _patch_schema() -> dict[str, Any]:
    line_array = {
        "type": "array",
        "items": {"type": "string"},
        "description": "Lines without newline characters.",
    }
    hunk = {
        "type": "object",
        "properties": {
            "old_lines": {
                **line_array,
                "minItems": 1,
                "description": (
                    "Exact consecutive lines that must occur once in the current file."
                ),
            },
            "new_lines": {
                **line_array,
                "description": "Replacement lines; use an empty array to delete.",
            },
        },
        "required": ["old_lines", "new_lines"],
        "additionalProperties": False,
    }
    file_patch = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to an existing UTF-8 file.",
            },
            "hunks": {
                "type": "array",
                "items": hunk,
                "minItems": 1,
            },
        },
        "required": ["path", "hunks"],
        "additionalProperties": False,
    }
    return object_schema(
        {
            "files": {
                "type": "array",
                "items": file_patch,
                "minItems": 1,
                "description": "All file patches in one atomic operation.",
            }
        },
        required=["files"],
    )


def _run_command(
    executor: CommandExecutor, arguments: Mapping[str, Any]
) -> dict[str, object]:
    command = _command_arguments(arguments.get("command"))
    cwd = _required_string(arguments, "cwd")
    timeout = arguments.get("timeout_seconds")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise ValueError("timeout_seconds must be a number")
    return executor.run(command, cwd=cwd, timeout_seconds=timeout).to_dict()


def _command_arguments(value: Any) -> list[str]:
    """Normalize an argv vector without ever invoking a shell.

    The function-call schema continues to prefer an explicit JSON array. The
    string fallback helps OpenAI-compatible providers that occasionally emit a
    command as text despite the schema; shell operators remain ordinary argv
    tokens and therefore cannot alter command execution semantics.
    """

    if isinstance(value, str):
        try:
            command = shlex.split(value, posix=True)
        except ValueError as error:
            raise ValueError(
                "command string could not be parsed; use an array of strings"
            ) from error
    elif not isinstance(value, (bytes, bytearray)) and isinstance(value, Sequence):
        command = list(value)
    else:
        raise ValueError("command must be an array of strings or one command string")
    if not command or any(not isinstance(part, str) or not part for part in command):
        raise ValueError("command must be an array of non-empty strings")
    return command


def _apply_patch(
    patch_applier: PatchApplier, arguments: Mapping[str, Any]
) -> ToolResult:
    result = patch_applier.apply(arguments.get("files"))
    return ToolResult(success=bool(result["applied"]), content=result)


def _create_file(workspace: Workspace, arguments: Mapping[str, Any]) -> dict[str, object]:
    path = workspace.resolve_for_create(_required_string(arguments, "path"))
    content = _content(arguments)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Re-resolve after directory creation to close the obvious symlink boundary
    # check before opening the target. The final ``x`` mode still prevents an
    # existing file from being overwritten.
    workspace.resolve_directory(str(path.parent))
    with path.open("x", encoding="utf-8", newline="") as file:
        file.write(content)
    relative_path = workspace.relative_name(path)
    return {
        "action": "created",
        "path": relative_path,
        "changed_files": [relative_path],
        "characters": len(content),
    }


def _write_file(workspace: Workspace, arguments: Mapping[str, Any]) -> dict[str, object]:
    path = workspace.resolve(_required_string(arguments, "path"))
    if not path.is_file():
        raise ValueError("path is not a file")
    content = _content(arguments)
    with path.open("w", encoding="utf-8", newline="") as file:
        file.write(content)
    relative_path = workspace.relative_name(path)
    return {
        "action": "wrote",
        "path": relative_path,
        "changed_files": [relative_path],
        "characters": len(content),
    }


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
