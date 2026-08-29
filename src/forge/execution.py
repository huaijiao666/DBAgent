"""Bounded subprocess execution inside a canonical workspace."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from forge.workspace import Workspace

DEFAULT_COMMAND_TIMEOUT_SECONDS = 60.0
MAX_COMMAND_TIMEOUT_SECONDS = 300.0
MAX_STREAM_CHARACTERS = 20_000

_ALLOWED_ENVIRONMENT_VARIABLES = frozenset(
    {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
)
_DENIED_EXECUTABLES = frozenset(
    {
        "bash",
        "cmd",
        "cmd.exe",
        "del",
        "diskpart",
        "format",
        "powershell",
        "powershell.exe",
        "pwsh",
        "pwsh.exe",
        "rm",
        "rmdir",
        "sh",
        "shutdown",
        "zsh",
    }
)
_DENIED_GIT_SUBCOMMANDS = frozenset(
    {
        "add",
        "am",
        "branch",
        "checkout",
        "clean",
        "commit",
        "merge",
        "mv",
        "push",
        "rebase",
        "reset",
        "restore",
        "rm",
        "stash",
        "switch",
        "tag",
    }
)


@dataclass(frozen=True, slots=True)
class CommandResult:
    """Structured result returned to the model for one subprocess."""

    command: tuple[str, ...]
    cwd: str
    return_code: int | None
    timed_out: bool
    stdout: str
    stderr: str
    stdout_truncated: bool
    stderr_truncated: bool

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["command"] = list(self.command)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False)


class CommandExecutor:
    """Run argument-vector commands without a shell or inherited secrets."""

    def __init__(
        self,
        workspace: Workspace,
        *,
        max_stream_characters: int = MAX_STREAM_CHARACTERS,
    ) -> None:
        if max_stream_characters <= 0:
            raise ValueError("max_stream_characters must be positive")
        self._workspace = workspace
        self._max_stream_characters = max_stream_characters

    def run(
        self,
        command: Sequence[str],
        *,
        cwd: str,
        timeout_seconds: float = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    ) -> CommandResult:
        arguments = self._validate_command(command)
        working_directory = self._workspace.resolve_directory(cwd)
        timeout = float(timeout_seconds)
        if timeout <= 0 or timeout > MAX_COMMAND_TIMEOUT_SECONDS:
            raise ValueError(
                "timeout_seconds must be greater than 0 and no greater than "
                f"{MAX_COMMAND_TIMEOUT_SECONDS:g}"
            )

        environment = _sanitized_environment()
        # Git normally walks from cwd through every parent directory looking
        # for .git. A workspace can itself be nested in another checkout, so
        # stop discovery at the workspace's parent to keep git observations
        # scoped to this workspace. A repository rooted at workspace is still
        # discovered before Git reaches the ceiling.
        environment["GIT_CEILING_DIRECTORIES"] = str(self._workspace.root.parent)

        try:
            completed = subprocess.run(
                arguments,
                cwd=working_directory,
                env=environment,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                shell=False,
                check=False,
            )
        except subprocess.TimeoutExpired as error:
            stdout, stdout_truncated = _truncate_stream(
                _timeout_text(error.stdout), self._max_stream_characters
            )
            stderr, stderr_truncated = _truncate_stream(
                _timeout_text(error.stderr), self._max_stream_characters
            )
            return CommandResult(
                command=arguments,
                cwd=self._workspace.relative_name(working_directory),
                return_code=None,
                timed_out=True,
                stdout=stdout,
                stderr=stderr,
                stdout_truncated=stdout_truncated,
                stderr_truncated=stderr_truncated,
            )

        stdout, stdout_truncated = _truncate_stream(
            completed.stdout, self._max_stream_characters
        )
        stderr, stderr_truncated = _truncate_stream(
            completed.stderr, self._max_stream_characters
        )
        return CommandResult(
            command=arguments,
            cwd=self._workspace.relative_name(working_directory),
            return_code=completed.returncode,
            timed_out=False,
            stdout=stdout,
            stderr=stderr,
            stdout_truncated=stdout_truncated,
            stderr_truncated=stderr_truncated,
        )

    def _validate_command(self, command: Sequence[str]) -> tuple[str, ...]:
        if isinstance(command, (str, bytes)) or not command:
            raise ValueError("command must be a non-empty array of strings")
        if len(command) > 100:
            raise ValueError("command contains too many arguments")
        if any(not isinstance(part, str) or not part for part in command):
            raise ValueError("every command argument must be a non-empty string")
        if any("\x00" in part or "\n" in part or "\r" in part for part in command):
            raise ValueError("command arguments must not contain control characters")

        arguments = tuple(command)
        executable = Path(arguments[0]).name.casefold()
        if executable in _DENIED_EXECUTABLES:
            raise PermissionError(f"command is blocked by policy: {arguments[0]}")
        if executable in {"git", "git.exe"} and len(arguments) > 1:
            if arguments[1].casefold() in _DENIED_GIT_SUBCOMMANDS:
                raise PermissionError(
                    f"git subcommand is blocked by policy: {arguments[1]}"
                )
        for argument in arguments[1:]:
            candidate = Path(argument)
            if candidate.is_absolute() and not self._workspace.contains(candidate):
                raise PermissionError(
                    f"absolute command argument escapes workspace: {argument}"
                )
        if (
            executable in {"python", "python.exe"}
            and Path(arguments[0]).name == arguments[0]
        ):
            arguments = (sys.executable, *arguments[1:])
        return arguments


def _sanitized_environment(
    source: Mapping[str, str] | None = None,
) -> dict[str, str]:
    environment = os.environ if source is None else source
    sanitized = {
        name: environment[name]
        for name in _ALLOWED_ENVIRONMENT_VARIABLES
        if name in environment
    }
    # Preserve the active virtual environment's executable directory. Resolving
    # sys.executable on Windows can follow the venv launcher to the base Python.
    runtime_directory = str(Path(sys.executable).parent)
    existing_path = sanitized.get("PATH", "")
    sanitized["PATH"] = (
        runtime_directory
        if not existing_path
        else runtime_directory + os.pathsep + existing_path
    )
    sanitized["PYTHONIOENCODING"] = "utf-8"
    sanitized["PYTHONUTF8"] = "1"
    sanitized["GIT_CONFIG_NOSYSTEM"] = "1"
    sanitized["GIT_CONFIG_GLOBAL"] = os.devnull
    sanitized["GIT_OPTIONAL_LOCKS"] = "0"
    sanitized["GIT_PAGER"] = "cat"
    return sanitized


def _truncate_stream(text: str, limit: int) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    marker = "\n...[truncated]...\n"
    if len(marker) >= limit:
        return marker[:limit], True
    available = limit - len(marker)
    head_length = available // 2
    tail_length = available - head_length
    return text[:head_length] + marker + text[-tail_length:], True


def _timeout_text(value: str | bytes | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value
