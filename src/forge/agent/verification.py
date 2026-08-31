"""Deterministic verification evidence and recovery signals."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from forge.llm import FunctionCall
from forge.tools.models import ToolObservation


class VerificationStatus(str, Enum):
    """Status of the newest deterministic evidence for the current files."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    STALE = "stale"


@dataclass(frozen=True, slots=True)
class VerificationRecord:
    """One recognized test, compiler, or linter invocation."""

    command: tuple[str, ...]
    cwd: str
    kind: str
    return_code: int | None
    timed_out: bool
    passed: bool
    mutation_generation: int
    stdout: str
    stderr: str
    failure_signature: str | None = None

    @property
    def status(self) -> VerificationStatus:
        return (
            VerificationStatus.PASSED
            if self.passed
            else VerificationStatus.FAILED
        )

    def summary(self, *, current_generation: int) -> str:
        status = self.status.value
        if self.mutation_generation != current_generation:
            status = VerificationStatus.STALE.value
        command = " ".join(self.command)
        return (
            f"status={status}; kind={self.kind}; command={command}; cwd={self.cwd}; "
            f"return_code={self.return_code}; timed_out={self.timed_out}; "
            f"stdout={_excerpt(self.stdout)}; stderr={_excerpt(self.stderr)}"
        )


@dataclass(frozen=True, slots=True)
class VerificationEvent:
    """Facts produced when one local tool observation is classified."""

    mutation: bool = False
    record: VerificationRecord | None = None
    repeated_failure_count: int = 0


@dataclass(slots=True)
class VerificationTracker:
    """Track evidence freshness, repeated failures, and code generations."""

    mutation_generation: int = 0
    latest: VerificationRecord | None = None
    history: list[VerificationRecord] = field(default_factory=list)
    repeated_failure_count: int = 0
    _last_failure_signature: str | None = None

    def observe(
        self, call: FunctionCall, observation: ToolObservation
    ) -> VerificationEvent:
        mutation = self._observe_mutation(call, observation)
        if call.name != "run_command" or not observation.success:
            return VerificationEvent(mutation=mutation)
        content = observation.content
        if not isinstance(content, Mapping):
            return VerificationEvent(mutation=mutation)
        command = _command_tuple(content.get("command"))
        kind = classify_verification_command(command)
        if kind is None:
            self.repeated_failure_count = 0
            self._last_failure_signature = None
            return VerificationEvent(mutation=mutation)

        return_code = _optional_int(content.get("return_code"))
        timed_out = bool(content.get("timed_out", False))
        passed = return_code == 0 and not timed_out
        stdout = str(content.get("stdout", ""))
        stderr = str(content.get("stderr", ""))
        failure_signature = (
            _failure_signature(command, content)
            if not passed
            else None
        )
        if failure_signature and failure_signature == self._last_failure_signature:
            self.repeated_failure_count += 1
        elif failure_signature:
            self.repeated_failure_count = 1
        else:
            self.repeated_failure_count = 0
        self._last_failure_signature = failure_signature

        record = VerificationRecord(
            command=command,
            cwd=str(content.get("cwd", ".")),
            kind=kind,
            return_code=return_code,
            timed_out=timed_out,
            passed=passed,
            mutation_generation=self.mutation_generation,
            stdout=stdout,
            stderr=stderr,
            failure_signature=failure_signature,
        )
        self.latest = record
        self.history.append(record)
        return VerificationEvent(
            mutation=mutation,
            record=record,
            repeated_failure_count=self.repeated_failure_count,
        )

    @property
    def status(self) -> VerificationStatus:
        if self.latest is None:
            return VerificationStatus.NOT_RUN
        if self.latest.mutation_generation != self.mutation_generation:
            return VerificationStatus.STALE
        return self.latest.status

    @property
    def requires_verification(self) -> bool:
        return self.mutation_generation > 0 or self.latest is not None

    @property
    def is_verified(self) -> bool:
        return (
            self.latest is not None
            and self.latest.passed
            and self.latest.mutation_generation == self.mutation_generation
        )

    def latest_summary(self) -> str:
        if self.latest is None:
            return "status=not_run; no deterministic test, compiler, or linter evidence"
        return self.latest.summary(current_generation=self.mutation_generation)

    def _observe_mutation(
        self, call: FunctionCall, observation: ToolObservation
    ) -> bool:
        if not observation.success or call.name not in {
            "apply_patch",
            "create_file",
            "write_file",
        }:
            return False
        if call.name == "apply_patch" and isinstance(observation.content, Mapping):
            if not bool(observation.content.get("applied")):
                return False
            changed_files = observation.content.get("changed_files")
            if isinstance(changed_files, Sequence) and not changed_files:
                return False
        self.mutation_generation += 1
        return True


def classify_verification_command(command: Sequence[str]) -> str | None:
    """Return a stable category for common local test/build checks."""

    if not command:
        return None
    executable = Path(command[0]).name.casefold()
    arguments = tuple(part.casefold() for part in command[1:])
    module = _python_module(arguments)
    first_argument = arguments[0] if arguments else ""

    if (
        executable in {"pytest", "pytest.exe", "py.test", "py.test.exe"}
        or module in {"pytest", "py.test", "unittest"}
        or (
            executable in {"node", "node.exe"}
            and (
                "--test" in arguments
                or any(_is_node_test_file(argument) for argument in arguments)
            )
        )
        or (executable in {"npm", "pnpm", "yarn"} and first_argument == "test")
        or (executable in {"cargo", "go"} and first_argument == "test")
    ):
        return "test"
    if (
        module == "compileall"
        or (executable in {"node", "node.exe"} and "--check" in arguments)
        or executable in {"mypy", "mypy.exe", "pyright", "pyright.exe", "tsc", "tsc.exe"}
        or (executable == "cargo" and first_argument == "check")
        or (executable == "go" and first_argument == "vet")
        or (
            executable in {"mvn", "mvn.cmd", "gradle", "gradlew", "gradlew.bat"}
            and first_argument == "compile"
        )
    ):
        return "compiler"
    if executable in {
        "ruff",
        "ruff.exe",
        "flake8",
        "flake8.exe",
        "pylint",
        "pylint.exe",
    } or (
        executable in {"black", "black.exe", "isort", "isort.exe"}
        and "--check" in arguments
    ):
        return "lint"
    return None


def _python_module(arguments: Sequence[str]) -> str | None:
    """Return the module passed to ``python -m`` without inspecting paths."""

    try:
        index = arguments.index("-m")
    except ValueError:
        return None
    return arguments[index + 1] if index + 1 < len(arguments) else None


def _is_node_test_file(argument: str) -> bool:
    """Recognize conventional deterministic JavaScript test file names."""

    filename = Path(argument).name
    return filename in {"test.js", "tests.js", "test.mjs", "tests.mjs"} or (
        filename.endswith((".test.js", ".spec.js", ".test.mjs", ".spec.mjs"))
    )


def _command_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(part) for part in value)
    return ()


def _optional_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _failure_signature(command: Sequence[str], content: Mapping[str, Any]) -> str:
    stdout = _normalize_failure_text(str(content.get("stdout", "")))
    stderr = _normalize_failure_text(str(content.get("stderr", "")))
    payload = {
        "command": list(command),
        "cwd": content.get("cwd", "."),
        "return_code": content.get("return_code"),
        "timed_out": bool(content.get("timed_out", False)),
        "stdout": stdout[-1_000:],
        "stderr": stderr[-1_000:],
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _normalize_failure_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _excerpt(value: str, limit: int = 260) -> str:
    normalized = _normalize_failure_text(value)
    if len(normalized) <= limit:
        return normalized
    return normalized[: limit // 2] + " ... " + normalized[-limit // 2 :]
