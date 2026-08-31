"""Explicit mutable state for one minimal agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from forge.agent.context import ContextUsage
from forge.agent.mode import TaskMode
from forge.agent.plan import TaskPlan
from forge.agent.verification import (
    VerificationRecord,
    VerificationStatus,
)
from forge.llm import FunctionCall
from forge.tools import ToolObservation


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"
    ERROR = "error"
    ABORTED = "aborted"
    BLOCKED = "blocked"


@dataclass(slots=True)
class AgentState:
    """All evolving data needed to inspect and explain one agent run."""

    task: str
    workspace: Path
    launch_directory: Path
    max_steps: int
    context: list[dict[str, Any]]
    mode: TaskMode = TaskMode.CODE
    step: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    response_ids: list[str] = field(default_factory=list)
    tool_calls: list[FunctionCall] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    context_usage: list[ContextUsage] = field(default_factory=list)
    plan: TaskPlan | None = None
    plan_history: list[TaskPlan] = field(default_factory=list)
    latest_verification: VerificationRecord | None = None
    verification_history: list[VerificationRecord] = field(default_factory=list)
    verification_status: VerificationStatus = VerificationStatus.NOT_RUN
    repeated_failure_count: int = 0
    no_progress_rounds: int = 0
    recovery_hints: list[str] = field(default_factory=list)
    final_answer: str | None = None

    @property
    def is_verified(self) -> bool:
        """Return whether current files have passing deterministic evidence."""

        return self.verification_status is VerificationStatus.PASSED

    @classmethod
    def start(
        cls,
        *,
        task: str,
        workspace: Path,
        launch_directory: Path | None = None,
        max_steps: int,
        mode: TaskMode = TaskMode.CODE,
    ) -> AgentState:
        if not task.strip():
            raise ValueError("task must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        resolved_workspace = workspace.resolve()
        resolved_launch_directory = (launch_directory or workspace).resolve()
        return cls(
            task=task,
            workspace=resolved_workspace,
            launch_directory=resolved_launch_directory,
            max_steps=max_steps,
            context=[{"role": "user", "content": task}],
            mode=mode,
        )
