"""Explicit mutable state for one minimal agent run."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from forge.agent.context import ContextUsage
from forge.llm import FunctionCall
from forge.tools import ToolObservation


class AgentStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    MAX_STEPS = "max_steps"


@dataclass(slots=True)
class AgentState:
    """All evolving data needed to inspect and explain one agent run."""

    task: str
    workspace: Path
    max_steps: int
    context: list[dict[str, Any]]
    step: int = 0
    status: AgentStatus = AgentStatus.RUNNING
    response_ids: list[str] = field(default_factory=list)
    tool_calls: list[FunctionCall] = field(default_factory=list)
    observations: list[ToolObservation] = field(default_factory=list)
    context_usage: list[ContextUsage] = field(default_factory=list)
    final_answer: str | None = None

    @classmethod
    def start(cls, *, task: str, workspace: Path, max_steps: int) -> AgentState:
        if not task.strip():
            raise ValueError("task must not be empty")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        return cls(
            task=task,
            workspace=workspace.resolve(),
            max_steps=max_steps,
            context=[{"role": "user", "content": task}],
        )
