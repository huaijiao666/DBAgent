"""Model-authored semantic routing for an automatic DBAgent turn.

The runtime deliberately does not infer whether a natural-language request is
an investigation or a coding task from a list of verbs.  In automatic mode the
same model makes that semantic choice through a native function call, while the
local handler validates and records the small piece of state it is allowed to
set.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from dbagent.agent.mode import TaskMode
from dbagent.llm import FunctionTool
from dbagent.tools.models import ToolDefinition, ToolResult, object_schema


@dataclass(frozen=True, slots=True)
class TaskModeDecision:
    """The model's one-time selection for an automatic user turn."""

    mode: TaskMode
    reason: str


@dataclass(slots=True)
class TaskModeStore:
    """Validate and retain a semantic ASK/CODE decision for one run."""

    _decision: TaskModeDecision | None = None

    @property
    def decision(self) -> TaskModeDecision | None:
        return self._decision

    def apply(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Accept exactly one supported non-auto mode and a concise rationale."""

        raw_mode = arguments.get("mode")
        reason = arguments.get("reason")
        if raw_mode not in {TaskMode.ASK.value, TaskMode.CODE.value}:
            return ToolResult(
                success=False,
                content={
                    "error": "mode must be exactly 'ask' or 'code'",
                    "selected_mode": (
                        self._decision.mode.value if self._decision else None
                    ),
                },
            )
        if not isinstance(reason, str) or not reason.strip():
            return ToolResult(
                success=False,
                content={
                    "error": "reason must be a non-empty string",
                    "selected_mode": (
                        self._decision.mode.value if self._decision else None
                    ),
                },
            )
        concise_reason = " ".join(reason.split())
        if len(concise_reason) > 280:
            return ToolResult(
                success=False,
                content={
                    "error": "reason must be at most 280 characters",
                    "selected_mode": (
                        self._decision.mode.value if self._decision else None
                    ),
                },
            )
        decision = TaskModeDecision(TaskMode(raw_mode), concise_reason)
        if self._decision is not None and decision != self._decision:
            return ToolResult(
                success=False,
                content={
                    "error": "task mode cannot change after routing",
                    "selected_mode": self._decision.mode.value,
                },
            )
        changed = decision != self._decision
        self._decision = decision
        return ToolResult(
            success=True,
            content={
                "selected_mode": decision.mode.value,
                "reason": decision.reason,
                "changed": changed,
            },
        )


def select_task_mode_tool(store: TaskModeStore) -> ToolDefinition:
    """Return the run-local native tool used only before automatic-mode work."""

    return ToolDefinition(
        schema=FunctionTool(
            name="select_task_mode",
            description=(
                "Choose how to handle the current user request. Select ask for an "
                "explanation, inspection, review, or other read-only answer; select "
                "code only when the user is asking the agent to change or create local "
                "workspace files. Base the decision on the complete request, not keywords."
            ),
            parameters=object_schema(
                {
                    "mode": {
                        "type": "string",
                        "enum": [TaskMode.ASK.value, TaskMode.CODE.value],
                        "description": "The semantic authority needed for this turn.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "One concise reason based on the complete request.",
                    },
                },
                required=["mode", "reason"],
            ),
        ),
        handler=store.apply,
    )
