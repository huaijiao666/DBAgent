"""Explicit task plans and the local update_plan tool."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any

from dbagent.llm import FunctionTool
from dbagent.tools.models import ToolDefinition, ToolResult, object_schema


class PlanStepStatus(str, Enum):
    """Lifecycle states for one actionable plan step."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class PlanStep:
    """One named step whose status can be updated by the model."""

    step_id: str
    description: str
    status: PlanStepStatus

    def to_dict(self) -> dict[str, str]:
        return {
            "id": self.step_id,
            "description": self.description,
            "status": self.status.value,
        }


@dataclass(frozen=True, slots=True)
class TaskPlan:
    """The model's persisted goal, criteria, and ordered work steps."""

    goal: str
    success_criteria: tuple[str, ...]
    steps: tuple[PlanStep, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> TaskPlan:
        if not isinstance(value, Mapping):
            raise ValueError("plan must be a JSON object")
        goal = _required_text(value, "goal")
        criteria = _required_text_list(value, "success_criteria")
        raw_steps = value.get("steps")
        if not isinstance(raw_steps, Sequence) or isinstance(
            raw_steps, (str, bytes)
        ):
            raise ValueError("steps must be a non-empty array")
        if not raw_steps:
            raise ValueError("steps must be a non-empty array")

        steps: list[PlanStep] = []
        seen_ids: set[str] = set()
        for raw_step in raw_steps:
            if not isinstance(raw_step, Mapping):
                raise ValueError("each step must be an object")
            step_id = _required_text(raw_step, "id")
            if step_id in seen_ids:
                raise ValueError(f"duplicate step id: {step_id}")
            seen_ids.add(step_id)
            description = _required_text(raw_step, "description")
            raw_status = raw_step.get("status")
            try:
                status = PlanStepStatus(raw_status)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid status for step {step_id}: {raw_status!r}"
                ) from error
            steps.append(PlanStep(step_id, description, status))
        return cls(goal, tuple(criteria), tuple(steps))

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "success_criteria": list(self.success_criteria),
            "steps": [step.to_dict() for step in self.steps],
        }

    def to_prompt(self) -> str:
        lines = [f"goal: {self.goal}", "success criteria:"]
        lines.extend(f"- {criterion}" for criterion in self.success_criteria)
        lines.append("steps:")
        lines.extend(
            f"- [{step.status.value}] {step.step_id}: {step.description}"
            for step in self.steps
        )
        return "\n".join(lines)

    @property
    def is_complete(self) -> bool:
        return all(step.status is PlanStepStatus.COMPLETED for step in self.steps)


@dataclass(slots=True)
class PlanStore:
    """Validate and retain plan updates for one AgentLoop run."""

    _plan: TaskPlan | None = None
    _history: list[TaskPlan] = field(default_factory=list)

    @classmethod
    def resume(cls, plan: TaskPlan | None) -> PlanStore:
        """Start a new loop with one explicit, unfinished session plan."""

        return cls(
            _plan=plan,
            _history=[plan] if plan is not None else [],
        )

    @property
    def plan(self) -> TaskPlan | None:
        return self._plan

    @property
    def history(self) -> tuple[TaskPlan, ...]:
        return tuple(self._history)

    def apply(self, arguments: Mapping[str, Any]) -> ToolResult:
        """Apply a complete plan snapshot or return a structured error."""

        try:
            candidate = TaskPlan.from_mapping(arguments)
            if self._plan is not None:
                self._validate_update(candidate)
            else:
                self._validate_initial(candidate)
        except ValueError as error:
            return ToolResult(
                success=False,
                content={
                    "error": str(error),
                    "current_plan": (
                        self._plan.to_dict() if self._plan is not None else None
                    ),
                },
            )

        changed = candidate != self._plan
        self._plan = candidate
        if changed:
            self._history.append(candidate)
        return ToolResult(
            success=True,
            content={
                "updated": changed,
                "changed": changed,
                "plan": candidate.to_dict(),
                "update_number": len(self._history),
            },
        )

    def advance(self, statuses: Mapping[str, PlanStepStatus]) -> bool:
        """Advance named steps from local runtime evidence.

        A provider may omit ``update_plan`` during a long tool run. The runtime
        still needs a truthful visible plan, so this method permits only the
        same forward-only transitions validated for model-authored updates.
        """

        if self._plan is None or not statuses:
            return False
        steps = tuple(
            replace(step, status=statuses[step.step_id])
            if step.step_id in statuses and step.status is not statuses[step.step_id]
            else step
            for step in self._plan.steps
        )
        candidate = TaskPlan(
            goal=self._plan.goal,
            success_criteria=self._plan.success_criteria,
            steps=steps,
        )
        if candidate == self._plan:
            return False
        self._validate_update(candidate)
        self._plan = candidate
        self._history.append(candidate)
        return True

    def _validate_update(self, candidate: TaskPlan) -> None:
        assert self._plan is not None
        if candidate.goal != self._plan.goal:
            raise ValueError("plan goal cannot change after the initial plan")
        if candidate.success_criteria[: len(self._plan.success_criteria)] != (
            self._plan.success_criteria
        ):
            raise ValueError(
                "existing success criteria cannot be removed or changed"
            )

        previous = {step.step_id: step for step in self._plan.steps}
        current = {step.step_id: step for step in candidate.steps}
        missing = sorted(set(previous) - set(current))
        if missing:
            raise ValueError(
                "existing plan steps cannot be removed: " + ", ".join(missing)
            )
        for step_id, old_step in previous.items():
            self._validate_transition(old_step, current[step_id])
        for step_id, new_step in current.items():
            if step_id not in previous and new_step.status is not PlanStepStatus.PENDING:
                raise ValueError(
                    f"new step {step_id} must start with pending status"
                )

    @staticmethod
    def _validate_initial(candidate: TaskPlan) -> None:
        """Reject a plan that claims completion before any local evidence exists."""

        statuses = [step.status for step in candidate.steps]
        if PlanStepStatus.COMPLETED in statuses:
            raise ValueError("an initial plan cannot contain completed steps")
        if statuses.count(PlanStepStatus.IN_PROGRESS) != 1:
            raise ValueError(
                "an initial plan must have exactly one in_progress step"
            )

    @staticmethod
    def _validate_transition(old: PlanStep, new: PlanStep) -> None:
        if old.status is new.status:
            return
        allowed = {
            PlanStepStatus.PENDING: {
                PlanStepStatus.IN_PROGRESS,
                PlanStepStatus.COMPLETED,
                PlanStepStatus.BLOCKED,
            },
            PlanStepStatus.IN_PROGRESS: {
                PlanStepStatus.COMPLETED,
                PlanStepStatus.BLOCKED,
            },
            PlanStepStatus.BLOCKED: {PlanStepStatus.IN_PROGRESS},
            PlanStepStatus.COMPLETED: set(),
        }
        if new.status not in allowed[old.status]:
            raise ValueError(
                f"invalid status transition for step {old.step_id}: "
                f"{old.status.value} -> {new.status.value}"
            )


def runtime_code_plan(task: str, *, chinese: bool) -> TaskPlan:
    """Create the provider-failure fallback, never a task interpretation.

    The normal CODE path obtains a semantic plan through the working model's
    native ``update_plan`` call. This fallback exists solely for a provider
    that repeatedly refuses that protocol. It intentionally has no lexical
    task classifier: local runtime code must not pretend to understand the
    user's product request.
    """

    if chinese:
        criteria = (
            "请求的核心行为已落实为工作区中的可运行代码。",
            "已运行与当前改动匹配的确定性检查，并记录结果。",
        )
        steps = (
            PlanStep("inspect", "检查相关实现、约束和可用验证方式", PlanStepStatus.IN_PROGRESS),
            PlanStep("implement", "完成必要的实现或修复", PlanStepStatus.PENDING),
            PlanStep("verify", "运行针对性的确定性验证", PlanStepStatus.PENDING),
            PlanStep("deliver", "整理变更、运行方式和验证结论", PlanStepStatus.PENDING),
        )
    else:
        criteria = (
            "The requested behavior exists as runnable workspace code.",
            "A deterministic check appropriate to the current change has run.",
        )
        steps = (
            PlanStep("inspect", "Inspect relevant implementation, constraints, and verification", PlanStepStatus.IN_PROGRESS),
            PlanStep("implement", "Complete the necessary implementation or fix", PlanStepStatus.PENDING),
            PlanStep("verify", "Run targeted deterministic verification", PlanStepStatus.PENDING),
            PlanStep("deliver", "Summarize changes, run instructions, and evidence", PlanStepStatus.PENDING),
        )
    return TaskPlan(goal=_plan_goal(task, chinese=chinese), success_criteria=criteria, steps=steps)


def _plan_goal(task: str, *, chinese: bool) -> str:
    """Use the request's first sentence as a compact, generic plan title."""

    compact = " ".join(task.split())
    first_sentence = re.split(r"[。！？.!?]", compact, maxsplit=1)[0].strip()
    candidate = first_sentence or compact
    limit = 72 if chinese else 96
    return candidate if len(candidate) <= limit else candidate[: limit - 1].rstrip() + "…"


def update_plan_tool(store: PlanStore) -> ToolDefinition:
    """Create a model-visible plan tool bound to one run-local store."""

    step_schema = {
        "type": "object",
        "properties": {
            "id": {"type": "string", "description": "Stable step identifier."},
            "description": {
                "type": "string",
                "description": "Action to perform for this step.",
            },
            "status": {
                "type": "string",
                "enum": [status.value for status in PlanStepStatus],
            },
        },
        "required": ["id", "description", "status"],
        "additionalProperties": False,
    }
    return ToolDefinition(
        schema=FunctionTool(
            name="update_plan",
            description=(
                "Create or update the task plan. Call once near task start with "
                "the complete goal, observable success criteria, and ordered steps. "
                "On the first call, set exactly one step to in_progress and no step "
                "to completed. Later calls may append evidence-based success criteria "
                "but may not remove or rewrite existing criteria. Update it only when "
                "a milestone or step status changes; do not replan for every ordinary "
                "tool call. Preserve existing step IDs."
            ),
            parameters=object_schema(
                {
                    "goal": {
                        "type": "string",
                        "description": "The stable objective for this task.",
                    },
                    "success_criteria": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "description": "Observable conditions proving the goal is done.",
                    },
                    "steps": {
                        "type": "array",
                        "items": step_schema,
                        "minItems": 1,
                        "description": "Ordered steps and their current statuses.",
                    },
                },
                required=["goal", "success_criteria", "steps"],
            ),
        ),
        handler=store.apply,
    )


def _required_text(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item.strip():
        raise ValueError(f"{key} must be a non-empty string")
    return item.strip()


def _required_text_list(value: Mapping[str, Any], key: str) -> list[str]:
    items = value.get(key)
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError(f"{key} must be a non-empty array")
    if not items:
        raise ValueError(f"{key} must be a non-empty array")
    result: list[str] = []
    for item in items:
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{key} must contain non-empty strings")
        result.append(item.strip())
    return result
