import json

from forge.agent import PlanStepStatus, PlanStore, update_plan_tool
from forge.llm import FunctionCall
from forge.tools import ToolRegistry


def _plan(
    *,
    goal: str = "Fix the bug",
    criteria: list[str] | None = None,
    statuses: tuple[str, ...] = ("pending", "pending"),
) -> dict:
    return {
        "goal": goal,
        "success_criteria": criteria or ["pytest passes"],
        "steps": [
            {
                "id": "inspect",
                "description": "Inspect the implementation",
                "status": statuses[0],
            },
            {
                "id": "verify",
                "description": "Run deterministic verification",
                "status": statuses[1],
            },
        ],
    }


def test_initial_plan_is_structured_and_persisted() -> None:
    store = PlanStore()

    result = store.apply(_plan())

    assert result.success is True
    assert store.plan is not None
    assert store.plan.goal == "Fix the bug"
    assert store.plan.success_criteria == ("pytest passes",)
    assert [step.status for step in store.plan.steps] == [
        PlanStepStatus.PENDING,
        PlanStepStatus.PENDING,
    ]
    assert len(store.history) == 1


def test_plan_step_status_transitions_are_explicit() -> None:
    store = PlanStore()
    assert store.apply(_plan()).success is True

    progressing = _plan(statuses=("in_progress", "pending"))
    completed = _plan(statuses=("completed", "in_progress"))
    finished = _plan(statuses=("completed", "completed"))

    assert store.apply(progressing).success is True
    assert store.apply(completed).success is True
    assert store.apply(finished).success is True
    assert [step.status.value for step in store.plan.steps] == [
        "completed",
        "completed",
    ]
    assert len(store.history) == 4


def test_invalid_transition_and_plan_drift_are_rejected_without_mutation() -> None:
    store = PlanStore()
    assert store.apply(_plan()).success is True

    direct_completion = store.apply(_plan(statuses=("completed", "pending")))
    assert direct_completion.success is True

    backwards = store.apply(_plan(statuses=("pending", "pending")))
    assert backwards.success is False
    assert "invalid status transition" in backwards.content["error"]
    assert store.plan.steps[0].status is PlanStepStatus.COMPLETED

    drift = store.apply(_plan(goal="A different goal", statuses=("completed", "pending")))
    assert drift.success is False
    assert "goal cannot change" in drift.content["error"]
    assert store.plan.goal == "Fix the bug"


def test_update_plan_is_a_local_function_tool() -> None:
    store = PlanStore()
    registry = ToolRegistry([update_plan_tool(store)])
    arguments = json.dumps(_plan())

    observation = registry.dispatch(
        FunctionCall(call_id="plan_1", name="update_plan", arguments_json=arguments)
    )

    assert observation.success is True
    assert observation.content["updated"] is True
    assert registry.schemas()[0].name == "update_plan"
