from types import SimpleNamespace

from dbagent.agent import (
    PlanStep,
    PlanStepStatus,
    SessionContext,
    SessionObservation,
    TaskPlan,
)
from dbagent.agent.verification import VerificationRecord, VerificationStatus
from dbagent.tools import ToolObservation


def test_session_context_keeps_plan_verification_and_key_observations() -> None:
    plan = TaskPlan(
        goal="Fix parser",
        success_criteria=("pytest passes",),
        steps=(PlanStep("test", "Run tests", PlanStepStatus.IN_PROGRESS),),
    )
    record = VerificationRecord(
        command=("python", "-m", "pytest", "-q"),
        cwd=".",
        kind="test",
        return_code=1,
        timed_out=False,
        passed=False,
        mutation_generation=1,
        stdout="1 failed",
        stderr="AssertionError: wrong result",
    )
    state = SimpleNamespace(
        plan=plan,
        verification_status=VerificationStatus.FAILED,
        latest_verification=record,
        observations=[
            ToolObservation(
                call_id="patch",
                tool_name="apply_patch",
                success=True,
                content={
                    "applied": True,
                    "changed_files": ["src/parser.py"],
                    "hunks_applied": 1,
                    "failure_reason": None,
                },
            ),
            ToolObservation(
                call_id="test",
                tool_name="run_command",
                success=True,
                content={
                    "command": ["python", "-m", "pytest", "-q"],
                    "cwd": ".",
                    "return_code": 1,
                    "timed_out": False,
                    "stdout": "1 failed",
                    "stderr": "AssertionError: wrong result",
                },
            ),
        ],
    )

    context = SessionContext()
    context.update_from_state(state)

    rendered = context.render()
    prompt = context.augment_prompt("Fix the failing assertion.")
    assert context.turns == 1
    assert "[Plan]" in rendered
    assert "goal: Fix parser" in rendered
    assert "status=failed" in rendered
    assert "return_code=1" in rendered
    assert "src/parser.py" in rendered
    assert "Persistent structured context" in prompt
    assert "[Current turn request]" in prompt


def test_session_context_compacts_routine_observations_first() -> None:
    context = SessionContext(max_characters=8_000, max_observations=2)
    state = SimpleNamespace(
        plan=None,
        verification_status=VerificationStatus.NOT_RUN,
        latest_verification=None,
        observations=[
            ToolObservation("read", "read_file", True, "routine source"),
            ToolObservation(
                "patch",
                "apply_patch",
                True,
                {"applied": True, "changed_files": ["parser.py"], "hunks_applied": 1},
            ),
            ToolObservation(
                "test",
                "run_command",
                True,
                {"command": ["pytest"], "return_code": 0, "stdout": "2 passed"},
            ),
        ],
    )

    context.update_from_state(state)

    assert len(context.observations) == 2
    rendered = context.render()
    assert "routine source" not in rendered
    assert "parser.py" in rendered
    assert "2 passed" in rendered


def test_session_context_clear_removes_structured_state() -> None:
    context = SessionContext()
    context.observations.append(
        # Direct insertion keeps this test focused on reset semantics.
        SessionObservation(1, "run_command", True, "return_code=0", True)
    )
    context.verification_status = "passed"
    context.turns = 1

    context.clear()

    assert context.render() == ""
    assert context.turns == 0
    assert context.observations == []
    assert context.verification_status == "not_run"


def test_session_context_preserves_passed_evidence_until_a_mutation() -> None:
    context = SessionContext()
    passed = VerificationRecord(
        command=("pytest",),
        cwd=".",
        kind="test",
        return_code=0,
        timed_out=False,
        passed=True,
        mutation_generation=0,
        stdout="2 passed",
        stderr="",
    )
    context.update_from_state(
        SimpleNamespace(
            plan=None,
            verification_status=VerificationStatus.PASSED,
            latest_verification=passed,
            observations=[],
        )
    )
    context.update_from_state(
        SimpleNamespace(
            plan=None,
            verification_status=VerificationStatus.NOT_RUN,
            latest_verification=None,
            observations=[],
        )
    )
    assert context.verification_status == "passed"

    context.update_from_state(
        SimpleNamespace(
            plan=None,
            verification_status=VerificationStatus.NOT_RUN,
            latest_verification=None,
            observations=[
                ToolObservation(
                    "patch",
                    "apply_patch",
                    True,
                    {"applied": True, "changed_files": ["parser.py"]},
                )
            ],
        )
    )
    assert context.verification_status == "stale"


def test_session_context_round_trips_through_validated_json_shape() -> None:
    context = SessionContext()
    context.plan = TaskPlan(
        goal="Finish snake game",
        success_criteria=("pytest passes",),
        steps=(PlanStep("verify", "Run tests", PlanStepStatus.IN_PROGRESS),),
    )
    context.verification_status = "stale"
    context.verification_summary = "code changed after the last test"
    context.recovery_hints = ["run pytest"]
    context.observations = [
        SessionObservation(2, "apply_patch", True, "game.py changed", True)
    ]
    context.turns = 2

    restored = SessionContext.from_dict(context.to_dict())

    assert restored.to_dict() == context.to_dict()
    assert restored.plan is not None
    assert restored.plan.goal == "Finish snake game"
