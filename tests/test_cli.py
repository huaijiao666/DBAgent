from types import SimpleNamespace
from unittest.mock import patch

from forge.agent import AgentStatus, PlanStep, PlanStepStatus, TaskPlan
from forge.agent.verification import VerificationStatus
from forge.cli import main


def test_cli_prints_completed_answer(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    completed = SimpleNamespace(
        status=AgentStatus.COMPLETED,
        final_answer="Architecture summary",
        max_steps=12,
    )

    with (
        patch("forge.cli.OpenAIResponsesClient"),
        patch("forge.cli.create_coding_registry"),
        patch("forge.cli.AgentLoop") as loop_type,
    ):
        loop_type.return_value.run.return_value = completed
        exit_code = main(["inspect", "--workspace", str(tmp_path)])

    assert exit_code == 0
    assert capsys.readouterr().out == "Architecture summary\n"


def test_cli_reports_max_step_termination(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    stopped = SimpleNamespace(
        status=AgentStatus.MAX_STEPS,
        final_answer=None,
        max_steps=2,
    )

    with (
        patch("forge.cli.OpenAIResponsesClient"),
        patch("forge.cli.create_coding_registry"),
        patch("forge.cli.AgentLoop") as loop_type,
    ):
        loop_type.return_value.run.return_value = stopped
        exit_code = main(
            ["inspect", "--workspace", str(tmp_path), "--max-steps", "2"]
        )

    assert exit_code == 2
    assert "max_steps=2" in capsys.readouterr().err


def test_cli_displays_plan_status_updates(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    plan = TaskPlan(
        goal="Inspect the repository",
        success_criteria=("Explain the architecture",),
        steps=(
            PlanStep("inspect", "Inspect files", PlanStepStatus.COMPLETED),
            PlanStep("explain", "Explain structure", PlanStepStatus.IN_PROGRESS),
        ),
    )
    completed = SimpleNamespace(
        status=AgentStatus.COMPLETED,
        final_answer="Architecture summary",
        max_steps=12,
        plan_history=[plan],
        verification_status=VerificationStatus.PASSED,
    )

    with (
        patch("forge.cli.OpenAIResponsesClient"),
        patch("forge.cli.create_coding_registry"),
        patch("forge.cli.AgentLoop") as loop_type,
    ):
        loop_type.return_value.run.return_value = completed
        exit_code = main(["inspect", "--workspace", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Architecture summary\n"
    assert "Plan status updates:" in captured.err
    assert "[completed] inspect" in captured.err
    assert "[in_progress] explain" in captured.err
    assert "VERIFIED" in captured.err
