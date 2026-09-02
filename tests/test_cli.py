from types import SimpleNamespace
from unittest.mock import patch

from dbagent.agent import AgentStatus, PlanStep, PlanStepStatus, TaskPlan
from dbagent.agent.verification import VerificationStatus
from dbagent.cli import main


def test_cli_prints_completed_answer(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    completed = SimpleNamespace(
        status=AgentStatus.COMPLETED,
        final_answer="Architecture summary",
        max_steps=12,
    )

    with (
        patch("dbagent.cli.OpenAIResponsesClient"),
        patch("dbagent.cli.create_coding_registry"),
        patch("dbagent.cli.AgentLoop") as loop_type,
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
        patch("dbagent.cli.OpenAIResponsesClient"),
        patch("dbagent.cli.create_coding_registry"),
        patch("dbagent.cli.AgentLoop") as loop_type,
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
        patch("dbagent.cli.OpenAIResponsesClient"),
        patch("dbagent.cli.create_coding_registry"),
        patch("dbagent.cli.AgentLoop") as loop_type,
    ):
        loop_type.return_value.run.return_value = completed
        exit_code = main(["inspect", "--workspace", str(tmp_path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out == "Architecture summary\n"
    assert "Current plan:" in captured.err
    assert "inspect: Inspect files [completed]" in captured.err
    assert "explain: Explain structure [in_progress]" in captured.err
    assert "VERIFIED" in captured.err


def test_cli_selects_chat_completions_client(monkeypatch, capsys, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "from-environment")
    monkeypatch.setenv("DBAGENT_API_MODE", "chat_completions")
    completed = SimpleNamespace(
        status=AgentStatus.COMPLETED,
        final_answer="Luna answer",
        max_steps=12,
    )

    with (
        patch("dbagent.cli.OpenAIResponsesClient") as responses_client,
        patch("dbagent.cli.OpenAIChatCompletionsClient") as chat_client,
        patch("dbagent.cli.create_coding_registry"),
        patch("dbagent.cli.AgentLoop") as loop_type,
    ):
        loop_type.return_value.run.return_value = completed
        exit_code = main(["inspect", "--workspace", str(tmp_path)])

    assert exit_code == 0
    responses_client.assert_not_called()
    chat_client.assert_called_once()
    assert capsys.readouterr().out == "Luna answer\n"
