import io
from pathlib import Path
from types import SimpleNamespace

from forge.agent import AgentStatus, PlanStep, PlanStepStatus, TaskPlan
from forge.agent.verification import VerificationStatus
from forge.config import ForgeConfig
from forge.repl import ForgeRepl
from forge.tools import ToolObservation


class _FakeLoop:
    calls: list[tuple[str, Path]] = []

    def __init__(self, _model, _registry, *, max_steps, context_budget, trace) -> None:
        self.max_steps = max_steps
        self.context_budget = context_budget
        self.trace = trace

    def run(self, prompt: str, *, workspace: Path):
        self.calls.append((prompt, workspace))
        return SimpleNamespace(
            status=AgentStatus.COMPLETED,
            is_verified=False,
            verification_status=VerificationStatus.NOT_RUN,
            final_answer=f"answer {len(self.calls)}",
            plan=TaskPlan(
                goal="Keep testing",
                success_criteria=("tests pass",),
                steps=(PlanStep("test", "Run tests", PlanStepStatus.IN_PROGRESS),),
            ),
            plan_history=[],
            latest_verification=None,
            observations=[
                ToolObservation(
                    "read",
                    "read_file",
                    True,
                    "parser source",
                )
            ],
            step=1,
            max_steps=self.max_steps,
        )


def test_repl_keeps_local_history_and_handles_commands(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter([
        "/model gpt-5.6-sol",
        "first request",
        "follow-up request",
        "/status",
        "/clear",
        "after clear",
        "/exit",
    ])

    def input_function(prompt: str) -> str:
        stream.write(prompt)
        return next(inputs)

    config = ForgeConfig(
        openai_api_key="fake-token",
        model="gpt-5.6-luna",
        reasoning_effort="max",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )
    created_configs: list[ForgeConfig] = []

    def model_factory(value: ForgeConfig):
        created_configs.append(value)
        return object()

    _FakeLoop.calls = []
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("forge.repl.AgentLoop", _FakeLoop)

    repl = ForgeRepl(
        workspace=tmp_path,
        max_steps=4,
        input_function=input_function,
        stream=stream,
        model_factory=model_factory,
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert len(_FakeLoop.calls) == 3
    assert _FakeLoop.calls[0][0] == "first request"
    assert "[assistant]\nanswer 1" in _FakeLoop.calls[1][0]
    assert "[Plan]" in _FakeLoop.calls[1][0]
    assert "[Key tool observations]" in _FakeLoop.calls[1][0]
    assert _FakeLoop.calls[2][0] == "after clear"
    assert [item.model for item in created_configs] == [
        "gpt-5.6-luna",
        "gpt-5.6-sol",
    ]
    output = stream.getvalue()
    assert "DBA interactive session" in output
    assert "Model changed to gpt-5.6-sol" in output
    assert "ASSISTANT" in output
    assert "Local conversation history cleared." in output
    assert "Session closed" in output


def test_repl_help_and_unknown_command_do_not_call_model(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["/help", "/not-a-command", "/exit"])

    def input_function(prompt: str) -> str:
        stream.write(prompt)
        return next(inputs)

    config = ForgeConfig(openai_api_key="fake-token")
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("forge.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []

    repl = ForgeRepl(
        workspace=tmp_path,
        input_function=input_function,
        stream=stream,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    output = stream.getvalue()
    assert "/model [NAME]" in output
    assert "Unknown command '/not-a-command'" in output
    assert not _FakeLoop.calls
