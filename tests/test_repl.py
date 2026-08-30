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
    initial_plans: list[TaskPlan | None] = []

    def __init__(
        self,
        _model,
        _registry,
        *,
        max_steps,
        mode,
        context_budget,
        initial_plan=None,
        verification_required=False,
        trace,
    ) -> None:
        self.max_steps = max_steps
        self.mode = mode
        self.context_budget = context_budget
        self.trace = trace
        self.initial_plans.append(initial_plan)

    def run(
        self,
        prompt: str,
        *,
        workspace: Path,
        launch_directory: Path | None = None,
    ):
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
        "/plan",
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
    _FakeLoop.initial_plans = []
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
    assert "Current plan:" in output
    assert "Local and saved conversation history cleared." in output
    assert "Session closed" in output


def test_repl_resumes_only_an_explicit_unfinished_continuation(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["implement the feature", "continue this task", "/exit"])
    config = ForgeConfig(openai_api_key="fake-token")
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("forge.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []

    repl = ForgeRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt: next(inputs),
        stream=stream,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert _FakeLoop.initial_plans[0] is None
    assert _FakeLoop.initial_plans[1] is not None
    assert _FakeLoop.initial_plans[1].goal == "Keep testing"


def test_repl_resume_restores_latest_workspace_session_across_processes(
    tmp_path: Path, monkeypatch
) -> None:
    config = ForgeConfig(openai_api_key="fake-token")
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("forge.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []

    first = ForgeRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(["start task", "/exit"]): next(values),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert first.run() == 0
    assert (tmp_path / ".forge" / "session.json").is_file()

    output = io.StringIO()
    second = ForgeRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(
            ["/resume", "continue this task", "/exit"]
        ): next(values),
        stream=output,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert second.run() == 0

    assert "[assistant]\nanswer 1" in _FakeLoop.calls[1][0]
    assert _FakeLoop.initial_plans[1] is not None
    assert "Resumed workspace session" in output.getvalue()


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
    assert "/resume" in output
    assert "Unknown command '/not-a-command'" in output
    assert not _FakeLoop.calls


def test_repl_mode_command_is_sticky_and_visible(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["/mode ask", "/mode", "/exit"])
    config = ForgeConfig(openai_api_key="fake-token")
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)

    def input_function(prompt: str) -> str:
        stream.write(prompt)
        return next(inputs)

    repl = ForgeRepl(
        workspace=tmp_path,
        input_function=input_function,
        stream=stream,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    output = stream.getvalue()
    assert "Task mode changed to ask" in output
    assert "Current task mode: ask" in output
    assert "DBA[ask]>" in output


def test_repl_rejects_invalid_pipeline_unicode_without_calling_model(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["\udcaf", "/exit"])
    config = ForgeConfig(openai_api_key="fake-token")
    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("forge.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []

    repl = ForgeRepl(
        workspace=tmp_path,
        input_function=lambda _prompt: next(inputs),
        stream=stream,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert "invalid Unicode" in stream.getvalue()
    assert not _FakeLoop.calls


def test_repl_applies_explicit_model_and_reasoning_overrides(
    tmp_path: Path,
    monkeypatch,
) -> None:
    stream = io.StringIO()
    inputs = iter(["/exit"])
    config = ForgeConfig(
        openai_api_key="fake-token",
        model="gpt-5.6-luna",
        reasoning_effort="xhigh",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )
    created_configs: list[ForgeConfig] = []

    monkeypatch.setattr("forge.repl.load_repl_config", lambda _path: config)

    repl = ForgeRepl(
        workspace=tmp_path,
        model_override="gpt-5.6-sol",
        reasoning_effort_override="medium",
        input_function=lambda _prompt: next(inputs),
        stream=stream,
        model_factory=lambda value: created_configs.append(value) or object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert created_configs[0].model == "gpt-5.6-sol"
    assert created_configs[0].reasoning_effort == "medium"
    assert created_configs[0].base_url == config.base_url
