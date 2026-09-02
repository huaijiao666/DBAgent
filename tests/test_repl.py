import io
from pathlib import Path
from types import SimpleNamespace

import pytest

from dbagent.agent import AgentStatus, ContextBudget, PlanStep, PlanStepStatus, TaskPlan
from dbagent.agent.verification import VerificationStatus
from dbagent.config import DBAgentConfig
from dbagent.llm import ModelCommunicationError
from dbagent.repl import DBAgentRepl, _apply_config_overrides, _default_context_budget
from dbagent.tools import ToolObservation


class _FakeLoop:
    calls: list[tuple[str, Path]] = []
    initial_plans: list[TaskPlan | None] = []
    budgets: list[ContextBudget] = []

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
        state_checkpoint=None,
    ) -> None:
        self.max_steps = max_steps
        self.mode = mode
        self.context_budget = context_budget
        self.budgets.append(context_budget)
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


def test_deepseek_repl_profile_reserves_context_for_tool_schema() -> None:
    budget = _default_context_budget(
        DBAgentConfig(
            openai_api_key="fake-token",
            model="deepseek-v4-flash",
            reasoning_effort="high",
            base_url="https://api.deepseek.com",
            api_mode="chat_completions",
            provider="deepseek",
        )
    )

    assert budget.max_context_characters == 24_000
    assert budget.max_single_observation_characters == 3_000
    assert budget.recent_observation_count == 2


def test_default_repl_profile_preserves_existing_large_context() -> None:
    budget = _default_context_budget(DBAgentConfig(openai_api_key="fake-token"))

    assert budget.max_context_characters == 80_000
    assert budget.max_task_characters == 30_000


def test_repl_rejects_unknown_ui_mode(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="ui_mode"):
        DBAgentRepl(workspace=tmp_path, ui_mode="web")


def test_model_switch_uses_the_deepseek_context_profile(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DBAGENT_DEEPSEEK_API_KEY", "test-key")
    config = DBAgentConfig(openai_api_key="configured-key")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []
    _FakeLoop.budgets = []
    inputs = iter(["/model deepseek-flash", "inspect files", "/exit"])
    repl = DBAgentRepl(
        workspace=tmp_path,
        input_function=lambda _prompt: next(inputs),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert _FakeLoop.budgets[-1].max_context_characters == 24_000


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

    config = DBAgentConfig(
        openai_api_key="fake-token",
        model="gpt-5.6-luna",
        reasoning_effort="max",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )
    created_configs: list[DBAgentConfig] = []

    def model_factory(value: DBAgentConfig):
        created_configs.append(value)
        return object()

    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)

    repl = DBAgentRepl(
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
    assert "DBAgent" in output
    assert "Model changed to gpt-5.6-sol" in output
    assert "ASSISTANT" in output
    assert "Current plan:" in output
    assert "Current conversation cleared. Other saved sessions were kept." in output
    assert "Session closed" in output


def test_repl_resumes_only_an_explicit_unfinished_continuation(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["implement the feature", "continue this task", "/exit"])
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []

    repl = DBAgentRepl(
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


def test_repl_auto_mode_continuation_restores_code_authority(
    tmp_path: Path, monkeypatch
) -> None:
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []
    inputs = iter(["implement a feature", "continue this task", "/exit"])
    repl = DBAgentRepl(
        workspace=tmp_path,
        input_function=lambda _prompt: next(inputs),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert _FakeLoop.initial_plans[1] is not None
    assert _FakeLoop.initial_plans[1].goal == "Keep testing"


def test_repl_resume_restores_latest_workspace_session_across_processes(
    tmp_path: Path, monkeypatch
) -> None:
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []

    first = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(["start task", "/exit"]): next(values),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert first.run() == 0
    saved = first._session_store.list_sessions()
    assert len(saved) == 1

    output = io.StringIO()
    second = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(
            ["/resume latest", "continue this task", "/exit"]
        ): next(values),
        stream=output,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert second.run() == 0

    assert "[assistant]\nanswer 1" in _FakeLoop.calls[1][0]
    assert _FakeLoop.initial_plans[1] is not None
    assert "Resumed context" in output.getvalue()


def test_repl_help_and_unknown_command_do_not_call_model(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["/help", "/not-a-command", "/exit"])

    def input_function(prompt: str) -> str:
        stream.write(prompt)
        return next(inputs)

    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []

    repl = DBAgentRepl(
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
    assert "/steps" in output
    assert "/continue" in output
    assert "/sessions" in output
    assert "Unknown command '/not-a-command'" in output
    assert not _FakeLoop.calls


def test_repl_steps_and_continue_start_a_fresh_bounded_run(
    tmp_path: Path, monkeypatch
) -> None:
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []
    output = io.StringIO()
    inputs = iter(["build a project", "/steps 7", "/continue 9", "/exit"])
    repl = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt: next(inputs),
        stream=output,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert len(_FakeLoop.calls) == 2
    assert _FakeLoop.calls[1][0].endswith("[user]\ncontinue this task")
    assert _FakeLoop.initial_plans[1] is not None
    assert repl.max_steps == 9
    rendered = output.getvalue()
    assert "Step budget changed to 7" in rendered
    assert "fresh 9-step budget" in rendered


def test_repl_lists_and_resumes_a_specific_session(
    tmp_path: Path, monkeypatch
) -> None:
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []
    _FakeLoop.initial_plans = []

    first = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(["first task", "/exit"]): next(values),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert first.run() == 0
    first_id = first._session_id

    second = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(["second task", "/exit"]): next(values),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert second.run() == 0
    assert second._session_id != first_id

    output = io.StringIO()
    third = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt, values=iter(
            ["/sessions", "/resume", f"/resume {first_id}", "continue this task", "/exit"]
        ): next(values),
        stream=output,
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )
    assert third.run() == 0

    rendered = output.getvalue()
    assert first_id in rendered
    assert "Use /resume <number>" in rendered
    assert "first task" in rendered
    assert "Plan restored true" in rendered
    assert "[assistant]\nanswer 1" in _FakeLoop.calls[-1][0]


def test_repl_mode_command_is_sticky_and_visible(
    tmp_path: Path, monkeypatch
) -> None:
    stream = io.StringIO()
    inputs = iter(["/mode ask", "/mode", "/exit"])
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)

    def input_function(prompt: str) -> str:
        stream.write(prompt)
        return next(inputs)

    repl = DBAgentRepl(
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
    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", _FakeLoop)
    _FakeLoop.calls = []

    repl = DBAgentRepl(
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
    config = DBAgentConfig(
        openai_api_key="fake-token",
        model="gpt-5.6-luna",
        reasoning_effort="xhigh",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )
    created_configs: list[DBAgentConfig] = []

    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)

    repl = DBAgentRepl(
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


def test_startup_deepseek_preset_switches_the_full_provider_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("DBAGENT_DEEPSEEK_API_KEY", "deepseek-test-secret")
    configured = DBAgentConfig(
        openai_api_key="configured-provider-secret",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )

    selected = _apply_config_overrides(
        configured,
        model="deepseek-flash",
        reasoning_effort="high",
    )

    assert selected.model == "deepseek-v4-flash"
    assert selected.provider == "deepseek"
    assert selected.base_url == "https://api.deepseek.com"
    assert selected.api_mode == "chat_completions"
    assert selected.reasoning_effort == "high"
    assert selected.openai_api_key == "deepseek-test-secret"


def test_repl_switches_presets_and_reasoning_without_persisting_a_key(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("DBAGENT_DEEPSEEK_API_KEY", "deepseek-test-secret")
    stream = io.StringIO()
    config = DBAgentConfig(
        openai_api_key="configured-provider-secret",
        model="gpt-5.6-luna",
        reasoning_effort="medium",
        base_url="https://provider.example/v1",
        api_mode="chat_completions",
    )
    created: list[DBAgentConfig] = []
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    inputs = iter(
        [
            "/models",
            "/model deepseek-flash",
            "/reasoning high",
            "/status",
            "/model terra",
            "/exit",
        ]
    )
    repl = DBAgentRepl(
        workspace=tmp_path,
        input_function=lambda _prompt: next(inputs),
        stream=stream,
        model_factory=lambda value: created.append(value) or object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    assert [item.model for item in created] == [
        "gpt-5.6-luna",
        "deepseek-v4-flash",
        "deepseek-v4-flash",
        "gpt-5.6-terra",
    ]
    assert created[1].base_url == "https://api.deepseek.com"
    assert created[1].provider == "deepseek"
    assert created[2].reasoning_effort == "high"
    assert created[3].openai_api_key == "configured-provider-secret"
    output = stream.getvalue()
    assert "Model options" in output
    assert "Reasoning effort changed to high" in output
    assert "deepseek-test-secret" not in output
    assert "configured-provider-secret" not in output


def test_repl_persists_completed_steps_when_a_model_request_fails(
    tmp_path: Path, monkeypatch
) -> None:
    class CheckpointThenFail:
        def __init__(self, _model, _registry, *, state_checkpoint=None, **_kwargs) -> None:
            self._checkpoint = state_checkpoint

        def run(self, _prompt, *, workspace: Path, launch_directory: Path | None = None):
            assert self._checkpoint is not None
            plan = TaskPlan(
                goal="Implement feature",
                success_criteria=("tests pass",),
                steps=(
                    PlanStep("implement", "Write code", PlanStepStatus.IN_PROGRESS),
                ),
            )
            self._checkpoint(
                SimpleNamespace(
                    plan=plan,
                    verification_status=VerificationStatus.NOT_RUN,
                    latest_verification=None,
                    recovery_hints=[],
                    observations=[
                        ToolObservation(
                            "call_create",
                            "create_file",
                            True,
                            {"path": "feature.py", "changed_files": ["feature.py"]},
                        )
                    ],
                )
            )
            raise ModelCommunicationError("provider stopped")

    config = DBAgentConfig(openai_api_key="fake-token")
    monkeypatch.setattr("dbagent.repl.load_repl_config", lambda _path: config)
    monkeypatch.setattr("dbagent.repl.AgentLoop", CheckpointThenFail)
    inputs = iter(["implement the feature", "/exit"])
    repl = DBAgentRepl(
        workspace=tmp_path,
        mode="code",
        input_function=lambda _prompt: next(inputs),
        stream=io.StringIO(),
        model_factory=lambda _config: object(),
        registry_factory=lambda _workspace: object(),
    )

    assert repl.run() == 0
    saved = repl._session_store.load(repl._session_id)

    assert saved is not None
    assert saved["run_state"] == "interrupted"
    assert saved["session_context"]["plan"]["goal"] == "Implement feature"
    assert saved["session_context"]["observations"][0]["tool_name"] == "create_file"
