import json
from pathlib import Path

import pytest

from forge.agent import AgentLoop, AgentRunControl, AgentStatus, ContextBudget
from forge.agent.loop import (
    _display_progress_text,
    _is_safe_parallel_create_batch,
    _is_safe_parallel_read_batch,
    _safe_command,
)
from forge.llm import (
    FunctionCall,
    FunctionTool,
    ModelConnectionError,
    ModelAPIError,
    ModelProtocolError,
    ModelTextualToolMarkupError,
    ModelResponse,
)
from forge.tools import ToolDefinition, ToolRegistry, ToolResult, create_coding_registry


class QueueModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests = []

    def create_response(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


def _response(
    response_id: str,
    *,
    text: str = "",
    calls: tuple[FunctionCall, ...] = (),
) -> ModelResponse:
    output_items: list[dict[str, object]] = [
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json,
            "status": "completed",
        }
        for call in calls
    ]
    if text:
        output_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            }
        )
    return ModelResponse(
        response_id=response_id,
        model="gpt-5.6-sol",
        status="completed",
        output_text=text,
        output_items=tuple(output_items),
        function_calls=calls,
        usage=None,
    )


def _registry(handler=lambda arguments: arguments["value"]) -> ToolRegistry:
    return ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="echo",
                    description="Echo one value.",
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "string"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
                handler=handler,
            )
        ]
    )


def _call(call_id: str, *, name: str = "echo", value: str = "hello") -> FunctionCall:
    return FunctionCall(
        call_id=call_id,
        name=name,
        arguments_json=json.dumps({"value": value}),
    )


def test_normal_termination_without_tool_call(tmp_path: Path) -> None:
    model = QueueModelClient([_response("resp_final", text="Final answer")])

    state = AgentLoop(model, _registry()).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.step == 1
    assert state.final_answer == "Final answer"
    assert state.tool_calls == []
    assert state.context[0] == {
        "role": "user",
        "content": "[Persistent task context]\ninspect",
    }
    assert len(state.context_usage) == 1
    assert state.context_usage[0].input_characters <= 48_000


def test_continuation_context_is_not_promoted_to_the_current_task(tmp_path: Path) -> None:
    model = QueueModelClient([_response("resp_final", text="已完成说明")])

    state = AgentLoop(model, _registry()).run(
        "说明当前项目如何运行",
        workspace=tmp_path,
        continuation_context="Earlier session fact: a test previously failed.",
    )

    assert state.task == "说明当前项目如何运行"
    assert model.requests[0].input[0] == {
        "role": "user",
        "content": "[Persistent task context]\n说明当前项目如何运行",
    }
    assert "Earlier session fact" in str(model.requests[0].input[1])


def test_runtime_plan_goal_excludes_continuation_context(tmp_path: Path) -> None:
    model = QueueModelClient([_response("resp_final", text="暂时无法完成")])

    state = AgentLoop(
        model,
        create_coding_registry(tmp_path),
        mode="code",
        max_steps=1,
    ).run(
        "修复当前测试失败",
        workspace=tmp_path,
        continuation_context="Earlier conversation: build a snake game.",
    )

    assert state.plan is not None
    assert state.plan.goal == "修复当前测试失败"


def test_low_signal_read_progress_is_hidden_from_the_conversation() -> None:
    call = FunctionCall("read", "read_file", "{}")

    assert _display_progress_text(
        "I will inspect the implementation first.", (call,), chinese=True
    ) == ""
    assert _display_progress_text("我会先检查实现。", (call,), chinese=True) == ""


def test_reasoned_chinese_progress_is_kept_for_the_execution_summary() -> None:
    text = "已确认工作区为空且 Tkinter 可用，因此先建立可测试的规则层，再添加图形界面。"

    assert _display_progress_text(
        text,
        (FunctionCall("create", "create_file", "{}"),),
        chinese=True,
    ) == text


def test_common_zuo_yi_ge_build_request_gets_a_runtime_plan(tmp_path: Path) -> None:
    model = QueueModelClient([_response("resp_final", text="尚未完成")])

    state = AgentLoop(
        model,
        create_coding_registry(tmp_path),
        mode="code",
        max_steps=1,
    ).run("做一个可运行的俄罗斯方块游戏", workspace=tmp_path)

    assert state.plan is not None
    assert state.plan.goal == "做一个可运行的俄罗斯方块游戏"
    assert "update_plan" not in [tool.name for tool in model.requests[0].tools]


def test_only_distinct_create_file_calls_may_share_one_local_turn() -> None:
    create_a = FunctionCall("a", "create_file", json.dumps({"path": "a.py"}))
    create_b = FunctionCall("b", "create_file", json.dumps({"path": "b.py"}))
    duplicate = FunctionCall("c", "create_file", json.dumps({"path": "a.py"}))

    assert _is_safe_parallel_create_batch((create_a, create_b))
    assert not _is_safe_parallel_create_batch((create_a, duplicate))
    assert not _is_safe_parallel_create_batch((create_a, _call("read")))


def test_only_read_only_evidence_calls_may_share_one_local_turn() -> None:
    read_a = FunctionCall("a", "read_file", json.dumps({"path": "a.py"}))
    read_b = FunctionCall("b", "read_file", json.dumps({"path": "b.py"}))

    assert _is_safe_parallel_read_batch((read_a, read_b))
    assert not _is_safe_parallel_read_batch((read_a, _call("command", name="run_command")))
    assert not _is_safe_parallel_read_batch((read_a, FunctionCall(
        "patch", "apply_patch", json.dumps({"patches": []})
    )))


def test_mutation_withholds_redundant_reads_until_a_local_command_runs(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool("create_file", "create", {"type": "object"}),
                handler=lambda _arguments: {"path": "game.py", "changed_files": ["game.py"]},
            ),
            ToolDefinition(
                schema=FunctionTool("read_file", "read", {"type": "object"}),
                handler=lambda _arguments: "source",
            ),
            ToolDefinition(
                schema=FunctionTool("run_command", "verify", {"type": "object"}),
                handler=lambda _arguments: {
                    "command": ["python", "-m", "pytest"],
                    "cwd": ".",
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "1 passed",
                    "stderr": "",
                },
            ),
        ]
    )
    create = FunctionCall("create", "create_file", "{}")
    verify = FunctionCall("verify", "run_command", "{}")
    model = QueueModelClient(
        [
            _response("created", calls=(create,)),
            _response("verified", calls=(verify,)),
            _response("final", text="已验证。"),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=4).run(
        "创建一个文件并验证", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.plan is not None
    assert all(step.status.value == "completed" for step in state.plan.steps)
    assert "read_file" not in [tool.name for tool in model.requests[1].tools]
    assert "run_command" in [tool.name for tool in model.requests[1].tools]


def test_provider_cannot_execute_a_tool_withheld_from_the_current_turn(
    tmp_path: Path,
) -> None:
    reads = 0

    def read_handler(_arguments):
        nonlocal reads
        reads += 1
        return "this handler must not be reached"

    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool("create_file", "create", {"type": "object"}),
                handler=lambda _arguments: {
                    "path": "game.py", "changed_files": ["game.py"]
                },
            ),
            ToolDefinition(
                schema=FunctionTool("read_file", "read", {"type": "object"}),
                handler=read_handler,
            ),
            ToolDefinition(
                schema=FunctionTool("run_command", "verify", {"type": "object"}),
                handler=lambda _arguments: {
                    "command": ["python", "-m", "pytest"],
                    "cwd": ".",
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "1 passed",
                    "stderr": "",
                },
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("create", calls=(FunctionCall("create", "create_file", "{}"),)),
            # DeepSeek-compatible routes occasionally return a stale function
            # name. It was not in the request tool set after the mutation.
            _response("stale", calls=(FunctionCall("read", "read_file", "{}"),)),
            _response("verify", calls=(FunctionCall("verify", "run_command", "{}"),)),
            _response("final", text="Verified."),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=5).run(
        "create a small project", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert reads == 0
    assert state.observations[1].success is False
    assert "not available in this turn" in state.observations[1].content


def test_unclassified_failed_local_command_starts_a_repair_diagnosis_phase(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool("run_command", "run", {"type": "object"}),
                handler=lambda _arguments: {
                    "command": ["python", "-c", "import missing_module"],
                    "cwd": ".",
                    "return_code": 1,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "ModuleNotFoundError: no module named missing_module",
                },
            ),
            ToolDefinition(
                schema=FunctionTool("read_file", "read", {"type": "object"}),
                handler=lambda _arguments: "source",
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("run", calls=(FunctionCall("run", "run_command", "{}"),)),
            _response("final", text="The import failure was recorded."),
        ]
    )

    state = AgentLoop(model, registry, mode="code").run(
        "inspect the project", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    second_tools = {tool.name for tool in model.requests[1].tools}
    assert "read_file" in second_tools
    second_input = json.dumps(model.requests[1].input, ensure_ascii=False)
    assert "ModuleNotFoundError" in second_input


def test_runtime_plan_does_not_verify_a_partial_explicit_file_delivery(
    tmp_path: Path,
) -> None:
    def create_main(_arguments):
        (tmp_path / "main.py").write_text("print('ok')\n", encoding="utf-8")
        return {"path": "main.py", "changed_files": ["main.py"]}

    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool("create_file", "create", {"type": "object"}),
                handler=create_main,
            ),
            ToolDefinition(
                schema=FunctionTool("run_command", "verify", {"type": "object"}),
                handler=lambda _arguments: {
                    "command": ["python", "-m", "py_compile", "main.py"],
                    "cwd": ".",
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "",
                    "stderr": "",
                },
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("create", calls=(FunctionCall("create", "create_file", "{}"),)),
            _response("verify", calls=(FunctionCall("verify", "run_command", "{}"),)),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=2).run(
        "Create main.py and README.md, then verify the project.", workspace=tmp_path
    )

    assert state.status is AgentStatus.MAX_STEPS
    assert state.plan is not None
    assert {step.step_id: step.status.value for step in state.plan.steps} == {
        "inspect": "completed",
        "implement": "in_progress",
        "verify": "in_progress",
        "deliver": "pending",
    }


def test_live_steering_is_added_to_the_next_local_context(tmp_path: Path) -> None:
    control = AgentRunControl()

    def steer_after_first_tool(arguments):
        assert arguments["value"] == "inspect"
        assert control.submit_steering("also add a README")
        return "inspected"

    model = QueueModelClient(
        [
            _response("tool", calls=(_call("call_1", value="inspect"),)),
            _response("final", text="Done."),
        ]
    )
    state = AgentLoop(
        model, _registry(steer_after_first_tool), run_control=control, mode="code"
    ).run("Create the project", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    second_input = model.requests[1].input
    assert any("also add a README" in str(item.get("content", "")) for item in second_input)


def test_abort_stops_before_the_next_model_turn(tmp_path: Path) -> None:
    control = AgentRunControl()

    def abort_after_first_tool(_arguments):
        control.request_abort("stop for review")
        return "done"

    model = QueueModelClient(
        [
            _response("tool", calls=(_call("call_1"),)),
            _response("should_not_run", text="This must not be requested"),
        ]
    )
    state = AgentLoop(
        model, _registry(abort_after_first_tool), run_control=control, mode="code"
    ).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.ABORTED
    assert state.step == 1
    assert len(model.requests) == 1


def test_safe_command_redacts_common_secret_arguments() -> None:
    assert _safe_command(
        ["python", "-m", "tool", "--api-key", "secret", "--token=other"]
    ) == ["python", "-m", "tool", "--api-key", "[REDACTED]", "--token=[REDACTED]"]


def test_transient_model_error_is_retried_without_consuming_an_agent_step(
    tmp_path: Path,
) -> None:
    class FlakyModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            if self.attempts == 1:
                raise ModelConnectionError("temporary provider outage")
            return _response("resp_final", text="Recovered")

    model = FlakyModel()
    state = AgentLoop(
        model,
        _registry(),
        max_steps=1,
        max_model_retries=1,
    ).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.step == 1
    assert model.attempts == 2


def test_default_policy_retries_five_transient_model_failures(
    tmp_path: Path,
) -> None:
    class VeryFlakyModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            if self.attempts <= 5:
                raise ModelConnectionError("temporary provider outage")
            return _response("resp_final", text="Recovered after five retries")

    model = VeryFlakyModel()
    state = AgentLoop(model, _registry(), max_steps=1).run(
        "inspect",
        workspace=tmp_path,
    )

    assert state.status is AgentStatus.COMPLETED
    assert model.attempts == 6


def test_non_retryable_model_error_is_not_retried_and_is_traced(
    tmp_path: Path,
) -> None:
    class BrokenModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            raise ModelProtocolError("invalid model response")

    class Trace:
        def __init__(self) -> None:
            self.events = []

        def record(self, event, *, step, payload=None):
            self.events.append((event, step, payload or {}))

    model = BrokenModel()
    trace = Trace()
    with pytest.raises(ModelProtocolError):
        AgentLoop(
            model,
            _registry(),
            max_steps=2,
            max_model_retries=3,
            mode="code",
            trace=trace,
        ).run("inspect", workspace=tmp_path)

    assert model.attempts == 1
    assert [event[0] for event in trace.events] == [
        "run_started",
        "model_request",
        "model_error",
        "final",
    ]


def test_textual_tool_markup_is_retried_as_required_native_call(
    tmp_path: Path,
) -> None:
    class DSMLThenNativeModel:
        def __init__(self) -> None:
            self.requests = []

        def create_response(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelTextualToolMarkupError("textual DSML was rejected")
            if len(self.requests) == 2:
                return _response("native", calls=(_call("call_1"),))
            return _response("final", text="Recovered with a native call.")

    model = DSMLThenNativeModel()
    state = AgentLoop(
        model,
        _registry(),
        max_steps=2,
        mode="code",
        max_model_retries=1,
    ).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert len(model.requests) == 3
    repaired_request = model.requests[1]
    assert repaired_request.tool_choice == "required"
    assert repaired_request.parallel_tool_calls is False
    assert "textual tool-call markup" in repaired_request.instructions


def test_textual_markup_on_finalization_retries_without_tool_history(
    tmp_path: Path,
) -> None:
    class DSMLThenFinalModel:
        def __init__(self) -> None:
            self.requests = []

        def create_response(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                raise ModelTextualToolMarkupError("textual DSML was rejected")
            return _response("final", text="Final answer without a tool call.")

    model = DSMLThenFinalModel()
    state = AgentLoop(
        model,
        _registry(),
        max_steps=1,
        mode="code",
        max_model_retries=1,
    ).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert len(model.requests) == 2
    repaired_request = model.requests[1]
    assert repaired_request.tool_choice == "none"
    assert "final answer now" in repaired_request.instructions
    assert all(
        item.get("type") not in {"function_call", "function_call_output"}
        for item in repaired_request.input
    )


def test_textual_tool_markup_repairs_are_bounded(tmp_path: Path) -> None:
    class AlwaysTextualMarkupModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            raise ModelTextualToolMarkupError("textual DSML was rejected")

    model = AlwaysTextualMarkupModel()
    with pytest.raises(ModelTextualToolMarkupError):
        AgentLoop(
            model,
            _registry(),
            max_steps=1,
            mode="code",
            max_model_retries=4,
        ).run("inspect", workspace=tmp_path)

    # Two safe transport repairs, then a clear failure rather than executing text
    # or spending all generic retry attempts on a deterministic protocol violation.
    assert model.attempts == 3


def test_transient_provider_api_error_is_retried(tmp_path: Path) -> None:
    class TemporarilyUnavailableModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            if self.attempts == 1:
                raise ModelAPIError("provider unavailable", status_code=503)
            return _response("done", text="Recovered.")

    model = TemporarilyUnavailableModel()
    state = AgentLoop(model, _registry(), max_steps=1).run(
        "inspect",
        workspace=tmp_path,
    )

    assert state.status is AgentStatus.COMPLETED
    assert model.attempts == 2


def test_invalid_provider_api_error_is_not_retried(tmp_path: Path) -> None:
    class InvalidRequestModel:
        attempts = 0

        def create_response(self, _request):
            self.attempts += 1
            raise ModelAPIError("bad request", status_code=400)

    model = InvalidRequestModel()
    with pytest.raises(ModelAPIError):
        AgentLoop(model, _registry(), max_steps=1).run(
            "inspect",
            workspace=tmp_path,
        )

    assert model.attempts == 1


def test_provider_parallel_tool_calls_are_rejected_after_the_first_call(
    tmp_path: Path,
) -> None:
    first_calls = (_call("call_1", value="one"), _call("call_2", value="two"))
    third_call = _call("call_3", value="three")
    model = QueueModelClient(
        [
            _response("resp_1", calls=first_calls),
            _response("resp_2", calls=(third_call,)),
            _response("resp_3", text="Finished"),
        ]
    )

    state = AgentLoop(model, _registry(), max_steps=5, mode="code").run(
        "inspect", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.step == 3
    assert [call.call_id for call in state.tool_calls] == [
        "call_1",
        "call_2",
        "call_3",
    ]
    assert [observation.content for observation in state.observations] == [
        "one",
        "Only one tool call is executed per model turn. The workspace may have changed after the first call; reissue this operation after reading that result.",
        "three",
    ]
    second_input = model.requests[1].input
    assert any(item.get("call_id") == "call_1" for item in second_input)
    assert any(item.get("call_id") == "call_2" for item in second_input)


def test_tool_failure_is_observed_and_loop_continues(tmp_path: Path) -> None:
    def fail(_arguments) -> str:
        raise PermissionError("denied")

    model = QueueModelClient(
        [
            _response("resp_1", calls=(_call("call_1"),)),
            _response("resp_2", text="I could not inspect that value."),
        ]
    )

    state = AgentLoop(model, _registry(fail), mode="code").run(
        "inspect", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].success is False
    assert state.observations[0].content == "PermissionError: denied"
    feedback = model.requests[1].input[-1]
    assert json.loads(feedback["output"])["ok"] is False


def test_patch_failure_adds_actionable_recovery_context(tmp_path: Path) -> None:
    definition = ToolDefinition(
        schema=FunctionTool(
            name="apply_patch",
            description="test patch",
            parameters={"type": "object", "properties": {}},
        ),
        handler=lambda _arguments: ToolResult(
            success=False,
            content={
                "applied": False,
                "changed_files": [],
                "hunks_applied": 0,
                "failure_reason": "PatchError: context did not match",
            },
        ),
    )
    call = FunctionCall("patch_1", "apply_patch", "{}")
    model = QueueModelClient(
        [
            _response("patch", calls=(call,)),
            _response("final", text="Patch failure reported"),
        ]
    )

    state = AgentLoop(model, ToolRegistry([definition]), mode="code").run(
        "fix it",
        workspace=tmp_path,
    )

    assert state.status is AgentStatus.COMPLETED
    assert any("rejected atomically" in hint for hint in state.recovery_hints)
    second_input = json.dumps(model.requests[1].input, ensure_ascii=False)
    assert "Do not repeat the identical patch" in second_input


def test_write_file_is_withheld_until_patch_fallback_is_needed(tmp_path: Path) -> None:
    definition = ToolDefinition(
        schema=FunctionTool(
            name="write_file",
            description="replace file",
            parameters={"type": "object", "properties": {}},
        ),
        handler=lambda _arguments: {"path": "target.py"},
    )
    model = QueueModelClient(
        [
            _response(
                "write",
                calls=(FunctionCall("write_1", "write_file", "{}"),),
            ),
            _response("final", text="I need a patch fallback first."),
        ]
    )

    state = AgentLoop(model, ToolRegistry([definition]), mode="code").run(
        "fix it", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert "write_file" not in [tool.name for tool in model.requests[0].tools]
    assert state.observations[0].success is False
    assert "withheld by default" in state.observations[0].content


def test_existing_create_file_adds_patch_oriented_recovery_hint(tmp_path: Path) -> None:
    definition = ToolDefinition(
        schema=FunctionTool(
            name="create_file",
            description="create file",
            parameters={"type": "object", "properties": {}},
        ),
        handler=lambda _arguments: (_ for _ in ()).throw(
            FileExistsError("path already exists: app.py")
        ),
    )
    model = QueueModelClient(
        [
            _response(
                "create",
                calls=(FunctionCall("create_1", "create_file", "{}"),),
            ),
            _response("final", text="I will patch the existing file."),
        ]
    )

    state = AgentLoop(model, ToolRegistry([definition]), mode="code").run(
        "create a file", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert any("create_file never overwrites" in hint for hint in state.recovery_hints)


def test_failed_verification_limits_repeated_diagnosis_reads(tmp_path: Path) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="run_command",
                    description="run test",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {
                    "command": ["pytest", "-q"],
                    "cwd": ".",
                    "return_code": 1,
                    "timed_out": False,
                    "stdout": "FAILED test_target",
                    "stderr": "",
                },
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="read_file",
                    description="read source",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: "def broken(): pass",
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="apply_patch",
                    description="patch source",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {"applied": False},
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("failed", calls=(FunctionCall("test", "run_command", "{}"),)),
            _response("read_1", calls=(FunctionCall("read_1", "read_file", "{}"),)),
            _response("read_2", calls=(FunctionCall("read_2", "read_file", "{}"),)),
            _response("final", text="Need to fix the test failure."),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=4).run(
        "fix the failing test", workspace=tmp_path
    )

    assert state.status is AgentStatus.MAX_STEPS
    assert "read_file" not in [tool.name for tool in model.requests[3].tools]
    assert "apply_patch" in [tool.name for tool in model.requests[3].tools]


def test_repeated_failed_verification_does_not_restart_read_budget(
    tmp_path: Path,
) -> None:
    """A weak model must not evade recovery by alternating test/read calls."""

    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool("run_command", "run test", {"type": "object"}),
                handler=lambda _arguments: {
                    "command": ["pytest", "-q"],
                    "cwd": ".",
                    "return_code": 1,
                    "timed_out": False,
                    "stdout": "FAILED test_target",
                    "stderr": "",
                },
            ),
            ToolDefinition(
                schema=FunctionTool("read_file", "read source", {"type": "object"}),
                handler=lambda _arguments: "def broken(): pass",
            ),
            ToolDefinition(
                schema=FunctionTool("apply_patch", "patch source", {"type": "object"}),
                handler=lambda _arguments: {"applied": False},
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("failed_1", calls=(FunctionCall("test_1", "run_command", "{}"),)),
            _response("read_1", calls=(FunctionCall("read_1", "read_file", "{}"),)),
            _response("failed_2", calls=(FunctionCall("test_2", "run_command", "{}"),)),
            _response("read_2", calls=(FunctionCall("read_2", "read_file", "{}"),)),
            _response("final", text="Need a focused fix."),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=5).run(
        "fix the failing test", workspace=tmp_path
    )

    assert state.status is AgentStatus.MAX_STEPS
    assert "read_file" not in [tool.name for tool in model.requests[4].tools]
    assert "apply_patch" in [tool.name for tool in model.requests[4].tools]


def test_overlapping_read_ranges_are_not_counted_as_new_progress(
    tmp_path: Path,
) -> None:
    class Events:
        def __init__(self) -> None:
            self.items = []

        def record(self, event, *, step, payload=None):
            self.items.append((event, step, payload or {}))

    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="read_file",
                    description="read source",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: "1: def answer():\n2:     return 42",
            )
        ]
    )
    model = QueueModelClient(
        [
            _response(
                "read_1",
                calls=(
                    FunctionCall(
                        "read_1",
                        "read_file",
                        json.dumps({"path": "app.py", "start_line": 1, "end_line": 20}),
                    ),
                ),
            ),
            _response(
                "read_2",
                calls=(
                    FunctionCall(
                        "read_2",
                        "read_file",
                        json.dumps({"path": "app.py", "start_line": 5, "end_line": 10}),
                    ),
                ),
            ),
            _response("final", text="The file contains the answer function."),
        ]
    )
    trace = Events()

    state = AgentLoop(model, registry, mode="ask", max_steps=4, trace=trace).run(
        "inspect the repository", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.no_progress_rounds == 1
    duplicate_results = [
        payload
        for event, _step, payload in trace.items
        if event == "tool_result" and payload.get("duplicate_evidence")
    ]
    assert duplicate_results
    assert any("repeating evidence" in hint for hint in state.recovery_hints)


def test_length_limited_response_does_not_execute_tool_calls(tmp_path: Path) -> None:
    definition = ToolDefinition(
        schema=FunctionTool(
            name="dangerous_write",
            description="would mutate a file",
            parameters={"type": "object", "properties": {}},
        ),
        handler=lambda _arguments: pytest.fail("truncated tool call was executed"),
    )
    call = FunctionCall("cut_off", "dangerous_write", "{}")
    truncated = ModelResponse(
        response_id="truncated",
        model="test-model",
        status="length",
        output_text="",
        output_items=(),
        function_calls=(call,),
        usage=None,
    )
    model = QueueModelClient(
        [truncated, _response("final", text="Please retry the tool call.")]
    )

    state = AgentLoop(model, ToolRegistry([definition]), mode="code").run(
        "change a file", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].success is False
    assert "output-length limit" in state.observations[0].content
    feedback = json.dumps(model.requests[1].input)
    assert "Tool call was not executed" in feedback


def test_code_mode_requests_serial_tool_decisions(tmp_path: Path) -> None:
    model = QueueModelClient([_response("done", text="Finished")])

    state = AgentLoop(model, _registry(), mode="code").run(
        "make a change", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert model.requests[0].parallel_tool_calls is False


def test_ask_mode_requests_serial_tool_decisions_for_provider_compatibility(
    tmp_path: Path,
) -> None:
    model = QueueModelClient([_response("done", text="Finished")])

    state = AgentLoop(model, _registry(), mode="ask").run(
        "inspect this repository", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert model.requests[0].parallel_tool_calls is False


def test_successful_patch_withholds_next_patch_until_fresh_evidence(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="apply_patch",
                    description="patch",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: ToolResult(
                    success=True,
                    content={
                        "applied": True,
                        "changed_files": [{"path": "target.py"}],
                        "hunks_applied": 1,
                    },
                ),
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="run_command",
                    description="verify",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {
                    "command": ["python", "-m", "pytest", "-q"],
                    "cwd": ".",
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "1 passed",
                    "stderr": "",
                },
            ),
        ]
    )
    model = QueueModelClient(
        [
            _response("patch", calls=(FunctionCall("patch_1", "apply_patch", "{}"),)),
            _response("verify", calls=(FunctionCall("verify_1", "run_command", "{}"),)),
            _response("final", text="Verified."),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=4).run(
        "fix it", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert "apply_patch" not in [tool.name for tool in model.requests[1].tools]
    assert "run_command" in [tool.name for tool in model.requests[1].tools]


def test_two_malformed_patch_calls_disable_patch_and_offer_write_fallback(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="apply_patch",
                    description="patch",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: "unreachable",
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="write_file",
                    description="write",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {"path": "small.txt"},
            ),
            ToolDefinition(
                schema=FunctionTool(
                    name="run_command",
                    description="verify",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {
                    "command": ["pytest"],
                    "cwd": ".",
                    "return_code": 0,
                    "timed_out": False,
                    "stdout": "passed",
                    "stderr": "",
                },
            ),
        ]
    )
    malformed_first = FunctionCall("patch_1", "apply_patch", "{not json")
    malformed_second = FunctionCall("patch_2", "apply_patch", "{not json")
    write = FunctionCall("write_1", "write_file", "{}")
    verify = FunctionCall("verify_1", "run_command", "{}")
    model = QueueModelClient(
        [
            _response("first", calls=(malformed_first,)),
            _response("second", calls=(malformed_second,)),
            _response("write", calls=(write,)),
            _response("verify", calls=(verify,)),
            _response("final", text="Used the fallback"),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=6).run(
        "fix it", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert "apply_patch" not in [tool.name for tool in model.requests[2].tools]
    assert "write_file" in [tool.name for tool in model.requests[2].tools]
    assert any("temporarily unavailable" in hint for hint in state.recovery_hints)

def test_unknown_tool_is_observed_and_loop_continues(tmp_path: Path) -> None:
    model = QueueModelClient(
        [
            _response(
                "resp_1",
                calls=(_call("call_unknown", name="unknown"),),
            ),
            _response("resp_2", text="Recovered"),
        ]
    )

    state = AgentLoop(model, _registry()).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].success is False
    assert state.observations[0].content == "Unknown tool: unknown"


def test_max_steps_is_a_hard_termination_condition(tmp_path: Path) -> None:
    model = QueueModelClient(
        [
            _response("resp_1", calls=(_call("call_1"),)),
            _response("resp_2", calls=(_call("call_2"),)),
        ]
    )

    state = AgentLoop(model, _registry(), max_steps=2).run(
        "keep inspecting", workspace=tmp_path
    )

    assert state.status is AgentStatus.MAX_STEPS
    assert state.step == 2
    assert state.final_answer is None
    assert len(model.requests) == 2


def test_repeated_prose_finalization_stops_as_unverified(tmp_path: Path) -> None:
    registry = ToolRegistry(
        [
            ToolDefinition(
                schema=FunctionTool(
                    name="create_file",
                    description="create source",
                    parameters={"type": "object", "properties": {}},
                ),
                handler=lambda _arguments: {
                    "path": "snake.py",
                    "changed_files": ["snake.py"],
                },
            )
        ]
    )
    model = QueueModelClient(
        [
            _response(
                "create",
                calls=(FunctionCall("create", "create_file", "{}"),),
            ),
            _response("final_1", text="The game is complete."),
            _response("final_2", text="The game is complete."),
        ]
    )

    state = AgentLoop(model, registry, mode="code", max_steps=40).run(
        "create a game", workspace=tmp_path
    )

    assert state.status is AgentStatus.UNVERIFIED
    assert state.step == 3
    assert state.final_answer == "The game is complete."
    assert len(model.requests) == 3


def test_agent_loop_sends_bounded_context_and_records_usage(tmp_path: Path) -> None:
    class Trace:
        def __init__(self) -> None:
            self.events = []

        def record(self, event, *, step, payload=None):
            self.events.append((event, step, payload or {}))

    model = QueueModelClient(
        [
            _response("resp_1", calls=(_call("call_large"),)),
            _response("resp_2", text="Finished"),
        ]
    )
    budget = ContextBudget(
        max_context_characters=10_000,
        max_task_characters=1_000,
        max_plan_characters=500,
        max_repository_map_characters=1_000,
        max_relevant_code_characters=1_500,
        max_compact_observations_characters=1_000,
        max_recent_observations_characters=3_000,
        max_single_observation_characters=1_000,
        max_call_arguments_characters=500,
        recent_observation_count=2,
    )

    trace = Trace()
    state = AgentLoop(
        model,
        _registry(lambda _arguments: "z" * 50_000),
        mode="code",
        context_budget=budget,
        trace=trace,
    ).run("inspect", workspace=tmp_path)

    second_input = model.requests[1].input
    serialized = json.dumps(
        second_input,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    assert len(serialized) <= budget.max_context_characters
    assert "z" * 10_000 not in serialized
    assert "context truncated" in serialized
    assert len(state.context_usage) == 2
    assert all(
        usage.input_characters <= budget.max_context_characters
        for usage in state.context_usage
    )
    compaction_events = [item for item in trace.events if item[0] == "context_compacted"]
    assert len(compaction_events) == 1
    assert compaction_events[0][2]["truncated_items"] == 1


def test_model_updates_one_persisted_plan_across_tool_turns(tmp_path: Path) -> None:
    initial_plan = {
        "goal": "Inspect the repository",
        "success_criteria": ["Explain the architecture"],
        "steps": [
            {
                "id": "inspect",
                "description": "Inspect relevant files",
                "status": "pending",
            },
            {
                "id": "explain",
                "description": "Write the architecture explanation",
                "status": "pending",
            },
        ],
    }
    updated_plan = {
        **initial_plan,
        "steps": [
            {**initial_plan["steps"][0], "status": "completed"},
            {**initial_plan["steps"][1], "status": "completed"},
        ],
    }
    plan_call = FunctionCall(
        call_id="plan_1",
        name="update_plan",
        arguments_json=json.dumps(initial_plan),
    )
    update_call = FunctionCall(
        call_id="plan_2",
        name="update_plan",
        arguments_json=json.dumps(updated_plan),
    )
    model = QueueModelClient(
        [
            _response("resp_plan", calls=(plan_call,)),
            _response("resp_inspect", calls=(_call("call_inspect"),)),
            _response("resp_update", calls=(update_call,)),
            _response("resp_final", text="Architecture explained"),
        ]
    )

    state = AgentLoop(model, _registry(), mode="code").run(
        "Inspect the repository", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.plan is not None
    assert state.plan.steps[0].status.value == "completed"
    assert state.plan.steps[1].status.value == "completed"
    assert len(state.plan_history) == 2
    plan_context = str(model.requests[2].input[1]["content"])
    assert "[Current plan]" in plan_context
    assert "[pending] inspect" in plan_context
    updated_context = str(model.requests[3].input[1]["content"])
    assert "[completed] inspect" in updated_context
