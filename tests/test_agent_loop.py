import json
from pathlib import Path

import pytest

from forge.agent import AgentLoop, AgentStatus, ContextBudget
from forge.llm import (
    FunctionCall,
    FunctionTool,
    ModelConnectionError,
    ModelAPIError,
    ModelProtocolError,
    ModelResponse,
)
from forge.tools import ToolDefinition, ToolRegistry, ToolResult


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


def test_multiple_consecutive_tool_calls_are_fed_back_to_model(
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
        "two",
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
