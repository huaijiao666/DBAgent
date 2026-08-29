import json
from pathlib import Path

from forge.agent import AgentLoop, AgentStatus
from forge.llm import FunctionCall, FunctionTool, ModelResponse
from forge.tools import ToolDefinition, ToolRegistry


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
    assert state.context[0] == {"role": "user", "content": "inspect"}


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

    state = AgentLoop(model, _registry(), max_steps=5).run(
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

    state = AgentLoop(model, _registry(fail)).run("inspect", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.observations[0].success is False
    assert state.observations[0].content == "PermissionError: denied"
    feedback = model.requests[1].input[-1]
    assert json.loads(feedback["output"])["ok"] is False


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
