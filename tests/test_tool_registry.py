import json

import pytest

from forge.llm import FunctionCall, FunctionTool
from forge.tools import ToolDefinition, ToolRegistry


def _definition(name: str, handler) -> ToolDefinition:
    return ToolDefinition(
        schema=FunctionTool(
            name=name,
            description=f"Run {name}.",
            parameters={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
    )


def _call(name: str, arguments: str = '{"value":"hello"}') -> FunctionCall:
    return FunctionCall(
        call_id="call_test",
        name=name,
        arguments_json=arguments,
    )


def test_registered_tool_is_dispatched() -> None:
    registry = ToolRegistry([_definition("echo", lambda args: args["value"])])

    observation = registry.dispatch(_call("echo"))

    assert observation.success is True
    assert observation.content == "hello"
    model_input = observation.to_model_input()
    assert model_input["call_id"] == "call_test"
    assert json.loads(model_input["output"]) == {"ok": True, "result": "hello"}


def test_unknown_tool_becomes_an_error_observation() -> None:
    observation = ToolRegistry().dispatch(_call("missing"))

    assert observation.success is False
    assert observation.content == "Unknown tool: missing"


def test_tool_failure_becomes_an_error_observation() -> None:
    def fail(_arguments) -> str:
        raise OSError("cannot read")

    registry = ToolRegistry([_definition("broken", fail)])

    observation = registry.dispatch(_call("broken"))

    assert observation.success is False
    assert observation.content == "OSError: cannot read"
    assert json.loads(observation.to_model_input()["output"])["ok"] is False


@pytest.mark.parametrize("arguments", ["not json", "[]"])
def test_invalid_arguments_become_an_error_observation(arguments: str) -> None:
    registry = ToolRegistry([_definition("echo", lambda args: args["value"])])

    observation = registry.dispatch(_call("echo", arguments))

    assert observation.success is False


def test_duplicate_registration_is_rejected() -> None:
    definition = _definition("echo", lambda args: args["value"])
    registry = ToolRegistry([definition])

    with pytest.raises(ValueError, match="already registered"):
        registry.register(definition)
