import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import httpx2
from openai import APIConnectionError, APITimeoutError

from forge.config import ForgeConfig
from forge.llm import (
    FunctionTool,
    ModelProtocolError,
    ModelRequest,
    OpenAIChatCompletionsClient,
    responses_items_to_chat_messages,
)


def _config() -> ForgeConfig:
    return ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "FORGE_BASE_URL": "https://provider.example/v1",
            "FORGE_API_MODE": "chat_completions",
            "FORGE_MODEL": "gpt-5.6-luna",
            "FORGE_REASONING_EFFORT": "max",
        }
    )


def _sdk(response: SimpleNamespace) -> Mock:
    sdk = Mock()
    sdk.chat.completions.create.return_value = response
    return sdk


def _response(*, content: str, tool_calls=(), finish_reason: str = "stop") -> SimpleNamespace:
    return SimpleNamespace(
        id="chat_test",
        model="gpt-5.6-luna",
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content, tool_calls=list(tool_calls)),
                finish_reason=finish_reason,
            )
        ],
        usage=SimpleNamespace(
            prompt_tokens=12,
            completion_tokens=5,
            total_tokens=17,
        ),
        _request_id="req_chat",
    )


def test_responses_items_are_translated_to_chat_messages() -> None:
    messages = responses_items_to_chat_messages(
        [
            {"role": "user", "content": "Inspect the file."},
            {"type": "reasoning", "encrypted_content": "must not forward"},
            {"role": "assistant", "content": "I will inspect it."},
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "read_file",
                "arguments": '{"path":"README.md"}',
            },
            {
                "type": "function_call",
                "call_id": "call_2",
                "name": "search_text",
                "arguments": '{"query":"Forge"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": '{"ok":true}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_2",
                "output": '{"ok":true}',
            },
        ]
    )

    assert messages == [
        {"role": "user", "content": "Inspect the file."},
        {
            "role": "assistant",
            "content": "I will inspect it.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": '{"path":"README.md"}',
                    },
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "search_text",
                        "arguments": '{"query":"Forge"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": '{"ok":true}'},
        {"role": "tool", "tool_call_id": "call_2", "content": '{"ok":true}'},
    ]
    assert "must not forward" not in json.dumps(messages)


def test_chat_request_uses_nested_function_tools_and_instructions() -> None:
    sdk = _sdk(_response(content="done"))
    client = OpenAIChatCompletionsClient(_config(), sdk_client=sdk)
    tool = FunctionTool(
        name="read_file",
        description="Read a file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )

    client.create_response(
        ModelRequest(
            input="Read README.md",
            instructions="You are a local coding agent.",
            tools=(tool,),
            max_output_tokens=64,
            parallel_tool_calls=True,
        )
    )

    parameters = sdk.chat.completions.create.call_args.kwargs
    assert parameters["model"] == "gpt-5.6-luna"
    assert parameters["reasoning_effort"] == "max"
    assert parameters["messages"] == [
        {"role": "system", "content": "You are a local coding agent."},
        {"role": "user", "content": "Read README.md"},
    ]
    assert parameters["tools"] == [tool.to_chat_api_dict()]
    assert parameters["tool_choice"] == "auto"
    assert parameters["parallel_tool_calls"] is True
    assert parameters["max_tokens"] == 64
    assert "previous_response_id" not in parameters
    assert "conversation" not in parameters


def test_chat_tool_calls_are_normalized_to_existing_function_call_contract() -> None:
    raw_tool_call = SimpleNamespace(
        id="call_123",
        type="function",
        function=SimpleNamespace(
            name="read_file",
            arguments='{"path":"README.md"}',
        ),
    )
    sdk = _sdk(
        _response(
            content="I will read it.",
            tool_calls=(raw_tool_call,),
            finish_reason="tool_calls",
        )
    )
    result = OpenAIChatCompletionsClient(_config(), sdk_client=sdk).create_response(
        ModelRequest(input="Read it")
    )

    assert result.response_id == "chat_test"
    assert result.status == "completed"
    assert result.output_text == "I will read it."
    assert result.function_calls[0].call_id == "call_123"
    assert result.function_calls[0].name == "read_file"
    assert result.function_calls[0].arguments_json == '{"path":"README.md"}'
    assert result.output_items[-1]["type"] == "function_call"
    assert result.usage is not None
    assert result.usage.total_tokens == 17


def test_chat_request_can_disable_tools_for_finalization() -> None:
    sdk = _sdk(_response(content="done"))
    client = OpenAIChatCompletionsClient(_config(), sdk_client=sdk)
    tool = FunctionTool(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )

    client.create_response(ModelRequest(input="finish", tools=(tool,), tool_choice="none"))

    assert sdk.chat.completions.create.call_args.kwargs["tool_choice"] == "none"


def test_chat_protocol_rejects_non_function_tool_calls() -> None:
    raw_tool_call = SimpleNamespace(
        id="call_123",
        type="web_search",
        function=SimpleNamespace(name="search", arguments="{}"),
    )
    sdk = _sdk(_response(content="", tool_calls=(raw_tool_call,)))

    with pytest.raises(ModelProtocolError, match="non-function tool call"):
        OpenAIChatCompletionsClient(_config(), sdk_client=sdk).create_response(
            ModelRequest(input="Search")
        )


@pytest.mark.parametrize(
    ("sdk_error", "expected_message"),
    [
        (
            APITimeoutError(
                request=httpx2.Request("POST", "https://provider.example/v1")
            ),
            "timed out",
        ),
        (
            APIConnectionError(
                request=httpx2.Request("POST", "https://provider.example/v1")
            ),
            "Unable to reach",
        ),
    ],
)
def test_chat_transport_errors_are_wrapped(
    sdk_error: Exception,
    expected_message: str,
) -> None:
    sdk = _sdk(_response(content="done"))
    sdk.chat.completions.create.side_effect = sdk_error

    with pytest.raises(RuntimeError, match=expected_message) as caught:
        OpenAIChatCompletionsClient(_config(), sdk_client=sdk).create_response(
            ModelRequest(input="hello")
        )

    assert caught.value.__cause__ is sdk_error
