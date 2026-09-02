from types import SimpleNamespace
from unittest.mock import Mock

import httpx2
import pytest
from openai import APIConnectionError, APIStatusError, APITimeoutError, RateLimitError

from forge.config import ForgeConfig
from forge.llm import (
    FunctionTool,
    ModelAPIError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelRequest,
    ModelTimeoutError,
    OpenAIResponsesClient,
)


def _config(*, reasoning_effort: str = "medium") -> ForgeConfig:
    return ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "FORGE_REASONING_EFFORT": reasoning_effort,
        }
    )


def _config_with_base_url() -> ForgeConfig:
    return ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": "from-environment",
            "FORGE_BASE_URL": "https://provider.example/v1",
        }
    )


def _response(*, output: list[dict[str, object]]) -> SimpleNamespace:
    return SimpleNamespace(
        id="resp_test",
        model="gpt-5.6-sol",
        status="completed",
        output_text="done",
        output=output,
        usage=SimpleNamespace(
            input_tokens=10,
            output_tokens=5,
            total_tokens=15,
        ),
        error=None,
        incomplete_details=None,
        _request_id="req_test",
    )


def _sdk_client(response: SimpleNamespace | None = None) -> Mock:
    sdk = Mock()
    sdk.responses.create.return_value = response or _response(output=[])
    return sdk


def test_text_request_is_stateless_and_has_no_server_tools() -> None:
    sdk = _sdk_client(_response(output=[{"type": "message"}]))
    client = OpenAIResponsesClient(_config(reasoning_effort="high"), sdk_client=sdk)

    result = client.create_response(ModelRequest(input="hello"))

    parameters = sdk.responses.create.call_args.kwargs
    assert parameters["model"] == "gpt-5.6-sol"
    assert parameters["reasoning"] == {"effort": "high", "context": "current_turn"}
    assert parameters["tools"] == []
    assert parameters["include"] == ["reasoning.encrypted_content"]
    assert parameters["store"] is False
    assert parameters["background"] is False
    assert "previous_response_id" not in parameters
    assert "conversation" not in parameters
    assert result.output_text == "done"
    assert result.request_id == "req_test"
    assert result.usage is not None
    assert result.usage.total_tokens == 15


def test_only_function_tools_are_serialized_and_calls_are_normalized() -> None:
    sdk = _sdk_client(
        _response(
            output=[
                {"type": "reasoning"},
                {
                    "type": "function_call",
                    "call_id": "call_123",
                    "name": "read_file",
                    "arguments": '{"path":"README.md"}',
                    "status": "completed",
                },
            ]
        )
    )
    tool = FunctionTool(
        name="read_file",
        description="Read a workspace file.",
        parameters={
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
            "additionalProperties": False,
        },
    )
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    result = client.create_response(ModelRequest(input="read it", tools=(tool,)))

    parameters = sdk.responses.create.call_args.kwargs
    assert parameters["tools"] == [
        {
            "type": "function",
            "name": "read_file",
            "description": "Read a workspace file.",
            "parameters": dict(tool.parameters),
            "strict": True,
        }
    ]
    assert parameters["tool_choice"] == "auto"
    assert parameters["parallel_tool_calls"] is False
    assert result.function_calls[0].call_id == "call_123"
    assert result.function_calls[0].arguments_json == '{"path":"README.md"}'


def test_responses_request_can_disable_tools_for_finalization() -> None:
    sdk = _sdk_client(_response(output=[{"type": "message"}]))
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)
    tool = FunctionTool(
        name="read_file",
        description="Read a file.",
        parameters={"type": "object", "properties": {}, "additionalProperties": False},
    )

    client.create_response(ModelRequest(input="finish", tools=(tool,), tool_choice="none"))

    assert sdk.responses.create.call_args.kwargs["tool_choice"] == "none"


def test_responses_stream_forwards_text_deltas_and_normalizes_completion() -> None:
    completed = _response(output=[{"type": "message"}])
    sdk = _sdk_client()
    sdk.responses.create.return_value = [
        SimpleNamespace(type="response.output_text.delta", delta="Hello "),
        SimpleNamespace(type="response.output_text.delta", delta="world"),
        SimpleNamespace(type="response.completed", response=completed),
    ]
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)
    events: list[tuple[str, dict[str, object]]] = []

    result = client.create_response_stream(
        ModelRequest(input="hello"),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )

    assert sdk.responses.create.call_args.kwargs["stream"] is True
    assert events == [
        ("text_delta", {"delta": "Hello "}),
        ("text_delta", {"delta": "world"}),
    ]
    assert result.response_id == "resp_test"


def test_responses_stream_requires_a_completed_response() -> None:
    sdk = _sdk_client()
    sdk.responses.create.return_value = [
        SimpleNamespace(type="response.output_text.delta", delta="partial")
    ]
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    with pytest.raises(ModelProtocolError, match="response.completed"):
        client.create_response_stream(ModelRequest(input="hello"), on_event=lambda *_: None)


def test_non_function_tool_values_are_rejected() -> None:
    sdk = _sdk_client()
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)
    request = ModelRequest(input="search", tools=({"type": "web_search"},))  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="FunctionTool values only"):
        client.create_response(request)

    sdk.responses.create.assert_not_called()


def test_missing_api_key_is_rejected_before_sdk_construction() -> None:
    with pytest.raises(ModelConfigurationError, match="OPENAI_API_KEY"):
        OpenAIResponsesClient(ForgeConfig.from_env({}))


def test_custom_base_url_is_passed_to_official_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdk = _sdk_client()
    constructor = Mock(return_value=sdk)
    monkeypatch.setattr("forge.llm.client.OpenAI", constructor)

    OpenAIResponsesClient(_config_with_base_url(), timeout_seconds=7, max_retries=3)

    constructor.assert_called_once_with(
        api_key="from-environment",
        timeout=7,
        max_retries=3,
        base_url="https://provider.example/v1",
    )


@pytest.mark.parametrize(
    ("sdk_error", "expected_error"),
    [
        (
            APITimeoutError(
                request=httpx2.Request("POST", "https://api.openai.com")
            ),
            ModelTimeoutError,
        ),
        (
            APIConnectionError(
                request=httpx2.Request("POST", "https://api.openai.com")
            ),
            ModelConnectionError,
        ),
    ],
)
def test_transport_errors_are_wrapped(
    sdk_error: Exception, expected_error: type[Exception]
) -> None:
    sdk = _sdk_client()
    sdk.responses.create.side_effect = sdk_error
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    with pytest.raises(expected_error) as caught:
        client.create_response(ModelRequest(input="hello"))

    assert caught.value.__cause__ is sdk_error


def test_rate_limit_error_preserves_diagnostic_metadata() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(
        429,
        request=request,
        headers={"x-request-id": "req_rate"},
    )
    sdk_error = RateLimitError("rate limited", response=response, body=None)
    sdk = _sdk_client()
    sdk.responses.create.side_effect = sdk_error
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    with pytest.raises(ModelRateLimitError) as caught:
        client.create_response(ModelRequest(input="hello"))

    assert caught.value.status_code == 429
    assert caught.value.request_id == "req_rate"


def test_other_api_status_errors_are_wrapped() -> None:
    request = httpx2.Request("POST", "https://api.openai.com/v1/responses")
    response = httpx2.Response(
        500,
        request=request,
        headers={"x-request-id": "req_server"},
    )
    sdk_error = APIStatusError("server error", response=response, body=None)
    sdk = _sdk_client()
    sdk.responses.create.side_effect = sdk_error
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    with pytest.raises(ModelAPIError) as caught:
        client.create_response(ModelRequest(input="hello"))

    assert caught.value.status_code == 500
    assert caught.value.request_id == "req_server"


def test_disallowed_server_tool_output_is_rejected() -> None:
    sdk = _sdk_client(_response(output=[{"type": "web_search_call"}]))
    client = OpenAIResponsesClient(_config(), sdk_client=sdk)

    with pytest.raises(ModelProtocolError, match="disallowed output item"):
        client.create_response(ModelRequest(input="hello"))
