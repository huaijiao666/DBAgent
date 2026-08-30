"""Stateless Chat Completions compatibility adapter.

The rest of Forge uses Responses-shaped context and normalized model values.  This
module translates that local representation at the provider boundary so a provider
that only exposes ``/chat/completions`` can still participate in the same agent loop.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from forge.config import ForgeConfig
from forge.llm.errors import (
    ModelAPIError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelTimeoutError,
)
from forge.llm.models import (
    FunctionCall,
    FunctionTool,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)


class _ChatCompletionsResource(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ChatSDKClient(Protocol):
    chat: Any


class OpenAIChatCompletionsClient:
    """Call ``chat.completions.create`` without server-side conversation state."""

    def __init__(
        self,
        config: ForgeConfig,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        sdk_client: _ChatSDKClient | None = None,
    ) -> None:
        if config.openai_api_key is None:
            raise ModelConfigurationError(
                "OPENAI_API_KEY is required for model communication"
            )
        if timeout_seconds <= 0:
            raise ModelConfigurationError("timeout_seconds must be positive")
        if max_retries < 0:
            raise ModelConfigurationError("max_retries must not be negative")

        self._model = config.model
        self._reasoning_effort = config.reasoning_effort
        if sdk_client is not None:
            self._client = sdk_client
        else:
            client_parameters: dict[str, Any] = {
                "api_key": config.openai_api_key,
                "timeout": timeout_seconds,
                "max_retries": max_retries,
            }
            if config.base_url is not None:
                client_parameters["base_url"] = config.base_url
            self._client = OpenAI(**client_parameters)

    def create_response(self, request: ModelRequest) -> ModelResponse:
        """Create one chat completion and normalize it to Forge's model contract."""

        parameters = self._build_parameters(request)
        try:
            response = self._client.chat.completions.create(**parameters)
        except APITimeoutError as error:
            raise ModelTimeoutError(
                "Chat Completions request timed out after configured SDK retries"
            ) from error
        except RateLimitError as error:
            raise ModelRateLimitError(
                "Chat Completions rate limit exceeded after configured SDK retries",
                status_code=getattr(error, "status_code", 429),
                request_id=getattr(error, "request_id", None),
            ) from error
        except APIConnectionError as error:
            raise ModelConnectionError(
                "Unable to reach the Chat Completions provider"
            ) from error
        except APIStatusError as error:
            raise ModelAPIError(
                "Chat Completions provider returned an unsuccessful status",
                status_code=getattr(error, "status_code", None),
                request_id=getattr(error, "request_id", None),
            ) from error
        except APIError as error:
            raise ModelAPIError("Chat Completions request failed") from error

        return self._normalize_response(response)

    def _build_parameters(self, request: ModelRequest) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if not isinstance(tool, FunctionTool):
                raise TypeError("ModelRequest.tools accepts FunctionTool values only")
            tools.append(tool.to_chat_api_dict())

        messages = responses_items_to_chat_messages(request.input)
        if request.instructions is not None:
            messages.insert(0, {"role": "system", "content": request.instructions})

        parameters: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "reasoning_effort": self._reasoning_effort,
        }
        if request.max_output_tokens is not None:
            # ``max_tokens`` remains the most widely supported spelling among
            # OpenAI-compatible third-party providers.
            parameters["max_tokens"] = request.max_output_tokens
        if tools:
            parameters["tools"] = tools
            parameters["tool_choice"] = request.tool_choice
            parameters["parallel_tool_calls"] = request.parallel_tool_calls
        return parameters

    def _normalize_response(self, response: Any) -> ModelResponse:
        response_id = _required_string(response, "id", "chat completion")
        choices = _read(response, "choices")
        if not isinstance(choices, Sequence) or isinstance(choices, (str, bytes)):
            raise ModelProtocolError(
                "Chat Completions response did not contain a choices sequence"
            )
        if not choices:
            raise ModelProtocolError("Chat Completions response contained no choices")

        choice = choices[0]
        message = _read(choice, "message")
        if message is None:
            raise ModelProtocolError(
                "Chat Completions response choice did not contain a message"
            )
        output_text = _content_to_text(_read(message, "content", ""))
        raw_tool_calls = _read(message, "tool_calls", ()) or ()
        if not isinstance(raw_tool_calls, Sequence) or isinstance(
            raw_tool_calls, (str, bytes)
        ):
            raise ModelProtocolError(
                "Chat Completions message tool_calls must be a sequence"
            )

        function_calls: list[FunctionCall] = []
        output_items: list[dict[str, Any]] = [
            {
                "type": "message",
                "role": "assistant",
                "content": output_text,
            }
        ]
        for raw_tool_call in raw_tool_calls:
            tool_type = _read(raw_tool_call, "type", "function")
            if tool_type != "function":
                raise ModelProtocolError(
                    "Chat Completions returned a non-function tool call"
                )
            call_id = _required_string(raw_tool_call, "id", "tool call")
            function = _read(raw_tool_call, "function")
            if function is None:
                raise ModelProtocolError("Chat Completions tool call lacks function data")
            name = _required_string(function, "name", "function tool call")
            arguments = _required_string(
                function, "arguments", "function tool call"
            )
            function_calls.append(
                FunctionCall(
                    call_id=call_id,
                    name=name,
                    arguments_json=arguments,
                    status="completed",
                )
            )
            output_items.append(
                {
                    "type": "function_call",
                    "call_id": call_id,
                    "name": name,
                    "arguments": arguments,
                    "status": "completed",
                }
            )

        finish_reason = _read(choice, "finish_reason")
        status = (
            "completed"
            if finish_reason in {None, "stop", "tool_calls"}
            else str(finish_reason)
        )
        return ModelResponse(
            response_id=response_id,
            model=str(_read(response, "model", self._model)),
            status=status,
            output_text=output_text,
            output_items=tuple(output_items),
            function_calls=tuple(function_calls),
            usage=_normalize_chat_usage(_read(response, "usage")),
            request_id=_read(response, "_request_id"),
        )


def responses_items_to_chat_messages(
    input_items: str | Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Translate locally-owned Responses input items to Chat messages.

    Reasoning items are intentionally omitted: they are Responses-specific and
    may contain encrypted provider data. Function calls and their outputs remain
    explicit so the next Chat Completions request has a valid tool-call history.
    """

    if isinstance(input_items, str):
        return [{"role": "user", "content": input_items}]

    messages: list[dict[str, Any]] = []
    pending_assistant: dict[str, Any] | None = None

    def flush_assistant() -> None:
        nonlocal pending_assistant
        if pending_assistant is not None and (
            pending_assistant.get("content")
            or pending_assistant.get("tool_calls")
        ):
            messages.append(pending_assistant)
        pending_assistant = None

    for item in input_items:
        if not isinstance(item, Mapping):
            raise ModelProtocolError("model input item must be a mapping")
        item_type = item.get("type")
        if item_type == "reasoning":
            continue
        if item_type == "function_call":
            call_id = _required_string(item, "call_id", "function call")
            name = _required_string(item, "name", "function call")
            arguments = _required_string(item, "arguments", "function call")
            if pending_assistant is None:
                pending_assistant = {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [],
                }
            pending_assistant.setdefault("tool_calls", []).append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": name, "arguments": arguments},
                }
            )
            continue
        if item_type == "function_call_output":
            flush_assistant()
            call_id = _required_string(item, "call_id", "function call output")
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": _content_to_text(item.get("output", "")),
                }
            )
            continue

        role = item.get("role")
        if item_type == "message" and role is None:
            role = "assistant"
        if role in {"assistant", "user", "system", "developer"}:
            if role == "assistant":
                flush_assistant()
                pending_assistant = {
                    "role": "assistant",
                    "content": _content_to_text(item.get("content", ""))
                    or None,
                }
            else:
                flush_assistant()
                messages.append(
                    {
                        "role": role,
                        "content": _content_to_text(item.get("content", "")),
                    }
                )
            continue

        raise ModelProtocolError(
            f"unsupported Responses input item for Chat Completions: {item_type!r}"
        )

    flush_assistant()
    return messages


def _read(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _required_string(value: Any, key: str, description: str) -> str:
    result = _read(value, key)
    if not isinstance(result, str) or not result:
        raise ModelProtocolError(f"{description} is missing string field {key!r}")
    return result


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, (str, bytes)):
        parts: list[str] = []
        for part in content:
            text = _read(part, "text")
            if isinstance(text, str):
                parts.append(text)
        if parts:
            return "".join(parts)
    return json.dumps(content, ensure_ascii=False, default=str)


def _normalize_chat_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    input_tokens = _first_int(usage, "prompt_tokens", "input_tokens")
    output_tokens = _first_int(usage, "completion_tokens", "output_tokens")
    total_tokens = _first_int(usage, "total_tokens")
    if input_tokens is None or output_tokens is None or total_tokens is None:
        raise ModelProtocolError(
            "Chat Completions usage did not contain input/output/total token counts"
        )
    return TokenUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=total_tokens,
    )


def _first_int(value: Any, *keys: str) -> int | None:
    for key in keys:
        candidate = _read(value, key)
        if candidate is not None:
            try:
                return int(candidate)
            except (TypeError, ValueError) as error:
                raise ModelProtocolError(
                    f"Chat Completions usage field {key!r} is not an integer"
                ) from error
    return None
