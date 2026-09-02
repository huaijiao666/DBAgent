"""Stateless Chat Completions compatibility adapter.

The rest of DBAgent uses Responses-shaped context and normalized model values.  This
module translates that local representation at the provider boundary so a provider
that only exposes ``/chat/completions`` can still participate in the same agent loop.
"""

from __future__ import annotations

import json
import re
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

from dbagent.config import DBAgentConfig
from dbagent.llm.errors import (
    ModelAPIError,
    ModelConfigurationError,
    ModelConnectionError,
    ModelProtocolError,
    ModelRateLimitError,
    ModelTextualToolMarkupError,
    ModelTimeoutError,
)
from dbagent.llm.models import (
    FunctionCall,
    FunctionTool,
    ModelRequest,
    ModelResponse,
    TokenUsage,
)
from dbagent.llm.provider_policy import provider_policy


class _ChatCompletionsResource(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _ChatSDKClient(Protocol):
    chat: Any


class OpenAIChatCompletionsClient:
    """Call ``chat.completions.create`` without server-side conversation state."""

    def __init__(
        self,
        config: DBAgentConfig,
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
        self._policy = provider_policy(
            provider=config.provider, api_mode=config.api_mode
        )
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
        """Create one chat completion and normalize it to DBAgent's model contract."""

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
            status_code = getattr(error, "status_code", None)
            request_id = getattr(error, "request_id", None)
            raise ModelAPIError(
                _provider_status_message(
                    "Chat Completions provider returned an unsuccessful status",
                    status_code=status_code,
                    request_id=request_id,
                    body=getattr(error, "body", None),
                ),
                status_code=status_code,
                request_id=request_id,
            ) from error
        except APIError as error:
            raise ModelAPIError("Chat Completions request failed") from error

        return self._normalize_response(
            response,
            retain_reasoning=self._policy.replay_chat_reasoning_content,
        )

    def _build_parameters(self, request: ModelRequest) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if not isinstance(tool, FunctionTool):
                raise TypeError("ModelRequest.tools accepts FunctionTool values only")
            tools.append(tool.to_chat_api_dict())

        # DeepSeek rejects ``tool_choice`` in thinking mode. A local agent's
        # finalization turn has already decided that no tools are allowed, so
        # omit tool definitions entirely rather than sending tool_choice=none.
        provider_tools = tools
        if (
            self._policy.controls_chat_thinking_per_turn
            and request.tool_choice == "none"
        ):
            provider_tools = []

        messages = responses_items_to_chat_messages(
            request.input,
            # A local Coding Agent must not mix DeepSeek thinking and
            # non-thinking requests in one transcript: after thinking is
            # enabled, the provider requires every prior reasoning_content
            # value to be replayed verbatim. That hidden, unbounded state would
            # violate DBAgent's local context budget. The reviewed DeepSeek
            # policy therefore retains no provider-private reasoning content.
            replay_reasoning_content=self._policy.replay_chat_reasoning_content,
            require_assistant_content_for_tool_calls=(
                self._policy.requires_assistant_content_for_tool_calls
            ),
        )
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
        if self._policy.controls_chat_thinking_per_turn:
            # Keep one coherent protocol for the *entire* locally-owned task,
            # including the final no-tool response. Re-enabling thinking only
            # for finalization causes HTTP 400 because previous tool-turn
            # assistant messages have no DeepSeek reasoning_content to replay.
            parameters["extra_body"] = {"thinking": {"type": "disabled"}}
        if provider_tools:
            parameters["tools"] = provider_tools
            parameters["tool_choice"] = request.tool_choice
            parameters["parallel_tool_calls"] = request.parallel_tool_calls
        return parameters

    def _normalize_response(
        self, response: Any, *, retain_reasoning: bool
    ) -> ModelResponse:
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
        if _contains_textual_tool_markup(output_text):
            # Textual markup is not a function call. Executing it would bypass
            # the tool schema/dispatch boundary, so fail clearly instead of
            # presenting it as a final assistant answer.
            raise ModelTextualToolMarkupError(
                "Chat provider returned textual DSML tool markup instead of "
                "native function calls; no tool was executed"
            )
        reasoning_content = _read(message, "reasoning_content")
        if not isinstance(reasoning_content, str):
            reasoning_content = ""
        raw_tool_calls = _read(message, "tool_calls", ()) or ()
        if not isinstance(raw_tool_calls, Sequence) or isinstance(
            raw_tool_calls, (str, bytes)
        ):
            raise ModelProtocolError(
                "Chat Completions message tool_calls must be a sequence"
            )

        function_calls: list[FunctionCall] = []
        output_items: list[dict[str, Any]] = []
        if reasoning_content and retain_reasoning:
            # DeepSeek requires this value to be replayed alongside an
            # assistant tool-call message in thinking mode. It is retained in
            # the locally-owned context only; it is not printed as an answer.
            output_items.append({"type": "reasoning", "content": reasoning_content})
        output_items.append(
            {
                "type": "message",
                "role": "assistant",
                "content": output_text,
            }
        )
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
            raw_arguments = _required_string(
                function, "arguments", "function tool call"
            )
            arguments = _unwrap_compatibility_arguments(raw_arguments)
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
    *,
    replay_reasoning_content: bool = False,
    require_assistant_content_for_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Translate locally-owned Responses input items to Chat messages.

    Encrypted Responses reasoning is always omitted. Plain local reasoning is
    replayed as Chat Completions ``reasoning_content`` only when the selected
    provider policy requires it. Function calls and outputs remain explicit.
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
            if not replay_reasoning_content:
                continue
            reasoning_content = _content_to_text(item.get("content", ""))
            if reasoning_content:
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                pending_assistant["reasoning_content"] = reasoning_content
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
            if (
                require_assistant_content_for_tool_calls
                and pending_assistant.get("content") is None
            ):
                pending_assistant["content"] = ""
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
                # A local reasoning item immediately precedes its assistant
                # message. Coalesce them so Chat Completions receives the
                # required reasoning_content and tool_calls in one message.
                if pending_assistant is not None and not (
                    "reasoning_content" in pending_assistant
                    and pending_assistant.get("content") is None
                    and not pending_assistant.get("tool_calls")
                ):
                    flush_assistant()
                if pending_assistant is None:
                    pending_assistant = {"role": "assistant", "content": None}
                pending_assistant["content"] = (
                    _content_to_text(item.get("content", "")) or None
                )
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


def _contains_textual_tool_markup(content: str) -> bool:
    """Recognize provider-emitted DSML without ever interpreting it as a tool."""

    normalized = content.lower()
    return "dsml" in normalized and "tool_calls" in normalized and "invoke" in normalized


def _unwrap_compatibility_arguments(arguments_json: str) -> str:
    """Normalize a known Chat-Completions wrapper without relaxing tool schemas.

    Some compatible providers occasionally encode a tool's JSON object as
    ``{"arguments": {...}}``. The wrapper is transport noise, not a project
    parameter. Unwrap only an exact single-key mapping; all resulting arguments
    still pass through the normal local ToolRegistry JSON/schema validation.
    Malformed JSON and all other shapes are intentionally returned unchanged so
    dispatch can report the ordinary, observable tool error.
    """

    try:
        value = json.loads(arguments_json)
    except json.JSONDecodeError:
        return arguments_json
    if (
        isinstance(value, Mapping)
        and set(value) == {"arguments"}
        and isinstance(value["arguments"], Mapping)
    ):
        return json.dumps(value["arguments"], ensure_ascii=False, separators=(",", ":"))
    return arguments_json


def _provider_status_message(
    prefix: str, *, status_code: object, request_id: object, body: object = None
) -> str:
    """Expose a bounded, redacted provider diagnostic without logging bodies."""

    details: list[str] = []
    if isinstance(status_code, int):
        details.append(f"HTTP {status_code}")
    if isinstance(request_id, str) and request_id:
        details.append(f"request_id={request_id}")
    detail = _safe_provider_error_detail(body)
    if detail:
        details.append(detail)
    return prefix if not details else f"{prefix} ({', '.join(details)})"


def _safe_provider_error_detail(body: object) -> str:
    """Extract only standard error metadata and redact likely credentials."""

    payload = body.get("error", body) if isinstance(body, Mapping) else None
    if not isinstance(payload, Mapping):
        return ""
    parts: list[str] = []
    for key in ("type", "code", "message"):
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and str(value).strip():
            text = _redact_provider_detail(str(value).strip())
            if text:
                parts.append(f"{key}={text}")
    return "; ".join(parts)[:360]


def _redact_provider_detail(value: str) -> str:
    value = re.sub(
        r"(?i)(api[_ -]?key|authorization|bearer)\s*[:=]\s*\S+",
        r"\1=[redacted]",
        value,
    )
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[redacted]", value)
    return " ".join(value.split())[:220]


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
