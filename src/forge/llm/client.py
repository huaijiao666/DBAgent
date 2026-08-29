"""Stateless OpenAI Responses API adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, cast

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

_ALLOWED_OUTPUT_ITEM_TYPES = frozenset({"message", "function_call", "reasoning"})


class _ResponsesResource(Protocol):
    def create(self, **kwargs: Any) -> Any: ...


class _SDKClient(Protocol):
    responses: _ResponsesResource


class OpenAIResponsesClient:
    """Call OpenAI without relying on server-side conversation state."""

    def __init__(
        self,
        config: ForgeConfig,
        *,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        sdk_client: _SDKClient | None = None,
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
        """Create one response and normalize provider-specific output."""

        parameters = self._build_parameters(request)
        try:
            response = self._client.responses.create(**parameters)
        except APITimeoutError as error:
            raise ModelTimeoutError(
                "OpenAI request timed out after configured SDK retries"
            ) from error
        except RateLimitError as error:
            raise ModelRateLimitError(
                "OpenAI rate limit exceeded after configured SDK retries",
                status_code=getattr(error, "status_code", 429),
                request_id=getattr(error, "request_id", None),
            ) from error
        except APIConnectionError as error:
            raise ModelConnectionError("Unable to reach the OpenAI API") from error
        except APIStatusError as error:
            raise ModelAPIError(
                "OpenAI API returned an unsuccessful status",
                status_code=getattr(error, "status_code", None),
                request_id=getattr(error, "request_id", None),
            ) from error
        except APIError as error:
            raise ModelAPIError("OpenAI API request failed") from error

        return self._normalize_response(response)

    def _build_parameters(self, request: ModelRequest) -> dict[str, Any]:
        tools: list[dict[str, Any]] = []
        for tool in request.tools:
            if not isinstance(tool, FunctionTool):
                raise TypeError("ModelRequest.tools accepts FunctionTool values only")
            tools.append(tool.to_api_dict())
        model_input: str | list[dict[str, Any]]
        if isinstance(request.input, str):
            model_input = request.input
        else:
            model_input = [dict(item) for item in request.input]

        parameters: dict[str, Any] = {
            "model": self._model,
            "input": model_input,
            "reasoning": {
                "effort": self._reasoning_effort,
                "context": "current_turn",
            },
            "text": {"format": {"type": "text"}},
            "tools": tools,
            "include": ["reasoning.encrypted_content"],
            "store": False,
            "background": False,
        }
        if request.instructions is not None:
            parameters["instructions"] = request.instructions
        if request.max_output_tokens is not None:
            parameters["max_output_tokens"] = request.max_output_tokens
        if tools:
            parameters["tool_choice"] = "auto"
            parameters["parallel_tool_calls"] = request.parallel_tool_calls
        return parameters

    def _normalize_response(self, response: Any) -> ModelResponse:
        response_id = getattr(response, "id", None)
        if not isinstance(response_id, str) or not response_id:
            raise ModelProtocolError("OpenAI response did not contain a response id")

        response_error = getattr(response, "error", None)
        if response_error is not None:
            error_data = _object_to_dict(response_error)
            code = error_data.get("code", "unknown")
            message = error_data.get("message", "no error message")
            raise ModelAPIError(
                f"OpenAI response failed ({code}): {message}",
                request_id=getattr(response, "_request_id", None),
                response_id=response_id,
            )

        output_items = tuple(
            _object_to_dict(item) for item in getattr(response, "output", ())
        )
        function_calls: list[FunctionCall] = []
        for item in output_items:
            item_type = item.get("type")
            if item_type not in _ALLOWED_OUTPUT_ITEM_TYPES:
                raise ModelProtocolError(
                    f"OpenAI returned disallowed output item type: {item_type!r}"
                )
            if item_type == "function_call":
                function_calls.append(
                    FunctionCall(
                        call_id=_required_string(item, "call_id"),
                        name=_required_string(item, "name"),
                        arguments_json=_required_string(item, "arguments"),
                        status=cast(str | None, item.get("status")),
                    )
                )

        usage = _normalize_usage(getattr(response, "usage", None))
        incomplete = getattr(response, "incomplete_details", None)
        return ModelResponse(
            response_id=response_id,
            model=str(getattr(response, "model", self._model)),
            status=str(getattr(response, "status", "unknown")),
            output_text=str(getattr(response, "output_text", "")),
            output_items=output_items,
            function_calls=tuple(function_calls),
            usage=usage,
            request_id=getattr(response, "_request_id", None),
            incomplete_details=(
                _object_to_dict(incomplete) if incomplete is not None else None
            ),
        )


def _object_to_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump(mode="json")
        if isinstance(dumped, Mapping):
            return dict(dumped)
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        if isinstance(dumped, Mapping):
            return dict(dumped)
    raise ModelProtocolError(
        f"OpenAI SDK value cannot be converted to a mapping: {type(value).__name__}"
    )


def _required_string(item: Mapping[str, Any], field_name: str) -> str:
    value = item.get(field_name)
    if not isinstance(value, str) or not value:
        raise ModelProtocolError(
            f"function_call item is missing string field {field_name!r}"
        )
    return value


def _normalize_usage(usage: Any) -> TokenUsage | None:
    if usage is None:
        return None
    return TokenUsage(
        input_tokens=int(getattr(usage, "input_tokens")),
        output_tokens=int(getattr(usage, "output_tokens")),
        total_tokens=int(getattr(usage, "total_tokens")),
    )
