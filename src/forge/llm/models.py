"""Provider-neutral request and response values for model communication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, TypeAlias

ModelInput: TypeAlias = str | Sequence[Mapping[str, Any]]


@dataclass(frozen=True, slots=True)
class FunctionTool:
    """A local function exposed to the model through JSON Schema."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    strict: bool = True

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("function tool name must not be empty")
        if not self.description.strip():
            raise ValueError("function tool description must not be empty")

    def to_api_dict(self) -> dict[str, Any]:
        """Return the Responses API representation of this function."""

        return {
            "type": "function",
            "name": self.name,
            "description": self.description,
            "parameters": dict(self.parameters),
            "strict": self.strict,
        }


@dataclass(frozen=True, slots=True)
class ModelRequest:
    """One stateless model request prepared by a future agent loop."""

    input: ModelInput
    instructions: str | None = None
    tools: tuple[FunctionTool, ...] = field(default_factory=tuple)
    max_output_tokens: int | None = None
    parallel_tool_calls: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.input, str) and not self.input.strip():
            raise ValueError("model input must not be empty")
        if self.max_output_tokens is not None and self.max_output_tokens <= 0:
            raise ValueError("max_output_tokens must be positive")


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Token counts reported by the Responses API."""

    input_tokens: int
    output_tokens: int
    total_tokens: int


@dataclass(frozen=True, slots=True)
class FunctionCall:
    """A function invocation requested by the model but not yet executed."""

    call_id: str
    name: str
    arguments_json: str
    status: str | None = None


@dataclass(frozen=True, slots=True)
class ModelResponse:
    """Normalized result of one Responses API call."""

    response_id: str
    model: str
    status: str
    output_text: str
    output_items: tuple[dict[str, Any], ...]
    function_calls: tuple[FunctionCall, ...]
    usage: TokenUsage | None
    request_id: str | None = None
    incomplete_details: dict[str, Any] | None = None
