"""Explicit values shared by local tool registration and dispatch."""

from __future__ import annotations

from copy import deepcopy
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from dbagent.llm import FunctionTool

ToolHandler: TypeAlias = Callable[[Mapping[str, Any]], Any]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A model-visible schema paired with one local implementation."""

    schema: FunctionTool
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Allow a handler to report a structured success or failure explicitly."""

    success: bool
    content: Any


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """The explicit result of dispatching one model-requested function call."""

    call_id: str
    tool_name: str
    success: bool
    content: Any

    def to_model_input(self) -> dict[str, str]:
        """Serialize this observation as a Responses API input item."""

        key = "result" if self.success else "error"
        output = json.dumps(
            {"ok": self.success, key: self.content},
            ensure_ascii=False,
        )
        return {
            "type": "function_call_output",
            "call_id": self.call_id,
            "output": output,
        }


def object_schema(
    properties: dict[str, Any], *, required: list[str]
) -> dict[str, Any]:
    """Build the strict object schema shared by local function tools.

    OpenAI-style strict function schemas require every property name to appear
    in ``required``. A field that is optional to the *local handler* is thus
    required-but-nullable at the model boundary. A handler can treat ``null``
    like an omitted optional argument while strict providers receive a schema
    they accept.
    """

    unknown = set(required).difference(properties)
    if unknown:
        raise ValueError(
            "required fields must be declared in properties: "
            + ", ".join(sorted(unknown))
        )

    required_names = set(required)
    normalized: dict[str, Any] = {}
    for name, definition in properties.items():
        if not isinstance(definition, Mapping):
            raise TypeError(f"schema for property '{name}' must be a mapping")
        normalized_definition = deepcopy(dict(definition))
        if name not in required_names:
            normalized_definition = _nullable_schema(normalized_definition)
            description = normalized_definition.get("description")
            if isinstance(description, str) and "null" not in description.lower():
                normalized_definition["description"] = (
                    description + " Use null to request the local default."
                )
        normalized[name] = normalized_definition

    return {
        "type": "object",
        "properties": normalized,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Make one simple JSON-schema property accept ``null`` as its default."""

    type_value = schema.get("type")
    if isinstance(type_value, str):
        schema["type"] = [type_value, "null"]
    elif isinstance(type_value, list):
        schema["type"] = [
            *type_value,
            *(["null"] if "null" not in type_value else []),
        ]
    else:
        # All current local schemas have ``type``. ``anyOf`` keeps the helper
        # correct and explicit should a future tool use schema composition.
        schema = {"anyOf": [schema, {"type": "null"}]}
    return schema
