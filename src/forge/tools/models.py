"""Explicit values shared by local tool registration and dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from forge.llm import FunctionTool

ToolHandler: TypeAlias = Callable[[Mapping[str, Any]], str]


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    """A model-visible schema paired with one local implementation."""

    schema: FunctionTool
    handler: ToolHandler


@dataclass(frozen=True, slots=True)
class ToolObservation:
    """The explicit result of dispatching one model-requested function call."""

    call_id: str
    tool_name: str
    success: bool
    content: str

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
