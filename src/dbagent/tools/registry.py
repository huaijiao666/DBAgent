"""Registration and safe dispatch of local function tools."""

from __future__ import annotations

import json
from collections.abc import Iterable
from json import JSONDecodeError
from typing import Any

from dbagent.llm import FunctionCall, FunctionTool
from dbagent.tools.models import ToolDefinition, ToolObservation, ToolResult


class ToolRegistry:
    """Own the model-visible schemas and corresponding local handlers."""

    def __init__(self, definitions: Iterable[ToolDefinition] = ()) -> None:
        self._definitions: dict[str, ToolDefinition] = {}
        for definition in definitions:
            self.register(definition)

    def register(self, definition: ToolDefinition) -> None:
        name = definition.schema.name
        if name in self._definitions:
            raise ValueError(f"tool is already registered: {name}")
        self._definitions[name] = definition

    def schemas(self) -> tuple[FunctionTool, ...]:
        """Return schemas in deterministic registration order."""

        return tuple(definition.schema for definition in self._definitions.values())

    @property
    def names(self) -> tuple[str, ...]:
        """Return registered names in deterministic order."""

        return tuple(self._definitions)

    def select(self, allowed_names: Iterable[str]) -> ToolRegistry:
        """Copy only explicitly allowed tools, preserving registration order."""

        allowed = set(allowed_names)
        return ToolRegistry(
            definition
            for name, definition in self._definitions.items()
            if name in allowed
        )

    def clone(self) -> ToolRegistry:
        """Copy registrations so a run can add run-local tools safely."""

        return ToolRegistry(self._definitions.values())

    def dispatch(self, call: FunctionCall) -> ToolObservation:
        """Execute one function call and convert every ordinary failure to output."""

        definition = self._definitions.get(call.name)
        if definition is None:
            return ToolObservation(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                content=f"Unknown tool: {call.name}",
            )

        try:
            arguments = json.loads(call.arguments_json)
        except JSONDecodeError as error:
            return ToolObservation(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                content=f"Invalid JSON arguments: {error.msg}",
            )
        if not isinstance(arguments, dict):
            return ToolObservation(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                content="Tool arguments must be a JSON object",
            )

        try:
            handler_result = definition.handler(arguments)
        except Exception as error:
            return ToolObservation(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                content=f"{type(error).__name__}: {error}",
            )
        try:
            result = (
                handler_result.content
                if isinstance(handler_result, ToolResult)
                else handler_result
            )
            json.dumps(result, ensure_ascii=False)
        except (TypeError, ValueError) as error:
            return ToolObservation(
                call_id=call.call_id,
                tool_name=call.name,
                success=False,
                content=f"Tool returned a non-JSON result: {error}",
            )

        return ToolObservation(
            call_id=call.call_id,
            tool_name=call.name,
            success=(
                handler_result.success
                if isinstance(handler_result, ToolResult)
                else True
            ),
            content=result,
        )
