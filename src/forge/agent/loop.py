"""The first minimal Coding Agent loop."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from forge.agent.state import AgentState, AgentStatus
from forge.llm import ModelRequest, ModelResponse
from forge.tools import ToolRegistry

DEFAULT_AGENT_INSTRUCTIONS = """You are Forge, a local coding agent.
Inspect the workspace, reproduce problems, make the smallest necessary edits, and
verify changes with deterministic commands. Tool paths and command working
directories are relative to the workspace root. For Python repositories, prefer
get_repo_map, search_symbol, and read_symbol before broad file reads. Prefer
apply_patch for changes to existing files, use create_file for new files, and
inspect git_diff after editing. Use write_file only when a patch cannot safely
express a small whole-file change. Never claim a command passed unless its returned
status proves it. When the task is verified, answer without calling another tool.
"""


class ModelClient(Protocol):
    def create_response(self, request: ModelRequest) -> ModelResponse: ...


class AgentLoop:
    """Alternate between model responses and local tool observations."""

    def __init__(
        self,
        model_client: ModelClient,
        tool_registry: ToolRegistry,
        *,
        max_steps: int = 12,
        instructions: str = DEFAULT_AGENT_INSTRUCTIONS,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self._model_client = model_client
        self._tool_registry = tool_registry
        self._max_steps = max_steps
        self._instructions = instructions

    def run(self, task: str, *, workspace: Path) -> AgentState:
        state = AgentState.start(
            task=task,
            workspace=workspace,
            max_steps=self._max_steps,
        )
        tool_schemas = self._tool_registry.schemas()

        while state.step < state.max_steps:
            state.step += 1
            response = self._model_client.create_response(
                ModelRequest(
                    input=tuple(state.context),
                    instructions=self._instructions,
                    tools=tool_schemas,
                    parallel_tool_calls=True,
                )
            )
            state.response_ids.append(response.response_id)
            state.context.extend(response.output_items)

            if not response.function_calls:
                state.final_answer = response.output_text
                state.status = AgentStatus.COMPLETED
                return state

            for call in response.function_calls:
                state.tool_calls.append(call)
                observation = self._tool_registry.dispatch(call)
                state.observations.append(observation)
                state.context.append(observation.to_model_input())

        state.status = AgentStatus.MAX_STEPS
        return state
