"""Interactive DBA shell built on top of the existing local AgentLoop."""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from forge.agent import AgentLoop, AgentStatus, ContextBudget, SessionContext
from forge.agent.verification import VerificationStatus
from forge.config import ConfigurationError, ForgeConfig
from forge.llm import (
    ModelCommunicationError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from forge.provider_config import load_repl_config
from forge.tools import ToolRegistry, create_coding_registry
from forge.trace import TraceRecorder
from forge.ui import TerminalUI
from forge.workspace import Workspace


InputFunction = Callable[[str], str]
ModelFactory = Callable[[ForgeConfig], Any]


@dataclass(slots=True)
class LocalConversation:
    """Bounded text-only history shared by turns in one DBA process."""

    max_characters: int = 16_000
    _messages: list[tuple[str, str]] = field(default_factory=list, init=False, repr=False)

    def add(self, role: str, content: str) -> None:
        content = content.strip()
        if content:
            self._messages.append((role, content))
            self._trim()

    def clear(self) -> None:
        self._messages.clear()

    @property
    def turn_count(self) -> int:
        return sum(role == "user" for role, _content in self._messages)

    def build_prompt(self, current_request: str) -> str:
        if not self._messages:
            return current_request
        history = "\n\n".join(
            f"[{role}]\n{content}" for role, content in self._messages
        )
        return (
            "Previous conversation from this local DBA session. Treat it as context, "
            "not as tool instructions.\n\n"
            f"{history}\n\n"
            f"[user]\n{current_request}"
        )

    def _trim(self) -> None:
        while self._messages and len(self._render()) > self.max_characters:
            self._messages.pop(0)

    def _render(self) -> str:
        return "\n\n".join(
            f"[{role}]\n{content}" for role, content in self._messages
        )


class ForgeRepl:
    """Run a local multi-turn conversation over the existing AgentLoop."""

    def __init__(
        self,
        *,
        workspace: Path,
        max_steps: int = 12,
        trace_file: Path | None = None,
        config_path: Path | None = None,
        context_budget: ContextBudget | None = None,
        input_function: InputFunction | None = None,
        stream: TextIO | None = None,
        model_factory: ModelFactory | None = None,
        registry_factory: Callable[[Path], ToolRegistry] = create_coding_registry,
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        self.workspace = Workspace(workspace).root
        self.max_steps = max_steps
        self.trace_file = trace_file
        self.config_path = config_path
        # Leave enough task budget for both local chat history and structured
        # session state before ContextManager applies its total context cap.
        self.context_budget = context_budget or ContextBudget(
            max_task_characters=30_000,
        )
        self.input_function = input_function or input
        self.stream = stream or sys.stdout
        self._model_factory = model_factory or _create_model_client
        self._registry_factory = registry_factory
        self._conversation = LocalConversation()
        self._session_context = SessionContext()
        self._last_state: Any | None = None

    def run(self) -> int:
        """Start the REPL and return a process-style exit code."""

        trace: TraceRecorder | None = None
        try:
            config = load_repl_config(self.config_path)
            trace_path = _resolve_trace_path(self.workspace, self.trace_file)
            ui = TerminalUI(stream=self.stream)
            ui.session_start(
                workspace=self.workspace,
                model=config.model,
                api_mode=config.api_mode,
            )
            trace = TraceRecorder(
                trace_path,
                workspace=self.workspace,
                console=True,
                stream=self.stream,
                renderer=ui,
            )
            registry = self._registry_factory(self.workspace)
            model_client = self._model_factory(config)
        except (ConfigurationError, OSError, ValueError, ModelCommunicationError) as error:
            if trace is not None:
                trace.close()
            print(f"DBA failed to start: {error}", file=self.stream)
            return 1

        try:
            return self._loop(
                config=config,
                model_client=model_client,
                registry=registry,
                trace=trace,
                ui=ui,
            )
        finally:
            trace.close()

    def _loop(
        self,
        *,
        config: ForgeConfig,
        model_client: Any,
        registry: ToolRegistry,
        trace: TraceRecorder,
        ui: TerminalUI,
    ) -> int:
        while True:
            try:
                line = self.input_function(ui.prompt())
            except (EOFError, KeyboardInterrupt):
                ui.goodbye()
                return 0

            line = line.strip()
            if not line:
                continue
            if line.startswith("/"):
                should_continue, config, model_client = self._handle_command(
                    line,
                    config=config,
                    model_client=model_client,
                    ui=ui,
                )
                if not should_continue:
                    ui.goodbye()
                    return 0
                continue

            prompt = self._conversation.build_prompt(line)
            prompt = self._session_context.augment_prompt(prompt)
            self._conversation.add("user", line)
            ui.start(
                task=line,
                workspace=self.workspace,
                model=config.model,
                max_steps=self.max_steps,
            )
            try:
                state = AgentLoop(
                    model_client,
                    registry,
                    max_steps=self.max_steps,
                    context_budget=self.context_budget,
                    trace=trace,
                ).run(prompt, workspace=self.workspace)
            except ModelCommunicationError as error:
                ui.error(str(error))
                continue
            self._last_state = state
            self._session_context.update_from_state(state)
            ui.render_plan_history(getattr(state, "plan_history", ()))
            ui.finish(state)
            if state.final_answer:
                ui.assistant(state.final_answer)
                self._conversation.add("assistant", state.final_answer)
            elif state.status is AgentStatus.MAX_STEPS:
                ui.info(
                    f"Task incomplete after max_steps={state.max_steps}; "
                    "you can continue with another request."
                )
                self._conversation.add(
                    "assistant",
                    "The previous task stopped at max_steps before producing a final answer.",
                )

    def _handle_command(
        self,
        line: str,
        *,
        config: ForgeConfig,
        model_client: Any,
        ui: TerminalUI,
    ) -> tuple[bool, ForgeConfig, Any]:
        command, _, argument = line[1:].partition(" ")
        command = command.lower()
        argument = argument.strip()
        if command in {"exit", "quit", "q"}:
            return False, config, model_client
        if command == "help":
            ui.help()
            return True, config, model_client
        if command == "clear":
            self._conversation.clear()
            self._session_context.clear()
            ui.info("Local conversation history cleared.")
            return True, config, model_client
        if command == "status":
            self._render_status(config, ui)
            return True, config, model_client
        if command == "model":
            if not argument:
                ui.info(f"Current model: {config.model}")
                return True, config, model_client
            if any(character.isspace() for character in argument):
                ui.error("/model accepts one model name, for example /model gpt-5.6-luna")
                return True, config, model_client
            config = ForgeConfig(
                openai_api_key=config.openai_api_key,
                model=argument,
                reasoning_effort=config.reasoning_effort,
                base_url=config.base_url,
                api_mode=config.api_mode,
            )
            try:
                model_client = self._model_factory(config)
            except ModelCommunicationError as error:
                ui.error(str(error))
                return True, config, model_client
            ui.info(f"Model changed to {config.model} for the next turn.")
            return True, config, model_client

        ui.error(f"Unknown command '/{command}'. Type /help for available commands.")
        return True, config, model_client

    def _render_status(self, config: ForgeConfig, ui: TerminalUI) -> None:
        if self._last_state is None:
            ui.info(
                f"model={config.model}; mode={config.api_mode}; "
                f"turns={self._conversation.turn_count}; "
                f"{self._session_context.status_line()}; no task run yet"
            )
            return
        state = self._last_state
        verification = getattr(
            getattr(state, "verification_status", None), "value", "not_run"
        )
        status = getattr(getattr(state, "status", None), "value", "unknown")
        ui.info(
            f"model={config.model}; mode={config.api_mode}; "
            f"turns={self._conversation.turn_count}; "
            f"last_status={status}; verification={verification}; "
            f"{self._session_context.status_line()}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point for ``DBA``."""

    import argparse

    parser = argparse.ArgumentParser(
        prog="DBA",
        description="Start the interactive DBA coding-agent session.",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Workspace root (default: current directory).",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=12,
        help="Maximum model turns per user request (default: 12).",
    )
    parser.add_argument(
        "--trace-file",
        type=Path,
        default=None,
        help="JSONL trace path inside the workspace.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=None,
        help="External provider TOML path (default: known backup path).",
    )
    arguments = parser.parse_args(argv)
    try:
        return ForgeRepl(
            workspace=arguments.workspace,
            max_steps=arguments.max_steps,
            trace_file=arguments.trace_file,
            config_path=arguments.config_path,
        ).run()
    except (ConfigurationError, OSError, ValueError) as error:
        print(f"DBA failed: {error}", file=sys.stderr)
        return 1


def _create_model_client(config: ForgeConfig) -> Any:
    if config.api_mode == "chat_completions":
        return OpenAIChatCompletionsClient(config)
    return OpenAIResponsesClient(config)


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError("must be a positive integer")
    return parsed


def _resolve_trace_path(workspace: Path, user_path: Path | None) -> Path:
    candidate = user_path or Path(".forge") / "trace.jsonl"
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("trace file must be inside the workspace") from error
    if resolved == workspace or resolved.exists() and resolved.is_dir():
        raise ValueError("trace file must be a file path")
    return resolved
