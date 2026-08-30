"""Interactive DBA shell built on top of the existing local AgentLoop."""

from __future__ import annotations

import re
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from forge.agent import (
    AgentLoop,
    AgentStatus,
    ContextBudget,
    SessionContext,
    TaskMode,
    resolve_task_mode,
)
from forge.agent.verification import VerificationStatus
from forge.config import (
    SUPPORTED_REASONING_EFFORTS,
    ConfigurationError,
    ForgeConfig,
)
from forge.console import safe_print
from forge.discovery import WorkspaceDiscovery, discover_workspace
from forge.llm import (
    ModelCommunicationError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from forge.provider_config import load_repl_config
from forge.session_store import SessionStore
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

    def to_list(self) -> list[dict[str, str]]:
        return [
            {"role": role, "content": content}
            for role, content in self._messages
        ]

    def restore(self, value: object) -> None:
        if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
            raise ValueError("saved conversation must be an array")
        messages: list[tuple[str, str]] = []
        for item in value:
            if not isinstance(item, dict):
                raise ValueError("each saved conversation message must be an object")
            role = item.get("role")
            content = item.get("content")
            if role not in {"user", "assistant"} or not isinstance(content, str):
                raise ValueError("saved conversation message is invalid")
            if content.strip():
                messages.append((role, content.strip()))
        self._messages = messages
        self._trim()

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
        model_override: str | None = None,
        reasoning_effort_override: str | None = None,
        mode: TaskMode | str = TaskMode.AUTO,
        discovery: WorkspaceDiscovery | None = None,
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
        self.model_override = _optional_override(model_override, "model_override")
        self.reasoning_effort_override = _optional_override(
            reasoning_effort_override,
            "reasoning_effort_override",
        )
        self._mode = mode if isinstance(mode, TaskMode) else TaskMode(mode)
        self._discovery = discovery
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
        self._session_store = SessionStore(self.workspace)
        self._last_state: Any | None = None

    def run(self) -> int:
        """Start the REPL and return a process-style exit code."""

        trace: TraceRecorder | None = None
        try:
            config = load_repl_config(self.config_path)
            config = _apply_config_overrides(
                config,
                model=self.model_override,
                reasoning_effort=self.reasoning_effort_override,
            )
            trace_path = _resolve_trace_path(self.workspace, self.trace_file)
            ui = TerminalUI(stream=self.stream)
            ui.set_mode(self._mode.value)
            ui.session_start(
                workspace=self.workspace,
                model=config.model,
                api_mode=config.api_mode,
                mode=self._mode.value,
                launch_directory=(
                    self._discovery.start if self._discovery is not None else None
                ),
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
            safe_print(f"DBA failed to start: {error}", stream=self.stream)
            return 1

        if self._session_store.exists:
            ui.info("Saved workspace session found. Type /resume to restore it.")

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
            try:
                line.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                ui.error(
                    "Input contains invalid Unicode from the terminal pipeline; "
                    "type or paste it directly in a UTF-8 terminal."
                )
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
            # Checkpoint the request before a potentially long provider call.
            # A process crash can restore the conversation even though step-level
            # AgentLoop state is intentionally not replayed.
            self._persist_session(ui)
            resolved_mode = resolve_task_mode(line, self._mode)
            resume_plan = (
                self._session_context.plan
                if (
                    resolved_mode is TaskMode.CODE
                    and _is_continuation_request(line)
                    and self._session_context.plan is not None
                    and not self._session_context.plan.is_complete
                )
                else None
            )
            verification_required = (
                resume_plan is not None
                and self._session_context.verification_status in {"failed", "stale"}
            )
            ui.start(
                task=line,
                workspace=self.workspace,
                model=config.model,
                max_steps=self.max_steps,
                mode=(
                    f"{resolved_mode.value} (auto)"
                    if self._mode is TaskMode.AUTO
                    else resolved_mode.value
                ),
            )
            try:
                state = AgentLoop(
                    model_client,
                    registry,
                    max_steps=self.max_steps,
                    mode=resolved_mode,
                    context_budget=self.context_budget,
                    initial_plan=resume_plan,
                    verification_required=verification_required,
                    trace=trace,
                ).run(
                    prompt,
                    workspace=self.workspace,
                    launch_directory=(
                        self._discovery.start
                        if self._discovery is not None
                        else self.workspace
                    ),
                )
            except ModelCommunicationError as error:
                ui.error(str(error))
                self._conversation.add(
                    "assistant",
                    "The previous model request failed after retrying; the task can "
                    "be resumed without losing the user request.",
                )
                self._persist_session(ui)
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
            self._persist_session(ui)

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
            self._last_state = None
            try:
                self._session_store.clear()
            except OSError as error:
                ui.error(f"Unable to clear saved session: {error}")
            else:
                ui.info("Local and saved conversation history cleared.")
            return True, config, model_client
        if command == "resume":
            if argument:
                ui.error("/resume does not accept arguments")
                return True, config, model_client
            self._resume_session(ui)
            return True, config, model_client
        if command == "status":
            self._render_status(config, ui)
            return True, config, model_client
        if command == "plan":
            if self._session_context.plan is None:
                ui.info("No plan is currently retained in this DBA session.")
            else:
                ui.render_plan_history([self._session_context.plan])
            return True, config, model_client
        if command == "mode":
            if not argument:
                ui.info(f"Current task mode: {self._mode.value}")
                return True, config, model_client
            try:
                self._mode = TaskMode(argument.lower())
            except ValueError:
                ui.error("/mode accepts auto, ask, or code")
                return True, config, model_client
            ui.set_mode(self._mode.value)
            ui.info(f"Task mode changed to {self._mode.value}.")
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

    def _persist_session(self, ui: TerminalUI) -> None:
        try:
            self._session_store.save(
                {
                    "conversation": self._conversation.to_list(),
                    "session_context": self._session_context.to_dict(),
                }
            )
        except (OSError, ValueError) as error:
            ui.error(f"Unable to save resumable session: {error}")

    def _resume_session(self, ui: TerminalUI) -> None:
        try:
            saved = self._session_store.load()
            if saved is None:
                ui.info("No saved DBA session exists in this workspace.")
                return
            self._conversation.restore(saved.get("conversation", []))
            context = saved.get("session_context", {})
            if not isinstance(context, dict):
                raise ValueError("saved session_context must be an object")
            self._session_context = SessionContext.from_dict(context)
            self._last_state = None
        except (OSError, ValueError) as error:
            ui.error(f"Unable to resume saved session: {error}")
            return
        ui.info(
            "Resumed workspace session: "
            f"{self._conversation.turn_count} user turn(s), "
            f"{self._session_context.status_line()}."
        )

    def _render_status(self, config: ForgeConfig, ui: TerminalUI) -> None:
        if self._last_state is None:
            ui.info(
                f"model={config.model}; api={config.api_mode}; task_mode={self._mode.value}; "
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
            f"model={config.model}; api={config.api_mode}; task_mode={self._mode.value}; "
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
        default=None,
        help="Workspace root (default: auto-detect from current directory).",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=24,
        help="Maximum model turns per user request (default: 24).",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TaskMode],
        default=TaskMode.AUTO.value,
        help="Task mode: auto, ask, or code (default: auto).",
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
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model from the backup config for this process.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(SUPPORTED_REASONING_EFFORTS),
        default=None,
        help="Override reasoning effort from the backup config for this process.",
    )
    arguments = parser.parse_args(argv)
    try:
        discovery = (
            discover_workspace(Path.cwd())
            if arguments.workspace is None
            else WorkspaceDiscovery(
                start=arguments.workspace.resolve(strict=True),
                root=arguments.workspace.resolve(strict=True),
                markers=(),
            )
        )
        return ForgeRepl(
            workspace=discovery.root,
            max_steps=arguments.max_steps,
            trace_file=arguments.trace_file,
            config_path=arguments.config_path,
            model_override=arguments.model,
            reasoning_effort_override=arguments.reasoning_effort,
            mode=arguments.mode,
            discovery=discovery,
        ).run()
    except (ConfigurationError, OSError, ValueError) as error:
        safe_print(f"DBA failed: {error}", stream=sys.stderr)
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


def _optional_override(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{name} must not be empty")
    return normalized


_CONTINUATION_REQUEST = re.compile(
    r"(?:继续|接着|续作|完成剩余|完成上次|继续完成|"
    r"\bcontinue\b|\bresume\b|\bfinish\s+(?:it|this|the\s+task)\b)",
    re.IGNORECASE,
)


def _is_continuation_request(text: str) -> bool:
    return bool(_CONTINUATION_REQUEST.search(text))


def _apply_config_overrides(
    config: ForgeConfig,
    *,
    model: str | None,
    reasoning_effort: str | None,
) -> ForgeConfig:
    """Apply explicit process-local overrides without touching backup or env."""

    if model is None and reasoning_effort is None:
        return config
    return ForgeConfig.from_env(
        {
            "OPENAI_API_KEY": config.openai_api_key or "",
            "FORGE_BASE_URL": config.base_url or "",
            "FORGE_API_MODE": config.api_mode,
            "FORGE_MODEL": model or config.model,
            "FORGE_REASONING_EFFORT": reasoning_effort or config.reasoning_effort,
        }
    )


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
