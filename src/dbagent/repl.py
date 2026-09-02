"""Interactive DBA shell built on top of the existing local AgentLoop."""

from __future__ import annotations

import re
import sys
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from dbagent.agent import (
    AgentLoop,
    AgentRunControl,
    AgentStatus,
    ContextBudget,
    SessionContext,
    TaskMode,
)
from dbagent.agent.verification import VerificationStatus
from dbagent.config import (
    SUPPORTED_REASONING_EFFORTS,
    ConfigurationError,
    DBAgentConfig,
)
from dbagent.console import safe_print
from dbagent.discovery import WorkspaceDiscovery, select_workspace
from dbagent.llm import (
    ModelCommunicationError,
    OpenAIChatCompletionsClient,
    OpenAIResponsesClient,
)
from dbagent.model_presets import model_presets, resolve_model_selection
from dbagent.llm.provider_policy import provider_policy
from dbagent.provider_config import load_repl_config
from dbagent.session_store import SessionStore
from dbagent.tools import ToolRegistry, create_coding_registry
from dbagent.trace import TraceRecorder
from dbagent.tui import FullscreenTUI
from dbagent.ui import TerminalUI
from dbagent.web_ui import run_browser_ui
from dbagent.workspace import Workspace


InputFunction = Callable[[str], str]
ModelFactory = Callable[[DBAgentConfig], Any]


def _default_context_budget(config: DBAgentConfig) -> ContextBudget:
    """Return a local context cap with room for provider prompt overhead.

    Chat-Completions compatible routes can impose a lower effective input limit
    than the advertised model context window. DeepSeek's tool schema and
    system prompt consume a material part of that limit on every request, so
    leave a deterministic margin instead of waiting for an HTTP 400 response.
    This only controls locally-owned history; it never uses provider-side
    conversation state.
    """

    if config.provider == "deepseek":
        return ContextBudget(
            max_context_characters=24_000,
            max_task_characters=8_000,
            max_plan_characters=3_000,
            max_repository_map_characters=4_000,
            max_relevant_code_characters=6_000,
            max_compact_observations_characters=4_000,
            max_recent_observations_characters=8_000,
            max_single_observation_characters=3_000,
            max_call_arguments_characters=1_000,
            recent_observation_count=2,
            max_verification_characters=2_000,
            max_runtime_guidance_characters=2_000,
        )
    # Leave enough task budget for both local chat history and structured
    # session state before ContextManager applies its total context cap.
    return ContextBudget(max_task_characters=30_000)


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


class DBAgentRepl:
    """Run a local multi-turn conversation over the existing AgentLoop."""

    def __init__(
        self,
        *,
        workspace: Path,
        max_steps: int = 60,
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
        ui_mode: str = "cli",
    ) -> None:
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        normalized_ui_mode = ui_mode.strip().lower()
        if normalized_ui_mode not in {"cli", "tui"}:
            raise ValueError("ui_mode must be cli or tui")
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
        self._has_explicit_context_budget = context_budget is not None
        # An explicitly supplied budget is useful for tests and advanced local
        # embedding. Otherwise the selected provider determines the safe
        # default after configuration has been loaded in ``run``.
        self.context_budget = context_budget
        self.input_function = input_function or input
        self.stream = stream or sys.stdout
        # Injected input functions are deliberately kept synchronous so unit
        # tests and programmatic embedding remain deterministic.  The actual
        # DBA terminal enables a small cross-platform live-control poller.
        self._supports_live_controls = input_function is None and stream is None
        self._live_input_buffer = ""
        self._model_factory = model_factory or _create_model_client
        self._registry_factory = registry_factory
        self.ui_mode = normalized_ui_mode
        self._conversation = LocalConversation()
        self._session_context = SessionContext()
        self._session_store = SessionStore(self.workspace)
        self._session_id = self._session_store.new_session_id()
        self._session_title = ""
        self._last_state: Any | None = None
        self._queued_task: str | None = None

    def run(self) -> int:
        """Start the REPL and return a process-style exit code."""

        trace: TraceRecorder | None = None
        ui: TerminalUI | None = None
        try:
            config = load_repl_config(self.config_path)
            config = _apply_config_overrides(
                config,
                model=self.model_override,
                reasoning_effort=self.reasoning_effort_override,
            )
            if not self._has_explicit_context_budget:
                self.context_budget = _default_context_budget(config)
            # Keep the credential-bearing startup configuration only in memory.
            # Named configured presets restore it after a DeepSeek switch.
            self._startup_config = config
            trace_path = _resolve_trace_path(self.workspace, self.trace_file)
            ui = self._create_ui()
            ui.set_mode(self._mode.value)
            ui.session_start(
                workspace=self.workspace,
                model=config.model,
                api_mode=config.api_mode,
                mode=self._mode.value,
                session_id=self._session_id,
                session_state="new",
                launch_directory=(
                    self._discovery.start if self._discovery is not None else None
                ),
            )
            try:
                active_policy = provider_policy(
                    provider=config.provider,
                    api_mode=config.api_mode,
                )
                ui.info(f"Provider capabilities: {active_policy.capability_summary}")
            except ValueError:
                # The model factory below will provide the authoritative error.
                pass
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
            if ui is not None:
                closer = getattr(ui, "close", None)
                if callable(closer):
                    closer()
            safe_print(f"DBA failed to start: {error}", stream=self.stream)
            return 1

        saved_sessions = self._session_store.list_sessions()
        if saved_sessions:
            ui.info(
                f"Started a new empty session; nothing was resumed automatically. "
                f"Found {len(saved_sessions)} saved session(s). Type /sessions, "
                "then /resume <ID>."
            )
        else:
            ui.info("Started a new empty session; nothing was resumed automatically.")

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
            if ui is not None:
                closer = getattr(ui, "close", None)
                if callable(closer):
                    closer()

    def _create_ui(self) -> TerminalUI:
        """Select an explicit renderer; CLI remains safe for logs and CI."""

        if self.ui_mode == "tui":
            return FullscreenTUI(stream=self.stream)
        return TerminalUI(stream=self.stream)

    def _loop(
        self,
        *,
        config: DBAgentConfig,
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
                if self._queued_task is None:
                    continue
                line = self._queued_task
                self._queued_task = None

            prompt = self._conversation.build_prompt(line)
            prompt = self._session_context.augment_prompt(prompt)
            self._conversation.add("user", line)
            if not self._session_title:
                self._session_title = _session_title(line)
            # Checkpoint the request before a potentially long provider call.
            # A process crash can restore the conversation even though step-level
            # AgentLoop state is intentionally not replayed.
            self._persist_session(ui, run_state="in_progress")
            # AUTO authority is selected by the same model from the complete
            # request through a native tool.  Do not pre-classify the turn from
            # a local keyword list here.
            resolved_mode = self._mode
            # A resumed unfinished coding plan is explicit user intent. In auto
            # mode, commands such as "continue" retain CODE authority without
            # requiring another semantic-routing round.
            if (
                self._mode is TaskMode.AUTO
                and _is_continuation_request(line)
                and self._session_context.plan is not None
                and not self._session_context.plan.is_complete
            ):
                resolved_mode = TaskMode.CODE
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
                    "auto (semantic)"
                    if self._mode is TaskMode.AUTO
                    else resolved_mode.value
                ),
            )
            checkpoint_context: SessionContext | None = None
            checkpoint_base = SessionContext.from_dict(self._session_context.to_dict())

            def checkpoint(state: Any) -> None:
                nonlocal checkpoint_context
                checkpoint_context = SessionContext.from_dict(checkpoint_base.to_dict())
                checkpoint_context.update_from_state(state)
                self._persist_session(
                    ui,
                    context=checkpoint_context,
                    run_state="in_progress",
                )

            try:
                run_control = (
                    AgentRunControl() if self._supports_live_controls else None
                )
                agent_loop_parameters: dict[str, Any] = {
                    "max_steps": self.max_steps,
                    "mode": resolved_mode,
                    "context_budget": self.context_budget,
                    "initial_plan": resume_plan,
                    "verification_required": verification_required,
                    "trace": trace,
                    "state_checkpoint": checkpoint,
                }
                if run_control is not None:
                    agent_loop_parameters["run_control"] = run_control
                agent_loop = AgentLoop(
                    model_client,
                    registry,
                    **agent_loop_parameters,
                )
                state = self._run_agent_task(
                    agent_loop,
                    prompt,
                    workspace=self.workspace,
                    launch_directory=(
                        self._discovery.start
                        if self._discovery is not None
                        else self.workspace
                    ),
                    ui=ui,
                    run_control=run_control,
                )
            except ModelCommunicationError as error:
                if checkpoint_context is not None:
                    self._session_context = checkpoint_context
                trace.record(
                    "final",
                    step=0,
                    payload={
                        "status": "ERROR",
                        "verification_status": self._session_context.verification_status,
                        "reason": type(error).__name__,
                    },
                )
                ui.error(str(error))
                self._conversation.add(
                    "assistant",
                    "The previous model request failed after retrying; the task can "
                    "be resumed without losing the user request.",
                )
                self._persist_session(ui, run_state="interrupted")
                continue
            except KeyboardInterrupt:
                if checkpoint_context is not None:
                    self._session_context = checkpoint_context
                trace.record(
                    "final",
                    step=0,
                    payload={
                        "status": "ABORTED",
                        "verification_status": self._session_context.verification_status,
                        "reason": "KeyboardInterrupt",
                    },
                )
                self._conversation.add(
                    "assistant",
                    "The previous task was interrupted locally; its completed steps "
                    "were checkpointed and can be resumed.",
                )
                self._persist_session(ui, run_state="interrupted")
                ui.info(
                    "Task interrupted. Completed steps were checkpointed; type "
                    "a continuation request or /resume latest to continue."
                )
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

    def _run_agent_task(
        self,
        agent_loop: AgentLoop,
        prompt: str,
        *,
        workspace: Path,
        launch_directory: Path,
        ui: TerminalUI,
        run_control: AgentRunControl | None,
    ) -> Any:
        """Run synchronously for embedding, or poll live controls in DBA itself."""

        if run_control is None:
            return agent_loop.run(
                prompt,
                workspace=workspace,
                launch_directory=launch_directory,
            )

        result: dict[str, Any] = {}
        failure: list[BaseException] = []

        def worker() -> None:
            try:
                result["state"] = agent_loop.run(
                    prompt,
                    workspace=workspace,
                    launch_directory=launch_directory,
                )
            except BaseException as error:  # re-raised on the REPL thread
                failure.append(error)

        thread = threading.Thread(target=worker, name="dba-agent-run", daemon=True)
        ui.info(
            "Live controls active: type /steer <instruction> (or plain text) to "
            "guide the next safe step; type /abort to stop before the next action."
        )
        thread.start()
        while thread.is_alive():
            try:
                self._poll_live_controls(run_control, ui)
            except KeyboardInterrupt:
                run_control.request_abort("KeyboardInterrupt")
                ui.info("Abort requested; waiting for the current safe boundary.")
            thread.join(timeout=0.05)
        if failure:
            raise failure[0]
        return result["state"]

    def _poll_live_controls(
        self,
        run_control: AgentRunControl,
        ui: TerminalUI,
    ) -> None:
        """Read complete terminal lines without blocking the running agent.

        Windows uses ``msvcrt`` because ``select`` does not support console
        handles.  POSIX terminals use ``select``.  This is intentionally a
        small command channel, not a fragile full-screen terminal emulator.
        """

        if sys.platform == "win32":
            import msvcrt

            while msvcrt.kbhit():
                character = msvcrt.getwch()
                if character in {"\r", "\n"}:
                    self._accept_live_control_line(self._live_input_buffer, run_control, ui)
                    self._live_input_buffer = ""
                    self._render_live_input(ui, "\n")
                elif character == "\x03":
                    raise KeyboardInterrupt
                elif character in {"\b", "\x7f"}:
                    if self._live_input_buffer:
                        self._live_input_buffer = self._live_input_buffer[:-1]
                        self._render_live_input(ui, "\b \b")
                elif character.isprintable():
                    self._live_input_buffer += character
                    self._render_live_input(ui, character)
            return

        try:
            import select

            readable, _, _ = select.select([sys.stdin], [], [], 0)
        except (OSError, ValueError):
            return
        if readable:
            line = sys.stdin.readline()
            if line:
                self._accept_live_control_line(line, run_control, ui)

    def _accept_live_control_line(
        self,
        line: str,
        run_control: AgentRunControl,
        ui: TerminalUI,
    ) -> None:
        normalized = line.strip()
        if not normalized:
            return
        if normalized.casefold() == "/abort":
            run_control.request_abort("user issued /abort")
            ui.info("Abort requested. DBA will stop before the next model or tool action.")
            return
        if normalized.casefold().startswith("/steer"):
            message = normalized[6:].strip()
        elif normalized.casefold().startswith("/followup"):
            message = normalized[9:].strip()
        else:
            message = normalized
        if run_control.submit_steering(message):
            ui.info("Steering accepted; it will be added to the next local model context.")
        else:
            ui.error("Use /steer <instruction>, /followup <instruction>, or /abort.")

    def _write_live_input(self, value: str) -> None:
        """Echo characters consumed by Windows' nonblocking console reader."""

        try:
            self.stream.write(value)
            self.stream.flush()
        except (AttributeError, OSError):
            # The live control channel is a UX enhancement; a redirected or
            # closed terminal must not affect the locally running agent.
            pass

    def _render_live_input(self, ui: TerminalUI, fallback: str) -> None:
        """Send live input to TUI's input row or echo it in line mode."""

        setter = getattr(ui, "set_live_input", None)
        if callable(setter):
            setter(self._live_input_buffer)
            return
        self._write_live_input(fallback)

    def _handle_command(
        self,
        line: str,
        *,
        config: DBAgentConfig,
        model_client: Any,
        ui: TerminalUI,
    ) -> tuple[bool, DBAgentConfig, Any]:
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
                self._session_store.clear(self._session_id)
            except OSError as error:
                ui.error(f"Unable to clear saved session: {error}")
            else:
                self._session_id = self._session_store.new_session_id()
                self._session_title = ""
                ui.set_session_id(self._session_id, state="new")
                ui.info("Current conversation cleared. Other saved sessions were kept.")
            return True, config, model_client
        if command == "new":
            if argument:
                ui.error("/new does not accept arguments")
                return True, config, model_client
            self._conversation.clear()
            self._session_context.clear()
            self._last_state = None
            self._session_id = self._session_store.new_session_id()
            self._session_title = ""
            ui.set_session_id(self._session_id, state="new")
            ui.info(f"Started new session {self._session_id}.")
            return True, config, model_client
        if command == "sessions":
            if argument:
                ui.error("/sessions does not accept arguments")
                return True, config, model_client
            ui.render_sessions(
                self._session_store.list_sessions(),
                active_session_id=self._session_id,
            )
            return True, config, model_client
        if command == "resume":
            if not argument:
                ui.render_sessions(
                    self._session_store.list_sessions(),
                    active_session_id=self._session_id,
                )
                ui.info(
                    "Use /resume <number>, /resume <ID prefix>, or /resume latest."
                )
                return True, config, model_client
            self._resume_session(ui, self._resolve_session_selection(argument))
            return True, config, model_client
        if command == "status":
            self._render_status(config, ui)
            return True, config, model_client
        if command == "context":
            ui.render_context(self._last_state)
            return True, config, model_client
        if command == "capabilities":
            try:
                ui.render_capabilities(
                    provider_policy(provider=config.provider, api_mode=config.api_mode)
                )
            except ValueError as error:
                ui.error(str(error))
            return True, config, model_client
        if command == "steps":
            if not argument:
                ui.info(
                    f"Current step budget: {self.max_steps} model turns per task. "
                    "Use /steps N to change it for this DBA session."
                )
                return True, config, model_client
            try:
                self.max_steps = _positive_integer(argument)
            except ValueError:
                ui.error("/steps accepts one positive integer, for example /steps 48")
                return True, config, model_client
            ui.info(f"Step budget changed to {self.max_steps} model turns per task.")
            return True, config, model_client
        if command == "continue":
            if self._session_context.plan is None or self._session_context.plan.is_complete:
                ui.error("No unfinished plan is available in this session to continue.")
                return True, config, model_client
            if argument:
                try:
                    self.max_steps = _positive_integer(argument)
                except ValueError:
                    ui.error(
                        "/continue accepts an optional positive integer, for example /continue 48"
                    )
                    return True, config, model_client
            self._queued_task = "continue this task"
            ui.info(
                f"Continuing the unfinished plan with a fresh {self.max_steps}-step budget."
            )
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
                ui.render_model_options(model_presets(), current_model=config.model)
                ui.info("Use /model <alias> or /model <provider model name>.")
                return True, config, model_client
            if any(character.isspace() for character in argument):
                ui.error("/model accepts one model name, for example /model gpt-5.6-luna")
                return True, config, model_client
            try:
                next_config = resolve_model_selection(
                    argument,
                    active_config=config,
                    startup_config=self._startup_config,
                )
                next_client = self._model_factory(next_config)
            except (ConfigurationError, ModelCommunicationError) as error:
                ui.error(str(error))
                return True, config, model_client
            if not self._has_explicit_context_budget:
                self.context_budget = _default_context_budget(next_config)
            ui.info(f"Model changed to {next_config.model} for the next turn.")
            ui.info(
                "Local context profile: "
                f"{self.context_budget.max_context_characters // 1_000}k characters."
            )
            policy = provider_policy(
                provider=next_config.provider, api_mode=next_config.api_mode
            )
            ui.info(f"Provider capabilities: {policy.capability_summary}")
            if policy.controls_chat_thinking_per_turn:
                ui.info(
                    "DeepSeek keeps thinking disabled for this entire local agent "
                    "task, including finalization, to keep tool history valid."
                )
            return True, next_config, next_client
        if command == "models":
            if argument:
                ui.error("/models does not accept arguments")
            else:
                ui.render_model_options(model_presets(), current_model=config.model)
            return True, config, model_client
        if command == "reasoning":
            if not argument:
                ui.info(
                    "Current reasoning effort: "
                    f"{config.reasoning_effort}. Available: "
                    + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
                )
                return True, config, model_client
            effort = argument.lower()
            if effort not in SUPPORTED_REASONING_EFFORTS:
                ui.error(
                    "/reasoning accepts: "
                    + ", ".join(sorted(SUPPORTED_REASONING_EFFORTS))
                )
                return True, config, model_client
            next_config = DBAgentConfig(
                openai_api_key=config.openai_api_key,
                model=config.model,
                reasoning_effort=effort,
                base_url=config.base_url,
                api_mode=config.api_mode,
                provider=config.provider,
            )
            try:
                next_client = self._model_factory(next_config)
            except ModelCommunicationError as error:
                ui.error(str(error))
                return True, config, model_client
            ui.info(f"Reasoning effort changed to {effort} for the next turn.")
            if next_config.provider == "deepseek":
                ui.info(
                    "DeepSeek reasoning is stored as a preference but remains "
                    "disabled during DBA tool tasks for protocol compatibility."
                )
            return True, next_config, next_client

        ui.error(f"Unknown command '/{command}'. Type /help for available commands.")
        return True, config, model_client

    def _persist_session(
        self,
        ui: TerminalUI,
        *,
        context: SessionContext | None = None,
        run_state: str = "active",
    ) -> None:
        try:
            self._session_id = self._session_store.save(
                {
                    "title": self._session_title or "Untitled session",
                    "conversation": self._conversation.to_list(),
                    "session_context": (context or self._session_context).to_dict(),
                    "run_state": run_state,
                },
                session_id=self._session_id,
            )
            ui.set_session_id(self._session_id, state="active")
        except (OSError, ValueError) as error:
            ui.error(f"Unable to save resumable session: {error}")

    def _resume_session(self, ui: TerminalUI, session_id: str) -> None:
        try:
            saved = self._session_store.load(session_id)
            if saved is None:
                ui.error(f"No saved DBA session matches '{session_id}'.")
                return
            self._conversation.restore(saved.get("conversation", []))
            context = saved.get("session_context", {})
            if not isinstance(context, dict):
                raise ValueError("saved session_context must be an object")
            self._session_context = SessionContext.from_dict(context)
            self._last_state = None
            restored_id = saved.get("session_id")
            # A legacy single-session checkpoint is migrated to a normal session
            # file on the next save rather than mutating it during a read command.
            self._session_id = (
                self._session_store.new_session_id()
                if restored_id == "legacy"
                else str(restored_id)
            )
            title = saved.get("title")
            self._session_title = (
                title.strip()
                if isinstance(title, str) and title.strip()
                else _first_user_message(self._conversation)
            )
            ui.set_session_id(self._session_id, state="resumed")
        except (OSError, ValueError) as error:
            ui.error(f"Unable to resume saved session: {error}")
            return
        ui.render_resume_summary(
            session_id=str(restored_id),
            title=self._session_title,
            turns=self._conversation.turn_count,
            verification=self._session_context.verification_status,
            observation_count=len(self._session_context.observations),
            has_plan=self._session_context.plan is not None,
            checkpoint_state=str(saved.get("run_state") or "active"),
        )
        if self._session_context.plan is not None:
            ui.render_plan_history([self._session_context.plan])
        if self._session_context.verification_summary:
            ui.info(
                "Restored verification evidence: "
                + self._session_context.verification_summary
            )

    def _resolve_session_selection(self, selection: str) -> str:
        """Accept a list number or unambiguous ID prefix for fast local resume."""

        normalized = selection.strip()
        if normalized.casefold() == "latest":
            return "latest"
        sessions = self._session_store.list_sessions()
        if normalized.isdecimal():
            index = int(normalized) - 1
            if 0 <= index < len(sessions):
                return sessions[index].session_id
            return normalized
        matches = [
            item.session_id for item in sessions if item.session_id.startswith(normalized)
        ]
        return matches[0] if len(matches) == 1 else normalized

    def _render_status(self, config: DBAgentConfig, ui: TerminalUI) -> None:
        if self._last_state is None:
            ui.info("Session status")
            ui.info(
                f"session={self._session_id}; model={config.model}; "
                f"provider={config.provider}; api={config.api_mode}"
            )
            ui.info(
                f"reasoning={config.reasoning_effort}; task_mode={self._mode.value}; "
                f"step_budget={self.max_steps}; turns={self._conversation.turn_count}"
            )
            ui.info(f"{self._session_context.status_line()}; no task run yet")
            return
        state = self._last_state
        verification = getattr(
            getattr(state, "verification_status", None), "value", "not_run"
        )
        status = getattr(getattr(state, "status", None), "value", "unknown")
        context_usage = getattr(state, "context_usage", ())
        context = "context=unknown"
        if context_usage:
            latest = context_usage[-1]
            context = (
                f"context={latest.approximate_tokens}~tok; "
                f"compacted={latest.compacted_observations}"
            )
        ui.info("Session status")
        ui.info(
            f"session={self._session_id}; model={config.model}; "
            f"provider={config.provider}; api={config.api_mode}"
        )
        ui.info(
            f"reasoning={config.reasoning_effort}; task_mode={self._mode.value}; "
            f"step_budget={self.max_steps}; turns={self._conversation.turn_count}"
        )
        ui.info(
            f"last_status={status}; verification={verification}; {context}; "
            f"{self._session_context.status_line()}"
        )


def _session_title(request: str) -> str:
    title = " ".join(request.split())
    return title if len(title) <= 72 else title[:69] + "..."


def _first_user_message(conversation: LocalConversation) -> str:
    for item in conversation.to_list():
        if item["role"] == "user":
            return _session_title(item["content"])
    return "Untitled session"


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
        help="Workspace root (default: the exact current directory).",
    )
    parser.add_argument(
        "--discover-workspace",
        action="store_true",
        help="Opt in to searching parent directories for a project root.",
    )
    parser.add_argument(
        "--max-steps",
        type=_positive_integer,
        default=60,
        help="Maximum model turns per user request (default: 60).",
    )
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in TaskMode],
        default=TaskMode.AUTO.value,
        help="Task mode: auto, ask, or code (default: auto).",
    )
    parser.add_argument(
        "--ui",
        choices=["cli", "tui", "web"],
        default="cli",
        help="Presentation mode: scrolling CLI, full-screen TUI, or local browser UI (default: cli).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Loopback browser UI port (0 chooses a free port). Only used with --ui web.",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Do not open the local browser automatically with --ui web.",
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
        help="Explicit local provider TOML path (otherwise use the configured local lookup order).",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Override the model from the loaded provider configuration for this process.",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=sorted(SUPPORTED_REASONING_EFFORTS),
        default=None,
        help="Override reasoning effort from the loaded provider configuration for this process.",
    )
    arguments = parser.parse_args(argv)
    try:
        discovery = select_workspace(
            arguments.workspace or Path.cwd(),
            discover_parent=arguments.discover_workspace,
        )
        if arguments.ui == "web":
            if arguments.port < 0 or arguments.port > 65535:
                raise ValueError("--port must be from 0 to 65535")
            return run_browser_ui(
                discovery.root,
                config_path=arguments.config_path,
                max_steps=arguments.max_steps,
                port=arguments.port,
                open_browser=not arguments.no_browser,
            )
        return DBAgentRepl(
            workspace=discovery.root,
            max_steps=arguments.max_steps,
            trace_file=arguments.trace_file,
            config_path=arguments.config_path,
            model_override=arguments.model,
            reasoning_effort_override=arguments.reasoning_effort,
            mode=arguments.mode,
            discovery=discovery,
            ui_mode=arguments.ui,
        ).run()
    except (ConfigurationError, OSError, ValueError) as error:
        safe_print(f"DBA failed: {error}", stream=sys.stderr)
        return 1


def _create_model_client(config: DBAgentConfig) -> Any:
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
    config: DBAgentConfig,
    *,
    model: str | None,
    reasoning_effort: str | None,
) -> DBAgentConfig:
    """Apply explicit process-local overrides without mutating the environment."""

    if model is None and reasoning_effort is None:
        return config
    # Startup options must have the same semantics as the interactive
    # ``/model`` command.  In particular, a named preset can change the
    # provider, base URL, API mode, and in-memory credential—not merely the
    # model string sent to the currently configured provider.
    selected = (
        resolve_model_selection(
            model,
            active_config=config,
            startup_config=config,
        )
        if model is not None
        else config
    )
    return DBAgentConfig(
        openai_api_key=selected.openai_api_key,
        model=selected.model,
        reasoning_effort=reasoning_effort or selected.reasoning_effort,
        base_url=selected.base_url,
        api_mode=selected.api_mode,
        provider=selected.provider,
    )


def _resolve_trace_path(workspace: Path, user_path: Path | None) -> Path:
    candidate = user_path or Path(".dbagent") / "trace.jsonl"
    candidate = candidate if candidate.is_absolute() else workspace / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(workspace)
    except ValueError as error:
        raise ValueError("trace file must be inside the workspace") from error
    if resolved == workspace or resolved.exists() and resolved.is_dir():
        raise ValueError("trace file must be a file path")
    return resolved
