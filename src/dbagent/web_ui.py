"""Local-only browser control surface for DBAgent.

The browser never receives provider credentials and never runs tools itself.
It talks to this process over a loopback HTTP server; every model request,
filesystem operation, patch, and command still crosses the existing local
AgentLoop and ToolRegistry boundaries.
"""

from __future__ import annotations

import json
import secrets
import threading
import time
import webbrowser
from collections import deque
from collections.abc import Mapping
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

from dbagent.agent import AgentLoop, AgentRunControl, AgentState, SessionContext, TaskMode
from dbagent.config import ConfigurationError, DBAgentConfig, SUPPORTED_REASONING_EFFORTS
from dbagent.execution import CommandExecutor
from dbagent.llm import ModelCommunicationError, OpenAIChatCompletionsClient, OpenAIResponsesClient
from dbagent.model_presets import model_presets, resolve_model_selection
from dbagent.provider_config import load_repl_config
from dbagent.session_store import SessionStore
from dbagent.trace import TraceRecorder, sanitize_for_storage
from dbagent.tools import create_coding_registry
from dbagent.workspace import Workspace, is_local_secret_name

_ASSET_DIRECTORY = Path(__file__).with_name("web_assets")
_MAX_EVENTS = 500
_MAX_PREVIEW_CHARACTERS = 160_000


class BrowserAgentController:
    """Own one local web session and expose only safe, presentation-ready data."""

    def __init__(
        self,
        workspace: Path,
        *,
        config_path: Path | None = None,
        max_steps: int = 80,
    ) -> None:
        self._lock = threading.RLock()
        self._event_condition = threading.Condition(self._lock)
        self._workspace = Workspace(workspace).root
        self._session_store = SessionStore(self._workspace)
        self._session_id = self._session_store.new_session_id()
        self._session_title = ""
        self._session_context = SessionContext()
        self._conversation: list[dict[str, str]] = []
        self._config = load_repl_config(config_path)
        self._startup_config = self._config
        self._max_steps = max_steps
        self._events: deque[dict[str, Any]] = deque(maxlen=_MAX_EVENTS)
        self._next_event_id = 1
        self._active = False
        self._state: AgentState | None = None
        self._last_error = ""
        self._started_at: float | None = None
        self._run_control: AgentRunControl | None = None
        self._run_id: str | None = None
        self._metrics: dict[str, Any] = _empty_run_metrics()
        # A completed AgentState is only available after the worker returns.
        # Keep the last trace-derived plan separately so the browser can show
        # live updates and preserve a completed plan until the next task makes
        # a genuine replacement plan.
        self._last_plan: dict[str, Any] | None = None

    @property
    def workspace(self) -> Path:
        with self._lock:
            return self._workspace

    def status(self) -> dict[str, Any]:
        with self._lock:
            state = self._state
            plan = getattr(state, "plan", None)
            elapsed = self._metrics["elapsed_seconds"]
            if self._active and self._started_at is not None:
                elapsed = round(time.monotonic() - self._started_at, 1)
            return {
                "workspace": str(self._workspace),
                "model": self._config.model,
                "reasoning_effort": self._config.reasoning_effort,
                "provider": self._config.provider,
                "api_mode": self._config.api_mode,
                "max_steps": self._max_steps,
                "active": self._active,
                "run_id": self._run_id,
                "elapsed_seconds": elapsed,
                "current_step": self._metrics["current_step"],
                "current_tool": self._metrics["current_tool"],
                "last_event": self._metrics["last_event"],
                "token_usage": dict(self._metrics["token_usage"]),
                "context_usage": dict(self._metrics["context_usage"]),
                "latest_verification": self._metrics["latest_verification"],
                "changed_files": list(self._metrics["changed_files"]),
                "last_error": self._last_error,
                "run": _state_summary(state),
                "plan": plan.to_dict() if plan is not None else self._last_plan,
                "session": {
                    "id": self._session_id,
                    "title": self._session_title or "未命名会话",
                    "turns": self._session_context.turns,
                    "verification": self._session_context.verification_status,
                },
                "model_options": [
                    {
                        "alias": preset.alias,
                        "model": preset.model,
                        "label": preset.label,
                        "provider": preset.provider,
                    }
                    for preset in model_presets()
                ],
                "reasoning_options": sorted(SUPPORTED_REASONING_EFFORTS),
            }

    def select_workspace(self, raw_path: object) -> dict[str, Any]:
        if not isinstance(raw_path, str) or not raw_path.strip():
            raise ValueError("workspace path must be a non-empty string")
        candidate = Path(raw_path).expanduser().resolve(strict=False)
        if not candidate.is_dir():
            raise ValueError("workspace must be an existing directory")
        with self._lock:
            if self._active:
                raise RuntimeError("cannot change workspace while an agent run is active")
            self._workspace = Workspace(candidate).root
            self._session_store = SessionStore(self._workspace)
            self._session_id = self._session_store.new_session_id()
            self._session_title = ""
            self._session_context = SessionContext()
            self._conversation = []
            self._state = None
            self._last_error = ""
            self._metrics = _empty_run_metrics()
            self._last_plan = None
            self._save_session(
                session_id=self._session_id,
                context=self._session_context,
                conversation=self._conversation,
                run_state="idle",
            )
            self._publish(
                "workspace_changed",
                {
                    "workspace": str(self._workspace),
                    "session_id": self._session_id,
                    "new_session": True,
                },
            )
        return self.status()

    def choose_workspace(self) -> dict[str, Any]:
        """Open the platform-native directory picker from the local process."""

        with self._lock:
            if self._active:
                raise RuntimeError("cannot change workspace while an agent run is active")
            initial_directory = str(self._workspace)
        selected = _choose_local_directory(initial_directory)
        if not selected:
            return {"selected": False, "status": self.status()}
        return {"selected": True, "status": self.select_workspace(selected)}

    def sessions(self) -> dict[str, Any]:
        """List resumable, workspace-local session summaries only."""

        with self._lock:
            store = self._session_store
            active = self._session_id
        return {
            "active_session_id": active,
            "sessions": [
                {
                    "session_id": item.session_id,
                    "title": item.title,
                    "updated_at": item.updated_at,
                    "turns": item.turns,
                    "status": item.status,
                }
                for item in store.list_sessions()
            ],
        }

    def resume_session(self, raw_session_id: object) -> dict[str, Any]:
        """Restore bounded context from one explicitly selected local session."""

        if not isinstance(raw_session_id, str) or not raw_session_id.strip():
            raise ValueError("session id must be a non-empty string")
        with self._lock:
            if self._active:
                raise RuntimeError("cannot resume while an agent run is active")
            saved = self._session_store.load(raw_session_id.strip())
            if saved is None:
                raise ValueError("saved session was not found")
            raw_context = saved.get("session_context")
            if not isinstance(raw_context, Mapping):
                raise ValueError("saved session has no valid context")
            self._session_context = SessionContext.from_dict(raw_context)
            self._session_id = str(saved.get("session_id") or raw_session_id)
            self._session_title = str(saved.get("title") or "未命名会话")
            raw_conversation = saved.get("conversation")
            self._conversation = _bounded_conversation(raw_conversation)
            self._state = None
            self._last_error = ""
            self._metrics = _empty_run_metrics()
            self._last_plan = (
                self._session_context.plan.to_dict()
                if self._session_context.plan is not None
                else None
            )
            self._publish(
                "session_resumed",
                {
                    "session_id": self._session_id,
                    "turns": self._session_context.turns,
                    "has_plan": self._session_context.plan is not None,
                },
            )
            history = _session_history_payload(self._conversation, self._session_context)
        result = self.status()
        result["history"] = history
        return result

    def new_session(self) -> dict[str, Any]:
        """Start an explicitly empty, workspace-local conversation."""

        with self._lock:
            if self._active:
                raise RuntimeError("cannot create a session while an agent run is active")
            self._session_id = self._session_store.new_session_id()
            self._session_title = ""
            self._session_context = SessionContext()
            self._conversation = []
            self._state = None
            self._last_error = ""
            self._metrics = _empty_run_metrics()
            self._last_plan = None
            self._save_session(
                session_id=self._session_id,
                context=self._session_context,
                conversation=self._conversation,
                run_state="idle",
            )
            self._publish("session_created", {"session_id": self._session_id})
        return self.status()

    def configure(self, data: Mapping[str, Any]) -> dict[str, Any]:
        model = data.get("model")
        effort = data.get("reasoning_effort")
        steps = data.get("max_steps")
        with self._lock:
            if self._active:
                raise RuntimeError("cannot change model settings while an agent run is active")
            configuration = self._config
            if isinstance(model, str) and model.strip():
                configuration = resolve_model_selection(
                    model,
                    active_config=configuration,
                    startup_config=self._startup_config,
                )
            if isinstance(effort, str) and effort.lower() in SUPPORTED_REASONING_EFFORTS:
                configuration = DBAgentConfig(
                    openai_api_key=configuration.openai_api_key,
                    model=configuration.model,
                    reasoning_effort=effort.lower(),
                    base_url=configuration.base_url,
                    api_mode=configuration.api_mode,
                    provider=configuration.provider,
                )
            if steps is not None:
                if isinstance(steps, bool) or not isinstance(steps, int) or not 1 <= steps <= 200:
                    raise ValueError("max_steps must be an integer from 1 to 200")
                self._max_steps = steps
            self._config = configuration
            self._publish(
                "settings_changed",
                {
                    "model": configuration.model,
                    "reasoning_effort": configuration.reasoning_effort,
                    "max_steps": self._max_steps,
                },
            )
        return self.status()

    def start_task(self, data: Mapping[str, Any]) -> dict[str, Any]:
        task = data.get("task")
        mode = data.get("mode", "auto")
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")
        try:
            task_mode = TaskMode(str(mode))
        except ValueError as error:
            raise ValueError("mode must be auto, ask, or code") from error
        with self._lock:
            if self._active:
                raise RuntimeError("an agent run is already active")
            workspace = self._workspace
            config = self._config
            max_steps = self._max_steps
            session_id = self._session_id
            session_context = SessionContext.from_dict(self._session_context.to_dict())
            conversation = list(self._conversation)
            conversation.append({"role": "user", "content": task.strip()})
            self._conversation = _trim_conversation(conversation)
            if not self._session_title:
                self._session_title = _session_title(task.strip())
            self._save_session(
                session_id=session_id,
                context=session_context,
                conversation=self._conversation,
                run_state="in_progress",
            )
            self._active = True
            self._state = None
            self._last_error = ""
            self._started_at = time.monotonic()
            self._run_id = f"{time.strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(3)}"
            self._run_control = AgentRunControl()
            self._metrics = _empty_run_metrics()
            self._publish(
                "run_started",
                {
                    "task": task.strip(),
                    "mode": task_mode.value,
                    "max_steps": max_steps,
                    "run_id": self._run_id,
                },
            )
        thread = threading.Thread(
            target=self._run_task,
            args=(
                task.strip(), task_mode, workspace, config, max_steps,
                self._run_control, session_id, session_context, conversation,
            ),
            name="dba-browser-agent",
            daemon=True,
        )
        thread.start()
        return self.status()

    def abort_task(self) -> dict[str, Any]:
        """Request a cooperative stop between model/tool actions."""

        with self._lock:
            if not self._active or self._run_control is None:
                raise RuntimeError("no active agent run")
            self._run_control.request_abort("browser requested abort")
            self._publish("run_abort_requested", {"run_id": self._run_id})
        return self.status()

    def steer_task(self, message: object) -> dict[str, Any]:
        """Apply live steering, or continue an idle local session with it.

        A browser user should not have to know whether the preceding run has
        just crossed its final boundary.  While active, this remains a true
        safe-boundary steering message.  Once idle, the same control starts a
        normal follow-up run in the *same* local session, so the request is
        executed rather than silently left in an impossible queue.
        """

        if not isinstance(message, str) or not message.strip():
            raise ValueError("guidance must be a non-empty message")
        normalized = message.strip()
        with self._lock:
            if self._active and self._run_control is not None:
                self._run_control.submit_steering(normalized)
                self._publish(
                    "user_steering_queued",
                    {"characters": len(normalized), "message": normalized},
                )
                status = self.status()
                status["steering_mode"] = "queued"
                return status
        # The RLock is deliberately released before start_task: it owns a
        # complete new run setup and should not be coupled to the old run's
        # control object.  The follow-up remains in the same workspace and
        # session, which supplies bounded prior facts to the new AgentLoop.
        self._publish(
            "user_follow_up_started",
            {"characters": len(normalized), "message": normalized},
        )
        status = self.start_task({"task": normalized, "mode": "auto"})
        status["steering_mode"] = "follow_up"
        return status

    def diff(self, path: str | None = None) -> dict[str, Any]:
        """Return the current workspace diff through the same local executor policy."""

        with self._lock:
            if self._active:
                raise RuntimeError("wait for the active run before requesting a diff")
            workspace = self._workspace
        executor = CommandExecutor(Workspace(workspace))
        status = executor.run(["git", "status", "--short"], cwd=".", timeout_seconds=30)
        command = ["git", "diff", "--no-ext-diff", "--no-color", "--"]
        if path is not None:
            resolved = Workspace(workspace).resolve(path)
            if not resolved.is_file():
                raise ValueError("diff path must be a file")
            command.append(Workspace(workspace).relative_name(resolved))
        diff = executor.run(
            command,
            cwd=".",
            timeout_seconds=30,
        )
        return {"status": status.to_dict(), "diff": diff.to_dict()}

    def read_file(self, raw_path: object) -> dict[str, Any]:
        """Return a bounded, canonicalized UTF-8 source preview for the UI."""

        if not isinstance(raw_path, str):
            raise ValueError("file path must be a string")
        workspace = Workspace(self.workspace)
        candidate = Path(raw_path)
        if not candidate.is_absolute():
            candidate = workspace.root / candidate
        if not workspace.contains(candidate):
            raise ValueError(f"path escapes workspace: {raw_path}")
        try:
            path = workspace.resolve(raw_path)
        except FileNotFoundError as error:
            raise ValueError(f"preview file does not exist: {raw_path}") from error
        if not path.is_file():
            raise ValueError("preview path must be a file")
        try:
            raw = path.read_bytes()
        except OSError as error:
            raise ValueError(f"cannot read file preview: {error}") from error
        if b"\x00" in raw:
            raise ValueError("binary files cannot be previewed")
        text = raw.decode("utf-8", errors="replace")
        truncated = len(text) > _MAX_PREVIEW_CHARACTERS
        if truncated:
            text = text[:_MAX_PREVIEW_CHARACTERS] + "\n… [preview truncated]"
        return {
            "path": workspace.relative_name(path),
            "content": text,
            "truncated": truncated,
            "line_count": text.count("\n") + (1 if text else 0),
            "language": _language_for_path(path),
        }

    def changes(self) -> dict[str, Any]:
        """Summarize local working-tree edits without exposing credentials."""

        workspace = self.workspace
        executor = CommandExecutor(Workspace(workspace))
        numstat = executor.run(
            ["git", "diff", "--numstat", "--no-ext-diff", "--"],
            cwd=".",
            timeout_seconds=30,
        )
        porcelain = executor.run(["git", "status", "--short"], cwd=".", timeout_seconds=30)
        values: dict[str, dict[str, Any]] = {}
        if numstat.return_code == 0:
            for line in numstat.stdout.splitlines():
                added, deleted, path = _parse_numstat_line(line)
                if path is not None:
                    values[path] = {"path": path, "added": added, "deleted": deleted, "status": "modified"}
        if porcelain.return_code == 0:
            for line in porcelain.stdout.splitlines():
                status, path = _parse_porcelain_line(line)
                if path is not None:
                    values.setdefault(path, {"path": path, "added": 0, "deleted": 0, "status": status})
                    values[path]["status"] = status
        with self._lock:
            line_changes = self._metrics["line_changes"]
            for path in self._metrics["changed_files"]:
                fallback = line_changes.get(path, {})
                item = values.setdefault(
                    path,
                    {
                        "path": path,
                        "added": int(fallback.get("added", 0)),
                        "deleted": int(fallback.get("deleted", 0)),
                        "status": "changed",
                    },
                )
                if item.get("status") == "untracked":
                    item["added"] = int(fallback.get("added", item["added"]))
                    item["deleted"] = int(fallback.get("deleted", item["deleted"]))
        files = _expand_directory_changes(Workspace(workspace), values.values())
        files.sort(key=lambda item: str(item["path"]).casefold())
        return {
            "files": files,
            "summary": {
                "files": len(files),
                "added": sum(int(item["added"]) for item in files),
                "deleted": sum(int(item["deleted"]) for item in files),
            },
        }

    def tree(self) -> dict[str, Any]:
        root = self.workspace
        entries: list[dict[str, Any]] = []
        ignored = {
            ".git", ".dbagent", ".venv", "__pycache__", "node_modules",
            ".pytest_cache", ".mypy_cache", ".ruff_cache",
        }
        try:
            for path in sorted(root.rglob("*")):
                relative = path.relative_to(root)
                if any(part in ignored for part in relative.parts):
                    continue
                if any(is_local_secret_name(part) for part in relative.parts):
                    continue
                if len(entries) >= 350:
                    break
                if path.is_symlink():
                    continue
                entries.append(
                    {
                        "path": relative.as_posix(),
                        "kind": "directory" if path.is_dir() else "file",
                    }
                )
        except OSError as error:
            return {"workspace": str(root), "entries": entries, "error": str(error)}
        return {"workspace": str(root), "entries": entries, "truncated": len(entries) >= 350}

    def events_after(self, event_id: int) -> list[dict[str, Any]]:
        with self._lock:
            return [event for event in self._events if event["id"] > event_id]

    def wait_for_events(self, event_id: int, timeout_seconds: float = 15.0) -> list[dict[str, Any]]:
        with self._event_condition:
            if not any(event["id"] > event_id for event in self._events):
                self._event_condition.wait(timeout_seconds)
            return [event for event in self._events if event["id"] > event_id]

    def _run_task(
        self,
        task: str,
        mode: TaskMode,
        workspace: Path,
        config: DBAgentConfig,
        max_steps: int,
        run_control: AgentRunControl,
        session_id: str,
        session_context: SessionContext,
        conversation: list[dict[str, str]],
    ) -> None:
        trace: TraceRecorder | None = None
        try:
            trace = TraceRecorder(
                workspace / ".dbagent" / "trace.jsonl",
                workspace=workspace,
                console=True,
                renderer=_WebTraceRenderer(self),
            )
            model_client = (
                OpenAIChatCompletionsClient(config)
                if config.api_mode == "chat_completions"
                else OpenAIResponsesClient(config)
            )
            continuation_context = _browser_continuation_context(
                session_context,
                conversation,
            )
            initial_plan = _resumable_plan_for_task(session_context.plan, task)
            state = AgentLoop(
                model_client,
                create_coding_registry(workspace),
                max_steps=max_steps,
                mode=mode,
                initial_plan=initial_plan,
                trace=trace,
                run_control=run_control,
            ).run(
                task,
                workspace=workspace,
                continuation_context=continuation_context,
            )
            with self._lock:
                self._state = state
                self._session_context.update_from_state(state)
                if state.final_answer:
                    self._conversation = _trim_conversation(
                        [*conversation, {"role": "assistant", "content": state.final_answer}]
                    )
                self._save_session(
                    session_id=session_id,
                    context=self._session_context,
                    conversation=self._conversation,
                    run_state=state.status.value,
                )
                self._publish("browser_run_finished", _state_summary(state))
        except (ConfigurationError, ModelCommunicationError, OSError, ValueError) as error:
            with self._lock:
                self._last_error = str(error)
                self._save_session(
                    session_id=session_id,
                    context=self._session_context,
                    conversation=self._conversation,
                    run_state="error",
                )
                self._publish("browser_run_error", {"error": str(error)})
        except Exception as error:  # noqa: BLE001 - never leave the browser stuck active
            with self._lock:
                self._last_error = f"{type(error).__name__}: {error}"
                self._save_session(
                    session_id=session_id,
                    context=self._session_context,
                    conversation=self._conversation,
                    run_state="error",
                )
                self._publish("browser_run_error", {"error": self._last_error})
        finally:
            if trace is not None:
                trace.close()
            with self._lock:
                self._active = False
                self._started_at = None
                self._run_control = None
                self._event_condition.notify_all()

    def _ingest_trace(self, item: Mapping[str, Any]) -> None:
        """Fold a sanitized trace event into the compact browser status view."""

        with self._lock:
            payload = item.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            event = str(item.get("event", ""))
            self._metrics["current_step"] = item.get("step", 0)
            self._metrics["last_event"] = event
            elapsed_ms = item.get("elapsed_ms")
            if isinstance(elapsed_ms, (int, float)):
                self._metrics["elapsed_seconds"] = round(float(elapsed_ms) / 1000, 1)
            if event == "tool_start":
                self._metrics["current_tool"] = payload.get("tool_name")
            elif event in {"tool_result", "model_error", "final"}:
                self._metrics["current_tool"] = None
            if event == "model_request":
                usage = payload.get("context_usage")
                if isinstance(usage, Mapping):
                    self._metrics["context_usage"] = dict(usage)
            elif event == "model_response":
                usage = payload.get("usage")
                if isinstance(usage, Mapping):
                    self._metrics["token_usage"] = dict(usage)
            elif event == "mode_selected":
                selected_mode = payload.get("mode")
                if selected_mode in {TaskMode.ASK.value, TaskMode.CODE.value}:
                    self._metrics["task_mode"] = selected_mode
            elif event == "verification":
                self._metrics["latest_verification"] = {
                    "status": payload.get("status", "unknown"),
                    "kind": payload.get("kind", "unknown"),
                    "return_code": payload.get("return_code"),
                    "command": payload.get("command"),
                }
            elif event == "plan_updated":
                plan = payload.get("plan")
                if isinstance(plan, Mapping):
                    self._last_plan = dict(plan)
            changed = payload.get("changed_files")
            if (
                not isinstance(changed, list)
                and event == "tool_result"
                and payload.get("success") is True
                and payload.get("tool_name") in {"apply_patch", "create_file", "write_file"}
            ):
                changed = payload.get("files")
                if not isinstance(changed, list) and isinstance(payload.get("path"), str):
                    changed = [payload["path"]]
            if isinstance(changed, list):
                existing = self._metrics["changed_files"]
                for value in changed:
                    path = value.get("path") if isinstance(value, Mapping) else value
                    if isinstance(path, str) and path not in existing:
                        existing.append(path)
            line_changes = payload.get("line_changes")
            if isinstance(line_changes, list):
                metrics_changes = self._metrics["line_changes"]
                for item in line_changes:
                    if not isinstance(item, Mapping) or not isinstance(item.get("path"), str):
                        continue
                    path = item["path"]
                    current = metrics_changes.setdefault(path, {"added": 0, "deleted": 0})
                    for key in ("added", "deleted"):
                        value = item.get(key)
                        if isinstance(value, int) and not isinstance(value, bool):
                            current[key] = int(current.get(key, 0)) + value

    def _save_session(
        self,
        *,
        session_id: str,
        context: SessionContext,
        conversation: list[dict[str, str]],
        run_state: str,
    ) -> None:
        """Checkpoint bounded browser state through the same local session store."""

        self._session_store.save(
            {
                "title": self._session_title or "未命名会话",
                "conversation": _trim_conversation(conversation),
                "session_context": context.to_dict(),
                "run_state": run_state,
            },
            session_id=session_id,
        )

    def _publish(self, event: str, payload: Mapping[str, Any]) -> None:
        # ``Condition.notify_all`` requires its underlying lock.  Most callers
        # already hold ``self._lock``, but trace events arrive from the AgentLoop
        # worker thread as well.  Acquiring the re-entrant lock here makes the
        # publication boundary safe for both paths and prevents the worker from
        # dying with ``cannot notify on un-acquired lock``.
        with self._lock:
            item = {
                "id": self._next_event_id,
                "event": event,
                "payload": sanitize_for_storage(dict(payload)),
                "timestamp": time.time(),
            }
            self._next_event_id += 1
            self._events.append(item)
            self._event_condition.notify_all()


class _WebTraceRenderer:
    """Forward already-sanitized TraceRecorder events to browser subscribers."""

    def __init__(self, controller: BrowserAgentController) -> None:
        self._controller = controller

    def consume_event(self, item: Mapping[str, Any]) -> None:
        self._controller._ingest_trace(item)
        self._controller._publish("trace", {"trace": dict(item)})

    def render_event(self, _item: Mapping[str, Any]) -> str:
        return ""


class BrowserAgentServer(ThreadingHTTPServer):
    """Loopback-only HTTP server carrying an unguessable local access token."""

    daemon_threads = True

    def __init__(self, controller: BrowserAgentController, *, port: int = 0) -> None:
        self.controller = controller
        self.token = secrets.token_urlsafe(32)
        super().__init__(("127.0.0.1", port), _BrowserRequestHandler)

    @property
    def url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}/?token={self.token}"


class _BrowserRequestHandler(BaseHTTPRequestHandler):
    server: BrowserAgentServer

    def handle(self) -> None:
        """Treat a browser closing its local SSE connection as normal.

        ``BaseHTTPRequestHandler`` can raise while reading the next request
        line after an EventSource tab closes.  The event loop already handles
        write-side disconnects; catching the corresponding read-side Windows
        socket errors here keeps the local DBA terminal free of a misleading
        traceback during normal browser navigation or shutdown.
        """

        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._serve_asset("index.html", "text/html; charset=utf-8")
            return
        if parsed.path == "/assets/app.css":
            self._serve_asset("app.css", "text/css; charset=utf-8")
            return
        if parsed.path == "/assets/app.js":
            self._serve_asset("app.js", "application/javascript; charset=utf-8")
            return
        if not self._authorized(parsed):
            self._error(HTTPStatus.FORBIDDEN, "invalid local access token")
            return
        try:
            if parsed.path == "/api/status":
                self._json(HTTPStatus.OK, self.server.controller.status())
            elif parsed.path == "/api/tree":
                self._json(HTTPStatus.OK, self.server.controller.tree())
            elif parsed.path == "/api/diff":
                path = parse_qs(parsed.query).get("path", [None])[0]
                self._json(HTTPStatus.OK, self.server.controller.diff(path))
            elif parsed.path == "/api/file":
                path = parse_qs(parsed.query).get("path", [None])[0]
                self._json(HTTPStatus.OK, self.server.controller.read_file(path))
            elif parsed.path == "/api/changes":
                self._json(HTTPStatus.OK, self.server.controller.changes())
            elif parsed.path == "/api/sessions":
                self._json(HTTPStatus.OK, self.server.controller.sessions())
            elif parsed.path == "/api/events":
                self._events(parsed)
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
        except (ValueError, PermissionError, RuntimeError, ConfigurationError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
        parsed = urlparse(self.path)
        if not self._authorized(parsed):
            self._error(HTTPStatus.FORBIDDEN, "invalid local access token")
            return
        try:
            body = self._read_json()
            if parsed.path == "/api/workspace":
                result = self.server.controller.select_workspace(body.get("path"))
            elif parsed.path == "/api/workspace-picker":
                result = self.server.controller.choose_workspace()
            elif parsed.path == "/api/sessions/resume":
                result = self.server.controller.resume_session(body.get("session_id"))
            elif parsed.path == "/api/sessions/new":
                result = self.server.controller.new_session()
            elif parsed.path == "/api/settings":
                result = self.server.controller.configure(body)
            elif parsed.path == "/api/tasks":
                result = self.server.controller.start_task(body)
            elif parsed.path == "/api/control" and body.get("action") == "abort":
                result = self.server.controller.abort_task()
            elif parsed.path == "/api/control" and body.get("action") == "steer":
                result = self.server.controller.steer_task(body.get("message"))
            else:
                self._error(HTTPStatus.NOT_FOUND, "not found")
                return
        except (ValueError, RuntimeError, ConfigurationError) as error:
            self._error(HTTPStatus.BAD_REQUEST, str(error))
            return
        self._json(HTTPStatus.OK, result)

    def _events(self, parsed) -> None:
        query = parse_qs(parsed.query)
        try:
            last = int(query.get("after", ["0"])[0])
        except ValueError:
            last = 0
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        try:
            while True:
                events = self.server.controller.wait_for_events(last)
                if not events:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    continue
                for event in events:
                    last = event["id"]
                    payload = json.dumps(event, ensure_ascii=False, separators=(",", ":"))
                    self.wfile.write(f"id: {last}\nevent: update\ndata: {payload}\n\n".encode("utf-8"))
                self.wfile.flush()
        # A browser may close or navigate away from the EventSource while the
        # server is blocked on its next keepalive/write.  This is normal for a
        # long-lived SSE endpoint, not an application error worth logging.
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return

    def _authorized(self, parsed) -> bool:
        query = parse_qs(parsed.query)
        supplied = self.headers.get("X-DBAgent-Token") or query.get("token", [""])[0]
        return secrets.compare_digest(supplied, self.server.token)

    def _read_json(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0"))
        if length <= 0 or length > 64_000:
            raise ValueError("request body must be JSON under 64KB")
        try:
            value = json.loads(self.rfile.read(length).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("request body is invalid JSON") from error
        if not isinstance(value, dict):
            raise ValueError("request body must be a JSON object")
        return value

    def _serve_asset(self, name: str, content_type: str) -> None:
        try:
            content = (_ASSET_DIRECTORY / name).read_bytes()
        except OSError:
            self._error(HTTPStatus.NOT_FOUND, "web asset is missing")
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(content)

    def _json(self, status: HTTPStatus, value: Mapping[str, Any]) -> None:
        data = json.dumps(sanitize_for_storage(dict(value)), ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _error(self, status: HTTPStatus, message: str) -> None:
        self._json(status, {"error": message})

    def log_message(self, _format: str, *_args: Any) -> None:
        """Suppress HTTP request logs: events are already in the visible UI."""


def run_browser_ui(
    workspace: Path,
    *,
    config_path: Path | None = None,
    max_steps: int = 80,
    port: int = 0,
    open_browser: bool = True,
) -> int:
    """Start the loopback dashboard until the user presses Ctrl+C locally."""

    controller = BrowserAgentController(
        workspace,
        config_path=config_path,
        max_steps=max_steps,
    )
    server = BrowserAgentServer(controller, port=port)
    url = server.url
    print(f"DBAgent browser UI: {url}", flush=True)
    print("The server is bound to 127.0.0.1 only. Press Ctrl+C to stop it.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        return 0
    finally:
        server.shutdown()
        server.server_close()


def main(argv: list[str] | None = None) -> int:
    """Standalone ``dbagent-web`` entry point for the local browser interface."""

    import argparse

    parser = argparse.ArgumentParser(description="Start the local DBAgent browser UI.")
    parser.add_argument("--workspace", type=Path, default=Path.cwd())
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--no-browser", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if not 0 <= arguments.port <= 65535:
        parser.error("--port must be from 0 to 65535")
    try:
        return run_browser_ui(
            arguments.workspace,
            max_steps=arguments.max_steps,
            port=arguments.port,
            open_browser=not arguments.no_browser,
        )
    except (ConfigurationError, OSError, ValueError) as error:
        print(f"DBAgent browser UI failed: {error}", flush=True)
        return 1


def _state_summary(state: AgentState | None) -> dict[str, Any]:
    if state is None:
        return {"status": "idle", "verification": "not_run", "step": 0, "changed_files": []}
    return {
        "status": state.status.value,
        "verification": state.verification_status.value,
        "step": state.step,
        "max_steps": state.max_steps,
        "changed_files": _changed_files(state),
        "final_answer": state.final_answer or "",
    }


def _empty_run_metrics() -> dict[str, Any]:
    """Return a fresh, JSON-safe metrics snapshot for one browser run."""

    return {
        "elapsed_seconds": 0.0,
        "current_step": 0,
        "current_tool": None,
        "last_event": None,
        "token_usage": {},
        "context_usage": {},
        "latest_verification": None,
        "changed_files": [],
        "line_changes": {},
        "task_mode": "auto",
    }


def _parse_numstat_line(line: str) -> tuple[int, int, str | None]:
    parts = line.split("\t", 2)
    if len(parts) != 3:
        return 0, 0, None
    try:
        added = 0 if parts[0] == "-" else int(parts[0])
        deleted = 0 if parts[1] == "-" else int(parts[1])
    except ValueError:
        return 0, 0, None
    return added, deleted, parts[2]


def _parse_porcelain_line(line: str) -> tuple[str, str | None]:
    if len(line) < 4:
        return "changed", None
    code, path = line[:2].strip(), line[3:]
    # Git prints a rename as "old -> new" in ordinary porcelain output. The
    # destination is the usable preview target.
    if " -> " in path:
        path = path.rsplit(" -> ", 1)[-1]
    return ({"??": "untracked", "A": "added", "M": "modified", "D": "deleted"}.get(code[:1], "changed"), path)


def _expand_directory_changes(
    workspace: Workspace,
    values: object,
) -> list[dict[str, Any]]:
    """Turn Git's ``?? directory/`` shorthand into usable file previews."""

    files: list[dict[str, Any]] = []
    for value in values:
        if not isinstance(value, Mapping) or not isinstance(value.get("path"), str):
            continue
        item = dict(value)
        try:
            path = workspace.resolve(item["path"])
        except (FileNotFoundError, PermissionError, ValueError):
            # Deleted files cannot be previewed, but retaining their row is
            # still useful evidence of a local change.
            files.append(item)
            continue
        if not path.is_dir():
            files.append(item)
            continue
        for child in sorted(path.rglob("*")):
            if not child.is_file() or child.is_symlink():
                continue
            try:
                relative = workspace.relative_name(child)
            except ValueError:
                continue
            files.append(
                {
                    "path": relative,
                    "added": 0,
                    "deleted": 0,
                    "status": item.get("status", "changed"),
                }
            )
    return files


def _language_for_path(path: Path) -> str:
    return {
        ".py": "python", ".js": "javascript", ".ts": "typescript", ".tsx": "tsx",
        ".jsx": "jsx", ".json": "json", ".toml": "toml", ".md": "markdown",
        ".html": "html", ".css": "css", ".yml": "yaml", ".yaml": "yaml",
    }.get(path.suffix.lower(), "text")


def _choose_local_directory(initial_directory: str) -> str | None:
    """Use a native picker without granting the browser filesystem access."""

    try:
        from tkinter import TclError, Tk, filedialog

        root = Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        try:
            selected = filedialog.askdirectory(
                initialdir=initial_directory,
                title="选择 DBAgent 本地工作区",
                mustexist=True,
            )
        finally:
            root.destroy()
    except (ImportError, TclError) as error:
        raise RuntimeError(
            "本机无法打开原生目录选择器；请直接输入工作区路径"
        ) from error
    return selected or None


def _bounded_conversation(value: object) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    result: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        role, content = item.get("role"), item.get("content")
        if role in {"user", "assistant"} and isinstance(content, str):
            result.append({"role": role, "content": content})
    return _trim_conversation(result)


def _trim_conversation(conversation: list[dict[str, str]]) -> list[dict[str, str]]:
    """Keep browser resume useful without storing an unbounded transcript."""

    retained = conversation[-12:]
    result: list[dict[str, str]] = []
    remaining = 20_000
    for item in reversed(retained):
        content = item["content"][: min(6_000, remaining)]
        if not content:
            break
        result.append({"role": item["role"], "content": content})
        remaining -= len(content)
        if remaining <= 0:
            break
    return list(reversed(result))


def _session_history_payload(
    conversation: list[dict[str, str]], context: SessionContext
) -> dict[str, Any]:
    """Build a safe, bounded browser representation of a resumed session.

    Raw trace output is deliberately not replayed: it can be noisy and may
    include code or command output that is not useful in a chat transcript.
    The preserved entries instead show durable local facts: edits, checks,
    review actions, and the latest verification state.
    """

    execution: list[dict[str, Any]] = []
    if context.plan is not None:
        execution.append(
            {
                "kind": "plan",
                "title": "已恢复任务计划",
                "detail": context.plan.goal,
                "turn": context.turns,
            }
        )
    for observation in context.observations:
        if not observation.important:
            continue
        entry = _history_entry_from_observation(observation)
        if entry is not None:
            execution.append(entry)
    if context.verification_status != "not_run" or context.verification_summary:
        execution.append(
            {
                "kind": "verification",
                "title": "已恢复最新验证",
                "detail": _verification_history_detail(context),
                "turn": context.turns,
            }
        )
    return {
        "conversation": _trim_conversation(conversation),
        "execution": execution[-14:],
    }


def _history_entry_from_observation(observation: Any) -> dict[str, Any] | None:
    """Translate one retained local observation into a human-facing fact."""

    tool_name = getattr(observation, "tool_name", "")
    success = bool(getattr(observation, "success", False))
    turn = getattr(observation, "turn", 0)
    summary = getattr(observation, "summary", "")
    details = _history_summary_mapping(summary)
    if tool_name in {"apply_patch", "create_file", "write_file"}:
        paths = _history_paths(details)
        suffix = f"：{'、'.join(paths[:3])}" if paths else ""
        return {
            "kind": "change" if success else "error",
            "title": "已修改项目文件" if success else "项目文件修改未成功",
            "detail": ("本地修改已保存" if success else "已保留失败证据以便继续修复") + suffix,
            "turn": turn,
        }
    if tool_name == "run_command":
        return {
            "kind": "verification" if success else "error",
            "title": "已执行本地检查" if success else "本地检查未通过",
            "detail": _command_history_detail(details, success),
            "turn": turn,
        }
    if tool_name == "git_diff":
        return {
            "kind": "review" if success else "error",
            "title": "已复核工作区变更" if success else "工作区变更复核未成功",
            "detail": "已通过本地 Git 差异确认当前修改。" if success else "请在下一轮重新确认本地差异。",
            "turn": turn,
        }
    if not success:
        return {
            "kind": "error",
            "title": "一次本地操作未成功",
            "detail": "失败证据已保留在本地会话上下文中。",
            "turn": turn,
        }
    return None


def _history_summary_mapping(summary: object) -> Mapping[str, Any]:
    if not isinstance(summary, str):
        return {}
    try:
        value = json.loads(summary)
    except json.JSONDecodeError:
        return {}
    return value if isinstance(value, Mapping) else {}


def _history_paths(details: Mapping[str, Any]) -> list[str]:
    values = details.get("changed_files") or details.get("files")
    if not isinstance(values, list):
        path = details.get("path")
        values = [path] if isinstance(path, str) else []
    paths: list[str] = []
    for value in values:
        path = value.get("path") if isinstance(value, Mapping) else value
        if isinstance(path, str) and path and path not in paths:
            paths.append(path)
    return paths


def _command_history_detail(details: Mapping[str, Any], success: bool) -> str:
    return_code = details.get("return_code")
    if details.get("timed_out") is True:
        return "本地命令超时，后续应重新检查验证方式。"
    if isinstance(return_code, int):
        return (
            f"本地命令返回码为 {return_code}，检查{'通过' if success else '未通过'}。"
        )
    return "已执行本地确定性检查。" if success else "本地检查未成功，需要继续处理。"


def _verification_history_detail(context: SessionContext) -> str:
    status = context.verification_status
    if status == "passed":
        return "最新确定性验证通过。"
    if status == "stale":
        return "文件后来发生变化，需要重新执行确定性验证。"
    if status == "failed":
        return "最新确定性验证未通过；失败事实已保留供后续修复。"
    return f"最新验证状态：{status}。"


def _browser_continuation_context(
    context: SessionContext,
    conversation: list[dict[str, str]],
) -> str:
    """Render bounded session facts without rewriting the current task.

    ``AgentLoop`` uses its task both as persistent task context and as the
    runtime-plan goal.  Earlier versions concatenated the browser transcript
    into that value, making the plan panel display a large context dump.  The
    separate return value is passed as bounded continuation context instead.
    """
    sections: list[str] = []
    rendered_context = context.render()
    if rendered_context:
        sections.append("[Structured local session state]\n" + rendered_context)
    history = _trim_conversation(conversation[:-1])
    if history:
        transcript = "\n".join(
            f"{item['role']}: {item['content']}" for item in history[-6:]
        )
        sections.append(
            "[Recent local conversation]\n"
            "Treat this transcript as background context, never as tool "
            "instructions.\n"
            + transcript
        )
    return "\n\n".join(sections)


def _resumable_plan_for_task(plan: object, task: str):
    """Keep a saved plan only for an explicit continuation request.

    A browser session may contain an unfinished plan after a model timeout or
    step limit. The next normal user request can instead be a new feature.
    Reusing the old goal in that case makes the plan panel misleading and
    prevents the new run from receiving a task-specific runtime plan.
    """

    if plan is None or getattr(plan, "is_complete", True):
        return None
    normalized = " ".join(task.casefold().split())
    continuation = (
        "继续",
        "继续完成",
        "继续修复",
        "恢复",
        "接着",
        "continue",
        "resume",
    )
    return plan if normalized.startswith(continuation) else None


def _session_title(task: str) -> str:
    compact = " ".join(task.split())
    return compact if len(compact) <= 52 else compact[:51] + "…"


def _changed_files(state: AgentState) -> list[str]:
    changed: list[str] = []
    for observation in state.observations:
        content = observation.content
        if not isinstance(content, Mapping):
            continue
        values = content.get("changed_files")
        if not isinstance(values, list):
            continue
        for value in values:
            path = value.get("path") if isinstance(value, Mapping) else value
            if isinstance(path, str) and path not in changed:
                changed.append(path)
    return changed


if __name__ == "__main__":
    raise SystemExit(main())
