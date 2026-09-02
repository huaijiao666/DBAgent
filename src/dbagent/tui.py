"""Dependency-free full-screen terminal dashboard for DBAgent.

This module intentionally owns presentation only.  Agent state, tool safety,
sessions, and JSONL tracing remain in the ordinary local harness modules.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from dbagent.ui import TerminalUI, _changed_files, _fit_display, _truncate


class FullscreenTUI(TerminalUI):
    """ANSI alternate-screen view of an existing DBA session.

    It works without a UI framework so the main project keeps a small,
    explainable dependency surface.  ``TraceRecorder`` recognizes
    ``consume_event`` and feeds already-redacted events here instead of writing
    scrolling lines.  The classic :class:`TerminalUI` remains the default for
    redirected output, CI, and terminal recording.
    """

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        force_terminal: bool = False,
    ) -> None:
        target = stream or sys.stdout
        if not force_terminal and not bool(getattr(target, "isatty", lambda: False)()):
            raise ValueError("TUI mode requires an interactive terminal (TTY)")
        super().__init__(stream=target, color=False)
        self._lock = threading.RLock()
        self._entered = False
        self._closed = False
        self._activity: deque[str] = deque(maxlen=200)
        self._notice = "Ready. Type a task or /help."
        self._assistant = ""
        self._plan_lines: list[str] = []
        self._run_status = "idle"
        self._verification = "not_run"
        self._files_changed: list[str] = []
        self._current_step = 0
        self._input_line = ""
        self._input_hint = "DBA> "

    @staticmethod
    def available(stream: TextIO | None = None) -> bool:
        """Return whether a stream can support the alternate-screen mode."""

        target = stream or sys.stdout
        return bool(getattr(target, "isatty", lambda: False)()) and not bool(
            os.environ.get("TERM") == "dumb"
        )

    def session_start(
        self,
        *,
        workspace: Path,
        model: str,
        api_mode: str,
        mode: str = "auto",
        session_id: str = "",
        session_state: str = "new",
        launch_directory: Path | None = None,
    ) -> None:
        self._started = time.monotonic()
        self._workspace = workspace
        self._model = model
        self._mode = mode
        self._session_id = session_id
        self._session_state = session_state
        self._notice = f"Session {session_state}; API {api_mode}."
        if launch_directory is not None and launch_directory != workspace:
            self._notice += " Workspace is the detected project root."
        self._enter()
        self._draw()

    def prompt(self) -> str:
        """Draw the input row; the existing REPL still owns line parsing."""

        with self._lock:
            self._input_line = ""
            self._input_hint = f"DBA[{self._mode}]> "
            self._draw()
        # The terminal already displays the prompt in the final screen row.
        return ""

    def set_live_input(self, value: str) -> None:
        """Show nonblocking steering input collected by the Windows poller."""

        with self._lock:
            self._input_line = value
            self._input_hint = "STEER> "
            self._draw()

    def set_mode(self, mode: str) -> None:
        self._mode = mode
        self._draw()

    def set_session_id(self, session_id: str, *, state: str = "active") -> None:
        self._session_id = session_id
        self._session_state = state
        self._draw()

    def info(self, message: str) -> None:
        self._notice = message
        self._activity.append("INFO  " + message)
        self._draw()

    def error(self, message: str) -> None:
        self._notice = "ERROR: " + message
        self._activity.append("ERROR " + message)
        self._draw()

    def assistant(self, message: str) -> None:
        self._assistant = message.strip()
        self._notice = "Agent response available below."
        if self._assistant:
            self._activity.append("AGENT  Final response updated.")
        self._draw()

    def help(self) -> None:
        self._notice = (
            "Commands: /model, /reasoning, /steps, /mode, /status, /plan, "
            "/sessions, /resume, /continue, /new, /clear, /exit. "
            "During a run: /steer, /followup, /abort."
        )
        self._activity.append("HELP  " + self._notice)
        self._draw()

    def start(
        self,
        *,
        task: str,
        workspace: Path,
        model: str,
        max_steps: int,
        mode: str = "auto",
    ) -> None:
        self._started = time.monotonic()
        self._task = task
        self._workspace = workspace
        self._model = model
        self._max_steps = max_steps
        self._mode = mode
        self._run_status = "running"
        self._verification = "not_run"
        self._files_changed = []
        self._current_step = 0
        self._assistant = ""
        self._notice = "Agent running. /steer adds guidance at the next safe boundary."
        self._draw()

    def consume_event(self, item: Mapping[str, Any]) -> None:
        """Receive a sanitized trace event without emitting a scrolling line."""

        with self._lock:
            event = str(item.get("event", ""))
            payload = item.get("payload")
            payload = payload if isinstance(payload, Mapping) else {}
            self._current_step = max(self._current_step, int(item.get("step", 0) or 0))
            rendered = self.render_event(item)
            self._activity.append(rendered)
            if event == "verification":
                self._verification = str(payload.get("status", self._verification))
            elif event == "tool_result":
                changed = payload.get("changed_files")
                if isinstance(changed, Sequence) and not isinstance(changed, (str, bytes)):
                    for value in changed:
                        path = value.get("path") if isinstance(value, Mapping) else value
                        if isinstance(path, str) and path not in self._files_changed:
                            self._files_changed.append(path)
            elif event == "plan_updated":
                current = payload.get("current_step_description") or payload.get("current_step")
                if current:
                    self._notice = "Plan updated: " + str(current)
            elif event == "final":
                self._run_status = str(payload.get("status", "completed")).lower()
                self._verification = str(payload.get("verification_status", self._verification))
                self._notice = "Run finished: " + self._run_status.upper()
            self._draw()

    def finish(self, state: Any) -> None:
        status = str(getattr(getattr(state, "status", None), "value", "unknown"))
        self._run_status = "incomplete" if status == "max_steps" else status
        self._verification = str(
            getattr(getattr(state, "verification_status", None), "value", self._verification)
        )
        self._files_changed = _changed_files(getattr(state, "observations", ()))
        self._current_step = int(getattr(state, "step", self._current_step))
        self._notice = f"Run summary: {self._run_status.upper()}"
        self._draw()

    def render_plan_history(self, plans: Sequence[Any]) -> None:
        if not plans:
            return
        plan = plans[-1]
        lines = [str(getattr(plan, "goal", "Current plan"))]
        for step in getattr(plan, "steps", ()):
            status = str(getattr(getattr(step, "status", None), "value", "pending"))
            marker = {"completed": "x", "in_progress": ">", "blocked": "!"}.get(status, ".")
            lines.append(
                f" {marker} {getattr(step, 'step_id', '?')}: "
                f"{getattr(step, 'description', '')} [{status}]"
            )
        self._plan_lines = lines
        self._draw()

    def render_context(self, state: Any | None) -> None:
        if state is None or not getattr(state, "context_usage", None):
            self.info("No completed Agent run in this session.")
            return
        usage = state.context_usage[-1]
        self.info(
            f"Context {usage.approximate_tokens}~tok; recent={usage.recent_observations}; "
            f"compacted={usage.compacted_observations}."
        )

    def render_capabilities(self, policy: Any) -> None:
        self.info(str(getattr(policy, "capability_summary", policy)))

    def render_model_options(self, presets: Sequence[Any], *, current_model: str) -> None:
        options = [
            f"{'>' if getattr(item, 'model', '') == current_model else ' '} "
            f"{getattr(item, 'alias', '?')}: {getattr(item, 'model', '?')}"
            for item in presets
        ]
        self._activity.extend(options)
        self._notice = "Model aliases added to activity log."
        self._draw()

    def render_sessions(self, sessions: Sequence[Any], *, active_session_id: str = "") -> None:
        lines = ["Saved sessions:"]
        for index, item in enumerate(sessions, start=1):
            active = ">" if getattr(item, "session_id", "") == active_session_id else " "
            lines.append(
                f"{active} {index}. {getattr(item, 'session_id', '?')} "
                f"{getattr(item, 'status', '?')} {getattr(item, 'title', '')}"
            )
        self._activity.extend(lines or ["No saved sessions."])
        self._notice = "Use /resume <number>, <ID prefix>, or latest."
        self._draw()

    def render_resume_summary(self, **values: Any) -> None:
        self._notice = (
            f"Resumed {values.get('session_id', '?')}; turns={values.get('turns', 0)}; "
            f"verification={values.get('verification', 'not_run')}."
        )
        self._activity.append("RESUME " + self._notice)
        self._draw()

    def goodbye(self) -> None:
        self._notice = "Session closed."
        self.close(message="DBAgent session closed. Workspace checkpoints remain available via /resume.")

    def close(self, *, message: str | None = None) -> None:
        """Leave the alternate screen exactly once, even after a startup error."""

        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._entered:
                self._raw("\x1b[?25h\x1b[?1049l")
            if message:
                self._raw(message + "\n")

    def _enter(self) -> None:
        if self._entered:
            return
        _enable_windows_virtual_terminal()
        self._raw("\x1b[?1049h\x1b[?25l")
        self._entered = True

    def _draw(self) -> None:
        if not self._entered or self._closed:
            return
        with self._lock:
            size = shutil.get_terminal_size(fallback=(100, 32))
            width, height = max(60, size.columns), max(18, size.lines)
            plan_height = max(4, min(8, height // 4))
            response_lines = (
                _wrap_for_screen(self._assistant, width)
                if self._assistant
                else []
            )
            response_height = min(6, len(response_lines))
            fixed = 10 + plan_height + (response_height + 1 if response_height else 0)
            activity_height = max(4, height - fixed)
            lines = [
                "DBAgent  |  Local Coding Agent  |  TUI",
                f"workspace  {_truncate(str(self._workspace), width - 12)}",
                f"model {self._model}   mode {self._mode}   session {self._session_id[:24]} [{self._session_state}]",
                "=" * width,
                f"TASK  {_truncate(self._task or 'No active task', width - 7)}",
                f"STATUS  {self._run_status.upper()}    step {self._current_step}/{self._max_steps or '?'}    verification {self._verification}",
                f"FILES   {_truncate(', '.join(self._files_changed) or 'none', width - 8)}",
                "PLAN",
            ]
            plan = self._plan_lines or ["  No structured plan yet."]
            lines.extend(_fit_display(line, width) for line in plan[:plan_height])
            while len(lines) < 8 + plan_height:
                lines.append("")
            if response_height:
                lines.append("RESPONSE")
                lines.extend(response_lines[-response_height:])
            lines.append("ACTIVITY")
            activity: list[str] = []
            for item in list(self._activity)[-activity_height:]:
                activity.extend(_wrap_for_screen(item, width))
            lines.extend(activity[-activity_height:])
            while len(lines) < height - 2:
                lines.append("")
            lines.append("-" * width)
            notice = _truncate(self._notice, width)
            lines.append(
                _fit_display(
                    f"{self._input_hint}{self._input_line}" if self._input_line else notice,
                    width,
                )
            )
            frame = "\n".join(_fit_display(line, width) for line in lines[:height])
            self._raw("\x1b[H\x1b[2J" + frame + f"\x1b[{height};1H")

    def _raw(self, value: str) -> None:
        try:
            self.stream.write(value)
            self.stream.flush()
        except (AttributeError, OSError):
            pass


def _wrap_for_screen(value: str, width: int) -> list[str]:
    if not value:
        return [""]
    result: list[str] = []
    remaining = value
    while remaining:
        fitted = _fit_display(remaining, width).rstrip()
        result.append(fitted)
        if len(remaining) <= width:
            break
        remaining = remaining[max(1, width - 3) :]
    return result


def _enable_windows_virtual_terminal() -> None:
    """Best-effort VT processing for modern Windows Console hosts."""

    if os.name != "nt":
        return
    try:
        import ctypes

        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
        mode = ctypes.c_uint()
        if handle and kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(handle, mode.value | 0x0004)
    except (AttributeError, OSError):
        # Windows Terminal and newer conhost versions generally already enable
        # this.  Falling back to the literal control sequences is harmless.
        return
