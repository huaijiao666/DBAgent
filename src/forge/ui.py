"""Small, dependency-free terminal UI for an observable Forge run."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TextIO

from forge.console import safe_print


class TerminalUI:
    """Render a compact, Codex-inspired terminal dashboard.

    The UI is deliberately a line-oriented renderer rather than a full-screen
    TUI. It remains readable in a recorded terminal, in CI logs, and when the
    stream is redirected to a file. ANSI color is enabled only for a TTY unless
    explicitly requested.
    """

    _RESET = "\x1b[0m"
    _COLORS = {
        "blue": "\x1b[36m",
        "green": "\x1b[32m",
        "yellow": "\x1b[33m",
        "red": "\x1b[31m",
        "muted": "\x1b[90m",
        "magenta": "\x1b[35m",
    }

    def __init__(
        self,
        *,
        stream: TextIO | None = None,
        color: bool | None = None,
    ) -> None:
        self.stream = stream or sys.stderr
        if color is None:
            is_tty = bool(getattr(self.stream, "isatty", lambda: False)())
            color = is_tty and not os.environ.get("NO_COLOR")
        self.color = color
        self._started = time.monotonic()
        self._task = ""
        self._workspace = Path(".")
        self._model = ""
        self._max_steps = 0
        self._mode = "auto"

    def session_start(
        self,
        *,
        workspace: Path,
        model: str,
        api_mode: str,
        mode: str = "auto",
        launch_directory: Path | None = None,
    ) -> None:
        """Print the persistent DBA session banner."""

        self._started = time.monotonic()
        self._write("")
        self._write(self._paint("+-- DBA interactive session " + "-" * 46 + "+", "blue"))
        self._write(f"| Workspace {_truncate(str(workspace), 70):<70} |")
        self._write(f"| Model     {_truncate(model, 70):<70} |")
        self._write(f"| API mode  {_truncate(api_mode, 70):<70} |")
        self._write(f"| Task mode {_truncate(mode, 70):<70} |")
        if launch_directory is not None and launch_directory != workspace:
            self._write(
                f"| Started   {_truncate(str(launch_directory), 70):<70} |"
            )
            self._write("| Project root was detected automatically.                              |")
        self._write("| Type /help for commands; /exit to leave the session.                 |")
        self._write(self._paint("+" + "-" * 76 + "+", "blue"))
        self._write("")

    def prompt(self) -> str:
        """Return the prompt passed to ``input``."""

        return self._paint(f"DBA[{self._mode}]", "blue") + "> "

    def set_mode(self, mode: str) -> None:
        """Update the mode shown in subsequent prompts."""

        self._mode = mode

    def info(self, message: str) -> None:
        """Render a non-error status message."""

        self._write(self._paint("INFO  ", "muted") + _truncate(message, 96))

    def assistant(self, message: str) -> None:
        """Render the assistant's final response for one user turn."""

        self._write(self._paint("ASSISTANT", "green"))
        for line in (message or "").splitlines() or [""]:
            self._write(f"  {line}")

    def help(self) -> None:
        """Render the commands supported by the interactive session."""

        self._write("Commands:")
        self._write("  /model [NAME]  show or change the model for the next turn")
        self._write("  /mode [auto|ask|code]  show or change task authority")
        self._write("  /status        show session and latest task status")
        self._write("  /plan          show the latest retained task plan")
        self._write("  /resume        restore the latest session saved in this workspace")
        self._write("  /clear         clear local conversation history")
        self._write("  /help          show this help")
        self._write("  /exit          leave DBA")

    def goodbye(self) -> None:
        """Render the session exit message."""

        self._write(self._paint("Session closed. Local conversation discarded.", "muted"))

    def start(
        self,
        *,
        task: str,
        workspace: Path,
        model: str,
        max_steps: int,
        mode: str = "auto",
    ) -> None:
        """Print the run header and reset elapsed-time accounting."""

        self._started = time.monotonic()
        self._task = task
        self._workspace = workspace
        self._model = model
        self._max_steps = max_steps
        self._mode = mode
        self._write("")
        self._write(self._paint("+-- DBAgent coding session " + "-" * 50 + "+", "blue"))
        self._write(f"| Task      {_truncate(task, 70):<70} |")
        self._write(f"| Workspace {_truncate(str(workspace), 70):<70} |")
        self._write(f"| Model     {_truncate(model, 70):<70} |")
        self._write(f"| Mode      {_truncate(mode, 70):<70} |")
        self._write(f"| Budget    {max_steps} model turns{' ' * max(0, 57 - len(str(max_steps)))} |")
        self._write(self._paint("+" + "-" * 76 + "+", "blue"))
        self._write("")

    def render_event(self, item: Mapping[str, Any]) -> str:
        """Return one styled line for a sanitized trace event."""

        elapsed = f"{float(item.get('elapsed_ms', 0)) / 1000:.1f}s"
        step = item.get("step", "?")
        payload = item.get("payload") or {}
        event = item.get("event")
        symbol, tone = _EVENT_STYLE.get(event, (".", "muted"))
        if event == "tool_result" and not payload.get("success"):
            symbol, tone = "!!", "red"
        if event == "verification" and payload.get("status") == "failed":
            symbol, tone = "!!", "red"
        prefix = self._paint(f"{symbol} {elapsed:>6} | {step!s:>2}/{self._max_steps or '?'} |", tone)

        if event == "run_started":
            detail = (
                f"START  mode={payload.get('mode')}  "
                f"tools={payload.get('tool_count')}"
            )
        elif event == "assistant_update":
            detail = "AGENT  " + _truncate(str(payload.get("text", "")), 90)
        elif event == "model_request":
            usage = payload.get("context_usage", {})
            detail = (
                "MODEL request  "
                f"context={usage.get('approximate_tokens', '?')}~tok  "
                f"tools={len(payload.get('tools', []))}"
            )
        elif event == "model_response":
            usage = payload.get("usage") or {}
            tokens = (
                f"  tokens={usage.get('total_tokens')}"
                if usage.get("total_tokens") is not None
                else ""
            )
            detail = (
                "MODEL response  "
                f"status={payload.get('status')}  "
                f"calls={payload.get('function_call_count')}{tokens}"
            )
        elif event == "model_error":
            action = "retrying" if payload.get("will_retry") else "stopped"
            detail = (
                "MODEL error  "
                f"type={payload.get('error_type')}  "
                f"attempt={payload.get('attempt')}/{payload.get('max_attempts')}  "
                f"{action}"
            )
        elif event == "tool_start":
            detail = _human_tool_start(payload)
        elif event == "tool_result":
            detail = _human_tool_result(payload)
            if payload.get("return_code") is not None:
                detail += f"  rc={payload['return_code']}"
            if payload.get("changed_files"):
                detail += f"  files={payload['changed_files']}"
            elif payload.get("path"):
                detail += f"  path={payload['path']}"
            if not payload.get("success") and payload.get("failure_reason"):
                detail += "  " + _truncate(
                    str(payload["failure_reason"]),
                    72,
                )
        elif event == "patch_applied":
            detail = (
                "PATCH  "
                f"files={payload.get('changed_files', [])}  "
                f"hunks={payload.get('hunks_applied', 0)}"
            )
        elif event == "plan_updated":
            detail = (
                "PLAN  "
                f"current={payload.get('current_step') or 'done'}  "
                f"{_compact_statuses(payload.get('step_statuses', {}))}"
            )
        elif event == "verification":
            status = payload.get("status")
            detail = (
                "VERIFY  "
                f"status={status}  kind={payload.get('kind')}  "
                f"return_code={payload.get('return_code')}"
            )
        elif event == "recovery":
            detail = f"RECOVERY  {_truncate(str(payload.get('reason', '')), 88)}"
        elif event == "step_summary":
            detail = (
                f"STEP  tools={payload.get('succeeded', 0)}/{payload.get('tools', 0)} ok"
                f"  failed={payload.get('failed', 0)}"
            )
            if payload.get("current_plan_step"):
                detail += f"  next={payload.get('current_plan_step')}"
            if payload.get("verification") != "not_run":
                detail += f"  verify={payload.get('verification')}"
        elif event == "final":
            detail = f"FINAL  status={payload.get('status')}"
        else:
            detail = str(event)
        return f"{prefix} {self._paint(detail, tone)}"

    def finish(self, state: Any) -> None:
        """Print a compact final dashboard without duplicating the answer."""

        state_status = str(
            getattr(getattr(state, "status", None), "value", "unknown")
        )
        if state_status == "max_steps":
            status = "INCOMPLETE"
        elif getattr(state, "is_verified", False):
            status = "VERIFIED"
        else:
            status = state_status.upper()
        verification = getattr(
            getattr(state, "verification_status", None), "value", "not_run"
        )
        changed_files = _changed_files(getattr(state, "observations", ()))
        elapsed = time.monotonic() - self._started
        tone = "green" if status == "VERIFIED" else "yellow"
        self._write("")
        self._write(self._paint("+-- Run summary " + "-" * 63 + "+", tone))
        self._write(f"| Status         {status:<62} |")
        self._write(
            f"| Steps          {getattr(state, 'step', '?')}/{getattr(state, 'max_steps', '?'):<59} |"
        )
        self._write(f"| Verification   {verification:<62} |")
        self._write(f"| Elapsed        {elapsed:.1f}s{' ' * max(0, 60 - len(f'{elapsed:.1f}s'))} |")
        self._write(f"| Files changed  {_truncate(', '.join(changed_files) or 'none', 62):<62} |")
        self._write(self._paint("+" + "-" * 76 + "+", tone))

    def render_plan_history(self, plans: Sequence[Any]) -> None:
        """Display one final plan instead of repeating every full snapshot."""

        if not plans:
            return
        self._write("")
        plan = plans[-1]
        self._write(self._paint("Current plan:", "magenta"))
        self._write(f"  {_truncate(str(plan.goal), 94)}")
        for step in plan.steps:
            status = getattr(step.status, "value", step.status)
            marker = {"completed": "x", "in_progress": ">", "blocked": "!"}.get(
                status, "."
            )
            self._write(f"    {marker} {step.step_id}: {step.description} [{status}]")

    def error(self, message: str) -> None:
        """Render an error consistently with other dashboard output."""

        self._write(self._paint(f"ERROR  {_truncate(message, 96)}", "red"))

    def _write(self, text: str) -> None:
        safe_print(text, stream=self.stream, flush=True)

    def _paint(self, text: str, tone: str) -> str:
        if not self.color:
            return text
        return f"{self._COLORS.get(tone, '')}{text}{self._RESET}"


_EVENT_STYLE = {
    "run_started": (">", "blue"),
    "assistant_update": ("·", "blue"),
    "model_request": (".", "blue"),
    "model_response": ("<-", "blue"),
    "model_error": ("!!", "red"),
    "tool_start": ("->", "yellow"),
    "tool_result": ("OK", "green"),
    "patch_applied": ("#", "magenta"),
    "plan_updated": ("*", "magenta"),
    "verification": ("OK", "green"),
    "recovery": ("~", "yellow"),
    "step_summary": ("=", "muted"),
    "final": ("[]", "green"),
}


_TOOL_LABELS = {
    "list_files": "查看文件",
    "read_file": "读取文件",
    "search_text": "搜索文本",
    "get_repo_map": "构建仓库地图",
    "search_symbol": "搜索符号",
    "read_symbol": "读取符号",
    "apply_patch": "应用补丁",
    "create_file": "创建文件",
    "write_file": "写入文件",
    "run_command": "运行命令",
    "git_diff": "检查改动",
    "update_plan": "更新计划",
}


def _human_tool_start(payload: Mapping[str, Any]) -> str:
    name = str(payload.get("tool_name", "tool"))
    label = _TOOL_LABELS.get(name, name)
    command = payload.get("command")
    if isinstance(command, Sequence) and not isinstance(command, (str, bytes)):
        return f"{label}  {' '.join(str(part) for part in command)}"
    files = payload.get("files")
    if isinstance(files, Sequence) and not isinstance(files, (str, bytes)) and files:
        return f"{label}  {', '.join(str(path) for path in files)}"
    if name in {"search_text", "search_symbol"} and payload.get("query"):
        location = payload.get("path")
        suffix = f"  in {location}" if location else ""
        return f"{label}  {payload['query']}{suffix}"
    if name == "read_symbol" and payload.get("symbol_id"):
        return f"{label}  {payload['symbol_id']}"
    target = payload.get("target")
    return f"{label}  {target}" if target else label


def _human_tool_result(payload: Mapping[str, Any]) -> str:
    name = str(payload.get("tool_name", "tool"))
    label = _TOOL_LABELS.get(name, name)
    return f"完成: {label}" if payload.get("success") else f"失败: {label}"


def _compact_statuses(value: Any) -> str:
    if not isinstance(value, Mapping):
        return ""
    markers = {
        "completed": "x",
        "in_progress": ">",
        "pending": ".",
        "blocked": "!",
    }
    return " ".join(
        f"{markers.get(str(status), '?')}{step}"
        for step, status in value.items()
    )


def _truncate(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    if limit <= 3:
        return value[:limit]
    return value[: limit - 3] + "..."


def _changed_files(observations: Sequence[Any]) -> list[str]:
    changed: list[str] = []
    for observation in observations:
        content = getattr(observation, "content", None)
        if not isinstance(content, Mapping):
            continue
        files = content.get("changed_files")
        if not isinstance(files, Sequence) or isinstance(files, (str, bytes)):
            continue
        for item in files:
            path = (
                item.get("path")
                if isinstance(item, Mapping)
                else item
            )
            if isinstance(path, str) and path not in changed:
                changed.append(path)
    return changed
