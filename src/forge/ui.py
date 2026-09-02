"""Small, dependency-free terminal UI for an observable Forge run."""

from __future__ import annotations

import os
import sys
import textwrap
import time
import unicodedata
from collections.abc import Mapping, Sequence
from datetime import datetime
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
    _BOX_WIDTH = 76

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
        self._session_id = ""
        self._session_state = "new"

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
        """Print the persistent DBA session banner."""

        self._started = time.monotonic()
        self._session_id = session_id
        self._session_state = session_state
        self._write("")
        self._write(self._paint("+-- DBAgent " + "-" * 65 + "+", "blue"))
        self._write(_box_row("Local coding agent | repository-aware | self-verifying"))
        self._write(_box_row(f"Workspace {_truncate(str(workspace), 70)}"))
        self._write(_box_row(f"Model     {_truncate(model, 70)}"))
        self._write(_box_row(f"API mode  {_truncate(api_mode, 70)}"))
        self._write(_box_row(f"Task mode {_truncate(mode, 70)}"))
        session_label = f"{session_id or 'not saved yet'} [{session_state}]"
        self._write(_box_row(f"Session   {_truncate(session_label, 70)}"))
        if launch_directory is not None and launch_directory != workspace:
            self._write(_box_row(f"Started   {_truncate(str(launch_directory), 70)}"))
            self._write(_box_row("Project root was detected automatically."))
        self._write(_box_row("/help commands | /sessions history | /status current state"))
        self._write(self._paint("+" + "-" * 76 + "+", "blue"))
        self._write("")

    def prompt(self) -> str:
        """Return the prompt passed to ``input``."""

        return self._paint(f"DBA[{self._mode}]", "blue") + "> "

    def set_mode(self, mode: str) -> None:
        """Update the mode shown in subsequent prompts."""

        self._mode = mode

    def set_session_id(self, session_id: str, *, state: str = "active") -> None:
        """Update the active session shown by status-oriented UI."""

        self._session_id = session_id
        self._session_state = state

    def info(self, message: str) -> None:
        """Render a non-error status message."""

        self._write_wrapped("INFO  ", message, tone="muted")

    def assistant(self, message: str) -> None:
        """Render the assistant's final response for one user turn."""

        self._write(self._paint("ASSISTANT", "green"))
        for line in (message or "").splitlines() or [""]:
            self._write(f"  {line}")

    def help(self) -> None:
        """Render the commands supported by the interactive session."""

        self._write("Commands:")
        self._write("  /models        show model aliases and provider routing")
        self._write("  /model [NAME]  show aliases or change the model for the next turn")
        self._write("  /reasoning [LEVEL]  show or set reasoning effort for the next turn")
        self._write("  /steps [N]     show or set the maximum model turns per task")
        self._write("  /mode [auto|ask|code]  show or change task authority")
        self._write("  /status        show session, context, and latest task status")
        self._write("  /context       show the latest local context budget and compaction")
        self._write("  /capabilities  show active provider capabilities and limitations")
        self._write("  /plan          show the latest retained task plan")
        self._write("  /sessions      list resumable sessions in this workspace")
        self._write("  /resume <ID|#> restore by ID, prefix, or list number (/resume latest works)")
        self._write("  /continue [N]  continue the unfinished plan; optionally change its step budget")
        self._write("  During a task: /steer or /followup <message>; /abort (interactive terminal only)")
        self._write("  /new           start a new conversation and keep saved sessions")
        self._write("  /clear         delete only the current saved conversation")
        self._write("  /help          show this help")
        self._write("  /exit          leave DBA")

    def render_context(self, state: Any | None) -> None:
        """Render local context facts without dumping prompt contents."""

        self._write("")
        self._write(self._paint("Local context", "magenta"))
        if state is None or not getattr(state, "context_usage", None):
            self._write("  No completed Agent run in this session.")
            return
        usage = state.context_usage[-1]
        self._write(
            "  latest="
            f"{usage.approximate_tokens}~tok / {usage.budget_characters // 4}~tok "
            f"({usage.input_characters} chars)"
        )
        self._write(
            "  observations="
            f"recent {usage.recent_observations}, compacted {usage.compacted_observations}, "
            f"truncated {usage.truncated_items}"
        )

    def render_capabilities(self, policy: Any) -> None:
        """Render provider behavior that is actually active for this session."""

        self._write("")
        self._write(self._paint("Provider capabilities", "magenta"))
        self._write(f"  {getattr(policy, 'capability_summary', policy)}")
        self._write(f"  tool turns: {getattr(policy, 'tool_turn_policy', 'unknown')}")

    def render_model_options(self, presets: Sequence[Any], *, current_model: str) -> None:
        """Render short aliases without disclosing any provider credentials."""

        self._write("")
        self._write(self._paint("Model options", "magenta"))
        self._write("  Alias              Model                 Provider     Description")
        for preset in presets:
            marker = ">" if getattr(preset, "model", "") == current_model else " "
            self._write(
                f" {marker} {str(getattr(preset, 'alias', '?')):<18} "
                f"{str(getattr(preset, 'model', '?')):<21} "
                f"{str(getattr(preset, 'provider', '?')):<12} "
                f"{_truncate(str(getattr(preset, 'label', '')), 35)}"
            )

    def goodbye(self) -> None:
        """Render the session exit message."""

        self._write(
            self._paint(
                "Session closed. Workspace checkpoints remain available via /resume.",
                "muted",
            )
        )

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
        self._write(_box_row(f"Task      {_truncate(task, 70)}"))
        self._write(_box_row(f"Workspace {_truncate(str(workspace), 70)}"))
        self._write(_box_row(f"Model     {_truncate(model, 70)}"))
        self._write(_box_row(f"Mode      {_truncate(mode, 70)}"))
        self._write(_box_row(f"Budget    {max_steps} model turns"))
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
        phase = _phase_label(payload.get("phase"))
        phase_prefix = f"{phase}  " if phase else ""

        if event == "run_started":
            detail = (
                f"START  mode={payload.get('mode')}  "
                f"tools={payload.get('tool_count')}"
            )
        elif event == "assistant_update":
            detail = "AGENT 进度  " + _truncate(str(payload.get("text", "")), 110)
        elif event == "model_request":
            usage = payload.get("context_usage", {})
            approximate_tokens = usage.get("approximate_tokens", "?")
            budget_characters = usage.get("budget_characters")
            budget_tokens = (
                (int(budget_characters) + 3) // 4
                if isinstance(budget_characters, int)
                else "?"
            )
            detail = (
                f"{phase_prefix}分析中  "
                f"context={approximate_tokens}/{budget_tokens}~tok  "
                f"recent={usage.get('recent_observations', 0)}  "
                f"compact={usage.get('compacted_observations', 0)}"
            )
        elif event == "model_response":
            usage = payload.get("usage") or {}
            tokens = (
                f"  tokens={usage.get('total_tokens')}"
                if usage.get("total_tokens") is not None
                else ""
            )
            detail = (
                f"{phase_prefix}模型响应  "
                f"status={payload.get('status')}  "
                f"calls={payload.get('function_call_count')}{tokens}"
            )
        elif event == "model_wait":
            detail = (
                "仍在等待模型  "
                f"provider request running for {payload.get('waiting_seconds', '?')}s"
            )
        elif event == "model_stream":
            if payload.get("kind") == "text_delta":
                # Do not add a newline per token: this is a genuine streaming
                # progress signal, but remains readable in normal terminals.
                detail = "模型流式输出  " + _truncate(
                    str(payload.get("delta", "")), 96
                )
            else:
                detail = "模型正在构造本地工具调用"
        elif event == "user_steering":
            detail = "已接收用户实时指令，将在下一安全边界生效"
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
            if payload.get("intent"):
                detail = f"{payload['intent']}  {detail}"
            if payload.get("evidence_status") == "duplicate":
                detail = "重复证据  " + detail
            if payload.get("plan_step"):
                detail = f"[{payload['plan_step']}]  {detail}"
        elif event == "tool_result":
            detail = _human_tool_result(payload)
            if payload.get("result_summary"):
                detail += f"  {payload['result_summary']}"
            if payload.get("duplicate_evidence"):
                detail += "  未产生新证据"
            if payload.get("return_code") is not None:
                detail += f"  rc={payload['return_code']}"
            if payload.get("content_characters") is not None:
                detail += f"  {_human_size(int(payload['content_characters']))}"
            if payload.get("stdout_characters") is not None:
                detail += f"  out={_human_size(int(payload['stdout_characters']))}"
            if payload.get("timed_out"):
                detail += "  TIMEOUT"
            if payload.get("stdout_truncated") or payload.get("stderr_truncated"):
                detail += "  TRUNCATED"
            if payload.get("changed_files"):
                detail += f"  files={payload['changed_files']}"
            elif payload.get("path"):
                detail += f"  path={payload['path']}"
            if payload.get("line_range"):
                detail += f"  lines={payload['line_range']}"
            if payload.get("query") and payload.get("tool_name") == "search_text":
                detail += f"  query={payload['query']}"
            if not payload.get("success") and payload.get("failure_reason"):
                detail += "  " + _truncate(
                    str(payload["failure_reason"]),
                    72,
                )
        elif event == "patch_applied":
            detail = (
                "修改完成  "
                f"files={payload.get('changed_files', [])}  "
                f"hunks={payload.get('hunks_applied', 0)}"
            )
        elif event == "plan_updated":
            completed = payload.get("completed_steps", 0)
            total = payload.get("total_steps", "?")
            detail = (
                f"计划 {completed}/{total}  "
                f"当前: {_truncate(str(payload.get('current_step_description') or payload.get('current_step') or 'done'), 56)}"
            )
        elif event == "context_compacted":
            detail = (
                "上下文摘要  "
                f"older={payload.get('compacted_observations', 0)}  "
                f"recent={payload.get('recent_observations', 0)}  "
                f"truncated={payload.get('truncated_items', 0)}  "
                f"context={payload.get('approximate_tokens', '?')}~tok"
            )
        elif event == "verification":
            status = payload.get("status")
            detail = (
                "验证 VERIFY  "
                f"status={status}  kind={payload.get('kind')}  "
                f"return_code={payload.get('return_code')}"
            )
        elif event == "recovery":
            detail = f"RECOVERY  {_truncate(str(payload.get('reason', '')), 88)}"
        elif event == "step_summary":
            detail = (
                f"STEP  {_phase_label(payload.get('phase')) or '处理'}  "
                f"tools={payload.get('succeeded', 0)}/{payload.get('tools', 0)} ok"
                f"  failed={payload.get('failed', 0)}"
            )
            if payload.get("current_plan_step"):
                detail += f"  next={payload.get('current_plan_step')}"
            if payload.get("verification") != "not_run":
                detail += f"  verify={payload.get('verification')}"
        elif event == "final":
            detail = (
                f"完成  status={payload.get('status')}  "
                f"verification={payload.get('verification_status', 'not_run')}"
            )
            if payload.get("reason"):
                detail += f"  reason={_truncate(str(payload['reason']), 56)}"
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
        self._write(_box_row(f"Status         {status}"))
        self._write(_box_row(
            f"Steps          {getattr(state, 'step', '?')}/{getattr(state, 'max_steps', '?')}"
        ))
        self._write(_box_row(f"Verification   {verification}"))
        plan = getattr(state, "plan", None)
        if plan is not None:
            completed = sum(
                getattr(step.status, "value", step.status) == "completed"
                for step in plan.steps
            )
            self._write(_box_row(
                f"Plan           {completed}/{len(plan.steps)} steps completed"
            ))
        self._write(_box_row(f"Elapsed        {elapsed:.1f}s"))
        self._write(_box_row(
            f"Files changed  {_truncate(', '.join(changed_files) or 'none', 62)}"
        ))
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

    def render_sessions(
        self,
        sessions: Sequence[Any],
        *,
        active_session_id: str = "",
    ) -> None:
        """Render a stable, copyable list for ``/resume <ID>``."""

        self._write("")
        self._write(self._paint("Saved sessions", "magenta"))
        if not sessions:
            self._write("  No saved sessions in this workspace.")
            return
        self._write("  #  ID                       Updated              Turns  Verify     Title")
        for index, item in enumerate(sessions, start=1):
            session_id = str(getattr(item, "session_id", "?"))
            active = ">" if session_id == active_session_id else " "
            updated = _display_timestamp(str(getattr(item, "updated_at", "")))
            turns = str(getattr(item, "turns", 0))
            status = str(getattr(item, "status", "unknown"))
            title = _truncate(str(getattr(item, "title", "Untitled")), 42)
            self._write(
                f" {active} {index:>2} {session_id:<24} {updated:<19} {turns:>5}  "
                f"{_truncate(status, 10):<10} {title}"
            )

    def render_resume_summary(
        self,
        *,
        session_id: str,
        title: str,
        turns: int,
        verification: str,
        observation_count: int,
        has_plan: bool,
        checkpoint_state: str,
    ) -> None:
        """Show exactly which persisted context became active after resume."""

        self._write("")
        self._write(self._paint("+-- Resumed context " + "-" * 56 + "+", "green"))
        self._write(f"| Session       {_truncate(session_id, 60):<60} |")
        self._write(f"| Title         {_truncate(title, 60):<60} |")
        self._write(f"| Chat turns    {turns:<60} |")
        self._write(f"| Verification  {_truncate(verification, 60):<60} |")
        self._write(f"| Observations  {observation_count:<60} |")
        self._write(f"| Plan restored {str(has_plan).lower():<60} |")
        self._write(f"| Checkpoint    {_truncate(checkpoint_state, 60):<60} |")
        self._write(self._paint("+" + "-" * 76 + "+", "green"))

    def error(self, message: str) -> None:
        """Render an error consistently with other dashboard output."""

        self._write_wrapped("ERROR ", message, tone="red")

    def _write_wrapped(self, label: str, message: str, *, tone: str) -> None:
        lines = textwrap.wrap(
            message,
            width=96,
            break_long_words=False,
            break_on_hyphens=False,
        ) or [""]
        self._write(self._paint(label, tone) + lines[0])
        continuation = " " * len(label)
        for line in lines[1:]:
            self._write(continuation + line)

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
    "model_wait": ("·", "muted"),
    "model_error": ("!!", "red"),
    "tool_start": ("->", "yellow"),
    "tool_result": ("OK", "green"),
    "patch_applied": ("#", "magenta"),
    "plan_updated": ("*", "magenta"),
    "context_compacted": ("~", "magenta"),
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


def _phase_label(value: Any) -> str:
    labels = {
        "inspect": "检查",
        "plan": "规划",
        "implement": "实现",
        "verify": "验证",
        "review": "复核",
        "recover": "恢复",
        "work": "处理",
    }
    return labels.get(str(value), "") if value else ""


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


def _display_width(value: str) -> int:
    """Approximate terminal cell width, including Chinese wide characters."""

    width = 0
    for character in value:
        if unicodedata.combining(character):
            continue
        width += 2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
    return width


def _fit_display(value: str, width: int) -> str:
    if width <= 0:
        return ""
    if _display_width(value) <= width:
        return value + " " * (width - _display_width(value))
    if width <= 3:
        result = ""
    else:
        result = ""
        current = 0
        for character in value:
            character_width = 0 if unicodedata.combining(character) else (
                2 if unicodedata.east_asian_width(character) in {"W", "F"} else 1
            )
            if current + character_width > width - 3:
                break
            result += character
            current += character_width
        result += "..."
    return result + " " * max(0, width - _display_width(result))


def _box_row(value: str, width: int = 76) -> str:
    """Render a dashboard row using terminal cells rather than Python len()."""

    return "|" + _fit_display(value, width) + "|"


def _human_size(characters: int) -> str:
    if characters < 1_000:
        return f"{characters} chars"
    return f"{characters / 1_000:.1f}k chars"


def _display_timestamp(value: str) -> str:
    if not value:
        return "unknown"
    try:
        return datetime.fromisoformat(value).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return value[:19].replace("T", " ")


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
