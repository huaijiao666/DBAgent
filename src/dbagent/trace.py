"""Small JSONL trace writer and human-readable terminal renderer."""

from __future__ import annotations

import json
import re
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO

from dbagent.console import safe_print


class ConsoleRenderer(Protocol):
    """Render one already-sanitized trace event for a human terminal."""

    def render_event(self, item: Mapping[str, Any]) -> str: ...

_SENSITIVE_KEY = re.compile(
    r"(?:api[_-]?key|access[_-]?token|auth(?:orization)?|password|passwd|secret)",
    re.IGNORECASE,
)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:"
    r"sk-[a-z0-9_-]{8,}|"
    r"authorization\s*[:=]\s*bearer\s+[^\s,;]+|"
    r"(?:bearer|token|password|secret|api[_-]?key)\s*[:=]\s*[^\s,;]+"
    r")"
)


class TraceRecorder:
    """Write sanitized structured events and optionally mirror them to stderr."""

    def __init__(
        self,
        path: Path,
        *,
        workspace: Path | None = None,
        console: bool = False,
        stream: TextIO | None = None,
        renderer: ConsoleRenderer | None = None,
        progress_interval_seconds: float = 15.0,
    ) -> None:
        if progress_interval_seconds < 0:
            raise ValueError("progress_interval_seconds must not be negative")
        self.path = path
        self.workspace = workspace.resolve() if workspace is not None else None
        self.console = console
        self._stream = stream or sys.stderr
        self._renderer = renderer
        self._progress_interval_seconds = progress_interval_seconds
        self._waiting_stop = threading.Event()
        self._waiting_thread: threading.Thread | None = None
        self._stream_text_buffers: dict[int, str] = {}
        self._started = time.monotonic()
        self._file = self._open(path)

    def record(
        self,
        event: str,
        *,
        step: int,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Append one flushed JSONL event and render its concise console line."""

        if event in {"model_response", "final"}:
            self._flush_stream_text(step)
        if event in {"model_response", "final"} or (
            event == "model_error" and not (payload or {}).get("will_retry")
        ):
            self._stop_waiting()
        elapsed_ms = round((time.monotonic() - self._started) * 1000)
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": elapsed_ms,
            "step": step,
            "event": event,
            "payload": _sanitize(dict(payload or {})),
        }
        self._file.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        self._file.write("\n")
        self._file.flush()
        if self.console:
            consumer = getattr(self._renderer, "consume_event", None)
            if callable(consumer):
                consumer(item)
            else:
                line = (
                    self._renderer.render_event(item)
                    if self._renderer is not None
                    else _format_console_line(item)
                )
                safe_print(line, stream=self._stream, flush=True)
            if event == "model_request":
                self._start_waiting(step)

    def publish(
        self,
        event: str,
        *,
        step: int,
        payload: Mapping[str, Any] | None = None,
    ) -> None:
        """Render a sanitized transient event without persisting it to JSONL.

        Model preambles improve the live terminal experience but are not stored:
        the durable trace retains structured metadata instead of arbitrary model
        prose that could echo sensitive user input.
        """

        if event == "model_stream":
            # The first real server-sent event is stronger feedback than the
            # periodic waiting heartbeat, so stop it immediately.
            self._stop_waiting()
            if (payload or {}).get("kind") == "text_delta":
                delta = str((payload or {}).get("delta", ""))
                buffered = self._stream_text_buffers.get(step, "") + delta
                # Streaming one terminal line per token is noisy and can make
                # a Windows console feel slower than the API.  Flush complete
                # lines or bounded chunks while retaining true incremental
                # rendering.
                if "\n" not in buffered and len(buffered) < 96:
                    self._stream_text_buffers[step] = buffered
                    return
                payload = {**(payload or {}), "delta": buffered}
                self._stream_text_buffers.pop(step, None)
        if not self.console:
            return
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000),
            "step": step,
            "event": event,
            "payload": _sanitize(dict(payload or {})),
        }
        consumer = getattr(self._renderer, "consume_event", None)
        if callable(consumer):
            consumer(item)
            return
        line = (
            self._renderer.render_event(item)
            if self._renderer is not None
            else _format_console_line(item)
        )
        safe_print(line, stream=self._stream, flush=True)

    def _flush_stream_text(self, step: int) -> None:
        """Render a final partial text chunk before a completed response line."""

        delta = self._stream_text_buffers.pop(step, "")
        if not delta or not self.console:
            return
        item = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "elapsed_ms": round((time.monotonic() - self._started) * 1000),
            "step": step,
            "event": "model_stream",
            "payload": _sanitize({"kind": "text_delta", "delta": delta}),
        }
        consumer = getattr(self._renderer, "consume_event", None)
        if callable(consumer):
            consumer(item)
            return
        line = (
            self._renderer.render_event(item)
            if self._renderer is not None
            else _format_console_line(item)
        )
        safe_print(line, stream=self._stream, flush=True)

    def close(self) -> None:
        self._stop_waiting()
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _open(path: Path) -> TextIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        # Preserve earlier evidence when DBA restarts or resumes a workspace.
        return path.open("a", encoding="utf-8", newline="\n")

    def _start_waiting(self, step: int) -> None:
        self._stop_waiting()
        if self._progress_interval_seconds == 0:
            return
        stop_event = threading.Event()
        self._waiting_stop = stop_event
        waiting_started = time.monotonic()

        def report() -> None:
            delay = self._progress_interval_seconds
            while not stop_event.wait(delay):
                waiting_seconds = round(time.monotonic() - waiting_started)
                item = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "elapsed_ms": round((time.monotonic() - self._started) * 1000),
                    "step": step,
                    "event": "model_wait",
                    "payload": {"waiting_seconds": waiting_seconds},
                }
                consumer = getattr(self._renderer, "consume_event", None)
                if callable(consumer):
                    consumer(item)
                    delay = min(delay * 2, 60.0)
                    continue
                line = (
                    self._renderer.render_event(item)
                    if self._renderer is not None
                    else _format_console_line(item)
                )
                safe_print(line, stream=self._stream, flush=True)
                delay = min(delay * 2, 60.0)

        self._waiting_thread = threading.Thread(
            target=report,
            name="dba-model-wait",
            daemon=True,
        )
        self._waiting_thread.start()

    def _stop_waiting(self) -> None:
        self._waiting_stop.set()
        thread = self._waiting_thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=0.2)
        self._waiting_thread = None


def _sanitize(value: Any, *, key: str = "") -> Any:
    if _SENSITIVE_KEY.search(key):
        return "[REDACTED]"
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub("[REDACTED]", value)
    if isinstance(value, Mapping):
        return {
            str(item_key): _sanitize(item_value, key=str(item_key))
            for item_key, item_value in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_sanitize(item, key=key) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def sanitize_for_storage(value: Any) -> Any:
    """Return the same recursively redacted representation used by traces."""

    return _sanitize(value)


def _format_console_line(item: Mapping[str, Any]) -> str:
    elapsed = f"{float(item['elapsed_ms']) / 1000:.1f}s"
    step = item.get("step", "?")
    event = item.get("event")
    payload = item.get("payload") or {}
    if event == "model_request":
        usage = payload.get("context_usage", {})
        return (
            f"[{elapsed}] step {step} MODEL request: "
            f"context={usage.get('approximate_tokens', '?')}~tok, "
            f"tools={len(payload.get('tools', []))}"
        )
    if event == "model_response":
        usage = payload.get("usage") or {}
        token_text = (
            f", tokens={usage.get('total_tokens')}"
            if usage.get("total_tokens") is not None
            else ""
        )
        return (
            f"[{elapsed}] step {step} MODEL response: "
            f"status={payload.get('status')}, calls={payload.get('function_call_count')}"
            f"{token_text}"
        )
    if event == "mode_selected":
        return (
            f"[{elapsed}] step {step} MODE: "
            f"selected={payload.get('mode')}, source={payload.get('source')}"
        )
    if event == "model_wait":
        return (
            f"[{elapsed}] step {step} MODEL waiting: "
            f"{payload.get('waiting_seconds', '?')}s"
        )
    if event == "context_compacted":
        return (
            f"[{elapsed}] step {step} CONTEXT summarized: "
            f"older={payload.get('compacted_observations', 0)}, "
            f"recent={payload.get('recent_observations', 0)}, "
            f"truncated={payload.get('truncated_items', 0)}"
        )
    if event == "tool_start":
        return f"[{elapsed}] step {step} TOOL -> {payload.get('tool_name')}"
    if event == "tool_result":
        status = "ok" if payload.get("success") else "error"
        detail = payload.get("return_code")
        if detail is not None:
            detail = f", return_code={detail}"
        else:
            detail = ""
        changed = payload.get("changed_files")
        if changed:
            detail += f", files={changed}"
        elif payload.get("path"):
            detail += f", path={payload.get('path')}"
        return (
            f"[{elapsed}] step {step} TOOL <- {payload.get('tool_name')} "
            f"{status}{detail}"
        )
    if event == "patch_applied":
        return (
            f"[{elapsed}] step {step} PATCH: "
            f"files={payload.get('changed_files', [])}, "
            f"hunks={payload.get('hunks_applied', 0)}"
        )
    if event == "plan_updated":
        statuses = payload.get("step_statuses", {})
        return (
            f"[{elapsed}] step {step} PLAN: goal={payload.get('goal')}, "
            f"current={payload.get('current_step')}, statuses={statuses}"
        )
    if event == "verification":
        return (
            f"[{elapsed}] step {step} VERIFY: status={payload.get('status')}, "
            f"kind={payload.get('kind')}, return_code={payload.get('return_code')}"
        )
    if event == "recovery":
        return f"[{elapsed}] step {step} RECOVERY: {payload.get('reason')}"
    if event == "final":
        return f"[{elapsed}] step {step} FINAL: status={payload.get('status')}"
    return f"[{elapsed}] step {step} {event}"
