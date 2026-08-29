"""Small JSONL trace writer and human-readable terminal renderer."""

from __future__ import annotations

import json
import re
import sys
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol, TextIO


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
    ) -> None:
        self.path = path
        self.workspace = workspace.resolve() if workspace is not None else None
        self.console = console
        self._stream = stream or sys.stderr
        self._renderer = renderer
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
            line = (
                self._renderer.render_event(item)
                if self._renderer is not None
                else _format_console_line(item)
            )
            print(line, file=self._stream, flush=True)

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> TraceRecorder:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    @staticmethod
    def _open(path: Path) -> TextIO:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path.open("w", encoding="utf-8", newline="\n")


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
