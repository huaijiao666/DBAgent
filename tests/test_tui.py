import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from forge.agent import AgentStatus
from forge.agent.verification import VerificationStatus
from forge.trace import TraceRecorder
from forge.tui import FullscreenTUI


def test_fullscreen_tui_draws_dashboard_and_restores_terminal(tmp_path: Path) -> None:
    stream = io.StringIO()
    ui = FullscreenTUI(stream=stream, force_terminal=True)
    ui.session_start(
        workspace=tmp_path,
        model="gpt-5.6-sol",
        api_mode="responses",
        mode="code",
        session_id="session-123",
    )
    ui.start(
        task="Fix the calculator",
        workspace=tmp_path,
        model="gpt-5.6-sol",
        max_steps=12,
        mode="code",
    )
    ui.consume_event(
        {
            "elapsed_ms": 100,
            "step": 2,
            "event": "verification",
            "payload": {"status": "passed", "kind": "test", "return_code": 0},
        }
    )
    ui.assistant("The calculator test now passes after the focused patch.")
    ui.finish(
        SimpleNamespace(
            status=AgentStatus.COMPLETED,
            verification_status=VerificationStatus.PASSED,
            step=2,
            observations=[],
        )
    )
    ui.close(message="closed")

    output = stream.getvalue()
    assert "\x1b[?1049h" in output
    assert "DBAgent  |  Local Coding Agent  |  TUI" in output
    assert "Fix the calculator" in output
    assert "verification passed" in output
    assert "RESPONSE" in output
    assert "calculator test now passes" in output
    assert "RUN SUMMARY" not in output  # dashboard, not scrolling CLI boxes
    assert "\x1b[?1049l" in output
    assert output.endswith("closed\n")


def test_fullscreen_tui_receives_trace_without_scrolling_console(tmp_path: Path) -> None:
    stream = io.StringIO()
    ui = FullscreenTUI(stream=stream, force_terminal=True)
    ui.session_start(
        workspace=tmp_path,
        model="model",
        api_mode="responses",
        session_id="session",
    )
    path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(path, console=True, stream=stream, renderer=ui)
    try:
        trace.record(
            "tool_start",
            step=1,
            payload={"tool_name": "read_file", "target": "README.md"},
        )
    finally:
        trace.close()
        ui.close()

    events = [json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()]
    assert events == ["tool_start"]
    assert "读取文件  README.md" in stream.getvalue()


def test_fullscreen_tui_requires_tty_unless_test_override() -> None:
    with pytest.raises(ValueError, match="interactive terminal"):
        FullscreenTUI(stream=io.StringIO())
