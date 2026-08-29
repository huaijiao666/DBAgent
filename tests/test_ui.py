import io
from pathlib import Path
from types import SimpleNamespace

from forge.agent import AgentStatus
from forge.agent.verification import VerificationStatus
from forge.ui import TerminalUI


def test_terminal_ui_renders_header_and_summary(tmp_path: Path) -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)
    ui.start(
        task="Fix the calculator bug",
        workspace=tmp_path,
        model="gpt-5.6-sol",
        max_steps=12,
    )
    ui.finish(
        SimpleNamespace(
            status=AgentStatus.COMPLETED,
            is_verified=True,
            verification_status=VerificationStatus.PASSED,
            step=4,
            max_steps=12,
            observations=[
                SimpleNamespace(content={"changed_files": ["calculator.py"]})
            ],
        )
    )

    output = stream.getvalue()
    assert "Forge coding agent" in output
    assert "Fix the calculator bug" in output
    assert "Status         VERIFIED" in output
    assert "Files changed  calculator.py" in output


def test_terminal_ui_formats_trace_events_without_ansi() -> None:
    ui = TerminalUI(stream=io.StringIO(), color=False)
    ui.start(task="task", workspace=Path("."), model="model", max_steps=3)

    request = ui.render_event(
        {
            "elapsed_ms": 1200,
            "step": 1,
            "event": "model_request",
            "payload": {
                "tools": ["read_file"],
                "context_usage": {"approximate_tokens": 321},
            },
        }
    )
    verification = ui.render_event(
        {
            "elapsed_ms": 2200,
            "step": 2,
            "event": "verification",
            "payload": {"status": "passed", "kind": "test", "return_code": 0},
        }
    )

    assert "MODEL request" in request
    assert "context=321~tok" in request
    assert "VERIFY" in verification
    assert "status=passed" in verification
    assert "\x1b[" not in request


def test_terminal_ui_does_not_label_max_steps_as_verified() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)
    ui.finish(
        SimpleNamespace(
            status=AgentStatus.MAX_STEPS,
            is_verified=True,
            verification_status=VerificationStatus.PASSED,
            step=12,
            max_steps=12,
            observations=[],
        )
    )

    assert "Status         INCOMPLETE" in stream.getvalue()
