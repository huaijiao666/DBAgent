import io

from dbagent.console import safe_print
from dbagent.trace import TraceRecorder
from dbagent.ui import TerminalUI


def _ascii_stream() -> tuple[io.BytesIO, io.TextIOWrapper]:
    buffer = io.BytesIO()
    return buffer, io.TextIOWrapper(buffer, encoding="ascii", write_through=True)


def test_safe_print_replaces_unrepresentable_terminal_characters() -> None:
    buffer, stream = _ascii_stream()

    safe_print("Español 中文", stream=stream, flush=True)

    assert buffer.getvalue().decode("ascii").splitlines() == ["Espa?ol ??"]


def test_trace_preamble_cannot_crash_legacy_windows_console(tmp_path) -> None:
    buffer, stream = _ascii_stream()
    ui = TerminalUI(stream=stream, color=False)
    ui.start(task="task", workspace=tmp_path, model="model", max_steps=3)
    trace = TraceRecorder(
        tmp_path / "trace.jsonl",
        workspace=tmp_path,
        console=True,
        stream=stream,
        renderer=ui,
    )

    trace.publish(
        "assistant_update",
        step=1,
        payload={"text": "Voy a revisar el diseño 中文"},
    )
    trace.close()

    output = buffer.getvalue().decode("ascii")
    assert "AGENT" in output
    assert "dise?o" in output
