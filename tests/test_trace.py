import io
import json
import threading
from pathlib import Path

from dbagent.agent import AgentLoop, AgentStatus
from dbagent.llm import FunctionCall, ModelResponse
from dbagent.trace import TraceRecorder
from dbagent.tools import create_coding_registry


def test_trace_is_jsonl_and_redacts_sensitive_values(tmp_path: Path) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, console=True, stream=console)
    recorder.record(
        "tool_start",
        step=2,
        payload={
            "tool_name": "run_command",
            "api_key": "synthetic-secret-value",
            "note": "Authorization: Bearer synthetic-token",
        },
    )
    recorder.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "tool_start"
    assert event["step"] == 2
    serialized = json.dumps(event, ensure_ascii=False)
    assert "synthetic-secret-value" not in serialized
    assert "hidden-value" not in serialized
    assert "TOOL -> run_command" in console.getvalue()


def test_trace_appends_across_process_restarts(tmp_path: Path) -> None:
    path = tmp_path / ".dbagent" / "trace.jsonl"

    with TraceRecorder(path, workspace=tmp_path) as first:
        first.record("run_started", step=0, payload={"run": 1})
    with TraceRecorder(path, workspace=tmp_path) as second:
        second.record("run_started", step=0, payload={"run": 2})

    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert [event["payload"]["run"] for event in events] == [1, 2]


def test_trace_console_renderer_shows_model_usage_and_verification(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(
        path,
        console=True,
        stream=console,
    )
    try:
        recorder.record(
            "model_request",
            step=1,
            payload={
                "tools": ["run_command"],
                "context_usage": {"approximate_tokens": 321},
            },
        )
        recorder.record(
            "verification",
            step=1,
            payload={
                "status": "passed",
                "kind": "test",
                "return_code": 0,
            },
        )
    finally:
        recorder.close()

    output = console.getvalue()
    assert "context=321~tok" in output
    assert "VERIFY: status=passed" in output


def test_transient_assistant_update_is_rendered_but_not_persisted(
    tmp_path: Path,
) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, console=True, stream=console)
    try:
        recorder.publish(
            "assistant_update",
            step=1,
            payload={"text": "I will inspect the entry point."},
        )
    finally:
        recorder.close()

    assert "assistant_update" in console.getvalue()
    assert path.read_text(encoding="utf-8") == ""


def test_streamed_text_is_batched_for_console_and_not_persisted(tmp_path: Path) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, console=True, stream=console)
    try:
        recorder.publish(
            "model_stream", step=1, payload={"kind": "text_delta", "delta": "hello"}
        )
        assert console.getvalue() == ""
        recorder.record(
            "model_response",
            step=1,
            payload={"status": "completed", "function_call_count": 0},
        )
    finally:
        recorder.close()

    assert "model_stream" in console.getvalue()
    assert [
        json.loads(line)["event"] for line in path.read_text(encoding="utf-8").splitlines()
    ] == ["model_response"]


def test_model_wait_heartbeat_is_visible_but_not_persisted(tmp_path: Path) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(
        path,
        console=True,
        stream=console,
        progress_interval_seconds=0.01,
    )
    recorder.record("model_request", step=1, payload={"context_usage": {}})
    threading.Event().wait(0.05)
    recorder.record(
        "model_response",
        step=1,
        payload={"status": "completed", "function_call_count": 0},
    )
    recorder.close()

    assert "MODEL waiting" in console.getvalue()
    events = [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert events == ["model_request", "model_response"]


def test_trace_transient_events_use_tui_consumer_instead_of_scrolling(
    tmp_path: Path,
) -> None:
    class Consumer:
        def __init__(self) -> None:
            self.events: list[str] = []

        def consume_event(self, item):
            self.events.append(str(item["event"]))

        def render_event(self, _item):
            return "unexpected scrolling output"

    stream = io.StringIO()
    consumer = Consumer()
    recorder = TraceRecorder(
        tmp_path / "trace.jsonl",
        console=True,
        stream=stream,
        renderer=consumer,
        progress_interval_seconds=0.01,
    )
    try:
        recorder.publish(
            "model_stream", step=1, payload={"kind": "text_delta", "delta": "hello"}
        )
        recorder.record("model_response", step=1, payload={"status": "completed"})
        recorder.record("model_request", step=2, payload={})
        threading.Event().wait(0.03)
        recorder.record("model_response", step=2, payload={"status": "completed"})
    finally:
        recorder.close()

    assert "model_stream" in consumer.events
    assert "model_wait" in consumer.events
    assert stream.getvalue() == ""


class _ScriptedModel:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses

    def create_response(self, _request):
        return self.responses.pop(0)


def _call_response(response_id: str, call_id: str, name: str, arguments: dict) -> ModelResponse:
    arguments_json = json.dumps(arguments)
    call = FunctionCall(call_id, name, arguments_json)
    return ModelResponse(
        response_id=response_id,
        model="gpt-5.6-sol",
        status="completed",
        output_text="",
        output_items=(
            {
                "type": "function_call",
                "call_id": call_id,
                "name": name,
                "arguments": arguments_json,
            },
        ),
        function_calls=(call,),
        usage=None,
    )


def test_loop_trace_contains_plan_and_recovery_events(tmp_path: Path) -> None:
    plan = {
        "goal": "Inspect repository",
        "success_criteria": ["Explain architecture"],
        "steps": [
            {
                "id": "inspect",
                "description": "Inspect files",
                "status": "in_progress",
            },
            {
                "id": "verify",
                "description": "Verify conclusions",
                "status": "pending",
            },
            {
                "id": "deliver",
                "description": "Summarize evidence",
                "status": "pending",
            },
        ],
    }
    final = ModelResponse(
        response_id="final",
        model="gpt-5.6-sol",
        status="completed",
        output_text="Done",
        output_items=({"type": "message"},),
        function_calls=(),
        usage=None,
    )
    model = _ScriptedModel(
        [
            _call_response("plan", "plan_call", "update_plan", plan),
            _call_response("unknown_1", "unknown_call_1", "missing", {}),
            _call_response("unknown_2", "unknown_call_2", "missing", {}),
            _call_response(
                "finish_plan",
                "finish_plan_call",
                "update_plan",
                {
                    **plan,
                    "steps": [
                        {**step, "status": "completed"}
                        for step in plan["steps"]
                    ],
                },
            ),
            final,
        ]
    )
    path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(path)
    try:
        state = AgentLoop(
            model,
            create_coding_registry(tmp_path),
            max_steps=5,
            mode="code",
            trace=trace,
        ).run("Inspect repository", workspace=tmp_path)
    finally:
        trace.close()

    assert state.status is AgentStatus.COMPLETED
    events = [
        json.loads(line)["event"]
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert "plan_updated" in events
    assert "recovery" in events


def test_loop_trace_keeps_successful_verification_command_details(
    tmp_path: Path,
) -> None:
    (tmp_path / "test_pass.py").write_text(
        "def test_passes():\n    assert True\n",
        encoding="utf-8",
    )
    final = ModelResponse(
        response_id="final",
        model="gpt-5.6-sol",
        status="completed",
        output_text="Verified.",
        output_items=({"type": "message"},),
        function_calls=(),
        usage=None,
    )
    model = _ScriptedModel(
        [
            _call_response(
                "test",
                "test_call",
                "run_command",
                {
                    "command": ["python", "-m", "pytest", "-q"],
                    "cwd": ".",
                    "timeout_seconds": 10,
                },
            ),
            final,
        ]
    )
    path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(path)
    try:
        state = AgentLoop(
            model,
            create_coding_registry(tmp_path),
            max_steps=3,
            mode="ask",
            trace=trace,
        ).run("Run the test and report the result.", workspace=tmp_path)
    finally:
        trace.close()

    assert state.status is AgentStatus.COMPLETED
    verification_events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if json.loads(line)["event"] == "verification"
    ]
    successful = [
        event
        for event in verification_events
        if event["payload"].get("kind") == "test"
    ]
    assert successful
    assert successful[-1]["payload"]["status"] == "passed"
    assert successful[-1]["payload"]["return_code"] == 0
    assert successful[-1]["payload"]["command"][-2:] == ["pytest", "-q"]
