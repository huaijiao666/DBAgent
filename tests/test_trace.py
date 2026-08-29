import io
import json
from pathlib import Path

from forge.agent import AgentLoop, AgentStatus
from forge.llm import FunctionCall, ModelResponse
from forge.trace import TraceRecorder
from forge.tools import create_coding_registry


def test_trace_is_jsonl_and_redacts_sensitive_values(tmp_path: Path) -> None:
    console = io.StringIO()
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder(path, console=True, stream=console)
    recorder.record(
        "tool_start",
        step=2,
        payload={
            "tool_name": "run_command",
            "api_key": "sk-test-secret-value",
            "note": "Authorization: Bearer hidden-value",
        },
    )
    recorder.close()

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["event"] == "tool_start"
    assert event["step"] == 2
    serialized = json.dumps(event, ensure_ascii=False)
    assert "sk-test-secret-value" not in serialized
    assert "hidden-value" not in serialized
    assert "TOOL -> run_command" in console.getvalue()


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
                "status": "pending",
            }
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
