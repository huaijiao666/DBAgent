import json
import shutil
from pathlib import Path

from forge.agent import (
    AgentLoop,
    AgentStatus,
    VerificationStatus,
    VerificationTracker,
)
from forge.llm import FunctionCall, ModelResponse
from forge.trace import TraceRecorder
from forge.tools import ToolObservation, create_coding_registry


class ScriptedModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self.responses = responses
        self.requests = []

    def create_response(self, request):
        self.requests.append(request)
        if not self.responses:
            raise AssertionError("model called more times than expected")
        return self.responses.pop(0)


def _tool_response(response_id: str, call_id: str, name: str, arguments: dict) -> ModelResponse:
    arguments_json = json.dumps(arguments)
    call = FunctionCall(
        call_id=call_id,
        name=name,
        arguments_json=arguments_json,
    )
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
                "status": "completed",
            },
        ),
        function_calls=(call,),
        usage=None,
    )


def _final_response() -> ModelResponse:
    return ModelResponse(
        response_id="resp_final",
        model="gpt-5.6-sol",
        status="completed",
        output_text="Fixed and verified.",
        output_items=({"type": "message", "role": "assistant"},),
        function_calls=(),
        usage=None,
    )


def _test_command() -> dict[str, object]:
    return {
        "command": ["python", "-m", "pytest", "-q"],
        "cwd": ".",
        "timeout_seconds": 30,
    }


def _patch(old_line: str, new_line: str) -> dict[str, object]:
    return {
        "files": [
            {
                "path": "calculator.py",
                "hunks": [{"old_lines": [old_line], "new_lines": [new_line]},],
            }
        ]
    }


def test_failed_test_feedback_then_second_patch_is_verified(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "simple_bug_repo"
    workspace = tmp_path / "simple_bug_repo"
    shutil.copytree(fixture, workspace)
    model = ScriptedModelClient(
        [
            _tool_response(
                "resp_bad_patch",
                "call_bad_patch",
                "apply_patch",
                _patch("    return left - right", "    return left * right"),
            ),
            _tool_response(
                "resp_failed_test",
                "call_failed_test",
                "run_command",
                _test_command(),
            ),
            _tool_response(
                "resp_good_patch",
                "call_good_patch",
                "apply_patch",
                _patch("    return left * right", "    return left + right"),
            ),
            _tool_response(
                "resp_passed_test",
                "call_passed_test",
                "run_command",
                _test_command(),
            ),
            _final_response(),
        ]
    )

    trace_path = tmp_path / "trace.jsonl"
    trace = TraceRecorder(trace_path, workspace=workspace)
    try:
        state = AgentLoop(
            model,
            create_coding_registry(workspace),
            max_steps=8,
            trace=trace,
        ).run("Fix the calculator bug and verify it.", workspace=workspace)
    finally:
        trace.close()

    assert state.status is AgentStatus.COMPLETED
    assert state.verification_status is VerificationStatus.PASSED
    assert len(state.verification_history) == 2
    assert state.latest_verification is not None
    assert state.latest_verification.return_code == 0
    assert "return_code=1" in str(model.requests[2].input[1]["content"])
    assert "return left + right" in (workspace / "calculator.py").read_text(
        encoding="utf-8"
    )
    events = [
        json.loads(line)["event"]
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    assert {
        "model_request",
        "model_response",
        "tool_start",
        "tool_result",
        "patch_applied",
        "verification",
        "final",
    } <= set(events)


def test_final_claim_without_current_evidence_becomes_incomplete(
    tmp_path: Path,
) -> None:
    (tmp_path / "calculator.py").write_text("old\n", encoding="utf-8")
    model = ScriptedModelClient(
        [
            _tool_response(
                "resp_patch",
                "call_patch",
                "apply_patch",
                _patch("old", "new"),
            ),
            _final_response(),
        ]
    )
    state = AgentLoop(
        model,
        create_coding_registry(tmp_path),
        max_steps=2,
    ).run("Change the file and verify it.", workspace=tmp_path)

    assert state.status is AgentStatus.MAX_STEPS
    assert state.final_answer is None
    assert state.verification_status is VerificationStatus.NOT_RUN
    assert any("targeted" in hint for hint in state.recovery_hints)


def test_repeated_failure_signature_is_detected() -> None:
    tracker = VerificationTracker()
    call = FunctionCall(
        call_id="test",
        name="run_command",
        arguments_json=json.dumps(
            {"command": ["pytest", "-q"], "cwd": "."}
        ),
    )
    content = {
        "command": ["pytest", "-q"],
        "cwd": ".",
        "return_code": 1,
        "timed_out": False,
        "stdout": "1 failed",
        "stderr": "AssertionError",
    }

    first = tracker.observe(
        call,
        ToolObservation("test", "run_command", True, content),
    )
    second = tracker.observe(
        call,
        ToolObservation("test", "run_command", True, content),
    )

    assert first.repeated_failure_count == 1
    assert second.repeated_failure_count == 2
    assert tracker.status is VerificationStatus.FAILED


def test_passing_evidence_becomes_stale_after_a_successful_edit() -> None:
    tracker = VerificationTracker()
    test_call = FunctionCall(
        call_id="test",
        name="run_command",
        arguments_json=json.dumps(
            {"command": ["pytest", "-q"], "cwd": "."}
        ),
    )
    passing_content = {
        "command": ["pytest", "-q"],
        "cwd": ".",
        "return_code": 0,
        "timed_out": False,
        "stdout": "2 passed",
        "stderr": "",
    }
    tracker.observe(
        test_call,
        ToolObservation("test", "run_command", True, passing_content),
    )
    patch_call = FunctionCall(
        call_id="patch",
        name="apply_patch",
        arguments_json="{}",
    )

    event = tracker.observe(
        patch_call,
        ToolObservation(
            "patch",
            "apply_patch",
            True,
            {"applied": True, "changed_files": [{"path": "app.py"}]},
        ),
    )

    assert event.mutation is True
    assert tracker.status is VerificationStatus.STALE
    assert tracker.is_verified is False


def test_no_progress_rounds_add_recovery_guidance_to_next_request(
    tmp_path: Path,
) -> None:
    unknown_one = _tool_response("resp_1", "call_1", "missing_tool", {})
    unknown_two = _tool_response("resp_2", "call_2", "missing_tool", {})
    model = ScriptedModelClient([unknown_one, unknown_two, _final_response()])

    state = AgentLoop(
        model,
        create_coding_registry(tmp_path),
        max_steps=4,
    ).run("Inspect and explain the repository.", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.no_progress_rounds == 2
    assert any("repeating evidence" in hint for hint in state.recovery_hints)
    next_context = str(model.requests[2].input[1]["content"])
    assert "Stop exploring and answer" in next_context


def test_meaningful_plan_transition_resets_no_progress_rounds(
    tmp_path: Path,
) -> None:
    initial_plan = {
        "goal": "Inspect and explain",
        "success_criteria": ["an explanation is produced"],
        "steps": [
            {"id": "inspect", "description": "Inspect files", "status": "in_progress"},
            {"id": "explain", "description": "Explain findings", "status": "pending"},
        ],
    }
    progressed_plan = {
        **initial_plan,
        "steps": [
            {"id": "inspect", "description": "Inspect files", "status": "completed"},
            {"id": "explain", "description": "Explain findings", "status": "completed"},
        ],
    }
    model = ScriptedModelClient(
        [
            _tool_response("resp_plan_1", "plan_1", "update_plan", initial_plan),
            _tool_response("resp_unknown_1", "unknown_1", "missing_tool", {}),
            _tool_response("resp_plan_2", "plan_2", "update_plan", progressed_plan),
            _tool_response("resp_unknown_2", "unknown_2", "missing_tool", {}),
            _final_response(),
        ]
    )

    state = AgentLoop(
        model,
        create_coding_registry(tmp_path),
        max_steps=5,
        mode="code",
    ).run("Inspect and explain the repository.", workspace=tmp_path)

    assert state.status is AgentStatus.COMPLETED
    assert state.no_progress_rounds == 1
    assert not any("No meaningful progress" in hint for hint in state.recovery_hints)
