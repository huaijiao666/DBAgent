import json
import shutil
from pathlib import Path

from dbagent.agent import AgentLoop, AgentStatus
from dbagent.llm import FunctionCall, ModelResponse
from dbagent.tools import create_coding_registry


class ScriptedModelClient:
    def __init__(self, responses: list[ModelResponse]) -> None:
        self._responses = responses

    def create_response(self, _request):
        if not self._responses:
            raise AssertionError("model called more times than scripted")
        return self._responses.pop(0)


def _tool_response(
    response_id: str,
    call_id: str,
    name: str,
    arguments: dict,
) -> ModelResponse:
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
    text = "Fixed calculator.add and verified the test suite passes."
    return ModelResponse(
        response_id="resp_final",
        model="gpt-5.6-sol",
        status="completed",
        output_text=text,
        output_items=(
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text}],
            },
        ),
        function_calls=(),
        usage=None,
    )


def test_agent_completes_read_test_edit_test_loop(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "fixtures" / "simple_bug_repo"
    workspace = tmp_path / "simple_bug_repo"
    shutil.copytree(fixture, workspace)
    test_command = {
        "command": ["python", "-m", "pytest", "-q"],
        "cwd": ".",
        "timeout_seconds": 30,
    }
    source_patch = {
        "files": [
            {
                "path": "calculator.py",
                "hunks": [
                    {
                        "old_lines": ["    return left - right"],
                        "new_lines": [
                            "    # Addition must combine both operands.",
                            "    return left + right",
                        ],
                    }
                ],
            }
        ]
    }
    model = ScriptedModelClient(
        [
            _tool_response(
                "resp_plan",
                "call_plan",
                "update_plan",
                {
                    "goal": "Fix the calculator bug and verify it.",
                    "success_criteria": ["The calculator tests pass"],
                    "steps": [
                        {"id": "inspect", "description": "Inspect calculator behavior", "status": "in_progress"},
                        {"id": "fix", "description": "Correct addition", "status": "pending"},
                        {"id": "verify", "description": "Run calculator tests", "status": "pending"},
                        {"id": "deliver", "description": "Summarize evidence", "status": "pending"},
                    ],
                },
            ),
            _tool_response(
                "resp_read",
                "call_read",
                "read_file",
                {"path": "calculator.py"},
            ),
            _tool_response(
                "resp_test_before",
                "call_test_before",
                "run_command",
                test_command,
            ),
            _tool_response(
                "resp_patch",
                "call_patch",
                "apply_patch",
                source_patch,
            ),
            _tool_response(
                "resp_test_after",
                "call_test_after",
                "run_command",
                test_command,
            ),
            _final_response(),
        ]
    )

    state = AgentLoop(
        model,
        create_coding_registry(workspace),
        max_steps=8,
        mode="code",
    ).run("Fix the calculator bug and verify it.", workspace=workspace)

    assert state.status is AgentStatus.COMPLETED
    assert state.step == 6
    assert "return left - right" in state.observations[1].content
    assert state.observations[2].content["return_code"] == 1
    assert state.observations[3].success is True
    assert state.observations[3].content["hunks_applied"] == 1
    assert state.observations[4].content["return_code"] == 0, state.observations[
        4
    ].content
    assert "return left + right" in (workspace / "calculator.py").read_text(
        encoding="utf-8"
    )
