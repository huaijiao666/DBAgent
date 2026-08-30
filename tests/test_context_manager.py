import json

from forge.agent import ContextBudget, ContextManager
from forge.llm import FunctionCall, ModelResponse
from forge.tools import ToolObservation


def _call(call_id: str, name: str, arguments: dict) -> FunctionCall:
    return FunctionCall(
        call_id=call_id,
        name=name,
        arguments_json=json.dumps(arguments),
        status="completed",
    )


def _response(response_id: str, call: FunctionCall) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        model="gpt-5.6-sol",
        status="completed",
        output_text="",
        output_items=(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_json,
                "status": "completed",
                "phase": "commentary",
            },
        ),
        function_calls=(call,),
        usage=None,
    )


def _record(
    manager: ContextManager,
    response_id: str,
    call: FunctionCall,
    observation: ToolObservation,
) -> None:
    manager.record_turn(_response(response_id, call), [(call, observation)])


def _serialized_size(items) -> int:
    return len(
        json.dumps(items, ensure_ascii=False, separators=(",", ":"))
    )


def test_context_categories_and_usage_are_explicit_and_bounded() -> None:
    budget = ContextBudget(
        max_context_characters=12_000,
        max_task_characters=1_000,
        max_plan_characters=1_000,
        max_repository_map_characters=2_000,
        max_relevant_code_characters=2_000,
        max_compact_observations_characters=1_500,
        max_recent_observations_characters=2_500,
        max_single_observation_characters=800,
        max_call_arguments_characters=300,
        recent_observation_count=2,
    )
    manager = ContextManager("Fix the failing parser tests.", budget=budget)
    manager.set_plan("1. Reproduce\n2. Patch parser\n3. Run tests")
    manager.set_repository_map("src/parser.py\n  function parse [L1-20]")
    manager.add_relevant_code("src/parser.py::parse", "def parse(value):\n    pass")

    snapshot = manager.build_context(step=3)
    rendered = json.dumps(snapshot.input_items, ensure_ascii=False)

    assert "[Persistent task context]" in rendered
    assert "[Current plan]" in rendered
    assert "[Repository map]" in rendered
    assert "[Working/relevant code]" in rendered
    assert "[Compacted older observations]" in rendered
    assert snapshot.usage.step == 3
    assert snapshot.usage.input_characters == _serialized_size(
        snapshot.input_items
    )
    assert snapshot.usage.input_characters <= budget.max_context_characters
    assert snapshot.usage.approximate_tokens == (
        snapshot.usage.input_characters + 3
    ) // 4
    assert set(snapshot.usage.category_characters) == {
        "persistent_task",
        "execution_context",
        "current_plan",
        "latest_verification",
        "runtime_guidance",
        "repository_map",
        "relevant_code",
        "compact_observations",
        "recent_observations",
    }


def test_long_pytest_output_keeps_status_and_diagnostic_edges() -> None:
    manager = ContextManager("Fix tests")
    call = _call(
        "call_pytest",
        "run_command",
        {
            "command": ["python", "-m", "pytest", "-q"],
            "cwd": ".",
            "timeout_seconds": 30,
        },
    )
    long_stdout = (
        "pytest session starts\n"
        + "x" * 20_000
        + "\nFAILED tests/test_parser.py::test_empty\n1 failed, 20 passed\n"
    )
    observation = ToolObservation(
        call_id=call.call_id,
        tool_name=call.name,
        success=True,
        content={
            "command": ["python", "-m", "pytest", "-q"],
            "cwd": ".",
            "return_code": 1,
            "timed_out": False,
            "stdout": long_stdout,
            "stderr": "AssertionError: expected value\n",
            "stdout_truncated": False,
            "stderr_truncated": False,
        },
    )

    _record(manager, "resp_pytest", call, observation)
    snapshot = manager.build_context(step=2)
    output_item = next(
        item
        for item in snapshot.input_items
        if item.get("type") == "function_call_output"
    )
    payload = json.loads(output_item["output"])["result"]
    rendered = json.dumps(snapshot.input_items, ensure_ascii=False)

    assert payload["return_code"] == 1
    assert payload["timed_out"] is False
    assert payload["stdout_truncated"] is True
    assert "pytest session starts" in payload["stdout"]
    assert "1 failed, 20 passed" in payload["stdout"]
    assert "AssertionError: expected value" in payload["stderr"]
    assert "x" * 10_000 not in rendered
    assert snapshot.usage.input_characters <= manager.budget.max_context_characters
    assert snapshot.usage.truncated_items == 1


def test_old_observations_become_compact_but_keep_key_outcomes() -> None:
    budget = ContextBudget(recent_observation_count=1)
    manager = ContextManager("Fix and verify", budget=budget)

    failed_test = _call(
        "call_fail",
        "run_command",
        {"command": ["pytest"], "cwd": ".", "timeout_seconds": 30},
    )
    _record(
        manager,
        "resp_fail",
        failed_test,
        ToolObservation(
            call_id=failed_test.call_id,
            tool_name=failed_test.name,
            success=True,
            content={
                "command": ["pytest"],
                "cwd": ".",
                "return_code": 1,
                "timed_out": False,
                "stdout": "1 failed\n",
                "stderr": "AssertionError\n",
                "stdout_truncated": False,
                "stderr_truncated": False,
            },
        ),
    )
    patch = _call(
        "call_patch",
        "apply_patch",
        {"files": [{"path": "parser.py", "hunks": []}]},
    )
    _record(
        manager,
        "resp_patch",
        patch,
        ToolObservation(
            call_id=patch.call_id,
            tool_name=patch.name,
            success=True,
            content={
                "applied": True,
                "changed_files": [{"path": "parser.py"}],
                "hunks_applied": 1,
                "failure_reason": None,
            },
        ),
    )
    passed_test = _call(
        "call_pass",
        "run_command",
        {"command": ["pytest"], "cwd": ".", "timeout_seconds": 30},
    )
    _record(
        manager,
        "resp_pass",
        passed_test,
        ToolObservation(
            call_id=passed_test.call_id,
            tool_name=passed_test.name,
            success=True,
            content={
                "command": ["pytest"],
                "cwd": ".",
                "return_code": 0,
                "timed_out": False,
                "stdout": "21 passed\n",
                "stderr": "",
                "stdout_truncated": False,
                "stderr_truncated": False,
            },
        ),
    )
    recent_read = _call("call_read", "read_file", {"path": "parser.py"})
    _record(
        manager,
        "resp_read",
        recent_read,
        ToolObservation(
            call_id=recent_read.call_id,
            tool_name=recent_read.name,
            success=True,
            content="1: def parse():\n2:     return True\n",
        ),
    )

    snapshot = manager.build_context(step=5)
    compact_section = str(snapshot.input_items[1]["content"])
    raw_call_ids = {
        item.get("call_id")
        for item in snapshot.input_items
        if item.get("type") == "function_call"
    }
    raw_output_ids = {
        item.get("call_id")
        for item in snapshot.input_items
        if item.get("type") == "function_call_output"
    }

    assert manager.raw_observation_count == 1
    assert raw_call_ids == {"call_read"}
    assert raw_output_ids == raw_call_ids
    assert "return_code=1" in compact_section
    assert "1 failed" in compact_section
    assert "AssertionError" in compact_section
    assert "applied=True" in compact_section
    assert "parser.py" in compact_section
    assert "return_code=0" in compact_section
    assert "21 passed" in compact_section
    assert snapshot.usage.recent_observations == 1
    assert snapshot.usage.compacted_observations == 3


def test_repository_and_read_results_update_their_context_categories() -> None:
    manager = ContextManager("Inspect architecture")
    repo_call = _call("call_map", "get_repo_map", {})
    _record(
        manager,
        "resp_map",
        repo_call,
        ToolObservation(
            call_id=repo_call.call_id,
            tool_name=repo_call.name,
            success=True,
            content="src/app.py\n  function main [L1-5]",
        ),
    )
    read_call = _call("call_symbol", "read_symbol", {"symbol_id": "app.py::main@1"})
    _record(
        manager,
        "resp_symbol",
        read_call,
        ToolObservation(
            call_id=read_call.call_id,
            tool_name=read_call.name,
            success=True,
            content={
                "source": (
                    "1: def main():\n"
                    + "y" * 20_000
                    + "\n2:     return 0"
                )
            },
        ),
    )

    snapshot = manager.build_context(step=3)
    context_message = str(snapshot.input_items[1]["content"])

    assert "src/app.py" in context_message
    assert "read_symbol: app.py::main@1" in context_message
    assert "def main" in context_message
    assert "return 0" in context_message
    assert "y" * 10_000 not in context_message
