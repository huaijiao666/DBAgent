import json
from pathlib import Path

from dbagent.agent import (
    AgentLoop,
    AgentStatus,
    TaskMode,
    resolve_task_mode,
)
from dbagent.agent.mode import instructions_for_mode
from dbagent.llm import FunctionCall, FunctionTool, ModelResponse
from dbagent.tools import ToolDefinition, ToolRegistry


class ScriptedModel:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create_response(self, request):
        self.requests.append(request)
        return self.responses.pop(0)


def _response(response_id: str, *, text: str = "", calls=()) -> ModelResponse:
    return ModelResponse(
        response_id=response_id,
        model="model",
        status="completed",
        output_text=text,
        output_items=tuple(
            {
                "type": "function_call",
                "call_id": call.call_id,
                "name": call.name,
                "arguments": call.arguments_json,
            }
            for call in calls
        ),
        function_calls=tuple(calls),
        usage=None,
    )


def _registry() -> ToolRegistry:
    def definition(name: str) -> ToolDefinition:
        return ToolDefinition(
            schema=FunctionTool(
                name=name,
                description=f"Test {name}.",
                parameters={
                    "type": "object",
                    "properties": {"path": {"type": "string"}},
                    "required": ["path"],
                    "additionalProperties": False,
                },
            ),
            handler=lambda arguments: arguments["path"],
        )

    return ToolRegistry(
        definition(name)
        for name in ("read_file", "apply_patch", "create_file", "run_command")
    )


def test_auto_mode_distinguishes_questions_from_mutations() -> None:
    assert resolve_task_mode("这个项目怎么运行，有什么功能？") is TaskMode.ASK
    assert resolve_task_mode("这个项目能实现什么功能，有哪些改进点？") is TaskMode.ASK
    assert resolve_task_mode("如何实现缓存？") is TaskMode.ASK
    assert resolve_task_mode("说明改进点，但不要修改代码") is TaskMode.ASK
    assert resolve_task_mode("review it without changing files") is TaskMode.ASK
    assert resolve_task_mode("inspect and explain this repository") is TaskMode.ASK
    assert resolve_task_mode("修复 runner.py 的反向移动 bug") is TaskMode.CODE
    assert resolve_task_mode("add a regression test") is TaskMode.CODE
    assert resolve_task_mode("写一个可玩的贪吃蛇游戏") is TaskMode.CODE
    assert resolve_task_mode("怎么写一个可玩的贪吃蛇游戏？") is TaskMode.ASK


def test_ask_mode_exposes_no_edit_or_plan_tools(tmp_path: Path) -> None:
    call = FunctionCall("read", "read_file", json.dumps({"path": "README.md"}))
    model = ScriptedModel(
        [_response("read", calls=(call,)), _response("final", text="运行说明")]
    )

    state = AgentLoop(model, _registry(), mode="ask").run(
        "怎么运行？", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.mode is TaskMode.ASK
    names = {tool.name for tool in model.requests[0].tools}
    assert names == {"read_file", "run_command"}
    assert "update_plan" not in names
    assert "apply_patch" not in names


def test_code_mode_keeps_editing_tools_and_uses_runtime_plan_when_applicable(
    tmp_path: Path,
) -> None:
    model = ScriptedModel([_response("final", text="No change needed")])

    state = AgentLoop(model, _registry(), mode="code").run(
        "修复问题", workspace=tmp_path
    )

    names = {tool.name for tool in model.requests[0].tools}
    assert state.mode is TaskMode.CODE
    assert {"apply_patch", "create_file"} <= names
    assert "update_plan" not in names
    assert state.plan is not None
    assert [step.step_id for step in state.plan.steps] == [
        "inspect",
        "implement",
        "verify",
        "deliver",
    ]


def test_code_mode_instructions_preserve_multifile_deliverables() -> None:
    instructions = instructions_for_mode(TaskMode.CODE)

    assert "Simplified Chinese" in instructions
    assert "multiple files, modules, or assets" in instructions
    assert "do not silently collapse a multi-file project" in instructions


def test_safe_last_turn_forces_text_instead_of_another_tool(tmp_path: Path) -> None:
    call = FunctionCall("read", "read_file", json.dumps({"path": "README.md"}))
    model = ScriptedModel(
        [_response("read", calls=(call,)), _response("final", text="结论")]
    )

    state = AgentLoop(model, _registry(), mode="ask", max_steps=2).run(
        "介绍项目", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.final_answer == "结论"
    assert model.requests[0].tool_choice == "auto"
    assert model.requests[1].tool_choice == "none"


def test_ask_mode_stops_after_repeating_the_same_read(tmp_path: Path) -> None:
    call_1 = FunctionCall("read1", "read_file", json.dumps({"path": "README.md"}))
    call_2 = FunctionCall("read2", "read_file", json.dumps({"path": "README.md"}))
    model = ScriptedModel(
        [
            _response("read1", calls=(call_1,)),
            _response("read2", calls=(call_2,)),
            _response("final", text="Enough evidence"),
        ]
    )

    state = AgentLoop(model, _registry(), mode="ask", max_steps=8).run(
        "介绍项目", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.step == 3
    assert model.requests[2].tool_choice == "none"


def test_every_turn_preserves_workspace_and_launch_directory_facts(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project"
    launch_directory = workspace / "src" / "package"
    launch_directory.mkdir(parents=True)
    model = ScriptedModel([_response("final", text="Use the documented root")])

    state = AgentLoop(model, _registry(), mode="ask").run(
        "How do I run this?",
        workspace=workspace,
        launch_directory=launch_directory,
    )

    snapshot = str(model.requests[0].input[1]["content"])
    assert f"Absolute workspace root: {workspace.resolve()}" in snapshot
    assert f"User launch directory: {launch_directory.resolve()}" in snapshot
    assert "For Python commands, use `python`" in snapshot
    assert "Do not replace it with a parent repository" in snapshot
    assert state.launch_directory == launch_directory.resolve()


def test_code_mode_redirects_repeated_reads_before_budget_exhaustion(
    tmp_path: Path,
) -> None:
    first = FunctionCall("read1", "read_file", json.dumps({"path": "README.md"}))
    repeated = FunctionCall(
        "read2", "read_file", json.dumps({"path": "README.md"})
    )
    model = ScriptedModel(
        [
            _response("read1", calls=(first,)),
            _response("read2", calls=(repeated,)),
            _response("final", text="Incomplete evidence reported honestly"),
        ]
    )

    AgentLoop(model, _registry(), mode="code", max_steps=3).run(
        "Check the implementation",
        workspace=tmp_path,
    )

    assert "hard step budget is nearly exhausted" in model.requests[0].instructions
    last_input = json.dumps(model.requests[2].input, ensure_ascii=False)
    assert "Stop rereading unchanged files" in last_input


def test_continuation_with_stale_evidence_cannot_claim_completion(
    tmp_path: Path,
) -> None:
    model = ScriptedModel([_response("final", text="Done")])

    state = AgentLoop(
        model,
        _registry(),
        mode="code",
        max_steps=1,
        verification_required=True,
    ).run("continue", workspace=tmp_path)

    assert state.status is AgentStatus.MAX_STEPS
    assert state.final_answer is None
