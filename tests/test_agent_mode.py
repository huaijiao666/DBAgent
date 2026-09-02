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


def test_auto_mode_defers_all_natural_language_interpretation_to_model() -> None:
    requests = [
        "这个项目怎么运行，有什么功能？",
        "说明改进点，但不要修改代码",
        "修复 runner.py 的反向移动 bug",
        "add a regression test",
        "怎么写一个可玩的贪吃蛇游戏？",
    ]

    assert all(resolve_task_mode(task) is TaskMode.AUTO for task in requests)


def test_auto_mode_uses_native_semantic_routing_before_planning(
    tmp_path: Path,
) -> None:
    plan = {
        "goal": "修复反向移动问题",
        "success_criteria": ["相关测试通过"],
        "steps": [
            {"id": "inspect", "description": "定位故障", "status": "in_progress"},
            {"id": "fix", "description": "修改实现", "status": "pending"},
            {"id": "verify", "description": "运行测试", "status": "pending"},
            {"id": "deliver", "description": "总结结果", "status": "pending"},
        ],
    }
    model = ScriptedModel(
        [
            _response(
                "route",
                calls=(
                    FunctionCall(
                        "route",
                        "select_task_mode",
                        json.dumps({"mode": "code", "reason": "用户要求修复本地代码"}),
                    ),
                ),
            ),
            _response(
                "plan",
                calls=(FunctionCall("plan", "update_plan", json.dumps(plan)),),
            ),
        ]
    )

    state = AgentLoop(model, _registry(), max_steps=2).run(
        "修复 runner.py 的反向移动 bug", workspace=tmp_path
    )

    assert state.mode is TaskMode.CODE
    assert state.plan is not None
    assert {tool.name for tool in model.requests[0].tools} == {"select_task_mode"}
    assert {tool.name for tool in model.requests[1].tools} == {"update_plan"}
    assert all(request.tool_choice == "required" for request in model.requests)


def test_auto_mode_can_semantically_select_read_only_tools(tmp_path: Path) -> None:
    model = ScriptedModel(
        [
            _response(
                "route",
                calls=(
                    FunctionCall(
                        "route",
                        "select_task_mode",
                        json.dumps({"mode": "ask", "reason": "用户仅要求运行说明"}),
                    ),
                ),
            ),
            _response("final", text="请在项目根目录运行启动命令。"),
        ]
    )

    state = AgentLoop(model, _registry(), max_steps=2).run(
        "这个项目怎么运行？", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.mode is TaskMode.ASK
    assert {tool.name for tool in model.requests[0].tools} == {"select_task_mode"}
    assert "apply_patch" not in {tool.name for tool in model.requests[1].tools}


def test_auto_mode_safely_falls_back_to_ask_after_two_invalid_routes(
    tmp_path: Path,
) -> None:
    model = ScriptedModel(
        [
            _response("prose_1", text="I will inspect the task."),
            _response("prose_2", text="I will inspect the task."),
            _response("final", text="请明确使用 code 模式以修改文件。"),
        ]
    )

    state = AgentLoop(model, _registry(), max_steps=3).run(
        "请解释这个项目", workspace=tmp_path
    )

    assert state.status is AgentStatus.COMPLETED
    assert state.mode is TaskMode.ASK
    assert all(
        {tool.name for tool in request.tools} == {"select_task_mode"}
        for request in model.requests[:2]
    )
    assert "apply_patch" not in {tool.name for tool in model.requests[2].tools}


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


def test_code_mode_requires_a_model_authored_semantic_plan_before_editing(
    tmp_path: Path,
) -> None:
    plan = {
        "goal": "修复问题",
        "success_criteria": ["修复后运行相关检查"],
        "steps": [
            {"id": "inspect", "description": "定位故障原因", "status": "in_progress"},
            {"id": "fix", "description": "修改受影响实现", "status": "pending"},
            {"id": "verify", "description": "运行针对性检查", "status": "pending"},
            {"id": "deliver", "description": "总结证据", "status": "pending"},
        ],
    }
    model = ScriptedModel(
        [
            _response(
                "semantic_plan",
                calls=(FunctionCall("plan", "update_plan", json.dumps(plan)),),
            )
        ]
    )

    state = AgentLoop(model, _registry(), mode="code", max_steps=1).run(
        "修复问题", workspace=tmp_path
    )

    names = {tool.name for tool in model.requests[0].tools}
    assert state.mode is TaskMode.CODE
    assert names == {"update_plan"}
    assert model.requests[0].tool_choice == "required"
    assert state.plan is not None
    assert state.plan.goal == "修复问题"
    assert [step.step_id for step in state.plan.steps] == [
        "inspect", "fix", "verify", "deliver"
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
