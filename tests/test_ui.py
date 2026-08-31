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
    assert "DBAgent coding session" in output
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

    assert "分析中" in request
    assert "context=321/?~tok" in request
    assert "VERIFY" in verification
    assert "status=passed" in verification
    assert "\x1b[" not in request


def test_terminal_ui_wraps_important_status_instead_of_truncating() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)

    ui.info("session=abc; " + "verification details " * 10)

    output = stream.getvalue()
    assert "session=abc" in output
    assert " ".join(output.split()).count("verification details") == 10
    assert "..." not in output


def test_terminal_ui_makes_new_and_resumed_context_explicit(tmp_path: Path) -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)
    ui.session_start(
        workspace=tmp_path,
        model="model",
        api_mode="responses",
        session_id="20260831-120000-abcdef",
        session_state="new",
    )
    ui.render_resume_summary(
        session_id="20260830-120000-fedcba",
        title="Fix parser",
        turns=3,
        verification="passed",
        observation_count=5,
        has_plan=True,
        checkpoint_state="interrupted",
    )

    output = stream.getvalue()
    assert "20260831-120000-abcdef [new]" in output
    assert "Resumed context" in output
    assert "20260830-120000-fedcba" in output
    assert "Plan restored true" in output
    assert "Checkpoint    interrupted" in output


def test_terminal_ui_explains_tool_actions_in_human_terms() -> None:
    ui = TerminalUI(stream=io.StringIO(), color=False)
    ui.start(task="task", workspace=Path("."), model="model", max_steps=3)

    read = ui.render_event(
        {
            "elapsed_ms": 100,
            "step": 1,
            "event": "tool_start",
            "payload": {"tool_name": "read_file", "target": "README.md"},
        }
    )
    command = ui.render_event(
        {
            "elapsed_ms": 200,
            "step": 1,
            "event": "tool_start",
            "payload": {
                "tool_name": "run_command",
                "command": ["python.exe", "-m", "pytest", "-q"],
            },
        }
    )
    search = ui.render_event(
        {
            "elapsed_ms": 300,
            "step": 1,
            "event": "tool_start",
            "payload": {
                "tool_name": "search_text",
                "path": ".",
                "query": "priority",
                "target": ".",
            },
        }
    )

    assert "读取文件  README.md" in read
    assert "运行命令  python.exe -m pytest -q" in command
    assert "搜索文本  priority  in ." in search


def test_terminal_ui_shows_patch_failure_reason() -> None:
    ui = TerminalUI(stream=io.StringIO(), color=False)
    ui.start(task="task", workspace=Path("."), model="model", max_steps=3)

    rendered = ui.render_event(
        {
            "elapsed_ms": 400,
            "step": 2,
            "event": "tool_result",
            "payload": {
                "tool_name": "apply_patch",
                "success": False,
                "failure_reason": "PatchError: app.py hunk 1: context did not match",
            },
        }
    )

    assert "失败: 应用补丁" in rendered
    assert "context did not match" in rendered


def test_terminal_ui_shows_non_patch_tool_failure_reason() -> None:
    ui = TerminalUI(stream=io.StringIO(), color=False)
    ui.start(task="task", workspace=Path("."), model="model", max_steps=3)

    rendered = ui.render_event(
        {
            "elapsed_ms": 400,
            "step": 2,
            "event": "tool_result",
            "payload": {
                "tool_name": "create_file",
                "success": False,
                "failure_reason": "FileNotFoundError: parent directory does not exist",
            },
        }
    )

    assert "失败: 创建文件" in rendered
    assert "parent directory does not exist" in rendered


def test_terminal_ui_shows_context_compaction_and_plan_progress() -> None:
    ui = TerminalUI(stream=io.StringIO(), color=False)
    ui.start(task="task", workspace=Path("."), model="model", max_steps=8)

    compacted = ui.render_event(
        {
            "elapsed_ms": 500,
            "step": 4,
            "event": "context_compacted",
            "payload": {
                "compacted_observations": 7,
                "recent_observations": 4,
                "truncated_items": 2,
                "approximate_tokens": 8000,
            },
        }
    )
    plan = ui.render_event(
        {
            "elapsed_ms": 600,
            "step": 4,
            "event": "plan_updated",
            "payload": {
                "completed_steps": 1,
                "total_steps": 3,
                "current_step": "implement",
                "current_step_description": "Implement parser changes",
            },
        }
    )

    assert "上下文摘要" in compacted
    assert "older=7" in compacted
    assert "计划 1/3" in plan
    assert "Implement parser changes" in plan


def test_terminal_ui_renders_only_latest_plan_snapshot() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)
    first = SimpleNamespace(
        goal="Explain project",
        steps=[SimpleNamespace(step_id="inspect", description="Inspect", status="in_progress")],
    )
    latest = SimpleNamespace(
        goal="Explain project",
        steps=[SimpleNamespace(step_id="inspect", description="Inspect", status="completed")],
    )

    ui.render_plan_history([first, latest])

    output = stream.getvalue()
    assert output.count("Explain project") == 1
    assert "Inspect [completed]" in output


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


def test_terminal_ui_lists_files_created_in_a_non_git_workspace() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)

    ui.finish(
        SimpleNamespace(
            status=AgentStatus.COMPLETED,
            is_verified=True,
            verification_status=VerificationStatus.PASSED,
            step=3,
            max_steps=12,
            observations=[
                SimpleNamespace(
                    content={
                        "action": "created",
                        "path": "hello.py",
                        "changed_files": ["hello.py"],
                    }
                )
            ],
        )
    )

    assert "Files changed  hello.py" in stream.getvalue()


def test_terminal_ui_lists_structured_apply_patch_files() -> None:
    stream = io.StringIO()
    ui = TerminalUI(stream=stream, color=False)

    ui.finish(
        SimpleNamespace(
            status=AgentStatus.COMPLETED,
            is_verified=True,
            verification_status=VerificationStatus.PASSED,
            step=4,
            max_steps=12,
            observations=[
                SimpleNamespace(
                    content={
                        "applied": True,
                        "changed_files": [
                            {
                                "path": "calculator.py",
                                "before_sha256": "before",
                                "after_sha256": "after",
                            }
                        ],
                    }
                )
            ],
        )
    )

    assert "Files changed  calculator.py" in stream.getvalue()
