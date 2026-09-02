import json
import threading
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import pytest

from forge.config import ForgeConfig
from forge.agent import AgentRunControl, SessionContext
from forge.web_ui import (
    _ASSET_DIRECTORY,
    _browser_continuation_context,
    BrowserAgentController,
    BrowserAgentServer,
    _WebTraceRenderer,
)


@pytest.fixture
def controller(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> BrowserAgentController:
    monkeypatch.setattr(
        "forge.web_ui.load_repl_config",
        lambda _path: ForgeConfig(openai_api_key="synthetic-test-key"),
    )
    return BrowserAgentController(tmp_path)


def test_browser_controller_selects_existing_local_workspace_and_lists_tree(
    controller: BrowserAgentController,
    tmp_path: Path,
) -> None:
    nested = tmp_path / "project"
    nested.mkdir()
    (nested / "app.py").write_text("print('ok')\n", encoding="utf-8")
    (nested / ".forge").mkdir()
    (nested / ".forge" / "hidden.json").write_text("{}", encoding="utf-8")

    status = controller.select_workspace(str(nested))
    tree = controller.tree()

    assert status["workspace"] == str(nested.resolve())
    assert tree["entries"] == [{"path": "app.py", "kind": "file"}]
    assert "synthetic-test-key" not in json.dumps(status)
    assert controller.sessions()["active_session_id"] == status["session"]["id"]


def test_browser_controller_reads_only_a_canonical_in_workspace_file(
    controller: BrowserAgentController,
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.py"
    source.write_text("def answer():\n    return 42\n", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=nope\n", encoding="utf-8")

    preview = controller.read_file("source.py")

    assert preview["path"] == "source.py"
    assert preview["language"] == "python"
    assert "return 42" in preview["content"]
    with pytest.raises(PermissionError):
        controller.read_file(".env")
    with pytest.raises(ValueError, match="escapes workspace"):
        controller.read_file("../outside.py")


@pytest.mark.parametrize("name", ["api_key.txt", "config.toml"])
def test_browser_controller_hides_local_credential_files(
    controller: BrowserAgentController, tmp_path: Path, name: str
) -> None:
    (tmp_path / name).write_text("secret-value", encoding="utf-8")

    tree = controller.tree()

    assert name not in {entry["path"] for entry in tree["entries"]}
    with pytest.raises(PermissionError, match="credential files"):
        controller.read_file(name)


def test_browser_controller_keeps_trace_plan_until_a_workspace_change(
    controller: BrowserAgentController,
    tmp_path: Path,
) -> None:
    plan = {
        "goal": "make the test pass",
        "success_criteria": ["pytest passes"],
        "steps": [{"step_id": "inspect", "description": "read code", "status": "completed"}],
    }
    controller._ingest_trace(
        {"step": 2, "elapsed_ms": 15, "event": "plan_updated", "payload": {"plan": plan}}
    )

    assert controller.status()["plan"] == plan
    controller.select_workspace(str(tmp_path))
    assert controller.status()["plan"] is None


def test_browser_controller_lists_and_restores_a_local_session(
    controller: BrowserAgentController,
) -> None:
    session_id = controller._session_store.save(
        {
            "title": "Repair calculator",
            "conversation": [{"role": "user", "content": "Fix the failing test"}],
            "session_context": {
                "plan": None,
                "verification_status": "failed",
                "verification_summary": "pytest failed",
                "recovery_hints": [],
            "observations": [
                {
                    "turn": 1,
                    "tool_name": "apply_patch",
                    "success": True,
                    "summary": json.dumps({"changed_files": ["calculator.py"]}),
                    "important": True,
                },
                {
                    "turn": 1,
                    "tool_name": "run_command",
                    "success": False,
                    "summary": json.dumps({"return_code": 1}),
                    "important": True,
                },
            ],
                "turns": 1,
            },
        }
    )

    listing = controller.sessions()
    status = controller.resume_session(session_id)

    assert listing["sessions"][0]["session_id"] == session_id
    assert status["session"]["id"] == session_id
    assert status["session"]["title"] == "Repair calculator"
    assert status["session"]["verification"] == "failed"
    assert controller._conversation == [{"role": "user", "content": "Fix the failing test"}]
    assert status["history"]["conversation"] == controller._conversation
    assert [item["title"] for item in status["history"]["execution"]] == [
        "已修改项目文件",
        "本地检查未通过",
        "已恢复最新验证",
    ]


def test_browser_controller_starts_an_explicit_empty_session(
    controller: BrowserAgentController,
) -> None:
    old_session = controller.status()["session"]["id"]
    controller._last_plan = {"goal": "old", "steps": []}
    controller._conversation = [{"role": "user", "content": "old task"}]

    status = controller.new_session()

    assert status["session"]["id"] != old_session
    assert status["plan"] is None
    assert controller._conversation == []
    assert controller.events_after(0)[-1]["event"] == "session_created"
    assert controller.sessions()["sessions"][0]["session_id"] == status["session"]["id"]


def test_browser_controller_uses_native_picker_without_browser_file_access(
    controller: BrowserAgentController,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chosen = tmp_path / "chosen"
    chosen.mkdir()
    monkeypatch.setattr("forge.web_ui._choose_local_directory", lambda _initial: str(chosen))

    result = controller.choose_workspace()

    assert result["selected"] is True
    assert result["status"]["workspace"] == str(chosen.resolve())


def test_browser_controller_rejects_missing_or_midrun_workspace(
    controller: BrowserAgentController,
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="existing directory"):
        controller.select_workspace(str(tmp_path / "missing"))
    controller._active = True
    with pytest.raises(RuntimeError, match="active"):
        controller.select_workspace(str(tmp_path))


def test_browser_controller_tracks_live_metrics_and_supports_cooperative_abort(
    controller: BrowserAgentController,
) -> None:
    controller._ingest_trace(
        {
            "step": 3,
            "elapsed_ms": 1250,
            "event": "tool_start",
            "payload": {"tool_name": "run_command"},
        }
    )
    controller._ingest_trace(
        {
            "step": 3,
            "elapsed_ms": 1600,
            "event": "model_request",
            "payload": {"context_usage": {"approximate_tokens": 420}},
        }
    )
    controller._active = True
    controller._run_id = "test-run"
    controller._run_control = AgentRunControl()

    status = controller.abort_task()

    assert status["current_step"] == 3
    assert status["current_tool"] == "run_command"
    assert status["context_usage"]["approximate_tokens"] == 420
    assert controller._run_control is not None
    assert controller._run_control.abort_requested
    assert controller.events_after(0)[-1]["event"] == "run_abort_requested"


def test_browser_controller_queues_live_steering_at_a_safe_boundary(
    controller: BrowserAgentController,
) -> None:
    controller._active = True
    controller._run_id = "test-run"
    controller._run_control = AgentRunControl()

    status = controller.steer_task("Do not add new dependencies.")

    assert controller._run_control is not None
    assert status["steering_mode"] == "queued"
    assert controller._run_control.drain_steering() == ("Do not add new dependencies.",)
    event = controller.events_after(0)[-1]
    assert event["event"] == "user_steering_queued"
    assert "Do not add" in event["payload"]["message"]


def test_browser_controller_turns_idle_guidance_into_a_same_session_follow_up(
    controller: BrowserAgentController,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    received: list[dict[str, object]] = []

    def fake_start(data: dict[str, object]) -> dict[str, object]:
        received.append(data)
        return {"active": True, "session": {"id": "same-session"}}

    monkeypatch.setattr(controller, "start_task", fake_start)

    status = controller.steer_task("继续完成淡色界面。")

    assert received == [{"task": "继续完成淡色界面。", "mode": "auto"}]
    assert status["steering_mode"] == "follow_up"
    assert controller.events_after(0)[-1]["event"] == "user_follow_up_started"


def test_browser_continuation_context_does_not_contaminate_current_task() -> None:
    context = SessionContext()
    context.verification_status = "failed"
    context.verification_summary = "previous pytest failure"
    rendered = _browser_continuation_context(
        context,
        [
            {"role": "user", "content": "Build the first version."},
            {"role": "assistant", "content": "The first version exists."},
            {"role": "user", "content": "Make the palette lighter."},
        ],
    )

    assert "[Structured local session state]" in rendered
    assert "Build the first version." in rendered
    assert "Make the palette lighter." not in rendered
    assert "[Current turn request]" not in rendered


def test_trace_renderer_publishes_from_worker_thread_without_lock_error(
    controller: BrowserAgentController,
) -> None:
    renderer = _WebTraceRenderer(controller)

    # TraceRecorder invokes consume_event on the AgentLoop worker thread, not
    # while the controller's request handler lock is held.
    renderer.consume_event(
        {
            "step": 1,
            "elapsed_ms": 25,
            "event": "tool_start",
            "payload": {"tool_name": "list_files"},
        }
    )
    renderer.consume_event(
        {
            "step": 2,
            "elapsed_ms": 45,
            "event": "tool_result",
            "payload": {
                "tool_name": "write_file",
                "success": True,
                "path": "src/example.py",
            },
        }
    )

    assert controller.status()["current_tool"] is None
    assert controller.status()["changed_files"] == ["src/example.py"]
    assert controller.events_after(0)[-1]["event"] == "trace"


def test_browser_controller_uses_trace_line_deltas_when_git_has_no_numstat(
    controller: BrowserAgentController,
) -> None:
    controller._ingest_trace(
        {
            "step": 2,
            "elapsed_ms": 45,
            "event": "tool_result",
            "payload": {
                "tool_name": "create_file",
                "success": True,
                "path": "game.py",
                "line_changes": [{"path": "game.py", "added": 18, "deleted": 0}],
            },
        }
    )

    changes = controller.changes()

    assert changes["summary"] == {"files": 1, "added": 18, "deleted": 0}
    assert changes["files"][0]["path"] == "game.py"


def test_loopback_server_requires_token_and_serves_the_browser_surface(
    controller: BrowserAgentController,
) -> None:
    server = BrowserAgentServer(controller)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with urlopen(server.url, timeout=3) as response:
            html = response.read().decode("utf-8")
        assert "DBAgent" in html
        with pytest.raises(HTTPError) as caught:
            urlopen(f"http://{server.server_address[0]}:{server.server_address[1]}/api/status", timeout=3)
        assert caught.value.code == 403
        protected_file = Request(
            f"http://{server.server_address[0]}:{server.server_address[1]}/api/file?path=.env",
            headers={"X-DBAgent-Token": server.token},
        )
        with pytest.raises(HTTPError) as caught:
            urlopen(protected_file, timeout=3)
        assert caught.value.code == 400
        inactive_control = Request(
            f"http://{server.server_address[0]}:{server.server_address[1]}/api/control",
            data=json.dumps({"action": "abort"}).encode("utf-8"),
            headers={"X-DBAgent-Token": server.token, "Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(HTTPError) as caught:
            urlopen(inactive_control, timeout=3)
        assert caught.value.code == 400
        request = Request(
            f"http://{server.server_address[0]}:{server.server_address[1]}/api/status",
            headers={"X-DBAgent-Token": server.token},
        )
        with urlopen(request, timeout=3) as response:
            body = json.loads(response.read().decode("utf-8"))
        assert body["workspace"] == str(controller.workspace)
        assert "synthetic-test-key" not in json.dumps(body)
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_browser_surface_keeps_scrollable_regions_and_resize_handles() -> None:
    html = (_ASSET_DIRECTORY / "index.html").read_text(encoding="utf-8")
    css = (_ASSET_DIRECTORY / "app.css").read_text(encoding="utf-8")
    javascript = (_ASSET_DIRECTORY / "app.js").read_text(encoding="utf-8")

    assert 'id="conversation-log"' in html
    assert 'id="left-resizer"' in html
    assert 'id="center-resizer"' in html
    assert 'id="right-resizer"' in html
    assert ".conversation {" in css and "overflow-y:auto" in css
    assert ".timeline-wrap {" in css and "overflow-y:auto" in css
    assert "startResize" in javascript and "localStorage" in javascript
    assert "pointercancel" in javascript and "--conversation-height" in javascript
    assert 'aria-label="调整仓库面板宽度"' in html
    assert 'id="steer"' in html
    assert 'data-tab="diff"' not in html
    assert "/api/diff" not in javascript
    assert 'id="repo-preview"' in html
    assert 'id="change-preview"' in html
    assert 'id="activity-details"' in html
    assert "/api/file" in javascript
    assert "/api/changes" in javascript
    assert "/api/sessions" in javascript
    assert "/api/workspace-picker" in javascript
    assert "renderMarkdown" in javascript
    assert ".show-repo-preview" in css
    assert ".show-change-preview" in css
    assert 'id="sessions-dialog"' in html
    assert 'id="choose-workspace"' in html
    assert 'id="file-reference-menu"' in html
    assert "repoPreview" in javascript and "changePreview" in javascript
    assert "conversationMilestone" in javascript
    assert 'appendSessionProcess("分析判断", payload.text, "thinking")' in javascript
    assert "restoreSessionHistory(next.history)" in javascript
    assert "applyFreshWorkspace" in javascript
    assert "已恢复的执行摘要" in javascript
    assert "scheduleWorkspaceRefresh" in javascript
    assert "isWorkspaceMutation" in javascript
    assert "collapseSessionProcess" in javascript
    assert "agent-turn" in javascript
    assert 'title === "正在理解任务" ? "initial"' in javascript
    assert "appendSessionStream" not in javascript
    assert "updateFileReferenceMenu" in javascript
    assert ".agent-turn" in css
    assert 'id="new-session"' in html
    assert "session-process-line small" in html
    assert "/api/sessions/new" in javascript
    assert '$("steer").disabled = false' in javascript
