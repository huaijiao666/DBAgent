import json
import os
import subprocess
from pathlib import Path

import pytest

from forge.llm import FunctionCall
from forge.tools import create_coding_registry


def _dispatch(registry, name: str, arguments: dict):
    return registry.dispatch(
        FunctionCall(
            call_id=f"call_{name}",
            name=name,
            arguments_json=json.dumps(arguments),
        )
    )


def test_coding_registry_exposes_only_current_tools(tmp_path: Path) -> None:
    registry = create_coding_registry(tmp_path)

    assert [schema.name for schema in registry.schemas()] == [
        "list_files",
        "read_file",
        "search_text",
        "get_repo_map",
        "search_symbol",
        "read_symbol",
        "apply_patch",
        "run_command",
        "create_file",
        "write_file",
        "git_diff",
    ]


def test_create_and_write_file(tmp_path: Path) -> None:
    registry = create_coding_registry(tmp_path)

    created = _dispatch(
        registry,
        "create_file",
        {"path": "new.py", "content": "value = 1\n"},
    )
    written = _dispatch(
        registry,
        "write_file",
        {"path": "new.py", "content": "value = 2\n"},
    )

    assert created.success is True
    assert written.success is True
    assert created.content["changed_files"] == ["new.py"]
    assert written.content["changed_files"] == ["new.py"]
    assert (tmp_path / "new.py").read_text(encoding="utf-8") == "value = 2\n"


def test_create_file_creates_missing_parent_directories(tmp_path: Path) -> None:
    registry = create_coding_registry(tmp_path)

    created = _dispatch(
        registry,
        "create_file",
        {"path": "src/package/module.py", "content": "value = 1\n"},
    )

    assert created.success is True
    assert created.content["changed_files"] == ["src/package/module.py"]
    assert (tmp_path / "src" / "package" / "module.py").read_text(
        encoding="utf-8"
    ) == "value = 1\n"


def test_create_file_does_not_overwrite_existing_file(tmp_path: Path) -> None:
    (tmp_path / "existing.txt").write_text("original", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "create_file",
        {"path": "existing.txt", "content": "replacement"},
    )

    assert observation.success is False
    assert observation.content.startswith("FileExistsError:")
    assert (tmp_path / "existing.txt").read_text(encoding="utf-8") == "original"


def test_write_file_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    registry = create_coding_registry(workspace)

    observation = _dispatch(
        registry,
        "write_file",
        {"path": "../outside.txt", "content": "changed"},
    )

    assert observation.success is False
    assert "path escapes workspace" in observation.content
    assert outside.read_text(encoding="utf-8") == "outside"


def test_external_file_symlink_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside", encoding="utf-8")
    link = workspace / "link.txt"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlinks unavailable: {error}")
    registry = create_coding_registry(workspace)

    read = _dispatch(registry, "read_file", {"path": "link.txt"})
    write = _dispatch(
        registry,
        "write_file",
        {"path": "link.txt", "content": "changed"},
    )

    assert read.success is False
    assert write.success is False
    assert "path escapes workspace" in read.content
    assert "path escapes workspace" in write.content
    assert outside.read_text(encoding="utf-8") == "outside"


def test_external_directory_symlink_cannot_be_used_for_create(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "linked-directory"
    try:
        link.symlink_to(outside, target_is_directory=True)
    except OSError as error:
        if os.name != "nt":
            pytest.skip(f"symlinks unavailable: {error}")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(link), str(outside)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
    registry = create_coding_registry(workspace)

    observation = _dispatch(
        registry,
        "create_file",
        {"path": "linked-directory/new.txt", "content": "blocked"},
    )

    assert observation.success is False
    assert "path escapes workspace" in observation.content
    assert not (outside / "new.txt").exists()


def test_local_environment_file_cannot_be_written(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text("TOKEN=original", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "write_file",
        {"path": ".env", "content": "TOKEN=changed"},
    )

    assert observation.success is False
    assert observation.content.startswith("PermissionError:")
    assert (tmp_path / ".env").read_text(encoding="utf-8") == "TOKEN=original"


def test_run_command_returns_structured_result(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "run_command",
        {
            "command": ["python", "check.py"],
            "cwd": ".",
            "timeout_seconds": 10,
        },
    )

    assert observation.success is True
    assert observation.content["return_code"] == 0
    assert observation.content["stdout"] == "ok\n"
    assert observation.content["stderr"] == ""


def test_run_command_accepts_a_string_without_enabling_a_shell(tmp_path: Path) -> None:
    (tmp_path / "check.py").write_text("print('ok')\n", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "run_command",
        {"command": "python check.py", "cwd": ".", "timeout_seconds": 10},
    )

    assert observation.success is True
    assert observation.content["command"][-1] == "check.py"
    assert observation.content["stdout"] == "ok\n"


def test_apply_patch_failure_is_a_structured_tool_observation(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("current\n", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "apply_patch",
        {
            "files": [
                {
                    "path": "target.txt",
                    "hunks": [
                        {
                            "old_lines": ["stale context"],
                            "new_lines": ["replacement"],
                        }
                    ],
                }
            ]
        },
    )

    assert observation.success is False
    assert observation.content["applied"] is False
    assert observation.content["changed_files"] == []
    assert observation.content["hunks_applied"] == 0
    assert "context did not match" in observation.content["failure_reason"]
    assert target.read_text(encoding="utf-8") == "current\n"


def test_apply_patch_normalizes_one_terminal_line_ending_from_provider(
    tmp_path: Path,
) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "apply_patch",
        {
            "files": [
                {
                    "path": "target.txt",
                    "hunks": [
                        {"old_lines": ["before\n"], "new_lines": ["after\r\n"]}
                    ],
                }
            ]
        },
    )

    assert observation.success is True
    assert target.read_text(encoding="utf-8") == "after\n"


def test_apply_patch_rejects_embedded_line_endings(tmp_path: Path) -> None:
    target = tmp_path / "target.txt"
    target.write_text("before\n", encoding="utf-8")
    registry = create_coding_registry(tmp_path)

    observation = _dispatch(
        registry,
        "apply_patch",
        {
            "files": [
                {
                    "path": "target.txt",
                    "hunks": [
                        {"old_lines": ["before\nextra"], "new_lines": ["after"]}
                    ],
                }
            ]
        },
    )

    assert observation.success is False
    assert "embedded line endings" in observation.content["failure_reason"]
    assert target.read_text(encoding="utf-8") == "before\n"


def test_git_diff_uses_fixed_read_only_commands(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("before\n", encoding="utf-8")
    subprocess.run(["git", "add", "tracked.txt"], cwd=tmp_path, check=True)
    registry = create_coding_registry(tmp_path)

    patch = _dispatch(
        registry,
        "apply_patch",
        {
            "files": [
                {
                    "path": "tracked.txt",
                    "hunks": [
                        {"old_lines": ["before"], "new_lines": ["after"]}
                    ],
                }
            ]
        },
    )

    observation = _dispatch(registry, "git_diff", {})

    assert patch.success is True
    assert observation.success is True
    assert observation.content["status"]["return_code"] == 0
    assert observation.content["diff"]["return_code"] == 0
    assert "+after" in observation.content["diff"]["stdout"]
