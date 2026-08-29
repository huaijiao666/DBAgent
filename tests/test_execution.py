import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from forge.execution import CommandExecutor
from forge.workspace import Workspace


def _write_script(workspace: Path, name: str, source: str) -> None:
    (workspace / name).write_text(source, encoding="utf-8")


def test_command_captures_stdout_stderr_and_return_code(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "report.py",
        "import sys\nprint('standard output')\nprint('standard error', file=sys.stderr)\nraise SystemExit(3)\n",
    )
    executor = CommandExecutor(Workspace(tmp_path))

    result = executor.run(
        ["python", "report.py"],
        cwd=".",
        timeout_seconds=10,
    )

    assert result.return_code == 3
    assert result.timed_out is False
    assert result.stdout == "standard output\n"
    assert result.stderr == "standard error\n"


def test_command_uses_the_active_python_environment(tmp_path: Path) -> None:
    executor = CommandExecutor(Workspace(tmp_path))

    result = executor.run(
        ["python", "-c", "import sys; print(sys.executable)"],
        cwd=".",
        timeout_seconds=10,
    )

    assert result.return_code == 0
    assert Path(result.stdout.strip()).parent == Path(sys.executable).parent


def test_command_cwd_cannot_escape_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    executor = CommandExecutor(Workspace(workspace))

    with pytest.raises(ValueError, match="path escapes workspace"):
        executor.run(["python", "anything.py"], cwd="..", timeout_seconds=1)


def test_command_cwd_symlink_escape_is_rejected(tmp_path: Path) -> None:
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
    executor = CommandExecutor(Workspace(workspace))

    with pytest.raises(ValueError, match="path escapes workspace"):
        executor.run(["python", "anything.py"], cwd="linked-directory")


def test_command_timeout_is_reported(tmp_path: Path) -> None:
    _write_script(tmp_path, "slow.py", "import time\ntime.sleep(10)\n")
    executor = CommandExecutor(Workspace(tmp_path))

    result = executor.run(
        ["python", "slow.py"],
        cwd=".",
        timeout_seconds=0.2,
    )

    assert result.timed_out is True
    assert result.return_code is None


def test_command_streams_are_bounded_and_marked(tmp_path: Path) -> None:
    _write_script(
        tmp_path,
        "large_output.py",
        "import sys\nprint('o' * 1000)\nprint('e' * 1000, file=sys.stderr)\n",
    )
    executor = CommandExecutor(Workspace(tmp_path), max_stream_characters=100)

    result = executor.run(
        ["python", "large_output.py"],
        cwd=".",
        timeout_seconds=10,
    )

    assert result.stdout_truncated is True
    assert result.stderr_truncated is True
    assert len(result.stdout) <= 100
    assert len(result.stderr) <= 100
    assert "[truncated]" in result.stdout
    assert "[truncated]" in result.stderr


def test_sensitive_environment_variables_are_not_inherited(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-reach-child")
    monkeypatch.setenv("PRIVATE_ACCESS_TOKEN", "must-not-reach-child")
    _write_script(
        tmp_path,
        "environment.py",
        "import json, os\n"
        "print(json.dumps({"
        "'api_key': os.getenv('OPENAI_API_KEY'), "
        "'token': os.getenv('PRIVATE_ACCESS_TOKEN')}))\n",
    )
    executor = CommandExecutor(Workspace(tmp_path))

    result = executor.run(
        ["python", "environment.py"],
        cwd=".",
        timeout_seconds=10,
    )

    assert json.loads(result.stdout) == {"api_key": None, "token": None}


@pytest.mark.parametrize(
    "command",
    [
        ["powershell", "-Command", "Get-ChildItem"],
        ["git", "commit", "-m", "not allowed"],
        ["rm", "-rf", "target"],
    ],
)
def test_dangerous_commands_are_rejected(tmp_path: Path, command: list[str]) -> None:
    executor = CommandExecutor(Workspace(tmp_path))

    with pytest.raises(PermissionError, match="blocked by policy"):
        executor.run(command, cwd=".", timeout_seconds=1)


def test_absolute_argument_outside_workspace_is_rejected(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    outside = tmp_path / "outside.py"
    outside.write_text("print('outside')", encoding="utf-8")
    executor = CommandExecutor(Workspace(workspace))

    with pytest.raises(PermissionError, match="escapes workspace"):
        executor.run(
            ["python", str(outside)],
            cwd=".",
            timeout_seconds=1,
        )


def test_git_does_not_discover_a_parent_checkout_outside_workspace(
    tmp_path: Path,
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    (tmp_path / "outside.txt").write_text("outside\n", encoding="utf-8")
    subprocess.run(["git", "add", "outside.txt"], cwd=tmp_path, check=True)
    workspace = tmp_path / "nested-workspace"
    workspace.mkdir()

    result = CommandExecutor(Workspace(workspace)).run(
        ["git", "status", "--short"],
        cwd=".",
        timeout_seconds=10,
    )

    assert result.return_code != 0
    assert "not a git repository" in result.stderr.casefold()
    assert "outside.txt" not in result.stdout
