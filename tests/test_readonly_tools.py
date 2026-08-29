import json
from pathlib import Path

from forge.llm import FunctionCall
from forge.tools import create_readonly_registry


def _dispatch(registry, name: str, arguments: dict[str, str]):
    return registry.dispatch(
        FunctionCall(
            call_id=f"call_{name}",
            name=name,
            arguments_json=json.dumps(arguments),
        )
    )


def test_readonly_tools_list_read_and_search(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    source = workspace / "src"
    source.mkdir(parents=True)
    (source / "app.py").write_text("def Hello():\n    return 'world'\n", encoding="utf-8")
    (workspace / "README.md").write_text("Architecture notes\n", encoding="utf-8")
    (workspace / ".env").write_text("OPENAI_API_KEY=not-for-tools", encoding="utf-8")
    (workspace / ".env.example").write_text("OPENAI_API_KEY=", encoding="utf-8")
    ignored = workspace / ".git"
    ignored.mkdir()
    (ignored / "config").write_text("secret-ish metadata", encoding="utf-8")
    registry = create_readonly_registry(workspace)

    listing = _dispatch(registry, "list_files", {"path": "."})
    read = _dispatch(registry, "read_file", {"path": "src/app.py"})
    search = _dispatch(
        registry,
        "search_text",
        {"query": "architecture", "path": "."},
    )

    assert listing.success is True
    assert listing.content.splitlines() == [
        ".env.example",
        "README.md",
        "src/app.py",
    ]
    assert read.success is True
    assert "1: def Hello():" in read.content
    assert search.success is True
    assert search.content == "README.md:1: Architecture notes"


def test_read_file_rejects_workspace_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (tmp_path / "outside.txt").write_text("outside", encoding="utf-8")
    registry = create_readonly_registry(workspace)

    observation = _dispatch(
        registry,
        "read_file",
        {"path": "../outside.txt"},
    )

    assert observation.success is False
    assert "path escapes workspace" in observation.content


def test_read_file_reports_missing_file_as_observation(tmp_path: Path) -> None:
    registry = create_readonly_registry(tmp_path)

    observation = _dispatch(
        registry,
        "read_file",
        {"path": "missing.txt"},
    )

    assert observation.success is False
    assert observation.content.startswith("FileNotFoundError:")


def test_local_environment_files_are_blocked(tmp_path: Path) -> None:
    (tmp_path / ".env.local").write_text("TOKEN=hidden", encoding="utf-8")
    registry = create_readonly_registry(tmp_path)

    observation = _dispatch(
        registry,
        "read_file",
        {"path": ".env.local"},
    )

    assert observation.success is False
    assert observation.content == (
        "PermissionError: access to local environment files is blocked"
    )
